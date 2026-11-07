
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / 'Scripts'))  # helper modules (run_config, obs, standardplots) live in LPNE/Scripts

from run_config import PROJECT  # project root (LPNE/), found relative to run_config.py - no hard-coded paths
import LPNE
import AREmulator
import run_config as rc
import statschecker as sc
import standardplots as sp
import obs as obs

import numpy as np
import numpy.random as npr
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K
import keras as ke
from tensorflow.keras import regularizers
import optuna as opt
import contextlib
import io
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from matplotlib.collections import LineCollection

STUDY_DIR = rc.RUNS_DIR / '_Ablations'   # default folder for the IC draws; each notebook passes its own as study_dir

def build_emu(cfg, params, weight_decay=1e-1):
    """
    untrained, compiled emulator for cfg (also used to reload saved weights without retraining)
    params       - model hyperparameters (usually cfg['params']; Optuna passes its own trial's)
    weight_decay - AdamW weight decay; None compiles with plain Adam (what MC_Rollouts and optimize_model use)
    """
    brain = LPNE.CNN_model(cfg['window_size'], params=params, n_features=rc.brain_inputs(cfg), n_out=rc.n_state(cfg))
    emu = AREmulator.RolloutEmulator(brain,
                                     rollout_length=cfg['rollout_length'],
                                     vali_rollout_length=cfg['rollout_length_validation'],
                                     window_size=cfg['window_size'],
                                     **rc.model_dims(cfg),   # n_state=3, n_forcing=1, state_names=['X', 'Y', 'Z']
                                     dwt_gamma=params['dwt gamma'],
                                     dwt_beta=params['dwt beta'],
                                     dwt_eps=params['dwt eps'],
                                     dwt_levels=params['dwt levels'],
                                     noise_level=cfg['noise_level'],
                                     )
    if weight_decay is None:
        optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning rate'], clipnorm=1.0)
    else:
        optimizer = tf.keras.optimizers.AdamW(learning_rate=params['learning rate'], weight_decay=weight_decay, clipnorm=1.0)
    emu.compile(optimizer=optimizer)
    _ = emu(tf.zeros((1, cfg['window_size'], rc.n_features(cfg))))
    return emu

def fit_emulator(params, data, cfg, acf_threshold, patience, weights_path, seed=None, quiet=False, weight_decay=1e-1):
    """
    Build, compile and train one RolloutEmulator on data (from LPNE.prepare_data, or the same dict layout).
    params - model hyperparameters
    data   - training data
    cfg    - config array
    
    seed   - None keeps the current random state (what the Optuna objective does, so every trial starts
             differently); an integer clears the Keras session and resets every seed first, so the same
             seed always gives the same run
    quiet  - suppress the per-epoch progress bars and checkpoint messages
    weight_decay - AdamW weight decay; None for plain Adam (see build_emu)

    Returns (model, history, phys_check). model holds the weights of the LAST epoch; the best physically-valid
    epoch is saved to weights_path, which only exists if at least one epoch passed the ACF gate.
    """
    if seed is not None:
        tf.keras.backend.clear_session()
        LPNE.set_seeds(seed)

    model = build_emu(cfg, params, weight_decay=weight_decay)

    # a weights file left over from an earlier run must never pass as this run's best epoch
    Path(weights_path).unlink(missing_ok=True)
    phys_check = sc.PhysicalAttractorCheckpoint(filepath=str(weights_path), acf_threshold=acf_threshold,
                                                patience=patience)

    with contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext():
        history = model.fit(
            *data['train'],
            validation_data=data['valid'],
            epochs=cfg['epochs'],
            callbacks=[phys_check],
            batch_size=cfg['batch_size'],
            validation_batch_size=cfg['batch_size'],
            shuffle=True,
            verbose=0 if quiet else 'auto',
        )
    return model, history, phys_check

def generate_IC(n_ic, params, train_start = 'mid', ic_randomstate = 42, study_dir = None):
    """
        n_ic GP draws of each state record (X, Y, Z in STATE_RECORDS order), in physical units: (n_ic, n_ages, 3).
        Draws of record k use ic_randomstate + k. They are saved to study_dir (None -> STUDY_DIR, i.e. _Ablations)
        the first time and loaded from there afterwards: obs.gp_reg fits each GP with random optimizer restarts, so
        refitting in a new session can give slightly different GPs, and with them different draws. Give each notebook
        its own study_dir so its draws (and the runs made from them) stay its own
    """
    study_dir = STUDY_DIR if study_dir is None else Path(study_dir)
    mc_ic_file = study_dir / f'mc_initial_conditions_{n_ic}.npz'
    _cfg_xyz = rc.make_config(rc.DATASET, seed=42, params=params, train_start=train_start)
    ic_ages = rc.split_data(rc.load_data(_cfg_xyz), _cfg_xyz)['time']  # ages [ka BP] of the record, oldest -> youngest (the row order of every setup's data)

    if mc_ic_file.exists():
        with np.load(mc_ic_file) as f:
            assert np.array_equal(f['ages'], ic_ages) and list(f['records']) == rc.STATE_RECORDS, \
                f"{mc_ic_file.name} was drawn for a different record; delete it to draw new initial conditions"
            ic_draws = f['draws']
        print(f"initial conditions loaded from {mc_ic_file}")
    else:
        gprs = {rec: obs.gp_reg(rec) for rec in rc.STATE_RECORDS}
        ic_draws = np.stack([gprs[rec].sample_y(ic_ages.reshape(-1, 1), n_ic, random_state=ic_randomstate + k)
                                for k, rec in enumerate(rc.STATE_RECORDS)], axis=-1).transpose(1, 0, 2)   # (N_IC, n_ages, 3)
        study_dir.mkdir(parents=True, exist_ok=True)
        np.savez(mc_ic_file, draws=ic_draws, ages=ic_ages, records=np.array(rc.STATE_RECORDS))
        print(f"initial conditions drawn and saved to {mc_ic_file}")
    print("IC_DRAWS (physical units, X, Y, Z in STATE_RECORDS order):", ic_draws.shape)

    return ic_draws

def make_ic(cfg, n_ic, params, scaler, splits, study_dir=None):
    """
        the shared initial conditions laid out for one setup, scaled with its scaler: (N_IC, n_ages, n_features)
        with the setup's state variables (taken from IC_DRAWS) and its own forcing columns (from its record). A
        rollout only uses each draw's first window_size rows of the states, and the forcing after that
    """
    ic_draws = generate_IC(n_ic, params, study_dir=study_dir)   # study_dir: folder of the saved draws (see generate_IC)
    S = rc.n_state(cfg)
    states = ic_draws[:, :, [rc.STATE_RECORDS.index(r) for r in rc.state_records(cfg)]]
    forcing = np.broadcast_to(splits['full'][None, :, S:], (n_ic,) + splits['full'][:, S:].shape)
    ic_raw = np.concatenate([states, forcing], axis=-1)
    return scaler.transform(ic_raw.reshape(-1, ic_raw.shape[-1])).reshape(ic_raw.shape)

### for paralellizing training process
def limit_threads(n_threads):
    """pool initializer: cap TensorFlow's threads in this worker process (runs before any training)"""
    tf.config.threading.set_intra_op_parallelism_threads(n_threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)

def train_mc_seed(dataset, cfg_kwargs, seed, runs_dir, params, patience=15, weight_decay=1e-1, retrain=False):
    """
    Train one Monte Carlo model in its own process. Everything it needs is passed in as plain data
    (no models or Keras objects), and it returns only small results.
    cfg_kwargs - the setup's make_config options, e.g. dict(train_start='mid', forcing=('ecc', 'prec', 'obl'))
    Returns (seed, best_epoch, history): best_epoch is None if no epoch passed the ACF gate (last epoch saved),
    'skipped' if the weights already existed
    """
    cfg = rc.make_config(dataset, seed=int(seed), params=params, **cfg_kwargs)
    paths = rc.run_paths(cfg['run_tag'], runs_dir)
    if paths['weights'].exists() and not retrain:
        return seed, 'skipped', None

    data = LPNE.prepare_data(cfg)   # cheap: rebuilt in the worker rather than sent over from the notebook
    rc.save_run(cfg, data['scaler'], data['splits'], runs_dir=runs_dir)
    model, history, check = fit_emulator(params, data, cfg, cfg['acf_threshold'], patience, paths['weights'],
                                         seed=int(seed), quiet=True, weight_decay=weight_decay)
    if paths['weights'].exists():
        best = check.best_epoch
    else:
        best = None
        model.save_weights(str(paths['weights']))
    return seed, best, history.history


### Monte Carlo ensembles (AblationTest): one model per seed, trained in parallel, each rolled out from every IC
def mc_paths(dataset, cfg_kwargs, seeds, runs_dir, params):
    """{seed: rc.run_paths(...)} - the run folder of each seed's model (where train_mc_seed saves it)"""
    return {s: rc.run_paths(rc.make_config(dataset, seed=int(s), params=params, **cfg_kwargs)['run_tag'], runs_dir)
            for s in seeds}

def train_mc(dataset, cfg_kwargs, seeds, runs_dir, params, n_workers=4, n_threads=2, retrain=False, patience=15,
             weight_decay=1e-1):
    """
    Train one model per seed in parallel: train_mc_seed in a pool of n_workers spawned processes, each limited to
    n_threads TensorFlow threads. Seeds whose weights already exist are skipped unless retrain, so an interrupted
    run resumes.
    Returns ({seed: best epoch, None (no epoch passed the ACF gate) or 'skipped'}, {seed: history dict or None})
    """
    Path(runs_dir).mkdir(parents=True, exist_ok=True)
    best_epoch, hist = {}, {}
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context('spawn'),
                             initializer=limit_threads, initargs=(n_threads,)) as pool:
        futures = [pool.submit(train_mc_seed, dataset, cfg_kwargs, s, runs_dir, params, patience=patience,
                               weight_decay=weight_decay, retrain=retrain) for s in seeds]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="training"):
            s, best, h = fut.result()     # re-raises here if that worker failed
            best_epoch[s], hist[s] = best, h
    best_epoch, hist = dict(sorted(best_epoch.items())), dict(sorted(hist.items()))

    no_gate = [s for s, b in best_epoch.items() if b is None]
    if no_gate:
        print("no epoch passed the ACF gate (last-epoch weights kept) for seeds:", no_gate)
    print("trained this session:", {s: b for s, b in best_epoch.items() if b != 'skipped'}
          or "none (all weights already existed)")
    return best_epoch, hist

def rollout_mc(cfg, params, paths, ic):
    """
    every seed's model rolled out from every initial condition -> (n_models, n_ic, n_ages, n_features), scaled
    (states + forcings). One model is built and reloaded with each seed's weights (paths from mc_paths);
    sp.rollout_ensemble advances all the ICs together. Returns (emulator, rollouts)
    """
    emu = build_emu(cfg, params)
    roll = np.empty((len(paths),) + ic.shape, dtype=np.float32)
    for m, p in enumerate(tqdm(paths.values(), desc="rollouts (per model)")):
        emu.load_weights(str(p['weights']))
        roll[m] = sp.rollout_ensemble(emu, ic, cfg['window_size'], progress=False)
    return emu, roll

def mc_l2(cfg, full, roll, seeds):
    """
    L2 against the ground truth full (scaled), per model, seed window excluded, for the whole record and the test set:
    the L2 of the model's mean rollout over its ICs, and for reference the mean and variance of the L2 of its
    individual rollouts. Returns (table: one row per model, summary: mean and variance over the models of the
    model-mean L2 - the numbers in plot_mc_rollouts' titles)
    """
    w = cfg['window_size']
    whole, test = slice(w, None), slice(w, cfg['test_size'])
    names = rc.state_names(cfg)
    mean_roll = roll.mean(axis=1)
    rows = {}
    for m, s in enumerate(seeds):
        row = {}
        for k, name in enumerate(names):
            for part, rows_ in (('whole', whole), ('test', test)):
                row[(name, f'L2 model mean ({part})')] = mean_squared_error(full[rows_, k], mean_roll[m, rows_, k])
                per_ic = ((roll[m, :, rows_, k] - full[None, rows_, k]) ** 2).mean(axis=1)   # L2 of each IC's rollout
                row[(name, f'L2 per IC mean ({part})')] = per_ic.mean()
                row[(name, f'L2 per IC var ({part})')] = per_ic.var()
        rows[f"seed {s}"] = row
    table = pd.DataFrame.from_dict(rows, orient='index')
    table.columns = pd.MultiIndex.from_tuples(table.columns)
    summary = pd.DataFrame({
        (name, part): {'mean': table[(name, f'L2 model mean ({part})')].mean(),
                       'var': table[(name, f'L2 model mean ({part})')].var()}
        for name in names for part in ('whole', 'test')})
    return table, summary

def _darker(color, f=0.35):
    c = np.array(plt.matplotlib.colors.to_rgb(color))
    return plt.matplotlib.colors.to_hex(c * (1 - f))

FORCING_COLORS = ['coral', 'seagreen', 'slateblue']

def plot_mc_rollouts(cfg, roll, full, fullt, summary, boundary, title=None, show_model_means=False):
    """
    Monte Carlo rollout plot: every model x IC rollout faint, the ground truth (GP mean) dashed, one panel per state
    variable of the setup plus a forcing panel (none if the setup has no forcing).
    roll     - (n_models, n_ic, n_ages, n_features) from rollout_mc;  full, fullt - scaled ground truth and its ages
    summary  - from mc_l2 (panel titles);  boundary - test | train age, marked in every panel
    show_model_means - also draw each model's mean rollout over its ICs
    """
    names, F = rc.state_names(cfg), rc.n_forcing(cfg)
    S = len(names)
    n_models, n_ic = roll.shape[:2]
    faint_alpha = float(np.clip(6 / (n_models * n_ic), 0.02, 0.3))   # fainter the more rollouts are drawn (0.02 for 3000)
    n_panels = S + (1 if F else 0)
    with plt.rc_context({'font.size': 16}):
        fig, axs = plt.subplots(n_panels, 1, figsize=(22, 3.9 * n_panels), sharex=True, squeeze=False)
        axs = axs[:, 0]

        for k, name in enumerate(names):
            c_truth, c_emu = sp.DEFAULT_COLORS['XYZ'.index(name)]   # the variable's own colors, also when Y or Z is dropped
            ax = axs[k]
            # all model x IC rollouts in one collection (thousands of lines, so not one plot call each)
            segs = np.stack([np.broadcast_to(fullt, roll.shape[2]), np.zeros(roll.shape[2])], axis=-1)
            segs = np.repeat(segs[None], n_models * n_ic, axis=0)
            segs[:, :, 1] = roll[:, :, :, k].reshape(-1, roll.shape[2])
            ax.add_collection(LineCollection(segs, colors=c_emu, linewidths=0.4, alpha=faint_alpha, zorder=1))
            ax.plot([], [], color=c_emu, lw=2, alpha=0.001, label=f"each rollout ({n_models} models x {n_ic} ICs)")
            if show_model_means:   # each model's mean rollout over its ICs
                ax.plot(fullt, roll[:, :, :, k].mean(axis=1).T, color=_darker(c_emu), lw=1.3, alpha=0.8, zorder=2)
                ax.plot([], [], color=_darker(c_emu), lw=2, label="each model's mean rollout")
            ax.plot(fullt, full[:, k], color=c_truth, linestyle='dashed', lw=2.2, zorder=3, label="GP mean (ground truth)")

            sw, st = summary[(name, 'whole')], summary[(name, 'test')]
            ax.set_title(r"Mean model $\mathcal{L}_{2}$: " + f"$\\mu={sw['mean']:.3f}$, $\\sigma^2={sw['var']:.4f}$"
                         + r"    (test set: " + f"$\\mu={st['mean']:.3f}$, $\\sigma^2={st['var']:.4f}$)")
            ax.set_ylabel(name)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left', fontsize=13, ncol=3)
            ax.autoscale_view()

        if F:   # forcing: given, not predicted (the same for every rollout)
            ax = axs[-1]
            fnames = rc.forcing_names(cfg)
            for j, fname in enumerate(fnames):
                ax.plot(fullt, full[:, S + j], color=FORCING_COLORS[j % len(FORCING_COLORS)], lw=1.5, label=fname)
            if F == 1:
                ax.set_title("Insolation forcing" if fnames[0] == 'insol' else f"{fnames[0]} forcing")
                ax.set_ylabel("Insolation (scaled)" if fnames[0] == 'insol' else "Forcing (scaled)")
            else:
                ax.set_title("Forcing: " + ", ".join(fnames))
                ax.set_ylabel("Forcing (scaled)")
                ax.legend(loc='upper left', fontsize=13, ncol=F)
            ax.grid(True, alpha=0.3)

        for ax in axs:
            ax.axvline(boundary, color='black', linestyle=':', lw=2)
        axs[0].annotate('test  |  train / validation', xy=(boundary, 1.0), xycoords=('data', 'axes fraction'),
                        xytext=(0, 26), textcoords='offset points', ha='center', va='bottom', fontsize=14)
        if title:
            fig.suptitle(title, fontweight='bold')
        axs[-1].invert_xaxis()
        axs[-1].set_xlabel("time [kyr]")
        plt.tight_layout()
        plt.show()
