"""
Shared run configuration + data preparation for the SR-LPNE notebooks
(DWT-LPNE.ipynb for training, MC_Rollouts_*.ipynb for evaluation).

Why this file exists
--------------------
Training and evaluation used to each load, order, split and scale the data on
their own, with slightly different conventions, and every training run wrote to
the same `best_state_space_model.weights.h5`. This module is the single source
of truth for all of that, so the two notebooks cannot drift apart:

  * `make_config(...)`  -> a dict describing one run (dataset, split, params,
                           seed, window/rollout lengths) + a deterministic
                           `run_tag` derived from those settings
  * `load_data(cfg)`    -> obs dataframe, ALWAYS ordered oldest -> youngest
  * `split_data(...)`   -> test / trainval / train / validation arrays
  * `fit_scaler(...)`   -> StandardScaler fit on the training portion only
  * `save_run(...)` / `load_run(...)` -> config + scaler stored next to the
                           weights, in  Scripts/runs/<run_tag>/

Each run lives in its own folder:
    Scripts/runs/<run_tag>/model.weights.h5
    Scripts/runs/<run_tag>/config.json      (all settings + scaler mean/scale)

The run_tag is  <dataset>_seed<seed>_<8-char hash of all settings>, so changing
the dataset, split, hyperparameters or seed automatically gives a new folder,
and re-running the exact same settings reuses (overwrites) the same folder.
"""

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parents[1]   # .../LPNE (this file lives in Scripts/)
RUNS_DIR = PROJECT / 'Scripts' / 'runs'
MODEL_OUTPUT_DIR = PROJECT / 'Data' / 'ModelOutput'

AGE_COL = 'Age[kaBP]'
STATE_COLS = ['dO18_LR04[pm]', 'CO2_Yamamoto[pm]', 'MgCa_Eld[mmol/mol]', 'Insolation[W/m2]']
STD_COLS = ['dO18_std', 'CO2_std', 'MgCa_std']

# dataset name -> csv (relative to PROJECT) + the Optuna-tuned hyperparameters for it
DATASETS = {
    '1243ka': {
        'csv': 'Data/InterpolatedObs/gpr_full_traj_1243ka.csv',
        'params': {'conv filters': 32, 'conv kernel': 9, 'dense unit': 10,
                   'dropout rate': 0.2632450164100955, 'learning rate': 0.0005192120606179918,
                   'dwt gamma': 0.3818453319933549, 'dwt beta': 0.21402907131795812,
                   'dwt eps': 3.61758437807838e-08, 'dwt levels': 3},
    },
    '1461ka': {
        'csv': 'Data/InterpolatedObs/gpr_full_traj_2diff.csv',
        'params': {'conv filters': 16, 'conv kernel': 9, 'dense unit': 21,
                   'dropout rate': 0.24199127728779032, 'learning rate': 0.0008599275973935869,
                   'dwt gamma': 0.4016606625962723, 'dwt beta': 0.4538313866456523,
                   'dwt eps': 1.5028933889868065e-06, 'dwt levels': 2},
    },
}

COLORS = {
    "EDC":        "#D6195E",
    "Eld": "#1E88E5",
    "LR04":       "#2C754B",
    "Hon":    "#FB8C00",
    "Yam":      "#6519C5",
    "insol":"coral",
    "X":"lightskyblue",
    "Y":"slategrey",
    "Z":"limegreen",
}

# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------
def make_config(dataset, seed=42, params=None, test_frac=0.2, train_frac=0.8, test_size=None,
                timestep=1, window_kyr=50, rollout_kyr=32, vali_rollout_kyr=128,
                noise_level=1e-2, acf_threshold=0.97, epochs=100, batch_size=64):
    """
    Build the config dict for one run.

    dataset    - key of DATASETS ('1243ka' or '1461ka')
    params     - hyperparameter dict; None -> the tuned set stored in DATASETS[dataset].
                 Pass another dataset's params here to swap hyperparameters.
    test_frac  - fraction of the record (oldest end) held out as test set
    test_size  - optional explicit number of test rows; overrides test_frac
                 (e.g. test_size=461 reproduces the old hard-coded split)
    """
    if dataset not in DATASETS:
        raise KeyError(f"unknown dataset {dataset!r}; choose from {list(DATASETS)}")
    params = dict(DATASETS[dataset]['params'] if params is None else params)

    cfg = {
        'dataset': dataset,
        'csv': DATASETS[dataset]['csv'],

        'params': params,
        'seed': int(seed),
        'test_frac': float(test_frac),
        'train_frac': float(train_frac),
        'test_size': None if test_size is None else int(test_size),
        'timestep': timestep,

        'window_size': int(window_kyr / timestep),
        'rollout_length': int(rollout_kyr / timestep),
        'rollout_length_validation': int(vali_rollout_kyr / timestep),

        'noise_level': noise_level,
        'acf_threshold': acf_threshold,
        'epochs': epochs,
        'batch_size': batch_size,
    }
    digest = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    cfg['run_tag'] = f"{dataset}_seed{seed}_{digest}"
    return cfg


def run_paths(run_tag):
    d = RUNS_DIR / run_tag
    return {
        'dir': d,
        'weights': d / 'model.weights.h5',        # keras 3 requires the .weights.h5 suffix
        'prev_weights': d / 'model.prev.weights.h5',
        'config': d / 'config.json',
    }


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def load_data(cfg):
    """Observational dataframe ordered OLDEST -> YOUNGEST (row 0 = oldest age)."""
    df = pd.read_csv(PROJECT / cfg['csv'])
    return df.sort_values(AGE_COL, ascending=False).reset_index(drop=True)


def split_data(df, cfg):
    """
    Split an oldest->youngest dataframe into:
      test       - the oldest `test_size` rows
      (1-row buffer)
      trainval   - everything younger than that
      train      - oldest `train_frac` of trainval
      validation - youngest remainder of trainval
    Returns a dict of numpy arrays (state columns incl. insolation as last column).
    """
    A = df[STATE_COLS].to_numpy(dtype=float)
    S = df[STD_COLS].to_numpy(dtype=float)
    t = df[AGE_COL].to_numpy(dtype=float)

    n_total = A.shape[0]
    test_size = cfg['test_size'] if cfg.get('test_size') is not None else int(round(cfg['test_frac'] * n_total))

    trainval, trainval_time = A[test_size + 1:], t[test_size + 1:]
    train_end = int(round(cfg['train_frac'] * trainval.shape[0]))

    return {
        'full': A, 'full_std': S, 'time': t,
        'test': A[:test_size], 'test_time': t[:test_size],
        'trainval': trainval, 'trainval_time': trainval_time,
        'train': trainval[:train_end], 'train_time': trainval_time[:train_end],
        'validation': trainval[train_end:], 'validation_time': trainval_time[train_end:],
        'test_size': test_size, 'train_end': train_end,
    }


def describe_split(s):
    rng = lambda a: f"{a.max():.0f}-{a.min():.0f} ka"
    return (f"test: {s['test'].shape} ({rng(s['test_time'])}) | "
            f"train: {s['train'].shape} ({rng(s['train_time'])}) | "
            f"validation: {s['validation'].shape} ({rng(s['validation_time'])})")


def fit_scaler(splits):
    """StandardScaler fit on the training portion only (used by all three notebooks)."""
    scaler = StandardScaler()
    scaler.fit(splits['train'])
    return scaler


def fit_and_transform_splits(splits):
    """fit_scaler + transform train / validation / test in one go."""
    scaler = fit_scaler(splits)

    train_scaled = scaler.transform(splits['train'])
    valid_scaled = scaler.transform(splits['validation'])
    tests_scaled = scaler.transform(splits['test'])
    return scaler, train_scaled, valid_scaled, tests_scaled


# ----------------------------------------------------------------------------
# saving / loading runs
# ----------------------------------------------------------------------------
def save_run(cfg, scaler, splits=None):
    """Write config.json (settings + scaler parameters) into the run folder."""
    p = run_paths(cfg['run_tag'])
    p['dir'].mkdir(parents=True, exist_ok=True)
    out = dict(cfg)
    out['scaler_mean'] = scaler.mean_.tolist()
    out['scaler_scale'] = scaler.scale_.tolist()
    if splits is not None:
        out['split_ages'] = {k: [float(splits[k + '_time'].max()), float(splits[k + '_time'].min())]
                             for k in ('test', 'train', 'validation')}
    with open(p['config'], 'w') as f:
        json.dump(out, f, indent=2)
    return p


def scaler_from_config(cfg):
    scaler = StandardScaler()
    scaler.mean_ = np.array(cfg['scaler_mean'])
    scaler.scale_ = np.array(cfg['scaler_scale'])
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)
    return scaler


def load_run(run_tag):
    """Return (cfg, scaler, paths) for a previously trained run."""
    p = run_paths(run_tag)
    with open(p['config']) as f:
        cfg = json.load(f)
    return cfg, scaler_from_config(cfg), p


def list_runs(dataset=None):
    """Run tags that have trained weights, newest first."""
    if not RUNS_DIR.exists():
        return []
    runs = [d for d in RUNS_DIR.iterdir()
            if (d / 'config.json').exists() and (d / 'model.weights.h5').exists()
            and (dataset is None or d.name.startswith(dataset + '_'))]
    return [d.name for d in sorted(runs, key=lambda d: (d / 'model.weights.h5').stat().st_mtime, reverse=True)]


def latest_run(dataset):
    runs = list_runs(dataset)
    if not runs:
        raise FileNotFoundError(f"no trained runs for {dataset!r} in {RUNS_DIR} - train one with DWT-LPNE first")
    return runs[0]
