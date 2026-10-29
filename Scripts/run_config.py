"""
Shared run configuration + data preparation for the SR-LPNE notebooks
(DWT-LPNE.ipynb for training, MC_Rollouts_*.ipynb for evaluation).

Why this file exists
--------------------
Training and evaluation used to each load, order, split and scale the data on
their own, with slightly different conventions, and every training run wrote to
the same `best_state_space_model.weights.h5`. This module is the single source
of truth for all of that, so the two notebooks cannot drift apart:

  * `make_config(...)`  -> a dict describing one run (parameter set, insolation record,
                           training start / split, seed, window/rollout lengths) + a
                           deterministic `run_tag` derived from those settings
  * `load_data(cfg)`    -> obs dataframe from GPR_CSV, ALWAYS ordered oldest -> youngest
  * `split_data(...)`   -> test / trainval / train / validation arrays
  * `fit_scaler(...)`   -> StandardScaler fit on the training portion only
  * `save_run(...)` / `load_run(...)` -> config + scaler stored next to the
                           weights, in  Scripts/runs/<run_tag>/

Each run lives in its own folder:
    Scripts/runs/<run_tag>/model.weights.h5
    Scripts/runs/<run_tag>/config.json      (all settings + scaler mean/scale)

The run_tag is  <train_start>_seed<seed>_<insol>_<8-char hash of all settings>, so changing
the training start, split, hyperparameters, insolation or seed automatically gives a new folder,
and re-running the exact same settings reuses (overwrites) the same folder.
"""

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

####################
#### File Names ####
####################
PROJECT = Path(__file__).resolve().parents[1]   # .../LPNE (this file lives in Scripts/)
RUNS_DIR = PROJECT / 'Scripts' / 'runs'
MODEL_OUTPUT_DIR = PROJECT / 'Data' / 'EmulatorOutput'   # exported rollouts (standardplots.export_rollout)

# GPR table written by running obs.py directly: 1 ka grid, 0-5320 ka. Per record (obs.py source key) a GPR
# mean column <key> and a GPR variance column Variance_<key>, plus the insolation records (no variance).
# Each record's columns are NaN past its oldest sample.
GPR_CSV = PROJECT / 'Data' / 'InterpolatedObs' / 'gpr_5320.0kaBP.csv'

##################################
#### Important Variable Names ####
##################################
AGE_COL = 'Age[kaBP]'
# state variables X, Y, Z (columns of GPR_CSV), in that order
# recall that it's ice sheet volume, CO2, and deep ocean temperature
STATE_RECORDS = ['dO18_LR04', 'co2_Yam', 'MgCa_Eld']
VAR_COLS = [f'Variance_{r}' for r in STATE_RECORDS]
# insolation forcing options -> column in GPR_CSV (imported by obs.py in obs.NAMES)
INSOL_COLS = {'insol_huy06': 'ISI_thresh0Wm2', 'insol_ber90': 'Q65N_solstice_raw',
              'insol_la04': 'Q65N_solstice_la04'}
# insolation records that are not stored in GPR_CSV: load_data computes them on the fly instead.
# insol_la04 = Laskar et al. (2004) 65N June-solstice insolation (longterm_insolation.solstice_insolation),
# which also covers the future (to +21 Ma), so it can force rollouts past the present
COMPUTED_INSOL = {'insol_la04'}

# ---- model inputs: which state variables and forcings a run uses (make_config(state=..., forcing=...)) ----
# short name of each state variable; also accepted in place of the record name in make_config(state=...)
STATE_NAMES = {'dO18_LR04': 'X', 'co2_Yam': 'Y', 'MgCa_Eld': 'Z'}
# forcing options -> column of the load_data frame. 'insol' is the run's insolation record (cfg['insol'], the
# default single forcing); the rest are orbital elements, computed on the fly (nothing is stored on disk):
#   ecc  - eccentricity
#   prec - climatic precession index e*sin(varpi) (varpi: longitude of perihelion from the moving equinox)
#   obl  - obliquity [rad]
# from Berger 1990 (palinsol, 0-1500 ka: the decomposition behind insol_ber90) or Laskar 2004 (any age, incl.
# the future: the decomposition behind insol_la04). The two agree closely (r = 0.98-0.996 over 0-1500 ka).
# The bare names 'ecc', 'prec', 'obl' resolve to the source matching cfg['insol'] (Laskar for insol_la04,
# Berger otherwise); add the suffix ('ecc_la04', ...) to choose explicitly. New forcings go here + in
# _forcing_values below
ORBITAL_SOURCES = ('ber90', 'la04')
FORCING_COLS = {'insol': None,   # -> INSOL_COLS[cfg['insol']]
                **{f'{el}_{src}': f'{el}_{src}' for el in ('ecc', 'prec', 'obl') for src in ORBITAL_SOURCES}}
DEFAULT_FORCING = ('insol',)
BER90_CSV = PROJECT / 'Data' / 'Observations' / 'palinsol_ber90_raw.csv'

# runs trained before GPR_CSV existed stored a 'csv' path in their config.json; load_data reads those
# files with these column names, mapped onto the GPR_CSV names (stds are squared into variances)
_LEGACY_COLS = {'dO18_LR04[pm]': 'dO18_LR04', 'CO2_Yamamoto[pm]': 'co2_Yam', 'MgCa_Eld[mmol/mol]': 'MgCa_Eld',
                'Insolation[W/m2]': INSOL_COLS['insol_huy06'],   # what gpr_full_traj_*.csv actually call it (Huybers ISI)
                'Insolation_huy06[W/m2]': INSOL_COLS['insol_huy06'], 'Insolation_ber90[W/m2]': INSOL_COLS['insol_ber90']}
_LEGACY_STD_COLS = {'dO18_std': 'dO18_LR04', 'CO2_std': 'co2_Yam', 'MgCa_std': 'MgCa_Eld'}

COLORS = {
    "EDC":        "#D6195E",
    "Eld": "#1E88E5",
    "LR04":       "#2C754B",
    "Hon":    "#FB8C00",
    "Yam":      "#6519C5",
    "insol":"coral",
    "insol_ber90":"#00897B",
    "X":"lightskyblue",
    "Y":"slategrey",
    "Z":"limegreen",
}


##########################
#### Important Numbers ###
##########################
NFEATURES = 4   # columns of a standard X, Y, Z + insolation input row; ablation runs: n_features(cfg)

# default parameter set
DEFAULT_PARAMS = {
            'conv filters': 16, 'conv kernel': 9, 'dense unit': 21,
            'dropout rate': 0.24199127728779032, 'learning rate': 0.0008599275973935869,
            'dwt gamma': 0.4016606625962723, 'dwt beta': 0.4538313866456523,
            'dwt eps': 1.5028933889868065e-06, 'dwt levels': 2,
}

DATASET = '1461ka'

STARTS = {   # age [ka] training+validation begins at (the oldest training age); older rows are the test set
    'early': int(1350),
    'mid': int(1170),
    'late': int(800),
    'balanced': int(1461),   # whole oldest end, so no test set; ends at TRAIN_ENDS['balanced']
    'default': None, # use default 80:20 train:test split
}
# age [ka] training+validation ends at (the youngest training age), for starts that don't train up to the present;
# younger rows are held out (split_data's 'holdout' split). 'balanced' trains on 1461-579 ka: ~260 kyr of
# early Pleistocene (1461-1200 ka), the MPT (1200-800 ka) and ~220 kyr of late Pleistocene (800-579 ka)
TRAIN_ENDS = {
    'balanced': int(579),
}
# age [ka] the validation block is centred on, for starts that don't validate on the youngest end of
# training+validation. 'balanced' validates on the MPT, so training keeps both the early (1461-1200 ka) and the
# late (800-579 ka) Pleistocene; training is then two pieces (split_data's 'train_segments')
VALID_CENTERS = {
    'balanced': int(1000),
}

# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------
def make_config(dataset = DATASET, seed=42, params=DEFAULT_PARAMS, insol='insol_ber90', max_age=None,
                train_start='default', train_frac=0.8, valid_frac=0.2,
                timestep=1, window_kyr=50, rollout_kyr=32, vali_rollout_kyr=128,
                noise_level=1e-2, acf_threshold=0.97, epochs=100, batch_size=64, state=None, forcing=None,
                state_feedback=True):
    """
    Build the config dict for one run.

    dataset     - the size of the dataset in ka; defaults to 1461, but if longer datasets become availble, can be changed. 
    params      - hyperparameter dictionary. Defaults to DEFAULT_PARAMS. 
                 Pass another dataset's params here to swap hyperparameters.
    insol       - insolation forcing, a key of INSOL_COLS ('insol_huy06' = Huybers 2006 ISI,
                 'insol_ber90' = Berger 1990 65N June-solstice insolation, 'insol_la04' = Laskar 2004
                 65N June-solstice insolation, computed on the fly; see COMPUTED_INSOL)
    max_age     - oldest age [ka] used from GPR_CSV; None -> the age in the dataset name (see 
                 dataset). Can use to cut the dataset to arbitrary lengths. 
    train_start - key of STARTS ('early', 'mid', 'late', 'balanced' or 'default'). A named start fixes the age
                 [ka] training+validation begins at; 'default' -> train_frac takes over. A start listed in
                 TRAIN_ENDS also stops training+validation at that age ('balanced': 1461-579 ka, no test set).
    train_frac  - fraction of the record used for training+validation (youngest end) when
                 train_start='default'; the oldest remainder is the test set.
    valid_frac  - fraction of training+validation held out (youngest end) as validation set; raised
                 if needed so the validation set holds at least one validation window.
    state       - state variables the model evolves, in this order: record names (STATE_RECORDS) or their short
                 names ('X', 'Y', 'Z'). None -> all three. An empty list is allowed by the config but the
                 RolloutEmulator needs at least one state variable to roll out.
    forcing     - forcings the model is driven by, keys of FORCING_COLS: 'insol' (cfg['insol']) and/or the
                 orbital elements 'ecc', 'prec', 'obl' (optionally with a source suffix, e.g. 'ecc_la04').
                 None -> ('insol',). Use () for no forcing.
    state_feedback - True (default): autoregressive - the model sees its own states. False: regression on the
                 forcing - the model sees only the forcing window and predicts the states directly, never seeing
                 its own predictions (AREmulator.RolloutEmulator(state_inputs=False); see model_dims, brain_inputs)
    state/forcing/state_feedback are only stored in the config (and named in the run tag) when they differ from the
    default, so the configs and run tags of the standard X, Y, Z + insolation runs are unchanged.

    cfg['test_size'] (rows older than training+validation) is what split_data uses; it comes from train_start if
    one is named, otherwise from train_frac. cfg['train_size'] is the span of training+validation [kyr] and
    cfg['train_frac'] its fraction of the record, for reference. cfg['training_end'] is only set for starts in
    TRAIN_ENDS (so the configs, and run tags, of the other starts are unchanged).
    """
    if train_start not in STARTS:
        raise KeyError(f"unknown training start {train_start!r}; choose from {list(STARTS)}")
    state = _resolve_state(state)
    forcing = _resolve_forcing(forcing, insol)
    if insol not in INSOL_COLS:
        raise KeyError(f"unknown insolation record {insol!r}; choose from {list(INSOL_COLS)}")
    max_age = float(dataset.removesuffix('ka')) if max_age is None else float(max_age)

    # training+validation span: a named start takes precedence over train_frac; it runs to the present unless
    # the start has an end age in TRAIN_ENDS
    start_age = STARTS[train_start] if STARTS[train_start] is not None else int(float(train_frac) * max_age)
    end_age = TRAIN_ENDS.get(train_start, 0)
    if not 0 < start_age <= max_age:
        raise ValueError(f"training start {start_age} ka lies outside the record (0-{max_age:.0f} ka)")
    if not 0 <= end_age < start_age:
        raise ValueError(f"training end {end_age} ka must lie between the present and the training start "
                         f"({start_age} ka)")
    train_size = start_age - end_age

    cfg = {
        'training_start': STARTS[train_start],
        'insol': insol,
        'max_age': max_age,

        'params': params,
        'seed': int(seed),
        'train_size': train_size,
        'train_frac': train_size / max_age,
        'test_size': int(max_age) - start_age,
        'valid_frac': float(valid_frac),
        'timestep': timestep,

        'window_size': int(window_kyr / timestep),
        'rollout_length': int(rollout_kyr / timestep),
        'rollout_length_validation': int(vali_rollout_kyr / timestep),

        'noise_level': noise_level,
        'acf_threshold': acf_threshold,
        'epochs': epochs,
        'batch_size': batch_size,
    }
    # only added when set, so the configs (and run tags) of the other starts don't change
    if end_age:
        cfg['training_end'] = end_age
    if train_start in VALID_CENTERS:
        cfg['validation_center'] = VALID_CENTERS[train_start]
    ablation = ''
    if not state_feedback:
        if not forcing:
            raise ValueError("state_feedback=False (regression on the forcing) needs at least one forcing")
        cfg['state_feedback'] = False
    if state != STATE_RECORDS or forcing != list(DEFAULT_FORCING) or not state_feedback:
        cfg['state'], cfg['forcing'] = state, forcing
        ablation = f"_{inputs_label(cfg)}"
    digest = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    cfg['run_tag'] = f"{train_start}_seed{seed}_{insol}{ablation}_{digest}"
    return cfg


def _resolve_state(state):
    """state as passed to make_config -> list of record names (STATE_RECORDS order is NOT imposed)"""
    if state is None:
        return list(STATE_RECORDS)
    short = {v: k for k, v in STATE_NAMES.items()}
    out = [short.get(v, v) for v in state]
    bad = [v for v in out if v not in STATE_RECORDS]
    if bad:
        raise KeyError(f"unknown state variable(s) {bad}; choose from {STATE_RECORDS} or {list(short)}")
    if len(set(out)) != len(out):
        raise ValueError(f"state variables repeated: {list(state)}")
    return out


def _resolve_forcing(forcing, insol):
    """forcing as passed to make_config -> list of FORCING_COLS keys, bare orbital names given their source"""
    if forcing is None:
        return list(DEFAULT_FORCING)
    src = 'la04' if insol == 'insol_la04' else 'ber90'
    out = [f if f in FORCING_COLS else f'{f}_{src}' for f in forcing]
    bad = [f for f in out if f not in FORCING_COLS]
    if bad:
        raise KeyError(f"unknown forcing(s) {bad}; choose from 'insol', 'ecc', 'prec', 'obl' (or {list(FORCING_COLS)})")
    if len(set(out)) != len(out):
        raise ValueError(f"forcings repeated: {list(forcing)}")
    return out


# ---- what a config feeds the model (defaults reproduce the standard X, Y, Z + insolation runs) ----
def state_records(cfg):
    """state variables of the run (record names), in model order"""
    return list(cfg.get('state', STATE_RECORDS))


def state_names(cfg):
    """short names ('X', 'Y', 'Z') of the run's state variables, in model order (RolloutEmulator state_names)"""
    return [STATE_NAMES[r] for r in state_records(cfg)]


def forcing_names(cfg):
    """the run's forcings (FORCING_COLS keys), in model order"""
    return list(cfg.get('forcing', DEFAULT_FORCING))


def forcing_cols(cfg):
    """load_data columns of the run's forcings"""
    return [INSOL_COLS[cfg.get('insol', 'insol_huy06')] if f == 'insol' else FORCING_COLS[f]
            for f in forcing_names(cfg)]


def var_cols(cfg):
    """GPR variance columns of the run's state variables"""
    return [f'Variance_{r}' for r in state_records(cfg)]


def n_state(cfg):
    return len(state_records(cfg))


def n_forcing(cfg):
    return len(forcing_names(cfg))


def n_features(cfg):
    """columns of a model input window: states, then forcings (LPNE.CNN_model n_features)"""
    return n_state(cfg) + n_forcing(cfg)


def state_feedback(cfg):
    """True if the model sees its own states (autoregressive), False for a regression on the forcing"""
    return cfg.get('state_feedback', True)


def brain_inputs(cfg):
    """columns the network itself sees (LPNE.CNN_model n_features=): all of them, or only the forcings when the
    states aren't fed back. The data windows always hold all n_features(cfg) columns"""
    return n_features(cfg) if state_feedback(cfg) else n_forcing(cfg)


def model_dims(cfg):
    """
    keyword arguments describing the run's inputs, for AREmulator.RolloutEmulator(..., **rc.model_dims(cfg));
    use brain_inputs(cfg) / n_state(cfg) for LPNE.CNN_model(n_features=, n_out=)
    """
    return {'n_state': n_state(cfg), 'n_forcing': n_forcing(cfg), 'state_names': state_names(cfg),
            'state_inputs': state_feedback(cfg)}


def inputs_label(cfg):
    """short label of the run's inputs, e.g. 'XYZ_insol', 'XZ_ecc-prec-obl', 'XYZ_ecc-prec-obl_regression'"""
    st = ''.join(state_names(cfg)) or 'nostate'
    fo = '-'.join(f.split('_')[0] if f != 'insol' else f for f in forcing_names(cfg)) or 'noforcing'
    return f"{st}_{fo}" + ('' if state_feedback(cfg) else '_regression')


def run_paths(run_tag, runs_dir=None):
    """file paths of one run, in <runs_dir>/<run_tag>/; runs_dir None -> RUNS_DIR (e.g. RUNS_DIR / '_Ablations'
    keeps a study's runs in their own folder)"""
    d = Path(runs_dir if runs_dir is not None else RUNS_DIR) / run_tag
    return {
        'dir': d,
        'weights': d / 'model.weights.h5',        # keras 3 requires the .weights.h5 suffix
        'prev_weights': d / 'model.prev.weights.h5',
        'config': d / 'config.json',
    }


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def state_cols(cfg):
    """
    Columns of the load_data frame that make up a model input row: the run's state variables (default X, Y, Z),
    then its forcings (default the insolation record chosen by cfg['insol'], which is set in make_config so it's
    part of the run tag; old configs without it used Huybers).
    """
    return state_records(cfg) + forcing_cols(cfg)


def _forcing_values(name, ages):
    """values of the computed forcing `name` (a FORCING_COLS key other than 'insol') at ages [ka BP]"""
    element, src = name.split('_')
    if src == 'ber90':
        b = pd.read_csv(BER90_CSV)
        b = b[np.isclose(b['age_kyr'] % 1, 0)].set_index('age_kyr')   # file is every 0.1 ka; keep the 1 ka ages
        b = b.reindex(np.round(ages))                                    # NaN beyond 1500 ka -> load_data reports it
        return {'ecc': np.hypot(b['Pi_P_raw'], b['Pi_C_raw']),          # Pi_P = e sin(varpi), Pi_C = e cos(varpi)
                'prec': b['Pi_P_raw'], 'obl': b['E_raw']}[element].to_numpy(dtype=float)
    import longterm_insolation as li   # imported here: it needs climlab, which the stored records don't
    la = li.load_la2004()
    t = -np.asarray(ages, dtype=float)   # age [ka BP] -> kyr from J2000
    ecc = np.interp(t, la['t'], la['ecc'])
    return {'ecc': ecc, 'prec': ecc * np.sin(np.interp(t, la['t'], la['varpi'])),
            'obl': np.interp(t, la['t'], la['obliq'])}[element]


def add_computed_forcing(df, cfg):
    """Copy of df with every computed forcing of the run (orbital elements) added as a column. Nothing is written."""
    names = [f for f in forcing_names(cfg) if f != 'insol']
    if not names:
        return df
    df = df.copy()
    ages = df[AGE_COL].to_numpy(dtype=float)
    for f in names:
        df[FORCING_COLS[f]] = _forcing_values(f, ages)
    return df


def forcing_record(cfg, ages):
    """
    (len(ages), n_forcing) array of the run's forcings at arbitrary ages [ka BP] (negative = future), e.g. to
    force a rollout past the observations. 'insol' is only available here for insol_la04 (the stored records
    end at the oldest observation and the present); the Laskar orbital elements cover -21 Ma to +51 Ma
    """
    ages = np.asarray(ages, dtype=float)
    cols = []
    for f in forcing_names(cfg):
        if f != 'insol':
            cols.append(_forcing_values(f, ages))
        elif cfg.get('insol') == 'insol_la04':
            import longterm_insolation as li
            cols.append(li.solstice_insolation(-ages))
        else:
            raise ValueError(f"insolation {cfg.get('insol')!r} is only stored for observed ages; use insol_la04 "
                             f"for arbitrary ages")
    return np.column_stack(cols) if cols else np.empty((ages.size, 0))


def add_computed_insolation(df, insol):
    """
    Copy of df with the insolation record `insol` (a key of COMPUTED_INSOL) added as column INSOL_COLS[insol],
    computed at every row's age. Nothing is written to disk.
    """
    if insol != 'insol_la04':
        raise KeyError(f"no computation defined for insolation record {insol!r}")
    import longterm_insolation as li   # imported here: it needs climlab, which the stored records don't
    df = df.copy()
    df[INSOL_COLS[insol]] = li.solstice_insolation(-df[AGE_COL].to_numpy(dtype=float))   # age [ka BP] -> kyr from J2000
    return df


def load_data(cfg):
    """
    Observational dataframe ordered OLDEST -> YOUNGEST (row 0 = oldest age), with columns AGE_COL,
    state_cols(cfg) and VAR_COLS, for ages 0 to cfg['max_age']. The insolation column is the one picked by
    cfg['insol']; for a different insolation record make a separate config (make_config(..., insol=...)).

    Reads GPR_CSV. Configs of runs trained before it existed carry a 'csv' path instead; that file is
    read and its columns renamed to the GPR_CSV names (stds squared into variances), so old runs still load.
    """
    if 'csv' in cfg:
        df = pd.read_csv(PROJECT / cfg['csv']).rename(columns=_LEGACY_COLS)
        for std_col, rec in _LEGACY_STD_COLS.items():
            df[f'Variance_{rec}'] = df[std_col] ** 2
        max_age = df[AGE_COL].max()
    else:
        df = pd.read_csv(GPR_CSV, index_col=0)
        max_age = cfg['max_age']
        if cfg.get('insol') in COMPUTED_INSOL:
            df = add_computed_insolation(df, cfg['insol'])
    df = add_computed_forcing(df, cfg)

    cols = [AGE_COL] + state_cols(cfg) + var_cols(cfg)
    df = df.loc[df[AGE_COL] <= max_age, cols]
    if df.isna().any().any():
        short = [c for c in cols if df[c].isna().any()]
        raise ValueError(f"max_age = {max_age} ka is older than the end of {short}; "
                         f"oldest complete age is {df.dropna()[AGE_COL].max()} ka")
    return df.sort_values(AGE_COL, ascending=False).reset_index(drop=True)


def split_data(df, cfg):
    """
    Split an oldest->youngest dataframe into:
      test       - the oldest `test_size` rows (empty if test_size = 0, e.g. the 'balanced' start)
      (1-row buffer, only if there is a test set)
      trainval   - everything younger than that, down to cfg['training_end'] if set (else the present)
      holdout    - the rows younger than cfg['training_end'] (empty if it isn't set): in neither set
      train      - the rest of trainval
      validation - valid_frac of trainval, but at least enough rows for one validation window
                   (window_size + rollout_length_validation + 1, see LPNE.create_windows): the youngest end,
                   or a block centred on cfg['validation_center'] if set, which leaves train in two pieces
    'train_segments' lists the (start, stop) rows of 'train' that are contiguous in time; pass it to
    LPNE.create_windows(..., segments=) so no training window spans the validation block.
    Returns a dict of numpy arrays (state columns incl. insolation as last column). The GPR standard
    deviations of the run's state variables (sqrt of var_cols(cfg), unscaled) are split the same way under
    '<split>_std'.
    """
    A = df[state_cols(cfg)].to_numpy(dtype=float)
    S = np.sqrt(df[var_cols(cfg)].to_numpy(dtype=float))
    t = df[AGE_COL].to_numpy(dtype=float)

    test_size = cfg['test_size']
    if test_size is None:   # legacy configs
        test_size = int(round(cfg['test_frac'] * A.shape[0]))
    first = test_size + 1 if test_size > 0 else 0                          # skip the buffer row after a test set
    last = A.shape[0] - int(np.sum(t < cfg.get('training_end', 0)))     # rows younger than training_end held out
    trainval, trainval_time, trainval_std = A[first:last], t[first:last], S[first:last]
    n_trainval = trainval.shape[0]

    if 'train_size' in cfg:
        min_valid = cfg['window_size'] + cfg['rollout_length_validation'] + 56
        n_valid = max(int(round(cfg['valid_frac'] * n_trainval)), min_valid)
        train_end = n_trainval - n_valid
    else:   # configs of runs trained before train_size existed: train_frac split trainval into train/validation
        train_end = int(round(cfg['train_frac'] * n_trainval))
    if train_end <= 0:
        raise ValueError(f"training+validation set ({n_trainval} rows) is too short for a training and a "
                         f"validation set")

    # validation rows [v0, v1) of trainval: the youngest end, or a block centred on validation_center
    v0, v1 = train_end, n_trainval
    if cfg.get('validation_center') is not None:
        n_valid = n_trainval - train_end
        centre = int(np.argmin(np.abs(trainval_time - cfg['validation_center'])))
        v0 = min(max(centre - n_valid // 2, 0), n_trainval - n_valid)
        v1 = v0 + n_valid
    train_idx = np.r_[0:v0, v1:n_trainval]
    train_segments = [(a, b) for a, b in [(0, v0), (v0, v0 + n_trainval - v1)] if b > a]

    return {
        'full': A, 'full_std': S, 'time': t,
        'test': A[:test_size], 'test_time': t[:test_size], 'test_std': S[:test_size],
        'trainval': trainval, 'trainval_time': trainval_time, 'trainval_std': trainval_std,
        'train': trainval[train_idx], 'train_time': trainval_time[train_idx], 'train_std': trainval_std[train_idx],
        'train_segments': train_segments,
        'validation': trainval[v0:v1], 'validation_time': trainval_time[v0:v1],
        'validation_std': trainval_std[v0:v1],
        'holdout': A[last:], 'holdout_time': t[last:], 'holdout_std': S[last:],
        'test_size': test_size, 'train_end': train_end,
    }


def scale_std(std, scaler):
    """
    GPR standard deviations (n, n_state; the run's state variables, in model order - split_data's '<split>_std')
    -> the scaler's units. A spread is only divided by the scale; the mean shift applies to values, not spreads.
    The states are the scaler's first columns, so their scales are its first n_state ones.
    """
    return std / scaler.scale_[:std.shape[1]]


def describe_split(s):
    rng = lambda a: f"{a.max():.0f}-{a.min():.0f} ka" if a.size else "none"
    segs = s.get('train_segments', [(0, len(s['train_time']))])
    train_rng = " + ".join(rng(s['train_time'][a:b]) for a, b in segs)
    out = (f"test: {s['test'].shape} ({rng(s['test_time'])}) | "
           f"train: {s['train'].shape} ({train_rng}) | "
           f"validation: {s['validation'].shape} ({rng(s['validation_time'])})")
    if s.get('holdout_time') is not None and s['holdout_time'].size:
        out += f" | held out: {s['holdout'].shape} ({rng(s['holdout_time'])})"
    return out


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
    # an empty test set (e.g. the 'balanced' start) stays empty: StandardScaler refuses 0-row input
    tests_scaled = scaler.transform(splits['test']) if len(splits['test']) else splits['test'].copy()
    return scaler, train_scaled, valid_scaled, tests_scaled


# ----------------------------------------------------------------------------
# saving / loading runs
# ----------------------------------------------------------------------------
def save_run(cfg, scaler, splits=None, runs_dir=None):
    """Write config.json (settings + scaler parameters) into the run folder (in runs_dir, default RUNS_DIR)."""
    p = run_paths(cfg['run_tag'], runs_dir)
    p['dir'].mkdir(parents=True, exist_ok=True)
    out = dict(cfg)
    out['scaler_mean'] = scaler.mean_.tolist()
    out['scaler_scale'] = scaler.scale_.tolist()
    if splits is not None:
        out['split_ages'] = {k: [float(splits[k + '_time'].max()), float(splits[k + '_time'].min())]
                             for k in ('test', 'train', 'validation', 'holdout')
                             if k + '_time' in splits and splits[k + '_time'].size}   # empty splits are left out
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


def load_run(run_tag, runs_dir=None):
    """Return (cfg, scaler, paths) for a previously trained run (in runs_dir, default RUNS_DIR)."""
    p = run_paths(run_tag, runs_dir)
    with open(p['config']) as f:
        cfg = json.load(f)
    return cfg, scaler_from_config(cfg), p


def list_runs(train_start=None, runs_dir=None):
    """Run tags that have trained weights, newest first; train_start (a key of STARTS) keeps only runs with that start.
    Looks in runs_dir (default RUNS_DIR); subfolders such as _Ablations are not searched."""
    runs_dir = Path(runs_dir if runs_dir is not None else RUNS_DIR)
    if not runs_dir.exists():
        return []
    runs = [d for d in runs_dir.iterdir()
            if (d / 'config.json').exists() and (d / 'model.weights.h5').exists()
            and (train_start is None or d.name.startswith(train_start + '_'))]
    return [d.name for d in sorted(runs, key=lambda d: (d / 'model.weights.h5').stat().st_mtime, reverse=True)]


def latest_run(train_start=None, runs_dir=None):
    runs = list_runs(train_start, runs_dir)
    if not runs:
        raise FileNotFoundError(f"no trained runs for training start {train_start!r} in "
                                f"{runs_dir if runs_dir is not None else RUNS_DIR} - "
                                f"train one with TestRollouts first")
    return runs[0]
