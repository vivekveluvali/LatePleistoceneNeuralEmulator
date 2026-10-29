import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
import multitaper as mt
from scipy import stats

from obs import select_data, AGE_COL, NAMES, VAR_COL
from run_config import COLORS, MODEL_OUTPUT_DIR, STATE_RECORDS

MAPPINGS = ["X","Y","Z"]
ORBITAL = {100: 'darkblue', 41: 'slateblue', 23: 'mediumpurple'}  # orbital periods [kyr] -> reference line color

def _inv(v):
    '''frequency <-> period conversion for a secondary x-axis (1/f both ways); ignores the f = 0 divide warning'''
    with np.errstate(divide='ignore'):
        return 1 / np.asarray(v, dtype=float)

DEFAULT_COLORS = [["#D6195E","#FFB600"],["#1E88E5","#6519C5"],["#05EFC5","#8DB1A7"]]   # (data, emulator) per X, Y, Z

def _state_dims(model=None, n_state=None, state_names=None):
    '''
        number and names of the state variables (the first columns of the data / rollout; the rest are forcings).
        Taken from the arguments if given, else from a RolloutEmulator (model.n_state / model.state_names), else
        the original X, Y, Z setup - so calls written before the ablation runs behave as they always did
    '''
    if n_state is None:
        n_state = len(state_names) if state_names is not None else getattr(model, 'n_state', 3)
    if state_names is None:
        state_names = getattr(model, 'state_names', None)
    if state_names is None or len(state_names) != n_state:
        state_names = MAPPINGS[:n_state] if n_state <= len(MAPPINGS) else [f"s{i}" for i in range(n_state)]
    return n_state, list(state_names)

def _var_colors(names, colors):
    '''(data, emulator) colors per state variable: X, Y, Z keep their usual colors even when some are dropped'''
    return [colors[MAPPINGS.index(n) % len(colors)] if n in MAPPINGS else colors[i % len(colors)]
            for i, n in enumerate(names)]

def LossPlots(history, phys_check):
    '''
        history    -  Model training history (by epoch); data generated from functions in
                      RolloutEmulator class (see AREmulator.py). History should be standard
                      keras .fit() output.
        phys_check -  the PhysicalAttractorCheckpoint callback used in .fit() (see
                      statschecker.py); only its best_epoch is used, for the red marker line
    '''
    Metrics = history.history
    best_ep = phys_check.best_epoch
    # state variables the model logged a point-to-point MSE for (RolloutEmulator: mse_<state name>)
    names = [k[len('mse_'):] for k in Metrics if k.startswith('mse_')]


    # PLOTTING LOSS INFORMATION
    # bottom row: one point-to-point MSE panel per state variable (e.g. X, Y, Z), sharing the y axis
    n_cols = max(len(names), 1)
    mosaic = [['A'] * n_cols, ['B'] * n_cols] + ([names] if names else [])
    fig,axs = plt.subplot_mosaic(mosaic, figsize = (22, 16 if names else 11))

    axs['A'].plot(history.epoch,Metrics["abs"], label = 'Training Error')
    axs['A'].plot(history.epoch,Metrics["val_abs"], label = 'Validation Error')
    axs['A'].set_title("hMSE vs. Epochs", fontweight = "bold")
    axs['A'].set_ylabel(r"$\mathcal{L}_{L_2}$")
    axs['A'].set_xlabel("epochs")
    axs['A'].grid(True, alpha=0.3)
    axs['A'].legend()

    axs['B'].plot(history.epoch,Metrics["val_C_2"],color = "lightseagreen", label="C$_2$")
    axs['B'].plot(history.epoch,Metrics["val_lag1_acf"],color = "indigo", label="ACF")
    axs['B'].set_title("Validation Metrics vs. Epochs", fontweight = "bold")
    axs['B'].set_ylabel("Value")
    axs['B'].set_xlabel("epochs")
    axs['B'].legend(loc='center right')
    axs['B'].grid(True, alpha=0.3)

    # training solid, validation dashed, in each variable's color
    for var in names:
        ax = axs[var]
        color = COLORS.get(var, 'gray')
        ax.plot(history.epoch,Metrics[f"mse_{var}"], color = color, label = 'Training')
        ax.plot(history.epoch,Metrics[f"val_mse_{var}"], color = color, linestyle = '--', label = 'Validation')
        ax.set_title(f"Point-to-Point MSE ({var})", fontweight = "bold")
        ax.set_xlabel("epochs")
        ax.grid(True, alpha=0.3)
        ax.legend()
        if var != names[0]:
            ax.sharey(axs[names[0]])
            ax.tick_params(labelleft=False)
    if names:
        axs[names[0]].set_ylabel("MSE")

    for key in axs:
        axs[key].axvline(x=best_ep, color='red', linestyle='--', alpha=0.6, label='Best Physical Model')

    handles, labels = axs['A'].get_legend_handles_labels()

    fig.legend(handles, labels,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.02), # Centered, just above the very bottom
            ncol=3,
            fontsize=22,
            frameon=True, # Adding a frame can make it look more like a formal table
            facecolor='white',
            edgecolor='gray')

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    #plt.savefig(f"{exportpath}/Loss_vs_Epochs_valid_rollout_longer.pdf",bbox_inches='tight')
    plt.show()

@tf.function
# predict_step used in rollout
def predict_step(model, w):
    '''
        model   -   RolloutEmulator class (see AREmulator.py) 
        w       -   input for predictions with model

        returns predictions made using the model
    '''
    return model(w, training=False)

def rollout(model, data, window_size, export_name=None, time=None, scaler=None, n_state=None):
    '''
        model         -   trained RolloutEmulator (or the bare CNN "brain")
        data          -   data used for autoregressive rollout; should be scaled. Columns: the state variables
                          first, then the forcings (run_config.state_cols)
        window_size   -   size of window from which initial conditions are drawn
        export_name   -   optional file name (no extension): if given, the rollout is also written to
                          Data/EmulatorOutput/<export_name>.csv (see export_rollout)
        time          -   optional ages [ka BP] of the rows of data, written as the age column of the csv
        scaler        -   optional scaler data was scaled with; adds the rollout in physical units to the csv
        n_state       -   number of state variables (the first columns of data); None -> model.n_state for a
                          RolloutEmulator, else 3 (X, Y, Z)

        return the autoregressive rollout trajectory from the initial conditions: the predicted states, with the
        forcings copied from data at every step
    '''
    S, names = _state_dims(model, n_state)

    window_seed = data[:window_size, :] # initial conditions
    ed = list(window_seed)

    steps = data.shape[0]-window_size # number of steps rolled out after initial condition
                                      # necessary to build off of data for the forcing record

    # 3. Correct Loop
    for t in tqdm(range(steps), desc="Test Forecasting"):

        nxt_step = np.zeros(data.shape[1])
        # Grab the window directly from the main array to ensure it's always up to date
        current_window = np.array(ed[-window_size:])

        # Predict
        Input = np.expand_dims(current_window, axis=0)

        nxt_increments = np.asarray(predict_step(model, Input)).reshape(-1)

        # increments added to the last state (autoregressive), or the states themselves for a RolloutEmulator
        # with state_inputs=False (regression on the forcing)
        if getattr(model, 'state_inputs', True):
            nxt_step[0:S] = current_window[-1,0:S] + nxt_increments
        else:
            nxt_step[0:S] = nxt_increments

        nxt_step[S:] = data[window_size+t,S:]   # forcings: given, not predicted

        ed.append(nxt_step)

    emulator_test_data = np.array(ed)
    if export_name is not None:
        export_rollout(emulator_test_data, export_name, time=time, scaler=scaler, state_names=names)
    return emulator_test_data

def rollout_ensemble(model, data, window_size, n_state=None, progress=True):
    '''
        rollout() for many initial conditions at once: every run is advanced one step per call to the model, so
        n_runs rollouts cost about as much as one (e.g. a Monte Carlo ensemble of initial conditions)

        model         -   trained RolloutEmulator (or the bare CNN "brain")
        data          -   (n_runs, n_steps, n_columns) scaled records, one per run, laid out as for rollout(): the
                          state variables first, then the forcings. Each run starts from its own first window_size
                          rows; its forcings are copied from data at every step
        window_size   -   size of the window the initial conditions are drawn from
        n_state       -   number of state variables; None -> model.n_state for a RolloutEmulator, else 3
        progress      -   show a progress bar

        returns (n_runs, n_steps, n_columns): each run is exactly what rollout(model, data[i], window_size) gives
    '''
    S, _ = _state_dims(model, n_state)
    absolute = not getattr(model, 'state_inputs', True)   # regression mode: the model predicts the states themselves
    out = np.array(data, dtype=np.float32)                # states past the seed window are overwritten step by step

    for t in tqdm(range(window_size, out.shape[1]), desc="Ensemble forecasting", disable=not progress):
        pred = np.asarray(predict_step(model, out[:, t - window_size:t, :]))   # (n_runs, S)
        out[:, t, :S] = pred if absolute else out[:, t - 1, :S] + pred
    return out

def export_rollout(rollout_data, name, time=None, scaler=None, state_names=None, forcing_names=None,
                   records=None):
    '''
        rollout_data  -   (n, n_state + n_forcing) scaled rollout from rollout(): states, then forcings
        name          -   file name (no extension), e.g. f"{cfg['run_tag']}_full_rollout"
        time          -   optional ages [ka BP] of the rows -> AGE_COL column
        scaler        -   optional scaler the data was scaled with -> extra columns in physical units
        state_names   -   names of the state columns (default X, Y, Z, as many as there are before the forcings)
        forcing_names -   names of the forcing columns (default 'Insolation' for a single forcing, else F0, F1, ...)
        records       -   names of the physical-unit columns (default run_config.STATE_RECORDS for the standard
                          X, Y, Z + insolation layout; else <state name>, <forcing name>)

        writes the rollout to MODEL_OUTPUT_DIR/<name>.csv (same layout as the existing files there for the
        standard X, Y, Z + insolation runs) and returns the path
    '''
    n_cols = rollout_data.shape[1]
    if state_names is None:
        state_names = MAPPINGS[:min(3, n_cols - 1)]
    n_forc = n_cols - len(state_names)
    if forcing_names is None:
        forcing_names = ['Insolation'] if n_forc == 1 else [f"F{k}" for k in range(n_forc)]
    df = pd.DataFrame(rollout_data, columns=[f"{c}_scaled" for c in list(state_names) + list(forcing_names)])
    if scaler is not None:
        if records is None:
            standard = list(state_names) == MAPPINGS and list(forcing_names) == ['Insolation']
            records = (STATE_RECORDS + ['Insolation']) if standard else list(state_names) + list(forcing_names)
        df[records] = scaler.inverse_transform(rollout_data)
    if time is not None:
        df.insert(0, AGE_COL, np.asarray(time)[:len(df)])

    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_OUTPUT_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"rollout saved to {path}")
    return path

def plot_rollout(model, data, time, window_size, flag = False, rollout_data = None,
            colors = DEFAULT_COLORS,
            insol_color = "coral", boundary = None, boundary_label = "test  |  train / validation",
            n_state = None, state_names = None, forcing_names = None, test_rows = None):
    '''
        model         -   trained RolloutEmulator
        data          -   scaled ground truth; columns: the state variables (default X, Y, Z), then the forcings
                          (default Insolation)
        time          -   times (in kabp) linked to data
        window_size   -   size of the window for initial conditions
        flag          -   adds a moving-average reference line + its RMSE; off by default
        rollout_data  -   optional precomputed rollout(model, data, window_size), so the
                          (slow) rollout isn't redone for every plot
        colors        -   colormap for plots, (data, emulator) per state variable in X, Y, Z order
        insol_color   -   color of the forcing panel (for a single forcing; several get one color each)
        boundary      -   optional time [ka] to mark with a vertical dotted line on every panel, e.g. the
                          boundary between the test period and the train/validation period; off by default
        boundary_label -  text written above the boundary line on the top panel
        n_state, state_names
                      -   number / names of the state variables; None -> taken from the RolloutEmulator
                          (model.n_state, model.state_names), else X, Y, Z
        forcing_names -   legend labels of the forcing columns (default: none for a single forcing, F0, F1, ...)
        test_rows     -   number of leading rows of data that are the test set (e.g. len(splits['test'])); each
                          panel's title then also gives the L2 over the test set alone (seed window excluded).
                          None -> the rows older than `boundary`, if a boundary is given and they come first
                          (the test set is the oldest part of the record); no boundary -> overall L2 only

        plots each state variable (emulator vs. data) and, in a last panel, the scaled forcing(s)
        that drove the rollout (left out if there is none)

        returns (overall_rmse, overall_r2, overall_ma_rmse); overall_ma_rmse is None if flag=False
    '''
    S, names = _state_dims(model, n_state, state_names)
    var_colors = _var_colors(names, colors)
    n_forc = data.shape[1] - S

    if rollout_data is None:
        rollout_data = rollout(model, data, window_size, n_state=S)
    leng = rollout_data.shape[0]

    # rows of the test set: given, or the leading rows older than the boundary (ages run oldest -> youngest)
    if test_rows is None and boundary is not None:
        older = np.asarray(time[:leng]) > boundary
        n_older = int(older.sum())
        test_rows = n_older if n_older and older[:n_older].all() else None
    # test-set L2 needs at least one predicted step after the seed window
    test_slice = slice(window_size, min(test_rows, leng)) if test_rows and test_rows > window_size else None

    # actual plotting step
    with plt.rc_context({'font.size': 16}):
        n_panels = S + (1 if n_forc else 0)
        fig,axs = plt.subplots(n_panels,1,figsize=(22,3.75 * n_panels),sharex=True, squeeze=False)
        axs = axs[:, 0]

        for i in range(S):
            axs[i].plot(
                time[:leng], # tie dimensions to rollout dimensions
                rollout_data[:,i],
                color = var_colors[i][1],
                lw = 2,
                label="emulator data")

            axs[i].plot(
                time[:leng],
                data[:leng,i],
                label="real scaled data",
                alpha = 0.6,
                linestyle = "dashed",
                lw = 2,
                color = var_colors[i][0])

            title = r"$\mathcal{L}_{L_2}$"+f"= {mean_squared_error(data[window_size:leng,i],rollout_data[window_size:,i]):.3f}"
            if test_slice is not None:
                title += (r"    (test set: $\mathcal{L}_{L_2}$" +
                          f"= {mean_squared_error(data[test_slice, i], rollout_data[test_slice, i]):.3f})")
            axs[i].set_title(title)
            axs[i].set_ylabel(names[i])
            axs[i].legend(loc='upper left', fontsize=14)
            axs[i].grid(True, alpha=0.3)

        # forcing: given, not predicted - the rollout takes it from data at every step
        if n_forc:
            ax_f = axs[S]
            if n_forc == 1:
                ax_f.plot(time[:leng], data[:leng, S], color=insol_color, lw=1.5,
                          label=forcing_names[0] if forcing_names else None)
            else:
                for k in range(n_forc):
                    ax_f.plot(time[:leng], data[:leng, S + k], lw=1.5,
                              label=forcing_names[k] if forcing_names else f"F{k}")
            ax_f.set_title("Insolation forcing" if n_forc == 1 and not forcing_names else "Forcing")
            ax_f.set_ylabel("Forcing (scaled)" if n_forc > 1 else "Insolation (scaled)")
            ax_f.grid(True, alpha=0.3)
            if n_forc > 1 or forcing_names:
                ax_f.legend(loc='upper left', fontsize=14, ncol=n_forc)

        overall_ma_rmse = None
        if flag:
            ma_window = window_size
            moving_avg = np.column_stack([
                pd.Series(data[:leng, i]).rolling(ma_window, min_periods=1).mean().to_numpy()
                for i in range(S)
            ])
            for i in range(S):
                axs[i].plot(time[:leng], moving_avg[:, i], color="gray", linewidth=1,
                            label=f"{ma_window}-step moving average")
                axs[i].legend(loc='upper left', fontsize=14)

            overall_ma_rmse = np.sqrt(mean_squared_error(data[window_size:leng, 0:S], moving_avg[window_size:leng, 0:S]))

        if boundary is not None:
            for ax in axs:
                ax.axvline(boundary, color='black', linestyle=':', lw=2)
            axs[0].annotate(boundary_label, xy=(boundary, 1.0), xycoords=('data', 'axes fraction'),
                            xytext=(0, 6), textcoords='offset points', ha='center', va='bottom', fontsize=14)

        # the panels share one x-axis, so invert it once (each call flips it)
        axs[-1].invert_xaxis()
        axs[-1].set_xlabel("time [kyr]")
        plt.tight_layout()
        plt.show()

    # overall error across all state variables combined, seed window excluded
    overall_rmse = np.sqrt(mean_squared_error(data[window_size:leng, 0:S], rollout_data[window_size:, 0:S]))
    overall_r2 = r2_score(data[window_size:leng, 0:S], rollout_data[window_size:, 0:S])

    return overall_rmse, overall_r2, overall_ma_rmse

def phase_space(model, data, window_size, rollout_data = None,
                colors = DEFAULT_COLORS, n_state = None, state_names = None):
    '''
        model         -   trained RolloutEmulator
        data          -   scaled ground truth data (state variables first)
        window_size   -   size of the window for initial conditions
        rollout_data  -   optional precomputed rollout(model, data, window_size)
        colors        -   color palette, one (data, emulator) pair per panel
        n_state, state_names
                      -   number / names of the state variables; None -> taken from the RolloutEmulator
                          (model.n_state, model.state_names), else X, Y, Z

        one panel per pair of state variables (X-Y, X-Z, Y-Z for the standard setup); nothing to plot with fewer
        than two state variables
    '''
    S, names = _state_dims(model, n_state, state_names)
    pairs = [(a, b) for a in range(S) for b in range(a + 1, S)]
    if not pairs:
        print(f"phase_space: needs at least two state variables, the model has {S} ({names}) - nothing to plot")
        return

    if rollout_data is None:
        rollout_data = rollout(model, data, window_size, n_state=S)


    # PLOTTING PHASE SPACE OF FULL ROLLOUT NN AND NUMERICAL MODEL
    fig, axs = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 6), squeeze=False)
    axs = axs[0]

    for k, (ax, (a, b)) in enumerate(zip(axs, pairs)):
        ax.plot(
            data[:,a],
            data[:,b],
            color = colors[k % len(colors)][0],
            linestyle = "dashed",
            label="scaled data")

        ax.plot(
            rollout_data[:,a],
            rollout_data[:,b],
            color = colors[k % len(colors)][1],
            label="emulator data")

        ax.set_xlabel(names[a])
        ax.set_ylabel(names[b])

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    handles, labels = axs[0].get_legend_handles_labels()

    fig.legend(handles, labels,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.005),
            ncol=3,
            fontsize=20,
            frameon=True,
            facecolor='white',
            edgecolor='gray')

    #plt.savefig(f"{exportpath}/Phase_Space_Full_Rollout.pdf",bbox_inches="tight")
    plt.show()

def multitaper_spectrum(x, dt, nw=2):
    """
        Multitaper spectrum + classical chi-squared CI + Thomson F-test significance mask.
    """
    kspec = int(2*nw - 1)   # number of tapers - more tapers = less variance, less resolution
    psd = mt.MTSpec(x, nw=nw, kspec=kspec, dt=dt, iadapt=0)

    freq, spec = psd.rspec()
    freq = freq.squeeze()
    spec = spec.squeeze()

    one_sided = psd.freq.squeeze() >= 0

    # jackspec is either buggy or I (and Claude) can't figure it out - classical CI instead
    dof = 2 * kspec
    ci_lower = spec * dof / stats.chi2.ppf(0.975, dof)
    ci_upper = spec * dof / stats.chi2.ppf(0.025, dof)

    F, _ = psd.ftest()
    F = F.squeeze()[one_sided]
    f_crit = stats.f.ppf(0.95, 2, 2*kspec - 2)   # df1=2, df2=2K-2 for the harmonic F-test
    significant = F > f_crit

    return freq, spec, ci_lower, ci_upper, significant

def frequency_spectra(model, data, window_size, timestep = 1, spec_var = 0, nw = 2,
                      rollout_data = None, n_state = None, state_names = None):
    '''
        model         -   trained RolloutEmulator
        data          -   scaled ground truth
        window_size   -   size of the window for initial conditions
        timestep      -   sampling interval in kyr
        spec_var      -   column of the state variable (0 = X, 1 = Y, 2 = Z for the standard setup; in an
                          ablation run, its position among the run's state variables). Same for both curves
        n_state, state_names
                      -   number / names of the state variables, for the title and the rollout; None -> taken
                          from the RolloutEmulator (model.n_state, model.state_names), else X, Y, Z
        nw            -   multitaper time-bandwidth product
        rollout_data  -   optional precomputed rollout(model, data, window_size)

        returns dict {"gt": {...}, "test": {...}} with freq/spec/CI/significance arrays
    '''
    S, names = _state_dims(model, n_state, state_names)
    if not 0 <= spec_var < S:
        raise ValueError(f"spec_var={spec_var}, but there are {S} state variables ({names})")
    if rollout_data is None:
        rollout_data = rollout(model, data, window_size, n_state=S)

    # key -> (label, series, color, marker)
    datasets = {
        "gt":   ("Ground truth (GPR mean)",  data[:, spec_var],         '#D6195E', 'o',  0.8),
        "test": ("Emulator rollout",         rollout_data[:, spec_var], '#082271', 's', 0.4),
    }

    results = {}
    with plt.rc_context({'font.size': 16}):
        fig, ax = plt.subplots(figsize=(18, 9))

        for key, (label, x, color, marker, alpha) in datasets.items():
            freq, spec, ci_lower, ci_upper, significant = multitaper_spectrum(x, timestep, nw)
            results[key] = {"freq": freq, "spec": spec, "ci_lower": ci_lower, "ci_upper": ci_upper, "significant": significant}

            # spectrum line, CI band, and markers on F-test-significant frequencies;
            # f = 0 (the mean) can't go on a log axis
            pos = freq > 0
            sig = pos & significant
            ax.plot(freq[pos], spec[pos], color=color, lw=2, label=label, alpha=alpha)
            ax.fill_between(freq[pos], ci_lower[pos], ci_upper[pos], color=color, alpha=0.15, linewidth=0)
            ax.scatter(freq[sig], spec[sig], color=color, marker=marker, edgecolor='black', s=45, zorder=5)

        # reference lines at the eccentricity (100 kyr), obliquity (41 kyr) and precession (23 kyr) periods
        for period, color in ORBITAL.items():
            ax.axvline(1/period, color=color, linestyle='dashed', alpha=0.6, label=f'{period} kyr cycle')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('frequency [1/kyr]')
        ax.set_ylabel(r'power [scaled units$^2$ kyr]')  # data are in the training scaler's units
        ax.secondary_xaxis('top', functions=(_inv, _inv)).set_xlabel('period [kyr]')  # same axis, labelled as period
        ax.set_title(f'Multitaper spectra ({names[spec_var]}): ground truth vs. autoregressive rollout',
                     fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower left', fontsize=14)
        fig.text(0.99, 0.005, r'bands: 95% $\chi^2$ CI   markers: Thomson F-test significant at 95%',
                 ha='right', va='bottom', fontsize=14, color='dimgray')
        plt.tight_layout(rect=[0, 0.02, 1, 1])
        plt.show()

    return results

def plot_resolution(ax, source, color, marker='.', xmax=None, label=None):
    '''
        ax      -   matplotlib axis to plot on
        source  -   a DATAFOLDER/NAMES key (see obs.py), e.g. "ch4_EDC"
        color   -   color palette to use
        marker  -   point marker (e.g. the source's marker, to match other plots)
        xmax    -   optional oldest age [ka] to include; None -> the whole record
        label   -   optional title name for the record; None -> the source key

        plots sample spacing vs. age for one observational source, to show
        how temporal resolution varies/degrades with age. Doesn't call plt.show(), so several records
        can be drawn on the panels of one figure; show the figure from the caller.

        returns the mean spacing between samples and the number of samples (over the ages included)
    '''
    age_sorted = select_data(source)[AGE_COL].to_numpy()
    if xmax is not None:
        age_sorted = age_sorted[age_sorted <= xmax]
    spacing = np.diff(age_sorted)  # kyr between consecutive samples
    ax.plot(age_sorted[1:], spacing, marker, markersize=4, linestyle='none', color=color, alpha=0.6)
    ax.set_yscale('log')
    ax.set_xlabel("Age [ka BP]")
    ax.set_ylabel("Sample spacing [kyr]")
    ax.set_title(f"{label or source}: mean spacing = {np.mean(spacing):.2f} kyr, n = {len(age_sorted)}")

    return np.mean(spacing), len(age_sorted)
