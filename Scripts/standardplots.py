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


    # PLOTTING LOSS INFORMATION
    # bottom row: one point-to-point MSE panel per state variable (keys X, Y, Z), sharing the y axis
    fig,axs = plt.subplot_mosaic(("AAA;BBB;XYZ"),figsize = (22,16))

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
    for var in MAPPINGS:
        ax = axs[var]
        ax.plot(history.epoch,Metrics[f"mse_{var}"], color = COLORS[var], label = 'Training')
        ax.plot(history.epoch,Metrics[f"val_mse_{var}"], color = COLORS[var], linestyle = '--', label = 'Validation')
        ax.set_title(f"Point-to-Point MSE ({var})", fontweight = "bold")
        ax.set_xlabel("epochs")
        ax.grid(True, alpha=0.3)
        ax.legend()
        if var != MAPPINGS[0]:
            ax.sharey(axs[MAPPINGS[0]])
            ax.tick_params(labelleft=False)
    axs[MAPPINGS[0]].set_ylabel("MSE")

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

def rollout(model, data, window_size, export_name=None, time=None, scaler=None):
    '''
        model         -   trained RolloutEmulator (or the bare CNN "brain")
        data          -   data used for autoregressive rollout; should be scaled
        window_size   -   size of window from which initial conditions are drawn
        export_name   -   optional file name (no extension): if given, the rollout is also written to
                          Data/EmulatorOutput/<export_name>.csv (see export_rollout)
        time          -   optional ages [ka BP] of the rows of data, written as the age column of the csv
        scaler        -   optional scaler data was scaled with; adds the rollout in physical units to the csv

        return the autoregressive rollout trajectory from the initial conditions
    '''

    window_seed = data[:window_size, :] # initial conditions
    ed = list(window_seed)

    steps = data.shape[0]-window_size # number of steps rolled out after initial condition
                                      # necessary to build off of data for insolation record

    # 3. Correct Loop
    for t in tqdm(range(steps), desc="Test Forecasting"):

        nxt_step = np.zeros(data.shape[1])
        # Grab the window directly from the main array to ensure it's always up to date
        current_window = np.array(ed[-window_size:])

        # Predict
        Input = np.expand_dims(current_window, axis=0)

        nxt_increments = predict_step(model, Input)

        nxt_step[0:3] = current_window[-1,0:3] + nxt_increments

        nxt_step[3] = data[window_size+t,-1]

        ed.append(nxt_step)

    emulator_test_data = np.array(ed)
    if export_name is not None:
        export_rollout(emulator_test_data, export_name, time=time, scaler=scaler)
    return emulator_test_data

def export_rollout(rollout_data, name, time=None, scaler=None):
    '''
        rollout_data  -   (n, 4) scaled rollout from rollout(): X, Y, Z, insolation
        name          -   file name (no extension), e.g. f"{cfg['run_tag']}_full_rollout"
        time          -   optional ages [ka BP] of the rows -> AGE_COL column
        scaler        -   optional scaler the data was scaled with -> extra columns in physical units,
                          named after run_config.STATE_RECORDS (+ 'Insolation')

        writes the rollout to MODEL_OUTPUT_DIR/<name>.csv (same layout as the existing files there) and
        returns the path
    '''
    df = pd.DataFrame(rollout_data, columns=['X_scaled', 'Y_scaled', 'Z_scaled', 'Insolation_scaled'])
    if scaler is not None:
        df[STATE_RECORDS + ['Insolation']] = scaler.inverse_transform(rollout_data)
    if time is not None:
        df.insert(0, AGE_COL, np.asarray(time)[:len(df)])

    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_OUTPUT_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"rollout saved to {path}")
    return path

def plot_rollout(model, data, time, window_size, flag = False, rollout_data = None,
            colors = [["#D6195E","#FFB600"],["#1E88E5","#6519C5"],["#05EFC5","#8DB1A7"]],
            insol_color = "coral"):
    '''
        model         -   trained RolloutEmulator
        data          -   scaled ground truth; columns X, Y, Z, Insolation
        time          -   times (in kabp) linked to data
        window_size   -   size of the window for initial conditions
        flag          -   adds a moving-average reference line + its RMSE; off by default
        rollout_data  -   optional precomputed rollout(model, data, window_size), so the
                          (slow) rollout isn't redone for every plot
        colors        -   colormap for plots
        insol_color   -   color of the insolation panel

        plots X, Y, Z (emulator vs. data) and, in a fourth panel, the scaled insolation forcing
        that drove the rollout

        returns (overall_rmse, overall_r2, overall_ma_rmse); overall_ma_rmse is None if flag=False
    '''

    if rollout_data is None:
        rollout_data = rollout(model, data, window_size)
    leng = rollout_data.shape[0]

    # actual plotting step
    with plt.rc_context({'font.size': 16}):
        fig,axs = plt.subplots(4,1,figsize=(22,15),sharex=True)

        for i in range(3):
            axs[i].plot(
                time[:leng], # tie dimensions to rollout dimensions
                rollout_data[:,i],
                color = colors[i][1],
                lw = 2,
                label="emulator data")

            axs[i].plot(
                time[:leng],
                data[:leng,i],
                label="real scaled data",
                alpha = 0.6,
                linestyle = "dashed",
                lw = 2,
                color = colors[i][0])

            axs[i].set_title(r"$\mathcal{L}_{L_2}$"+f"= {mean_squared_error(data[window_size:,i],rollout_data[window_size:,i]):.3f}")
            axs[i].set_ylabel(MAPPINGS[i])
            axs[i].legend(loc='upper left', fontsize=14)
            axs[i].grid(True, alpha=0.3)

        # insolation: given, not predicted - the rollout takes it from data at every step
        axs[3].plot(time[:leng], data[:leng, 3], color=insol_color, lw=1.5)
        axs[3].set_title("Insolation forcing")
        axs[3].set_ylabel("Insolation (scaled)")
        axs[3].grid(True, alpha=0.3)

        overall_ma_rmse = None
        if flag:
            ma_window = window_size
            moving_avg = np.column_stack([
                pd.Series(data[:leng, i]).rolling(ma_window, min_periods=1).mean().to_numpy()
                for i in range(3)
            ])
            for i in range(3):
                axs[i].plot(time[:leng], moving_avg[:, i], color="gray", linewidth=1,
                            label=f"{ma_window}-step moving average")
                axs[i].legend(loc='upper left', fontsize=14)

            overall_ma_rmse = np.sqrt(mean_squared_error(data[window_size:leng, 0:3], moving_avg[window_size:leng, 0:3]))

        # the panels share one x-axis, so invert it once (each call flips it)
        axs[-1].invert_xaxis()
        axs[-1].set_xlabel("time [kyr]")
        plt.tight_layout()
        plt.show()

    # overall error across all three state variables combined (X, Y, Z), seed window excluded
    overall_rmse = np.sqrt(mean_squared_error(data[window_size:leng, 0:3], rollout_data[window_size:, 0:3]))
    overall_r2 = r2_score(data[window_size:leng, 0:3], rollout_data[window_size:, 0:3])

    return overall_rmse, overall_r2, overall_ma_rmse

def phase_space(model, data, window_size, rollout_data = None,
                colors = [["#D6195E","#FFB600"],["#1E88E5","#6519C5"],["#05EFC5","#8DB1A7"]]):
    '''
        model         -   trained RolloutEmulator
        data          -   scaled ground truth data
        window_size   -   size of the window for initial conditions
        rollout_data  -   optional precomputed rollout(model, data, window_size)
        colors        -   color palette
    '''
    if rollout_data is None:
        rollout_data = rollout(model, data, window_size)


    # PLOTTING PHASE SPACE OF FULL ROLLOUT NN AND NUMERICAL MODEL
    fig, axs = plt.subplots(1,3,figsize=(18,6))

    axs[0].plot(
        data[:,0],
        data[:,1],
        color = colors[0][0],
        linestyle = "dashed",
        label="scaled data")

    axs[0].plot(
        rollout_data[:,0],
        rollout_data[:,1],
        color = colors[0][1],
        label="emulator data")

    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")


    axs[1].plot(
        data[:,0],
        data[:,2],
        color = colors[1][0],
        linestyle = "dashed")

    axs[1].plot(
        rollout_data[:,0],
        rollout_data[:,2],
        color = colors[1][1])


    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Z")

    axs[2].plot(
        data[:,1],
        data[:,2],
        color = colors[2][0],
        linestyle = "dashed")

    axs[2].plot(
        rollout_data[:,1],
        rollout_data[:,2],
        color = colors[2][1])

    axs[2].set_xlabel("Y")
    axs[2].set_ylabel("Z")

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
                      rollout_data = None):
    '''
        model         -   trained RolloutEmulator
        data          -   scaled ground truth
        window_size   -   size of the window for initial conditions
        timestep      -   sampling interval in kyr
        spec_var      -   0 = X, 1 = Y, 2 = Z (same variable for ground truth and emulator)
        nw            -   multitaper time-bandwidth product
        rollout_data  -   optional precomputed rollout(model, data, window_size)

        returns dict {"gt": {...}, "test": {...}} with freq/spec/CI/significance arrays
    '''
    if rollout_data is None:
        rollout_data = rollout(model, data, window_size)

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
        ax.set_title(f'Multitaper spectra ({MAPPINGS[spec_var]}): ground truth vs. autoregressive rollout',
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
