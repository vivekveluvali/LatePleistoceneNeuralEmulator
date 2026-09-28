import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
import multitaper as mt
from scipy import stats

MAPPINGS = ["X","Y","Z"]

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
    fig,axs = plt.subplot_mosaic(("A;B;C"),figsize = (22,16))

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

    axs['C'].plot(history.epoch,Metrics["mse"], label = 'Training Error')
    axs['C'].plot(history.epoch,Metrics["val_mse"], label = 'Validation Error')
    axs['C'].set_title("Point-to-Point MSE vs. Epochs", fontweight = "bold")
    axs['C'].set_ylabel("MSE")
    axs['C'].set_xlabel("epochs")
    axs['C'].grid(True, alpha=0.3)
    axs['C'].legend()

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
def predict_step(model, w):
    '''
        model   -   RolloutEmulator class (see AREmulator.py) 
        w       -   input for predictions with model

        returns predictions made using the model
    '''
    return model(w, training=False)

def rollout(model, data, window_size):
    '''
        model         -   trained RolloutEmulator (or the bare CNN "brain")
        data          -   data used for autoregressive rollout; should be scaled
        window_size   -   size of window from which initial conditions are drawn

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
    return emulator_test_data

def plot_rollout(model, data, time, window_size, flag = False, rollout_data = None,
            colors = [["#D6195E","#FFB600"],["#1E88E5","#6519C5"],["#05EFC5","#8DB1A7"]]):
    '''
        model         -   trained RolloutEmulator
        data          -   scaled ground truth; columns X, Y, Z, Insolation
        time          -   times (in kabp) linked to data
        window_size   -   size of the window for initial conditions
        flag          -   adds a moving-average reference line + its RMSE; off by default
        rollout_data  -   optional precomputed rollout(model, data, window_size), so the
                          (slow) rollout isn't redone for every plot
        colors        -   colormap for plots

        returns (overall_rmse, overall_r2, overall_ma_rmse); overall_ma_rmse is None if flag=False
    '''

    if rollout_data is None:
        rollout_data = rollout(model, data, window_size)
    leng = rollout_data.shape[0]

    # actual plotting step
    fig,axs = plt.subplots(3,1,figsize=(20,6),sharex=True)

    for i in range(len(axs)):
        axs[i].plot(
            time[:leng], # tie dimensions to rollout dimensions
            rollout_data[:,i],
            color = colors[i][1],
            label="emulator data")

        axs[i].plot(
            time[:leng],
            data[:leng,i],
            label="real scaled data",
            alpha = 0.4,
            linestyle = "dashed",
            color = colors[i][0])

        axs[i].set_title(r"$\mathcal{L}_{L_2}$"+f"= {mean_squared_error(data[window_size:,i],rollout_data[window_size:,i]):.3f}")
        axs[i].invert_xaxis()
        axs[i].set_ylabel(MAPPINGS[i])

    overall_ma_rmse = None
    if flag:
        ma_window = window_size
        moving_avg = np.column_stack([
            pd.Series(data[:leng, i]).rolling(ma_window, min_periods=1).mean().to_numpy()
            for i in range(3)
        ])
        for i in range(len(axs)):
            axs[i].plot(time[:leng], moving_avg[:, i], color="gray", linewidth=1,
                        label=f"{ma_window}-step moving average")

        overall_ma_rmse = np.sqrt(mean_squared_error(data[window_size:leng, 0:3], moving_avg[window_size:leng, 0:3]))


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

    dt = timestep
    SPEC_VAR = spec_var

    # label -> (series, color)
    datasets = {
        "gt":   ("Ground truth",                data[:, SPEC_VAR],             '#D6195E'),
        #"tv":   ("Training/validation rollout", full_rollout[:, SPEC_VAR],            '#1E88E5'),
        "test": ("Test rollout (long)",         rollout_data[:, SPEC_VAR], '#43A047'),
    }

    results = {}
    fig, ax = plt.subplots(figsize=(12, 6))

    for key, (label, x, color) in datasets.items():
        freq, spec, ci_lower, ci_upper, significant = multitaper_spectrum(x, dt, nw)
        results[key] = {"freq": freq, "spec": spec, "ci_lower": ci_lower, "ci_upper": ci_upper, "significant": significant}

        ax.plot(freq, spec, color=color, label=f"{label} spectrum")
        ax.fill_between(freq, ci_lower, ci_upper, color=color, alpha=0.15)
        ax.scatter(freq[significant], spec[significant], color=color, edgecolor="black", s=20, zorder=5)

        #print(f"{label} significant periods (kyr):", 1 / freq[significant])

    ymin = min(r["spec"].min() for r in results.values())
    ymax = max(r["spec"].max() for r in results.values())
    ax.vlines(1/100, ymin=ymin, ymax=ymax, alpha=0.5, linestyle="dashed", label="100 kyr cycle", color="darkblue")
    ax.vlines(1/41,  ymin=ymin, ymax=ymax, alpha=0.5, linestyle="dashed", label="41 kyr cycle",  color="slateblue")

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("frequency [1/kyr]")
    ax.set_ylabel("power")
    ax.set_title(f"Multitaper spectra ({MAPPINGS[SPEC_VAR]}): ground truth vs. autoregressive rollouts")
    ax.legend()
    plt.tight_layout()
    plt.show()

    return results
