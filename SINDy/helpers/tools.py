import matplotlib as mpl
import matplotlib.pyplot as plt
import functools
from types import SimpleNamespace
import numpy as np
import random
from sklearn.preprocessing import PolynomialFeatures
from scipy.integrate import solve_ivp



''' 
Integration Stuff
'''

def SM90_deterministic(t, system, R):

    X_t = system[0]
    Y_t = system[1]
    Z_t = system[2]
    R_t = R(t)
    
    v = 0.2
    u = 0.6
    p = 1.0
    r = 0.9
    s = 1.0
    w = 0.5
    q = 2.5
    
    dX_dt = - X_t - Y_t - v*Z_t - u*R_t 
    dY_dt = - p*Z_t + r*Y_t + s*Z_t**2 - w*Y_t*Z_t - Z_t**2*Y_t
    dZ_dt = - q*(X_t + Z_t)

    return np.array([dX_dt, dY_dt, dZ_dt])


def SM90_stochastic(t, system, R, W, n=0):

    X_t = system[0]
    Y_t = system[1]
    Z_t = system[2]
    R_t = R(t)
    W_t = W(t)
    
    v = 0.2
    u = 0.6
    p = 1.0
    r = 0.9
    s = 1.0
    w = 0.5
    q = 2.5
    
    dX_dt = - X_t - Y_t - v*Z_t - u*R_t + W_t*n 
    dY_dt = - p*Z_t + r*Y_t + s*Z_t**2 - w*Y_t*Z_t - Z_t**2*Y_t 
    dZ_dt = - q*(X_t + Z_t)

    return np.array([dX_dt, dY_dt, dZ_dt])


def ou_noise(n, dt, tau, sigma, rng):
    a = np.exp(-dt / tau)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = a * x[i-1] + sigma * np.sqrt(1 - a**2) * rng.standard_normal()
    return x                      # stationary std = sigma


def white_noise(n, sigma, rng):
    return sigma * rng.standard_normal(n)      # independent draws, std = sigma


def integrate_white_noise(rhs, t_grid, y0, noise, t_eval=None, args=(), component=0, n_sub=10):
    """
    Fixed-step integration of dy/dt = rhs(t, y, *args) + noise_k, with the noise added to y[component] only.
    noise_k is held constant over each grid step [t_k, t_k+1), so its std is the std of the noise term
    (Euler-Maruyama for the noise: increment noise_k * dt; RK4 sub-steps for the drift).
    t_grid: evenly spaced integration grid; noise: len(t_grid) - 1 values, one per step
    t_eval: output times, must lie on t_grid (default: all of t_grid)
    Returns an object with .t and .y, like solve_ivp.
    """
    t_grid = np.asarray(t_grid, float)
    e = np.zeros(len(y0)); e[component] = 1.0
    y = np.zeros((len(y0), len(t_grid)))
    y[:, 0] = y0

    for k in range(len(t_grid) - 1):
        f = lambda tt, yy: rhs(tt, yy, *args) + noise[k] * e
        h = (t_grid[k + 1] - t_grid[k]) / n_sub
        tk, yk = t_grid[k], y[:, k].copy()
        for _ in range(n_sub):
            k1 = f(tk, yk)
            k2 = f(tk + h/2, yk + h/2 * k1)
            k3 = f(tk + h/2, yk + h/2 * k2)
            k4 = f(tk + h, yk + h * k3)
            yk = yk + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            tk += h
        y[:, k + 1] = yk

    t_eval = t_grid if t_eval is None else np.asarray(t_eval, float)
    idx = np.searchsorted(t_grid, t_eval - 1e-9)
    if not np.allclose(t_grid[idx], t_eval):
        raise ValueError("t_eval must lie on t_grid")
    return SimpleNamespace(t=t_eval, y=y[:, idx])


'''
SINDy stuff
'''



def sindy_to_rhs(model, n_degree = 3, var_order=('X', 'Y', 'Z', 'R')):
    """
    Extract a fitted SINDy model's three equations as a solve_ivp-compatible RHS:
    rhs(t, system, R_forcing) -> [dX/dt, dY/dt, dZ/dt]
    """
    coefs = model.coefficients()  # shape (3, n_features)

    poly = PolynomialFeatures(degree=n_degree, include_bias=True)
    poly.fit(np.zeros((1, len(var_order))))

    def rhs(t, system, R_forcing):
        X, Y, Z = system
        R = R_forcing(t)

        values = {'X': X, 'Y': Y, 'Z': Z, 'R': R}
        point = np.array([[values[name] for name in var_order]])
        theta = poly.transform(point)[0]

        return coefs @ theta  # [dX/dt, dY/dt, dZ/dt]

    return rhs

def sindy_to_rhs_dR(model, n_degree = 3, var_order=('X', 'Y', 'Z', 'R', 'dR')):
    """
    Non-Markovian variant: R_forcing(t) must return [R, dR].
    rhs(t, system, R_forcing) -> [dX/dt, dY/dt, dZ/dt]
    """
    coefs = model.coefficients()  # shape (3, n_features)

    poly = PolynomialFeatures(degree=n_degree, include_bias=True)
    poly.fit(np.zeros((1, len(var_order))))

    def rhs(t, system, R_forcing):
        X, Y, Z = system
        R, dR = R_forcing(t)

        values = {'X': X, 'Y': Y, 'Z': Z, 'R': R, 'dR': dR}
        point = np.array([[values[name] for name in var_order]])
        theta = poly.transform(point)[0]

        return coefs @ theta

    return rhs

def sindy_to_rhs_t(model, n_degree = 3, var_order=('X', 'Y', 'Z', 'R', 't'), t_mean = 0, t_std = 1):
    """
    Extract a fitted SINDy model's three equations as a solve_ivp-compatible RHS:
    rhs(t, system, R_forcing) -> [dX/dt, dY/dt, dZ/dt]
    """
    coefs = model.coefficients()  # shape (3, n_features)

    poly = PolynomialFeatures(degree=n_degree, include_bias=True)
    poly.fit(np.zeros((1, len(var_order))))

    def rhs(t, system, R_forcing):
        X, Y, Z = system
        R = R_forcing(t)
        t_standardised = (t-t_mean)/t_std

        values = {'X': X, 'Y': Y, 'Z': Z, 'R': R, 't': t_standardised}
        point = np.array([[values[name] for name in var_order]])
        theta = poly.transform(point)[0]

        return coefs @ theta

    return rhs


''' 
Plotting stuff
'''



def plot_spread(t, true_values, stacked_values, labels=('X', 'Y', 'Z'), fontsize=22, titles = ['spread', 'ensamble mean', 'SM90'], legend_fontsize=None, ylim=None, savefig=None):
    """
    true_values: shape (n_timesteps, 3) -- SM90 ground truth
    stacked_values: shape (n_seeds, n_timesteps, 3) -- ensemble across seeds
    ylim: optional (lo, hi) applied to all panels, or a sequence of (lo, hi) per panel
    """
    if legend_fontsize is None:
        legend_fontsize = fontsize - 2

    mean_values = np.mean(stacked_values, axis=0)
    min_values = np.min(stacked_values, axis=0)
    max_values = np.max(stacked_values, axis=0)

    fig, axs = plt.subplots(3, 1, figsize=(18, 18), sharex=True)

    for i, label in enumerate(labels):
        axs[i].fill_between(t, min_values[:, i], max_values[:, i], color='grey', alpha=0.4, label=titles[0])
        axs[i].plot(t, mean_values[:, i], color='black', linestyle='--', label=titles[1])
        axs[i].plot(t, true_values[:, i], color='darkred', label=titles[2])
        axs[i].set_ylabel(label, fontsize=fontsize)
        axs[i].tick_params(labelsize=fontsize - 2)
        if ylim is not None:
            axs[i].set_ylim(ylim[i] if isinstance(ylim[0], (tuple, list)) else ylim)

    axs[0].legend(fontsize=legend_fontsize)
    axs[-1].set_xlabel('time', fontsize=fontsize)
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(f"./figures/{savefig}")

    return fig, axs


def plot_spaghetti(t, true_values, stacked_values, labels=('X', 'Y', 'Z'), fontsize=22, titles = ['trajectories', 'ensamble mean', 'SM90'], legend_fontsize=None, ylim=None, alpha=0.1, savefig=None):
    """
    true_values: shape (n_timesteps, 3) -- SM90 ground truth
    stacked_values: shape (n_seeds, n_timesteps, 3) -- ensemble across seeds
    ylim: optional (lo, hi) applied to all panels, or a sequence of (lo, hi) per panel
    """
    if legend_fontsize is None:
        legend_fontsize = fontsize - 2

    mean_values = np.mean(stacked_values, axis=0)

    fig, axs = plt.subplots(3, 1, figsize=(18, 18), sharex=True)

    for i, label in enumerate(labels):
        for j, seed_values in enumerate(stacked_values):
            axs[i].plot(t, seed_values[:, i], color='grey', alpha=alpha, label=titles[0] if j == 0 else None)
        axs[i].plot(t, mean_values[:, i], color='black', linestyle='--', label=titles[1])
        axs[i].plot(t, true_values[:, i], color='darkred', label=titles[2])
        axs[i].set_ylabel(label, fontsize=fontsize)
        axs[i].tick_params(labelsize=fontsize - 2)
        if ylim is not None:
            axs[i].set_ylim(ylim[i] if isinstance(ylim[0], (tuple, list)) else ylim)

    axs[0].legend(fontsize=legend_fontsize)
    axs[-1].set_xlabel('time', fontsize=fontsize)
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(f"./figures/{savefig}")

    return fig, axs



def plot_comparison(t, X_states, sol_1, sol_2=None, titles = ('0', '1', '2'), labels=('X', 'Y', 'Z'), fontsize=22, savefig = None, ylim = None):
    ncols = 1 if sol_2 is None else 2
    figsize = (12, 15) if ncols == 1 else (20, 20)
    fig, axs = plt.subplots(3, ncols, figsize=figsize, sharex=True, sharey='row')
    if ncols == 1:
        axs = axs.reshape(3, 1)

    for i, label in enumerate(labels):
        axs[i, 0].plot(t, sol_1.y[i, :], color='firebrick', label=titles[0])
        axs[i, 0].plot(t, X_states[:, i], color='black', linestyle='--', alpha=0.6, label=titles[2])
        axs[i, 0].set_ylabel(label, fontsize=fontsize)
        axs[i, 0].tick_params(labelsize=fontsize - 2)

        if ylim is not None:
            axs[i, 0].set_ylim(ylim)  # sharey='row' propagates this to col 1

        if sol_2 is not None:
            axs[i, 1].plot(t, sol_2.y[i, :], color='steelblue', label=titles[1])
            axs[i, 1].plot(t, X_states[:, i], color='black', linestyle='--', alpha=0.6, label=titles[2])
            axs[i, 1].tick_params(labelsize=fontsize - 2)

    axs[-1, 0].set_xlabel('time', fontsize=fontsize)
    axs[0, 0].legend(fontsize=fontsize - 2)

    if sol_2 is not None:
        axs[0, 1].set_title(titles[1], fontsize=fontsize + 2)
        axs[-1, 1].set_xlabel('time', fontsize=fontsize)
        axs[0, 1].legend(fontsize=fontsize - 2)

    plt.tight_layout()
    if savefig is not None:
        plt.savefig(f"./figures/{savefig}")

    return fig, axs

def rmse_vs_integration_step(t, X_states, sol_A, sol_B, labels=('X', 'Y', 'Z'),
                              titles=('0', '1'), window=50,
                              colors=('black', 'grey', 'orange'), fontsize=22, savefig = None):
    def sliding_rmse(true, pred, window):
        squared_error = (pred - true) ** 2
        kernel = np.ones(window) / window
        mse = np.convolve(squared_error, kernel, mode='valid')
        return np.sqrt(mse)

    t_rmse = t[window - 1:]

    fig, axs = plt.subplots(1, 2, figsize=(18, 7), sharex=True, sharey=True)

    for i, label in enumerate(labels):
        true = X_states[:, i]
        rmse_A = sliding_rmse(true, sol_A.y[i, :], window)
        rmse_B = sliding_rmse(true, sol_B.y[i, :], window)

        axs[0].plot(t_rmse, rmse_A, color=colors[i], label=label)
        axs[1].plot(t_rmse, rmse_B, color=colors[i], label=label)

    axs[0].set_title(titles[0], fontsize=fontsize + 2)
    axs[1].set_title(titles[1], fontsize=fontsize + 2)
    axs[0].set_ylabel('RMSE', fontsize=fontsize)
    axs[0].set_xlabel('time', fontsize=fontsize)
    axs[1].set_xlabel('time', fontsize=fontsize)
    axs[0].tick_params(labelsize=fontsize - 2)
    axs[1].tick_params(labelsize=fontsize - 2)
    axs[0].legend(fontsize=fontsize - 2)
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(f"./figures/{savefig}")

    return fig, axs

def plot_state_space(X_states, sol_1, sol_2=None, titles = ('0', '1', '2'), colors=('black', 'firebrick', 'steelblue'), fontsize=22, savefig = None):
    pairs = [(0, 1, 'X', 'Y'), (0, 2, 'X', 'Z'), (1, 2, 'Y', 'Z')]

    fig, axs = plt.subplots(1, 3, figsize=(24, 7))

    for ax, (i, j, label_i, label_j) in zip(axs, pairs):
        ax.plot(X_states[:, i], X_states[:, j], color=colors[0], alpha=0.7, label=titles[2])
        ax.plot(sol_1.y[i, :], sol_1.y[j, :], color=colors[1], alpha=0.7, label=titles[0])
        if sol_2 is not None:
            ax.plot(sol_2.y[i, :], sol_2.y[j, :], color=colors[2], alpha=0.7, label=titles[1])

        ax.set_xlabel(label_i, fontsize=fontsize)
        ax.set_ylabel(label_j, fontsize=fontsize)
        ax.tick_params(labelsize=fontsize - 2)

    axs[0].legend(fontsize=fontsize - 2)
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(f"./figures/{savefig}")

    return fig, axs


def plot_frequency_spectrum(X_states, sol_1, sol_2=None, titles = ('0', '1', '2'), colors=('black', 'firebrick', 'steelblue'), fontsize=22, savefig = None):
    def spectrum(series, d=1.0):
        fourier = np.fft.fft(series)
        coeff = np.abs(fourier)
        freq = np.fft.fftfreq(n=len(series), d=d)
        mask = freq >= 0
        return freq[mask], coeff[mask]

    freq_true, coeff_true = spectrum(X_states[:, 0])
    freq_fin, coeff_fin = spectrum(sol_1.y[0, :])

    panels = [(freq_fin, coeff_fin, colors[1], titles[0])]
    if sol_2 is not None:
        freq_spl, coeff_spl = spectrum(sol_2.y[0, :])
        panels.append((freq_spl, coeff_spl, colors[2], titles[1]))

    ncols = len(panels)
    figsize = (18, 7) if ncols == 2 else (9, 7)
    fig, axs = plt.subplots(1, ncols, figsize=figsize, sharex=True, sharey=True)
    if ncols == 1:
        axs = [axs]

    for ax, (freq, coeff, color, label) in zip(axs, panels):
        ax.plot(freq_true, coeff_true, color=colors[0], label=titles[2])
        ax.plot(freq, coeff, color=color, label=label)
        ax.vlines(1/100, ymin=coeff_true.min(), ymax=coeff_true.max(),
                  linestyle='dashed', color='darkblue', alpha=0.5, label='100 kyr cycle')
        ax.vlines(1/41, ymin=coeff_true.min(), ymax=coeff_true.max(),
                  linestyle='dashed', color='slateblue', alpha=0.5, label='41 kyr cycle')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel('frequency [1/kyr]', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize - 2)
        ax.legend(fontsize=fontsize - 4)

    axs[0].set_ylabel('coefficients', fontsize=fontsize)
    axs[0].set_title(titles[0], fontsize=fontsize + 2)
    if ncols == 2:
        axs[1].set_title(titles[1], fontsize=fontsize + 2)
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(f"./figures/{savefig}")

    return fig, axs

def C_2(true_states, predicted_states):
    true_std = np.std(true_states, axis=0)
    pred_std = np.std(predicted_states, axis=0)
    return np.mean(np.abs(1.0 - (pred_std / (true_std + 1e-6))))



# Shared look for plot_timeseries, GPR.sanity, GPR.plot_gp_spaghetti and customSINDy.plot_cv
PLOT_STYLE = {
    'figure.dpi': 150,
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linewidth': 0.6,
    'lines.linewidth': 1.2, 'lines.markersize': 3,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'axes.titleweight': 'bold',
}

def styled(func):
    """Run a plotting function with PLOT_STYLE applied."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Set up the backend (e.g. Jupyter's inline one) before the style context, so that leaving the
        # context cannot undo that setup and stop later figures from showing.
        plt.get_backend()
        with mpl.rc_context(PLOT_STYLE):
            result = func(*args, **kwargs)
            interactive = mpl.is_interactive()
        mpl.interactive(interactive)
        return result
    return wrapper

# Fixed margins. Figures saved with the same width and margins have plot boxes of exactly the same size,
# whether or not they show y labels, so they line up side by side in LaTeX.
# Left/right are fixed in inches (room for the y labels does not grow with the figure), so wide figures stay
# centred; at width 4 in they equal the old fractions 0.24 / 0.96. Bottom/top/hspace are fractions.
DEFAULT_MARGIN_INCHES = dict(left=0.96, right=0.16)
DEFAULT_MARGINS = dict(bottom=0.08, top=0.98, hspace=0.12)

def finish_layout(fig, margins=None):
    """margins=None: tight_layout (boxes shrink to fit their labels).
    margins=True: DEFAULT_MARGIN_INCHES + DEFAULT_MARGINS;
    margins=dict(...): those defaults with these values replaced (given as fractions of the figure)."""
    if margins is None or margins is False:
        fig.tight_layout()
    else:
        width = fig.get_figwidth()
        defaults = dict(left=DEFAULT_MARGIN_INCHES['left'] / width,
                        right=1 - DEFAULT_MARGIN_INCHES['right'] / width, **DEFAULT_MARGINS)
        fig.subplots_adjust(**{**defaults, **(margins if isinstance(margins, dict) else {})})

def save_figure(fig, save_path, dpi=300):
    """Save at high resolution; the figure size is kept as is so side-by-side figures still line up.
    Use a .pdf path for vector output (sharpest in LaTeX)."""
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi)

@styled
def plot_timeseries(x, ys, ys_true=None, x_true=None, colors=None, labels=None, legends=None, xlabel="Age [ka BP]", width=8, figsize_per_panel=2, ylim = None, true_as_points=False, invert_xaxis = False, yticks=True, save_path=None, margins=None, true_color='black', bands=None, band_color='#f2b8b5'):
    colors = colors or ["firebrick"]
    n = len(ys)
    x_true = x if x_true is None else x_true
    per_panel = np.ndim(next(iter(x_true))) > 0          # True if x_true is a list of time arrays, one per series

    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(width, figsize_per_panel * n))
    if n == 1:
        axes = [axes]

    lab_y    = legends[0] if legends else None
    lab_true = legends[1] if legends and len(legends) > 1 else None
    lab_band = legends[2] if legends and len(legends) > 2 else None

    for i, (ax, y) in enumerate(zip(axes, ys)):
        first = (i == 0)
        if bands is not None:                            # optional (lo, hi) band per panel, drawn underneath
            lo, hi = bands[i]
            ax.fill_between(x, lo, hi, color=band_color, linewidth=0, label=lab_band if first else None)
        ax.plot(x, y, color=colors[i % len(colors)], label=lab_y if first else None)
        if ys_true is not None:
            xt = x_true[i] if per_panel else x_true
            style = dict(linestyle='none', marker='x', markersize=3) if true_as_points else dict(linestyle='--')
            ax.plot(xt, ys_true[i], color=true_color, label=lab_true if first else None, **style)
        if legends and first:
            ax.legend()
        if labels:
            ax.set_ylabel(labels[i])
        if ylim is not None:
            per_panel_ylim = any(l is None or np.ndim(l) > 0 for l in ylim)
            lim = ylim[i] if per_panel_ylim else ylim   # one (lo, hi) for all panels, or a list with one per panel
            if lim is not None:
                ax.set_ylim(lim)
        if not yticks:
            ax.tick_params(axis='y', left=False, labelleft=False)


    axes[-1].set_xlabel(xlabel)
    if invert_xaxis:
        axes[-1].invert_xaxis() 
    finish_layout(fig, margins)
    save_figure(fig, save_path)
    return fig, axes


def _term_tex(term, names=None):
    r"""'Y*Y*Z' -> '$Y^2 Z$'; names maps variable names to LaTeX (e.g. {'Esi': r'\Pi_P'})."""
    if term == '1':
        return '$1$'
    names = names or {}
    factors = term.split('*')
    parts = []
    for v in dict.fromkeys(factors):                                   # keep order, count powers
        n = factors.count(v)
        tex = names.get(v, v)
        parts.append(tex + (f'^{n}' if n > 1 else ''))
    return '$' + r'\,'.join(parts) + '$'


@styled
def plot_term_contributions(models, Theta, dots=None, names=None, absolute=False, logy=False, whis=(0, 100), ylabels=None, ylim=None,
                            yticklabels=True, row=False, width=4, figsize_per_panel=2, save_path=None, margins=None):
    r"""Box plots of each term's contribution c_k * theta_k(t) over time, one panel per equation.
    models: dict eq -> model Series (term -> coefficient), e.g. {'X': Xmodels[4], ...}
    Theta:  library the models are evaluated on (DataFrame, columns = term names)
    dots:   optional dict eq -> derivative; drawn as a grey reference box first
    names:  LaTeX names for variables in the term labels, e.g. {'Esi': r'\Pi_P', 'Eco': r'\Pi_c'}
    absolute=True plots |c_k * theta_k(t)| (size of each term's contribution; logy=True for a log axis).
    row=True puts the panels side by side, each 20% taller than in a column (width = whole figure width).
    Box = 25-75th percentile, line = median, whiskers = whis percentiles (default min / max)."""
    n = len(models)
    if row:
        fig, axes = plt.subplots(1, n, figsize=(width, 1.2 * figsize_per_panel))
    else:
        fig, axes = plt.subplots(n, 1, figsize=(width, figsize_per_panel * n))
    axes = np.atleast_1d(axes)
    box = dict(whis=whis, showfliers=False, patch_artist=True, widths=0.6,
               medianprops=dict(color='black', lw=1.2),
               whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))

    for i, (ax, (eq, w)) in enumerate(zip(axes, models.items())):
        Th = Theta[eq] if isinstance(Theta, dict) else Theta      # per-equation library, or one shared library
        data = [w[t] * Th[t].to_numpy() for t in w.index]
        labels = [_term_tex(t, names) for t in w.index]
        colors = ['#f2b8b5'] * len(data)                                 # GP_BAND: identified terms
        if dots is not None:
            data = [np.asarray(dots[eq])] + data
            labels = [rf'$\dot{{{eq}}}$'] + labels
            colors = ['0.85'] + colors                                    # grey: reference derivative
        if absolute:
            data = [np.abs(d) for d in data]
        bp = ax.boxplot(data, positions=np.arange(1, len(data) + 1), **box)
        for patch, c in zip(bp['boxes'], colors):
            patch.set(facecolor=c, edgecolor='black', lw=0.8)
        if logy:
            ax.set_yscale('log')
        if not absolute:
            ax.axhline(0, color='0.5', lw=0.6, zorder=0)
        if dots is not None:
            ax.axvline(1.5, color='0.5', lw=0.6, ls='--')
        ax.set_xticks(np.arange(1, len(data) + 1))
        ax.set_xticklabels(labels)
        ax.grid(axis='x', visible=False)
        if ylabels and yticklabels:
            ax.set_ylabel(ylabels[i])
        if ylim is not None:
            lim = ylim[i] if any(l is None or np.ndim(l) > 0 for l in ylim) else ylim
            if lim is not None:
                ax.set_ylim(lim)
        if not yticklabels:
            ax.tick_params(axis='y', which='both', labelleft=False)
    finish_layout(fig, margins)
    save_figure(fig, save_path)
    return fig, axes
