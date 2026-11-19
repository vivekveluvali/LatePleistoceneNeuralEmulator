import numpy as np
import pandas as pd
import scipy.interpolate as sciin
import sklearn as sk
from sklearn.gaussian_process.kernels import Matern
import matplotlib.pyplot as plt
from tools import styled, save_figure, finish_layout

# GP colours, shared by sanity and plot_gp_spaghetti: dark red for the mean, lighter reds for its uncertainty
GP_COLOR = 'darkred'    # GP mean
GP_BAND  = '#f2b8b5'    # ±1 std band
GP_DRAWS = '#e06c66'    # MC draws

def fitGPRtodata(x, y, x_fine, variance=None):
    # normalize_y=True: Matern/RBF have no independent amplitude term (k(x,x)=1
    # is fixed), so without this the kernel's implicit unit variance can badly
    # mismatch a dataset's raw scale and the noise term ends up absorbing most
    # of the real signal, flattening the fit.
    if variance is not None:
        variance = variance / np.var(y)  # normalize variance to the same scale as normalize_y
        kernel = Matern(length_scale=50, length_scale_bounds=(1, 1000), nu=2.5)
        gpr = sk.gaussian_process.GaussianProcessRegressor(kernel=kernel, alpha=variance,
                                                             n_restarts_optimizer=5, normalize_y=True)
    else:
        kernel = (Matern(length_scale=50, length_scale_bounds=(1, 1000), nu=2.5)
                  + sk.gaussian_process.kernels.WhiteKernel(noise_level_bounds=(1e-5, 1e3)))
        gpr = sk.gaussian_process.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)

    gpr.fit(x.reshape(-1, 1), y)
    y_fine, y_std = gpr.predict(x_fine.reshape(-1, 1), return_std=True)
    return y_fine, y_std, gpr


def _matern_length_scale(kernel_):
    """Pull the length scale out of the fitted kernel, whether it's a bare
    Matern or a Sum(Matern, WhiteKernel)."""
    components = [kernel_.k1, kernel_.k2] if hasattr(kernel_, "k1") else [kernel_]
    for k in components:
        if isinstance(k, Matern):
            return k.length_scale
    raise ValueError("fitted kernel has no Matern component")


def gpr_derivative(gpr, x_query, y):
    """Analytic d(GP posterior mean)/dx at x_query, for a Matern(nu=2.5)-based
    GaussianProcessRegressor fitted with normalize_y=True (see markdown above).
    `y` is the raw (un-normalized) training target array passed to fitGPRtodata,
    needed to reproduce the y_std that normalize_y applied internally.
    """
    length_scale = _matern_length_scale(gpr.kernel_)
    a = np.sqrt(5) / length_scale
    X_train = gpr.X_train_.ravel()
    alpha = gpr.alpha_.ravel()
    y_std = y.std()  # ddof=0, matching normalize_y's internal convention

    d = x_query.reshape(-1, 1) - X_train.reshape(1, -1)
    abs_d = np.abs(d)
    dk_dx = -(a**2 / 3) * d * (1 + a * abs_d) * np.exp(-a * abs_d)
    return y_std * (dk_dx @ alpha)

@styled
def sanity(series, window, tgrid, xlabel="Age [ka BP]", width=8, figsize_per_panel=2, ylim=None, yticks=True,
           save_path=None, margins=None):
    # Same layout as tools.plot_timeseries so the two figures line up side by side.
    n = len(series)
    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(width, figsize_per_panel * n))
    if n == 1:
        axes = [axes]

    grid_mask = (tgrid >= window[0]) & (tgrid <= window[1])
    for i, (ax, (label, (x_raw, y_raw, mean, std))) in enumerate(zip(axes, series.items())):
        first = (i == 0)
        raw_mask = (x_raw >= window[0]) & (x_raw <= window[1])

        ax.fill_between(tgrid[grid_mask], (mean - std)[grid_mask], (mean + std)[grid_mask],
                        color=GP_BAND, linewidth=0, label='GP mean ± 1 std' if first else None)
        ax.plot(tgrid[grid_mask], mean[grid_mask], color=GP_COLOR, label='GP mean' if first else None)
        ax.plot(x_raw[raw_mask], y_raw[raw_mask], linestyle='none', marker='x', markersize=3,
                color='black', label='Proxy sampled' if first else None)
        if first:
            ax.legend()
        ax.set_ylabel(label)
        if ylim is not None:
            per_panel_ylim = any(l is None or np.ndim(l) > 0 for l in ylim)
            lim = ylim[i] if per_panel_ylim else ylim   # one (lo, hi) for all panels, or a list with one per panel
            if lim is not None:
                ax.set_ylim(lim)
        if not yticks:
            ax.tick_params(axis='y', left=False, labelleft=False)
            ax.set_ylabel('')        # also drop the axis label

    axes[-1].set_xlabel(xlabel)
    axes[-1].invert_xaxis()  # oldest on the left, present on the right
    finish_layout(fig, margins)
    save_figure(fig, save_path)
    return fig, axes


''' 
MC draws
'''

from scipy.linalg import cho_factor, cho_solve


def _matern52_blocks(t1, t2, ell):
    """Matern 5/2 kernel k(t1,t2), dk/dt1 and d2k/dt1dt2 (unit amplitude)."""
    tau = t1[:, None] - t2[None, :]
    r = np.abs(tau)
    a = np.sqrt(5.0) / ell
    e = np.exp(-a * r)
    k   = (1 + a*r + (a*r)**2 / 3) * e
    dk1 = -(a**2 / 3) * tau * (1 + a*r) * e          # dk/dt1  (dk/dt2 = -dk1)
    d2k = (a**2 / 3) * (1 + a*r - (a*r)**2) * e      # d2k/dt1dt2
    return k, dk1, d2k


def gp_joint_posterior(gpr, X, y, tgrid):
    """Posterior mean and covariance of the latent [f(tgrid); f'(tgrid)], original units."""
    kern = gpr.kernel_
    matern = kern.k1 if hasattr(kern, "k1") else kern          # strip WhiteKernel if present
    if not np.isclose(matern.nu, 2.5):
        raise ValueError(f"Expected Matern nu=2.5, got {matern.nu}")
    ell = float(matern.length_scale)
    white = kern.k2.noise_level if hasattr(kern, "k2") else 0.0

    X = np.asarray(X, float).ravel()
    y = np.asarray(y, float).ravel()
    y_mean, y_sd = y.mean(), y.std()                 # same as normalize_y=True
    yn = (y - y_mean) / y_sd

    # Training covariance: Matern + observation noise (alpha and/or WhiteKernel)
    Kxx, _, _ = _matern52_blocks(X, X, ell)
    Kxx[np.diag_indices_from(Kxx)] += np.broadcast_to(gpr.alpha, X.shape) + white
    cf = cho_factor(Kxx, lower=True)

    # Covariance between [f; f'] on the grid and the observations
    k_gx, dk_gx, _ = _matern52_blocks(tgrid, X, ell)
    Kcross = np.vstack([k_gx, dk_gx])                # (2N, M)

    # Prior covariance of [f; f'] on the grid (latent, no observation noise)
    k_gg, dk_gg, d2k_gg = _matern52_blocks(tgrid, tgrid, ell)
    Kprior = np.block([[k_gg, -dk_gg],
                       [dk_gg, d2k_gg]])

    mu_n  = Kcross @ cho_solve(cf, yn)
    cov_n = Kprior - Kcross @ cho_solve(cf, Kcross.T)
    cov_n = 0.5 * (cov_n + cov_n.T)                  # enforce symmetry

    N = len(tgrid)
    mu = y_sd * mu_n
    mu[:N] += y_mean                                 # offset applies to f, not f'
    return mu, y_sd**2 * cov_n, ell


def draw_joint_samples(mu, cov, n, rng):
    """n joint draws of [x; xdot]. Returns two arrays of shape (n, N)."""
    w, V = np.linalg.eigh(cov)
    L = V * np.sqrt(np.clip(w, 0.0, None))           # matrix square root, L @ L.T = cov
    z = mu[:, None] + L @ rng.standard_normal((len(mu), n))
    N = len(mu) // 2
    return z[:N].T, z[N:].T


@styled
def plot_gp_spaghetti(samples, tgrid, data=None, n_show=30, tmax=800, labels=None, derivative=False, seed=0, color=None,
                      invert_x=True, means=None, window=None, xlabel="Age [ka BP]", width=8, figsize_per_panel=2,
                      ylim=None, yticks=True, save_path=None, margins=None):
    """Spaghetti plot of GP draws, laid out like tools.plot_timeseries so the figures line up side by side.
    samples: {name: {"x": (n_draws, len(tgrid)), "xdot": (n_draws, len(tgrid))}}
    tgrid:   ages (kyr BP) of the grid the draws are on
    means:   {name: GP mean on tgrid}; if given, the mean is drawn as a line and the proxy points are left out
    data:    {name: (age, value)} proxy records, drawn as crosses when means is not given
    n_show:  number of draws shown
    window:  (lo, hi) age range shown; if None, everything up to tmax (kyr BP) is shown (tmax=None: full grid)
    labels:  optional {name: (ylabel, derivative_ylabel)};  derivative=True adds a second column with the xdot draws
    ylim:    one (lo, hi) for every panel, or a list with one per row (None entries stay automatic); first column only
    yticks:  False hides the y tick marks and numbers"""
    color = GP_DRAWS if color is None else color
    tgrid = np.asarray(tgrid)
    lo, hi = window if window is not None else (-np.inf, tmax if tmax is not None else np.inf)
    in_range = lambda a: (a >= lo) & (a <= hi)
    mask = in_range(tgrid)
    t = tgrid[mask]
    names = list(samples)
    ncol = 2 if derivative else 1
    fig, axes = plt.subplots(len(names), ncol, figsize=(width * ncol, figsize_per_panel * len(names)),
                             sharex=True, squeeze=False)
    for row, name in enumerate(names):
        first = (row == 0)
        x_s = samples[name]["x"][:, mask]
        idx = np.random.default_rng(seed).choice(len(x_s), min(n_show, len(x_s)), replace=False)
        ylab, dlab = (labels or {}).get(name, (name, f"d{name}/d(age)"))

        ax = axes[row, 0]
        ax.plot(t, x_s[idx].T, color=color, lw=0.5, alpha=0.5)
        if first:
            ax.plot([], [], color=color, lw=1, label=f"{len(idx)} MC draws")
        if means is not None:
            ax.plot(t, np.ravel(means[name])[mask], color=GP_COLOR, label="GP mean" if first else None)
        elif data is not None:
            age, val = (np.asarray(a) for a in data[name])
            keep = in_range(age)
            ax.plot(age[keep], val[keep], linestyle="none", marker="x", markersize=3, color="black",
                    label="Proxy sampled" if first else None)
        if first:
            ax.legend()
        ax.set_ylabel(ylab)
        if ylim is not None:
            per_panel_ylim = any(l is None or np.ndim(l) > 0 for l in ylim)
            lim = ylim[row] if per_panel_ylim else ylim   # one (lo, hi) for all panels, or a list with one per panel
            if lim is not None:
                ax.set_ylim(lim)

        if derivative:
            axes[row, 1].plot(t, samples[name]["xdot"][:, mask][idx].T, color=color, lw=0.5, alpha=0.6)
            axes[row, 1].axhline(0, color="0.5", lw=0.5); axes[row, 1].set_ylabel(dlab)

        if not yticks:
            for a in axes[row]:
                a.tick_params(axis='y', left=False, labelleft=False)
                a.set_ylabel('')        # also drop the axis label

    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    if invert_x:
        axes[0, 0].invert_xaxis()        # the axes share x, so this flips every panel: oldest age on the left

    finish_layout(fig, margins)
    save_figure(fig, save_path)
    return fig, axes

