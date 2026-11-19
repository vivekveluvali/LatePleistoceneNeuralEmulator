from sklearn.linear_model import ridge_regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tools import styled, save_figure, finish_layout
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from sklearn.preprocessing import PolynomialFeatures
import pysindy as ps
from gurobipy import GRB

'''
Libraries and forcing
'''

def make_crucifix_forcing(df, a):
    w = (df['age_kyr'] <= 1000).values  # the paper normalises over the last 1000 kyr (the file extends further back)
    norm = lambda x: (x - x[w].mean()) / x[w].std()
    Esi, Eco, O = (norm(df[c].values) for c in ('Pi_P_raw', 'Pi_C_raw', 'E_raw'))
    f = lambda x: np.where(x <= 0, x + np.sqrt(4*a**2 + x**2) - 2*a, x) # for the pp12 transformed columns
    cols = dict(Esi=Esi, Eco=Eco, O=O, Esi_tr=norm(f(Esi)), Eco_tr=norm(f(Eco)))
    return {n: interp1d(-df['age_kyr'].values, v, kind='cubic') for n, v in cols.items()}  # t = -age (kyr)

def _rename(c):                                   # 'X^2 Y' -> 'X*X*Y', 'X^3' -> 'X*X*X'
    parts = []
    for tok in c.split(' '):
        v, _, p = tok.partition('^')
        parts += [v] * (int(p) if p else 1)
    return '*'.join(parts)

def build_theta_SM90(X, Y, Z, R, degree=3):
    """Polynomial library in the three states and a single forcing variable R.
    All inputs are 1-D arrays on the same time grid; returns the DataFrame and the list of its column names."""
    variables = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z, 'R': R})
    pf = PolynomialFeatures(degree=degree, include_bias=True)
    Theta = pd.DataFrame(pf.fit_transform(variables), columns=pf.get_feature_names_out(variables.columns))
    Theta.columns = [_rename(c) if c != '1' else '1' for c in Theta.columns]
    return Theta, list(Theta.columns)

def build_theta_crucifix(X, Y, Z, Esi, Eco, O, degree=3, max_forcing_degree=None, max_forcing_term_degree=None):
    """Polynomial library in the three states and the three forcing components.
    max_forcing_degree:      max number of forcing factors in a term (e.g. 2 drops Esi^3, Esi*Eco*O)
    max_forcing_term_degree: max total degree of any term that contains forcing (e.g. 2 drops X^2*Esi, X*Esi^2)"""
    variables = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z, 'Esi': Esi, 'Eco': Eco, 'O': O})
    pf = PolynomialFeatures(degree=degree, include_bias=True)
    Theta = pd.DataFrame(pf.fit_transform(variables), columns=pf.get_feature_names_out(variables.columns))
    Theta.columns = [_rename(c) if c != '1' else '1' for c in Theta.columns]

    powers = pf.powers_                        # one row per term: exponent of each variable
    f_deg = powers[:, 3:].sum(axis=1)          # forcing factors per term
    t_deg = powers.sum(axis=1)                 # total degree per term
    keep = np.ones(len(powers), bool)
    if max_forcing_degree is not None:
        keep &= f_deg <= max_forcing_degree
    if max_forcing_term_degree is not None:
        keep &= (f_deg == 0) | (t_deg <= max_forcing_term_degree)
    Theta = Theta.loc[:, keep]
    return Theta, list(Theta.columns)




''' 
SINDy helpers
'''

def _ridge_step(Theta, Y, support, alpha=0.05, normalize=False):
    A = Theta[support].to_numpy()
    scale = np.linalg.norm(A, axis=0) if normalize else np.ones(A.shape[1])   
    w = ridge_regression(A / scale, Y, alpha) / scale                         
    return pd.Series(w, index=support)

def _drop_by_coefficient(Theta, Y, support, alpha=0.05, normalize=False):
    coef = _ridge_step(Theta, Y, support, alpha, normalize)
    if normalize:
        coef = coef * np.linalg.norm(Theta[support].to_numpy(), axis=0)   
    return coef.abs().idxmin()

def _drop_by_residual(Theta, Y, support, alpha=0.05, normalize=False):
    rss = {}
    for term in support:
        s = [t for t in support if t != term]
        coef = _ridge_step(Theta, Y, s, alpha, normalize)
        rss[term] = np.sum((Y - Theta[s].to_numpy() @ coef.to_numpy()) ** 2)
    return min(rss, key=rss.get)

def _cross_validation_SSR(Theta, Y, support, n_blocks, alpha = 0.05, normalize = False):

    folds = np.array_split(np.arange(len(Y)), n_blocks)   
    MSEs = []
    records = {}
    for fold, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(np.arange(len(Y)), test_idx)
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        Theta_train, Theta_test = Theta.iloc[train_idx], Theta.iloc[test_idx]
        w = _ridge_step(Theta_train, Y_train, support, alpha=alpha, normalize=normalize)
        pred = Theta_test[support].to_numpy() @ w.to_numpy()
        MSE = np.mean((pred - Y_test)**2)
        MSEs.append(MSE)
        records[fold] = w.reindex(Theta.columns)
    
    MMSE = np.mean(MSEs)
    coefs = pd.DataFrame(records).T                 
    coefs.index.name = 'fold'

    return MMSE, coefs


def _cross_validation_MIOSR(Theta, Y, k, n_blocks, alpha=1e-8, normalize=False, timeout=30):
    Y = np.asarray(Y).ravel()
    folds = np.array_split(np.arange(len(Y)), n_blocks)
    MSEs, records, optimal = [], {}, []
    for fold, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(np.arange(len(Y)), test_idx)
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        Theta_train, Theta_test = Theta.iloc[train_idx], Theta.iloc[test_idx]

        A = Theta_train.to_numpy()
        scale = np.linalg.norm(A, axis=0) if normalize else np.ones(A.shape[1])
        opt = ps.MIOSR(target_sparsity=k, alpha=alpha, regression_timeout=timeout)
        opt.fit(A / scale, Y_train)
        w = pd.Series(opt.coef_[0] / scale, index=Theta.columns)
        optimal.append(opt.model.Status == GRB.OPTIMAL)

        pred = Theta_test.to_numpy() @ w.to_numpy()
        MSEs.append(np.mean((pred - Y_test) ** 2))
        records[fold] = w

    coefs = pd.DataFrame(records).T
    coefs.index.name = "fold"
    coefs["optimal"] = optimal
    return np.mean(MSEs), coefs




''' 
CustomSSR
'''

def customSSR(Y, Theta, iterations, n_blocks, drop_by_residual = False, alpha = 0.05, normalize = False):

    Y = np.asarray(Y)
    support = list(Theta.columns)

    MMSEs = {}
    dfs = {}
    models = {} 
    R2s = {}
    for i in range(iterations):

        w = _ridge_step(Theta, Y, support, alpha=alpha, normalize=normalize)
        pred = Theta[w.index].to_numpy() @ w.to_numpy()   # new

        MMSE, coefs = _cross_validation_SSR(Theta, Y, support, n_blocks, alpha = alpha, normalize = normalize)
        n = len(support)
        models[n], (MMSEs[n], dfs[n]) = w, (MMSE, coefs)
        R2s[n] = 1 - np.sum((Y - pred) ** 2) / np.sum((Y - Y.mean()) ** 2)

        if drop_by_residual:
            drop = _drop_by_residual(Theta, Y, support, alpha=alpha, normalize=normalize)
        else:
            drop = _drop_by_coefficient(Theta, Y, support, alpha=alpha, normalize=normalize)
    
        support.remove(drop)

    return models, MMSEs, R2s, dfs



''' 
customMOISR
'''

def customMIOSR(Y, Theta, sparsity_range, n_blocks, alpha=1e-8, normalize=False, timeout=30):
    Y = np.asarray(Y).ravel()
    A = Theta.to_numpy()
    scale = np.linalg.norm(A, axis=0) if normalize else np.ones(A.shape[1])

    MMSEs, dfs, models, R2s, optimal = {}, {}, {}, {}, {}
    for k in range(sparsity_range[0], sparsity_range[1] + 1):
        MMSE, coefs = _cross_validation_MIOSR(Theta, Y, k, n_blocks, alpha=alpha,
                                              normalize=normalize, timeout=timeout)

        opt = ps.MIOSR(target_sparsity=k, alpha=alpha, regression_timeout=timeout)
        opt.fit(A / scale, Y)
        w = pd.Series(opt.coef_[0] / scale, index=Theta.columns)
        optimal[k] = opt.model.Status == GRB.OPTIMAL

        n_bad = (~coefs["optimal"]).sum() + (not optimal[k])
        if n_bad:
            print(f"k={k}: {n_bad} fit(s) not proven optimal")

        pred = A @ w.to_numpy()
        models[k], MMSEs[k], dfs[k] = w[w != 0], MMSE, coefs
        R2s[k] = 1 - np.sum((Y - pred) ** 2) / np.sum((Y - Y.mean()) ** 2)

    return models, MMSEs, R2s, dfs, optimal



''' 
Results
'''

@styled
def plot_cv(MMSEs, logy=True, title=None, xlabel='', ylabel='', best=True, sel=None, selstring='', save_path=None, ylim=None, yticklabels=True, legend=True):
    """CV curve: held-out MSE against the number of terms, minimum marked."""
    n = sorted(MMSEs)
    best_line = min(MMSEs, key=MMSEs.get)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(n, [MMSEs[k] for k in n], 'o-', color='black')
    if best:
        ax.axvline(best_line, color='firebrick', ls='--', label=f'min at n = {best}')
    if sel is not None:
        ax.axvline(sel, color='grey', ls='--', label=f'{selstring} = {sel}')
    ax.set_xlabel(f"{xlabel}"); ax.set_ylabel(f"{ylabel}")
    if legend and (best or sel is not None): ax.legend()
    ax.set_title(title)
    if logy: ax.set_yscale('log')
    if ylim is not None: ax.set_ylim(ylim)   # set after the scale; on a log axis the lower limit must be > 0
    if not yticklabels: ax.tick_params(axis='y', which='both', labelleft=False)   # keep the ticks, hide the numbers
    fig.tight_layout()
    save_figure(fig, save_path)
    return fig, ax

@styled
def plot_cv_column(MMSEs_list, logy=True, ylabels=None, xlabel='', ylim=None, best=False, sel=None, selstring='',
                   legend=True, yticklabels=True, width=5, figsize_per_panel=2, title=None, save_path=None,
                   margins=None):
    """CV curves for several equations stacked in one column (one panel per equation, shared x-axis).
    MMSEs_list: list of {n_terms: held-out MSE}, one dict per panel (e.g. [XMMSEs, YMMSEs, ZMMSEs])
    ylabels:    list with one y-label per panel;  xlabel: shown under the bottom panel only
    ylim:       one (lo, hi) for every panel, or a list with one per panel (None entries stay automatic)
    best:       mark the CV minimum of each panel (red dashed line)
    sel:        list with one selected n per panel (grey dashed line); None entries draw no line
    selstring:  legend text for the sel line;  legend: show the legend (top panel only)
    yticklabels=False keeps the y ticks but hides the numbers;  title goes above the top panel"""
    n_panels = len(MMSEs_list)
    fig, axes = plt.subplots(n_panels, 1, sharex=True, figsize=(width, figsize_per_panel * n_panels))
    if n_panels == 1:
        axes = [axes]
    sels = sel if sel is not None else [None] * n_panels

    for i, (ax, MMSEs) in enumerate(zip(axes, MMSEs_list)):
        first = (i == 0)
        n = sorted(MMSEs)
        ax.plot(n, [MMSEs[k] for k in n], 'o-', color='black')
        if best:
            ax.axvline(min(MMSEs, key=MMSEs.get), color='firebrick', ls='--', label='CV minimum' if first else None)
        if sels[i] is not None:
            ax.axvline(sels[i], color='grey', ls='--', label=(selstring or 'selected') if first else None)
        if ylabels:
            ax.set_ylabel(ylabels[i])
        if logy:
            ax.set_yscale('log')
        if ylim is not None:
            per_panel_ylim = any(l is None or np.ndim(l) > 0 for l in ylim)
            lim = ylim[i] if per_panel_ylim else ylim   # set after the scale; on a log axis the lower limit must be > 0
            if lim is not None:
                ax.set_ylim(lim)
        if not yticklabels:
            ax.tick_params(axis='y', which='both', labelleft=False)   # keep the ticks, hide the numbers

    if legend and (best or any(s is not None for s in sels)):
        axes[0].legend()
    if title:
        axes[0].set_title(title)
    axes[-1].set_xlabel(xlabel)
    finish_layout(fig, margins)
    save_figure(fig, save_path)
    return fig, axes


def all_subsets_R2(Y, Theta, k, n_top=1000, batch=20000):
    """Least-squares fit of every k-term subset of the library; returns the n_top best by R^2.
    Uses the Gram matrix: with G = A^T A and b = A^T y precomputed, a subset S only needs a k x k solve,
    RSS_S = y^T y - b_S^T G_SS^-1 b_S. Columns are normalised first (better conditioning; R^2 is unchanged).
    Returns a DataFrame with rank, R2, dR2 (best R^2 minus R^2) and the subset's terms."""
    from itertools import combinations
    from math import comb
    y = np.asarray(Y, float).ravel()
    A = Theta.to_numpy(float)
    A = A / np.linalg.norm(A, axis=0)
    G, b, yy = A.T @ A, A.T @ y, y @ y
    tss = np.sum((y - y.mean()) ** 2)

    n_sub = comb(A.shape[1], k)
    subsets = np.fromiter((i for S in combinations(range(A.shape[1]), k) for i in S),
                          dtype=np.int32, count=n_sub * k).reshape(n_sub, k)
    rss = np.empty(n_sub)
    for s in range(0, n_sub, batch):                                     # batched k x k solves
        idx = subsets[s:s + batch]
        G_S = G[idx[:, :, None], idx[:, None, :]]
        b_S = b[idx]
        c = np.einsum('nij,nj->ni', np.linalg.pinv(G_S, hermitian=True), b_S)   # pinv: safe for collinear subsets
        rss[s:s + batch] = yy - np.einsum('ni,ni->n', b_S, c)
    R2 = 1 - rss / tss

    order = np.argsort(-R2)[:n_top]
    names = np.asarray(Theta.columns)
    return pd.DataFrame({'rank': np.arange(1, len(order) + 1),
                         'R2': R2[order],
                         'dR2': R2[order[0]] - R2[order],
                         'terms': [' + '.join(names[subsets[i]]) for i in order]})


@styled
def plot_rank_dR2(tables, y='dR2', ylabels=None, logy=False, ylim=None, yticklabels=True, markers=False,
                  ms=3, drop_best=False, width=4, figsize_per_panel=2, save_path=None, margins=None):
    """R^2 or dR^2 against rank, one panel per equation (tables: list of all_subsets_R2 outputs).
    y: 'dR2' (best R^2 minus R^2) or 'R2' (absolute R^2).
    Each rank is a step of width 1 centred on it, so the curve starts at rank 1. On a log axis the
    rank-1 value dR^2 = 0 cannot be drawn: its step sits below the axis and the curve rises from the floor.
    markers=True draws one marker per rank instead of steps (on a log axis rank 1, dR^2 = 0, is not shown).
    ms: marker size.  drop_best=True leaves out the best model and numbers the alternatives from 1
    (x = rank among the alternatives, rank 2 overall at x = 1).
    yticklabels=False hides the y label and numbers but keeps the ticks."""
    n = len(tables)
    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(width, figsize_per_panel * n))
    axes = np.atleast_1d(axes)
    for i, (ax, df) in enumerate(zip(axes, tables)):
        if drop_best:
            df = df.iloc[1:]
        r = np.arange(1, len(df) + 1)
        v = df[y].to_numpy(float)
        if logy:
            v = np.where(v > 0, v, np.finfo(float).tiny)                  # zero -> below the axis floor
            ax.set_yscale('log')
        if markers:
            ax.plot(r, v, 'o', color='black', ms=ms, clip_on=logy)          # unclipped: edge markers show in full
        else:
            ax.stairs(v, np.append(r - 0.5, r[-1] + 0.5), baseline=None, color='black', lw=1)
        pad = 1.0 if markers else 0.5                                     # markers: a full step clear of the frame
        ax.set_xlim(r[0] - pad, r[-1] + pad)
        if ylabels and yticklabels:
            ax.set_ylabel(ylabels[i])
        if ylim is not None:
            lim = ylim[i] if any(l is None or np.ndim(l) > 0 for l in ylim) else ylim
            if lim is not None:
                ax.set_ylim(lim)
        if not yticklabels:
            ax.tick_params(axis='y', which='both', labelleft=False)      # keep the ticks, hide the numbers
    axes[-1].set_xlabel('Rank')
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
        data = [w[t] * Theta[t].to_numpy() for t in w.index]
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


def plot_R2(R2s):
    """Training R^2 against the number of terms."""
    n = sorted(R2s)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(n, [R2s[k] for k in n], 'o-', color='#2A4E6C')
    ax.set_xlabel('number of terms'); ax.set_ylabel('R$^2$ (full data)')
    fig.tight_layout()
    return fig, ax


def print_model(models, n, name="X'"):
    """Print the n-term model, every term and its coefficient."""
    w = models[n]
    print(f"{name} ({n} terms) = " + " + ".join(f"{c:.3f} {t}" for t, c in w.items()))


from collections import Counter

def summary_V1(models, n, min_n=100):
    df = pd.DataFrame([m[n] for m in models])
    count = df.notna().sum()
    out = pd.DataFrame({'n': count,
                        'inclusion': count / len(df),
                        'mean': df.mean(),
                        'lo': df.quantile(0.025).where(count >= min_n),
                        'hi': df.quantile(0.975).where(count >= min_n)})
    out = (out.assign(abs_mean=out['mean'].abs())
              .sort_values(['inclusion', 'abs_mean'], ascending=False)
              .drop(columns='abs_mean'))

    supports = [frozenset(m[n].index) for m in models]              # the set of terms in each trajectory's model
    best, k = Counter(supports).most_common(1)[0]                    # the most frequent equation and how often it occurs
    common = (out.loc[[t for t in out.index if t in best]]
                 .drop(columns='n')
                 .assign(occurrence=k / len(models)))
    return out, common

def summary(fixed_df, flex_df, ref):
    """
    fixed_df: MC fits with exactly the reference terms. Gives mu / lo / hi.
    flex_df:  flexible MC fits (NaN = term not included). Gives inc / occurrence.
    ref:      reference fit, a pd.Series of term -> coef.
    """
    # mu / lo / hi from the fixed fits
    fixed = fixed_df.drop(columns='optimal', errors='ignore').reindex(columns=ref.index)

    # inc / occurrence from the flexible fits
    flex = flex_df.drop(columns='optimal', errors='ignore')
    cols = flex.columns.union(ref.index)
    included = flex.reindex(columns=cols).fillna(0).ne(0)
    same_support = (included == cols.isin(ref.index)).all(axis=1)

    return pd.DataFrame({
        'GP':  ref,
        'mu':  fixed.mean(),
        'med': fixed.median(),
        'lo':  fixed.quantile(0.025),
        'hi':  fixed.quantile(0.975),
        'inc': included[ref.index].mean(),
        'occurrence': same_support.mean(),
    })


def integrate_model(w, Theta, t, X0=None, switches=('XT', 'XP'), blowup=10):
    """Integrate dX/dt = sum_k w_k * term_k for a model Series w (term name -> coefficient).
    X is the state; every other variable a term uses is read from Theta and interpolated in time
    (cubic for continuous series, step ('previous') for the 0/1 switches)."""
    t = np.asarray(t, float)
    X0 = Theta['X'].iloc[0] if X0 is None else X0
    used = {f for term in w.index for f in term.split('*')} - {'1', 'X'}        # arrays this model needs
    interp = {n: interp1d(t, Theta[n].to_numpy(), kind='previous' if n in switches else 'cubic',
                          bounds_error=False, fill_value=(Theta[n].iloc[0], Theta[n].iloc[-1])) for n in used}
    def term_value(term, x, tt):
        v = 1.0
        for f in term.split('*'):
            if f == 'X': v *= x
            elif f != '1': v *= float(interp[f](tt))
        return v
    rhs = lambda tt, y: [sum(c * term_value(k, y[0], tt) for k, c in w.items())]
    blow = lambda tt, y: abs(y[0]) - blowup; blow.terminal = True               # stop if the run diverges
    sol = solve_ivp(rhs, (t[0], t[-1]), [X0], t_eval=t, events=blow, max_step=1.0, rtol=1e-7, atol=1e-9)
    out = np.full(len(t), np.nan); out[:sol.y.shape[1]] = sol.y[0]
    return out


def integrate_system(ws, Theta, t, y0=None, blowup=10):
    """Integrate the coupled system d(state)/dt = sum_k w_k * term_k for every state in ws.
    ws: dict state -> model Series (term name -> coefficient), e.g. {'X': wX, 'Y': wY, 'Z': wZ}.
    The states are evolved together; every other variable (the forcing) is read from Theta
    and interpolated in time (cubic). Returns a dict state -> trajectory (NaN after a blow-up)."""
    t = np.asarray(t, float)
    states = list(ws)
    y0 = [Theta[s].iloc[0] for s in states] if y0 is None else y0
    used = {f for w in ws.values() for term in w.index for f in term.split('*')} - {'1', *states}
    interp = {n: interp1d(t, Theta[n].to_numpy(), kind='cubic',
                          bounds_error=False, fill_value=(Theta[n].iloc[0], Theta[n].iloc[-1])) for n in used}
    def term_value(term, y, tt):
        v = 1.0
        for f in term.split('*'):
            if f in states: v *= y[states.index(f)]
            elif f != '1': v *= float(interp[f](tt))
        return v
    rhs = lambda tt, y: [sum(c * term_value(k, y, tt) for k, c in ws[s].items()) for s in states]
    blow = lambda tt, y: np.max(np.abs(y)) - blowup; blow.terminal = True       # stop if the run diverges
    sol = solve_ivp(rhs, (t[0], t[-1]), y0, t_eval=t, events=blow, max_step=1.0, rtol=1e-7, atol=1e-9)
    out = {s: np.full(len(t), np.nan) for s in states}
    for i, s in enumerate(states): out[s][:sol.y.shape[1]] = sol.y[i]
    return out

