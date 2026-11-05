"""Figures for M0 vs M1, built from the SAME pipeline as run_m0.py / run_m1.py.

`collect()` calls the pipeline modules (m0, gate, cointegration, ecm, nested, race, robustness, noise) with
the same settings and seeds as the two run scripts, so every number in a figure is a number in the console
report (tests/test_plots.py checks this against a real `run_m1.py --fast` run). Nothing is re-estimated here
except what only a figure needs: ACF, Q-Q, rolling OLS, variance ratios (all statsmodels).

One figure per question the proposal asks, in the order of its phases:

    01_data            Phase 0   what the two series do, and the gap between them
    02_m0_gate         Phase 1   M0: is c1 below 1? what reversion speed does that allow?
    03_m1_gate         Phase 1   M1: unit roots, Engle-Granger, ECM bootstrap test, Gregory-Hansen
    04_anchor          Phase 2   the long-run level each model uses (the M0 vs M1 difference)
    05_anchor_test     Phase 3   H0: beta1 = beta2 = 0, classical / HAC / bootstrap
    06_instability     Phase 2/3 rolling slope and start-date sensitivity
    07_forecast_race   Phase 4   out-of-sample RMSE vs random walk, Clark-West
    08_m0_vs_sora      M0 / noise  what a fitted Vasicek implies vs what SORA does; residual tails
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.stattools import acf

from . import cointegration as C
from . import ecm, gate, loaders, m0, nested, noise, race, robustness
from .ols import exclusion_f, exclusion_wald_hac, subset

M0C, M1C, GREY, RED = "#4c72b0", "#dd8452", "#9a9a9a", "#d62728"     # seaborn "deep": blue / orange; grey = data
T1C, T2C = "#55a868", "#8172b3"                                       # two further colours for pairs of tests
MIN_OBS_START = 400                                                   # shortest sample used in the start-date sweep


def _style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"axes.titleweight": "bold", "axes.titlesize": 13, "figure.dpi": 100})


# ------------------------------------------------------------------ numbers only a figure needs
def variance_ratio(x: np.ndarray, hs) -> np.ndarray:
    """Var(r_{t+h} - r_t) / (h Var(r_{t+1} - r_t)); rows of x are paths. 1 for a random walk, < 1 for mean reversion."""
    x = np.atleast_2d(x)
    v1 = np.diff(x, axis=1).var(axis=1)
    return np.stack([(x[:, h:] - x[:, :-h]).var(axis=1) / (h * v1) for h in hs], axis=1)


def anchor_path(model: ecm.M1ECM, df: pd.DataFrame) -> pd.Series:
    """M1's long-run level on each date: alpha + beta1 * SOFR (+ level shift after the break date)."""
    a = model.alpha + model.beta1 * df["x"]
    if model.has_shift:
        a = a + model.level_shift * (df.index >= pd.Timestamp(model.break_date))
    return a


def start_dates(index: pd.DatetimeIndex, min_obs: int = MIN_OBS_START) -> list:
    """Quarterly start dates from the first observation, keeping samples of at least `min_obs` days."""
    grid = pd.date_range(index[0].to_period("Q").start_time, index[-1], freq="QS-OCT")      # first quarter start at/before the data
    return [d for d in grid if (index >= d).sum() >= min_obs]


# ------------------------------------------------------------------ the pipeline, once
def collect(fast: bool = False, seed: int = 2027, verbose: bool = True) -> dict:
    """Same data, lags, break date, bootstrap sizes and seeds as run_m0.py / run_m1.py (fast = their --fast)."""
    say = print if verbose else (lambda *a, **k: None)
    B, gh_sim, n_boot = (99, 40, 99) if fast else (499, 200, 299)
    R: dict = {"fast": fast, "seed": seed}

    r = loaders.load_sora()
    sofr = loaders.load_sofr()
    z = loaders.make_z(loaders.load_usdsgd())
    df = loaders.build_dataset(r, sofr, z)
    p = int(ecm.select_lags(df, 10, 0, True)["bic"].idxmin())
    R.update(r=r, sofr=sofr, z=z, df=df, p=p)

    say("M0: calibration and gate ...")
    R["fit0"] = m0.calibrate(r)
    R["tab0"] = R["fit0"].params_table()
    R["gate0"] = gate.adf_gate(r, p=None, n_boot=n_boot, seed=seed)

    say("M1: unit roots, Engle-Granger, Gregory-Hansen, ECM bootstrap ...")
    R["ur"] = C.unit_root_table(df)
    R["eg"] = C.engle_granger(df["r"], df["x"])
    ghC = C.gregory_hansen(df["r"], df["x"], "C", p, n_sim=gh_sim, seed=seed, dates=df.index, return_sims=True)
    R["gh"], bd = ghC, ghC["break_date"]
    R["break_date"] = bd
    R["coint0"] = nested.cointegration_ecm_test(df, p, 0, True, B, seed=seed)

    say("M1: ECM and anchor tests ...")
    R["m1"] = ecm.fit_ecm(df, p, 0, True)
    R["m1b"] = ecm.fit_ecm(df, p, 0, True, break_date=bd)
    R["anchor"] = {"M0": pd.Series(R["fit0"].theta, index=df.index), "M1": anchor_path(R["m1"], df),
                   "M1 + level shift": anchor_path(R["m1b"], df)}
    tests = {}
    for key, model, b in (("no break", R["m1"], None), ("level shift", R["m1b"], bd)):
        drop = ecm.nested_columns(model.fit.names, "anchor")
        tests[key] = {"classical": exclusion_f(model.fit, subset(model.fit, drop)),
                      "hac": exclusion_wald_hac(model.fit, drop),
                      "boot": nested.anchor_test(df, p, 0, True, "unit_root", B, seed=seed, break_date=b)}
    R["anchor_tests"] = tests

    say("Slope stability and start dates ...")
    roll = RollingOLS(df["r"], sm.add_constant(df["x"]), window=500).fit()
    R["roll"] = pd.DataFrame({"slope": roll.params["x"], "se": roll.bse["x"]}).dropna()
    R["slope_full"] = C.slope_stability(df, windows=())["full"]
    R["starts"] = robustness.start_date_sensitivity(r, sofr, z, start_dates(df.index), p, 0, gh_lags=p)

    say("Forecast race ...")
    R["race"] = race.run_race(df, (1, 20, 60), 500, True, B=B, null="unit_root", p=p, seed=seed)["summary"]

    say("M0 simulation and noise ...")
    n_paths = 200 if fast else 500
    k = 10
    dr = np.diff(r.to_numpy())
    R["lags"] = k
    R["acf_act"] = acf(dr, nlags=k)[1:]
    hs = np.array([1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 60, 90, 120])
    R["hs"], R["vr_act"] = hs, variance_ratio(r.to_numpy(), hs)[0]
    R["paths"] = None
    if R["fit0"].valid:                                                     # the pipeline's own M0 simulator
        paths = R["fit0"].simulate(r.iloc[0], len(r) - 1, n_paths, seed)
        R["paths"] = paths
        R["acf_band"] = np.percentile([acf(np.diff(q), nlags=k)[1:] for q in paths], [5, 50, 95], axis=0)
        R["vr_band"] = np.percentile(variance_ratio(paths, hs), [5, 50, 95], axis=0)
    nested_m0 = subset(R["m1"].fit, ecm.nested_columns(R["m1"].fit.names, "all"))      # M0 on the SAME sample as M1
    R["resid0"], R["resid1"] = nested_m0.resid, R["m1"].fit.resid
    R["kurt0"] = float(noise.noise_diagnostics(R["resid0"]).loc["excess kurtosis", "value"])
    R["kurt1"] = float(noise.noise_diagnostics(R["resid1"]).loc["excess kurtosis", "value"])
    return R


# ------------------------------------------------------------------ helpers for drawing
def _vline_break(ax, bd, label=None):
    ax.axvline(bd, color="black", lw=0.9, ls="--", label=label)


def _null_hist(ax, draws, obs, cv, xlabel, obs_label, bins=30):
    sns.histplot(np.asarray(draws), bins=bins, color=GREY, edgecolor="white", ax=ax)
    ax.axvline(obs, color=RED, lw=2, label=obs_label)
    ax.axvline(cv, color="black", lw=1, ls="--", label="5% critical value")
    ax.set(xlabel=xlabel, ylabel="Bootstrap draws")
    ax.legend(loc="best")


# ------------------------------------------------------------------ figures
def fig_data(R: dict):
    df, bd = R["df"], R["break_date"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    ax.plot(df.index, df["r"] * 100, color=GREY, lw=1.2, label="SORA (daily)")
    ax.plot(df.index, df["x"] * 100, color=M0C, lw=1.0, label="SOFR (last published)")
    _vline_break(ax, bd, f"Level shift, {bd:%d %b %Y} (Gregory-Hansen)")
    ax.set(title="SORA and SOFR", ylabel="Rate (% per annum)")
    ax.legend(loc="lower right")
    ax = axes[1]
    ax.plot(df.index, (df["x"] - df["r"]) * 100, color="black", lw=1.0)
    ax.axhline(0, color=GREY, lw=0.8)
    _vline_break(ax, bd)
    ax.set(title="SOFR minus SORA", xlabel="Year", ylabel="Spread (pp)")
    fig.tight_layout()
    return fig


def fig_m0_gate(R: dict):
    """c1 against the unit root (top) and what each value of c1 means for the reversion speed (bottom)."""
    ag, t0, dt = R["gate0"], R["tab0"].loc["c1"], R["fit0"].dt
    lo = min(t0["ci_low"], ag.ci[0])
    hi = max(t0["ci_high"], ag.ci[1], 1.0)
    pad = 0.15 * (hi - lo)
    xmin, xmax = lo - pad, hi + pad
    fig, (a, b) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [1, 2]})
    rows = [(f"M0 calibration: AR(1) OLS, c1 = {R['fit0'].c1:.4f}\n(normal 95% CI)", R["fit0"].c1, t0["ci_low"], t0["ci_high"], GREY),
            (f"Gate: OLS with {ag.lags} lagged differences, c1 = {ag.c1_ols:.4f}", ag.c1_ols, None, None, GREY),
            (f"Gate: bias-corrected, c1 = {ag.c1_bc:.4f}\n(grid-bootstrap 95% CI)", ag.c1_bc, ag.ci[0], ag.ci[1], M0C)]
    for y, (lab, est, l, h, col) in enumerate(rows[::-1]):
        if l is not None:
            a.hlines(y, l, h, color=col, lw=3)
        a.plot(est, y, "o", color=col, ms=8)
    a.set_yticks(range(len(rows)))
    a.set_yticklabels([r_[0] for r_ in rows[::-1]])
    a.set_ylim(-0.6, len(rows) - 0.4)
    a.axvline(1.0, color=RED, lw=1.5)
    a.set_title("M0 Gate: Persistence c1")
    c = np.linspace(xmin, 0.99995, 600)
    c = c[c < 1.0]
    kap = -np.log(c) / dt
    b.plot(c, kap, color=M0C, lw=2)
    b.axvspan(ag.ci[0], min(ag.ci[1], xmax), color=M0C, alpha=0.15, label="Bootstrap 95% CI")
    b.axvline(1.0, color=RED, lw=1.5, label="c1 = 1 (unit root: no reversion)")
    b.axvspan(1.0, xmax, color=GREY, alpha=0.25)
    k_hi = -np.log(ag.ci[0]) / dt if 0 < ag.ci[0] < 1 else np.nan
    b.set_ylim(0, (k_hi if np.isfinite(k_hi) else kap.max()) * 1.4)
    b.set(xlabel="Daily persistence c1", ylabel="Reversion speed kappa (per year)", xlim=(xmin, xmax))
    b.set_title("Reversion Speed Implied by c1")
    b.legend(loc="upper left")
    fig.tight_layout()
    return fig


def fig_m1_gate(R: dict):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    # (a) unit roots
    ax = axes[0, 0]
    ur = R["ur"]
    names = {"r (SORA)": "SORA", "x (SOFR)": "SOFR", "x - r (spread)": "Spread", "z (FX return)": "USD/SGD return"}
    keys = [(s, f) for s in ur.index.get_level_values(0).unique() for f in ("level", "diff")]
    y = np.arange(len(keys))[::-1]
    ax.scatter([ur.loc[k, "adf_p"] for k in keys], y, marker="o", s=45, color=T1C, label="ADF (H0: unit root)", zorder=3)
    ax.scatter([ur.loc[k, "kpss_p"] for k in keys], y, marker="s", s=45, color=T2C, label="KPSS (H0: stationary)", zorder=3)
    ax.axvline(0.05, color="black", lw=1, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{names[s]} ({'level' if f == 'level' else 'change'})" for s, f in keys])
    ax.set(title="Unit-Root Tests", xlabel="p-value (dashed: 0.05)")
    ax.legend(loc="lower right")
    # (b) Engle-Granger over the ADF lag
    ax = axes[0, 1]
    eg = R["eg"]
    for tr, lab, col in (("c", "Constant", M1C), ("ct", "Constant + trend", M0C)):
        s = eg.loc[tr].xs("fixed", level="rule")["p_value"]
        ax.plot(s.index, s.to_numpy(), marker="o", ms=4, color=col, label=lab)
        bic = eg.loc[tr].xs("BIC", level="rule")
        ax.plot(bic.index[0], bic["p_value"].iloc[0], marker="*", ms=15, color=col, mec="black", ls="none")
    ax.axhline(0.05, color="black", lw=1, ls="--")
    ax.plot([], [], marker="*", ms=12, color="white", mec="black", ls="none", label="BIC lag choice")
    ax.set(title="Engle-Granger", xlabel="ADF lags on the residual", ylabel="p-value (dashed: 0.05)")
    ax.legend(loc="upper left")
    # (c) ECM bootstrap, no break
    c0 = R["coint0"]
    _null_hist(axes[1, 0], c0["t_boot"], c0["t"], c0["cv5_t_left"], "t-statistic on r(t-1)",
               f"Observed {c0['t']:.2f} (p = {c0['p_boot_t_left']:.3f})")
    axes[1, 0].set_title("ECM Bootstrap Test, No Break")
    # (d) Gregory-Hansen
    g = R["gh"]
    _null_hist(axes[1, 1], g["sims"], g["stat"], g["cv_5pct"], "Most negative ADF over break dates",
               f"Observed {g['stat']:.2f} (p = {g['p_value']:.3f})")
    axes[1, 1].set_title(f"Gregory-Hansen, Break {R['break_date']:%b %Y}")
    fig.suptitle("M1 Gate: Is SORA Tied to SOFR?", fontweight="bold", fontsize=14)
    fig.tight_layout()
    return fig


def fig_anchor(R: dict):
    df, bd = R["df"], R["break_date"]
    a = R["anchor"]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["r"] * 100, color=GREY, lw=1.2, label="SORA (daily)")
    ax.axhline(R["fit0"].theta * 100, color=M0C, lw=2, label=f"M0: constant long-run level ({R['fit0'].theta * 100:.2f}%)")
    ax.plot(df.index, a["M1"] * 100, color=M1C, lw=1.6, ls=":", label="M1: level moves with SOFR")
    ax.plot(df.index, a["M1 + level shift"] * 100, color=M1C, lw=2, label=f"M1 + level shift ({bd:%b %Y}, exploratory)")
    _vline_break(ax, bd)
    ax.set(title="Long-Run Level: M0 vs M1", xlabel="Year", ylabel="Rate (% per annum)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def fig_anchor_test(R: dict):
    """H0: beta1 = beta2 = 0 (SOFR level and FX add nothing to M0), by three ways of getting the p-value."""
    T = R["anchor_tests"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    ax = axes[0]
    specs = list(T)
    methods = (("Classical F", lambda t: t["classical"]["p_classical"], GREY),
               ("HAC Wald", lambda t: t["hac"]["p_hac_F"], "#c8c8c8"),
               ("Bootstrap (valid)", lambda t: t["boot"]["p_boot_F"], M1C))
    h = 0.22
    for j, (lab, fn, col) in enumerate(methods):
        pos = np.arange(len(specs))[::-1] + (1 - j) * h
        ax.scatter([fn(T[s]) for s in specs], pos, s=110, color=col, edgecolor="black", linewidth=0.6, label=lab, zorder=3)
    ax.set_xscale("log")
    ax.set_yticks(np.arange(len(specs))[::-1])
    ax.set_yticklabels([s.capitalize() for s in specs])
    ax.axvline(0.05, color="black", lw=1, ls="--")
    ax.set(title="p-value of the Anchor Test", xlabel="p-value (log scale, dashed: 0.05)")
    ax.set_ylim(-0.6, len(specs) - 0.4)
    ax.legend(loc="center left")
    for ax, s in zip(axes[1:], specs):
        bt = T[s]["boot"]
        _null_hist(ax, bt["F_boot"], bt["F"], bt["cv5_F"], "F-statistic",
                   f"Observed {bt['F']:.2f} (p = {bt['p_boot_F']:.3f})")
        ax.set_title(f"Bootstrap F, {s.capitalize()}")
    fig.tight_layout()
    return fig


def fig_instability(R: dict):
    roll, st = R["roll"], R["starts"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), gridspec_kw={"width_ratios": [2, 1, 1]})
    ax = axes[0]
    ax.plot(roll.index, roll["slope"], color=M1C, lw=1.8, label="500-day window")
    ax.fill_between(roll.index, roll["slope"] - 1.96 * roll["se"], roll["slope"] + 1.96 * roll["se"], color=M1C, alpha=0.2,
                    label="95% band (OLS)")
    ax.axhline(R["slope_full"], color="black", lw=0.9, ls="--", label=f"Full sample ({R['slope_full']:.2f})")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set(title="Rolling Slope of SORA on SOFR", xlabel="Window end", ylabel="Slope")
    ax.legend(loc="best")
    d = pd.to_datetime(st["start"])
    ax = axes[1]
    ax.plot(d, st["slope_r_on_x"], marker="o", color=M1C)
    ax.set(title="Slope by Start Date", xlabel="Sample start", ylabel="Slope")
    ax = axes[2]
    ax.plot(d, st["EG_p_const(BIC)"], marker="o", color=M0C)
    ax.axhline(0.05, color="black", lw=1, ls="--")
    ax.set(title="Engle-Granger by Start Date", xlabel="Sample start", ylabel="p-value (dashed: 0.05)")
    for ax in axes[1:]:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    return fig


def fig_race(R: dict):
    s = R["race"]
    rows = [{"h": h, "model": nm, "rel": 100 * row[f"rmse_{key}_bp"] / row["rmse_rw_bp"]}
            for h, row in s.iterrows() for nm, key in (("Random walk", "rw"), ("M0", "m0"), ("M1", "m1"))]
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=d, x="h", y="rel", hue="model", palette={"Random walk": GREY, "M0": M0C, "M1": M1C}, ax=ax)
    ax.axhline(100, color="black", lw=0.8)
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels([f"{h} day{'s' if h > 1 else ''}\n(Clark-West p, M1 vs M0: {s.loc[h, 'p_boot M1 vs M0']:.3f})" for h in s.index])
    ax.set_ylim(min(90, d["rel"].min() - 3), d["rel"].max() + 3)
    ax.set(title="Out-of-Sample RMSE Relative to Random Walk", xlabel="Forecast horizon", ylabel="RMSE (random walk = 100)")
    ax.legend(title=None)
    fig.tight_layout()
    return fig


def fig_m0_vs_sora(R: dict):
    """A fitted M0 (Gaussian Vasicek, the pipeline's own simulator) vs the data: dynamics (top) and noise (bottom)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    k = R["lags"]
    lag = np.arange(1, k + 1)
    ok = R["paths"] is not None
    ax = axes[0, 0]
    if ok:
        lo, mid, hi = R["acf_band"]
        ax.fill_between(lag, lo, hi, color=M0C, alpha=0.25, label="Fitted M0, 90% range")
        ax.plot(lag, mid, color=M0C, lw=1.5)
    ax.plot(lag, R["acf_act"], color="black", marker="o", ms=4, lw=1.2, label="SORA")
    ax.axhline(0, color="black", lw=0.6)
    ax.set(title="ACF of Daily Changes", xlabel="Lag (days)", ylabel="Autocorrelation")
    ax.legend()
    ax = axes[0, 1]
    if ok:
        lo, mid, hi = R["vr_band"]
        ax.fill_between(R["hs"], lo, hi, color=M0C, alpha=0.25, label="Fitted M0, 90% range")
        ax.plot(R["hs"], mid, color=M0C, lw=1.5)
    ax.plot(R["hs"], R["vr_act"], color="black", marker="o", ms=4, lw=1.2, label="SORA")
    ax.axhline(1, color="black", lw=0.6, ls="--")
    ax.set(title="Variance Ratio", xlabel="Horizon h (days)", ylabel="Var(h-day change) / (h x Var(1-day))")
    ax.legend()
    ax = axes[1, 0]
    sm.qqplot(R["resid0"].to_numpy() * 1e4, line="s", ax=ax, markerfacecolor=M0C, markeredgecolor=M0C, alpha=0.5, ms=4,
              label=f"M0, same sample (excess kurtosis {R['kurt0']:.1f})")
    sm.qqplot(R["resid1"].to_numpy() * 1e4, line="s", ax=ax, markerfacecolor=M1C, markeredgecolor=M1C, alpha=0.5, ms=4,
              label=f"M1 (excess kurtosis {R['kurt1']:.1f})")
    ax.set(title="Normal Q-Q of Residuals", xlabel="Theoretical quantiles", ylabel="Residual (bp)")
    ax.legend(loc="upper left")
    ax = axes[1, 1]
    plot_acf(R["resid0"].to_numpy() ** 2, lags=k, ax=ax, zero=False, color=M0C, vlines_kwargs={"colors": M0C},
             title=None, alpha=0.05, auto_ylims=True)
    ax.set(title="ACF of Squared M0 Residuals", xlabel="Lag (days)", ylabel="Autocorrelation")
    fig.tight_layout()
    return fig


FIGURES = [("01_data", fig_data), ("02_m0_gate", fig_m0_gate), ("03_m1_gate", fig_m1_gate), ("04_anchor", fig_anchor),
           ("05_anchor_test", fig_anchor_test), ("06_instability", fig_instability), ("07_forecast_race", fig_race),
           ("08_m0_vs_sora", fig_m0_vs_sora)]


def make_all(out_dir="results/figures", fast: bool = False, formats=("png", "pdf"), seed: int = 2027, dpi: int = 200,
             verbose: bool = True, R: dict | None = None) -> list:
    _style()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    R = R or collect(fast, seed, verbose)
    files = []
    for name, fn in FIGURES:
        fig = fn(R)
        for ext in formats:
            f = out / f"{name}.{ext}"
            fig.savefig(f, dpi=dpi, bbox_inches="tight")
            files.append(f)
        plt.close(fig)
    return files
