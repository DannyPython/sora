"""Figures for M0 vs M1 (seaborn look, statsmodels for the numbers).

Every figure is drawn from the data by `collect()`; nothing is typed in by hand.
statsmodels does the estimation (OLS, ACF, Q-Q, rolling OLS); `ssef` supplies the dataset, the ECM design,
the Gregory-Hansen break date and the forecast race, so the numbers match run_m0.py / run_m1.py.

    from ssef import plots
    plots.make_all("results/figures")
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
from . import ecm, loaders, race

DT = 1.0 / 252.0
M0C, M1C, GREY, RED = "#4c72b0", "#dd8452", "#9a9a9a", "#d62728"     # seaborn "deep" blue / orange; grey for data


def _style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"axes.titleweight": "bold", "axes.titlesize": 13, "figure.dpi": 100})


# ------------------------------------------------------------------ numbers (statsmodels)
def fit_m0(r: pd.Series) -> dict:
    """Classical Vasicek by OLS on the exact AR(1): r_t = c0 + c1 r_{t-1} + e."""
    res = sm.OLS(r.iloc[1:].to_numpy(), sm.add_constant(r.iloc[:-1].to_numpy())).fit()
    c0, c1 = res.params
    resid = pd.Series(res.resid, index=r.index[1:])
    ok = 0.0 < c1 < 1.0
    return {"res": res, "c0": c0, "c1": c1, "resid": resid, "valid": ok,
            "theta": c0 / (1 - c1) if ok else float(r.mean()),            # fall back to the sample mean if no reversion
            "kappa": -np.log(c1) / DT if ok else np.nan}


def fit_m1(df: pd.DataFrame, p: int, break_date=None) -> dict:
    """ECM by OLS: dr = c + a r_{t-1} + b x_{t-1} + ... ; long-run anchor = -(c + b x [+ d D]) / a."""
    y, X = ecm.design(df, p, 0, True, break_date)
    res = sm.OLS(y, X).fit()
    a = res.params["r_lag"]
    anchor = -(res.params["const"] + res.params["x_lag"] * df["x"]) / a
    if break_date is not None:
        anchor = anchor - res.params["shift"] * (df.index >= pd.Timestamp(break_date)) / a
    return {"res": res, "a": a, "c1": 1 + a, "anchor": anchor, "resid": res.resid,
            "alpha": -res.params["const"] / a, "beta1": -res.params["x_lag"] / a,
            "shift": (-res.params["shift"] / a if break_date is not None else np.nan)}


def variance_ratio(x: np.ndarray, hs) -> np.ndarray:
    """Var(r_{t+h} - r_t) / (h Var(r_{t+1} - r_t)); rows of x are paths. 1 for a random walk, < 1 for mean reversion."""
    x = np.atleast_2d(x)
    v1 = np.diff(x, axis=1).var(axis=1)
    return np.stack([(x[:, h:] - x[:, :-h]).var(axis=1) / (h * v1) for h in hs], axis=1)


def simulate_m0(m: dict, r0: float, n: int, n_paths: int, seed: int) -> np.ndarray:
    """Paths of the fitted AR(1) with Gaussian errors (what M0 says the data should look like)."""
    rng = np.random.default_rng(seed)
    s = float(np.sqrt(m["res"].scale))
    out = np.empty((n_paths, n + 1))
    out[:, 0] = r0
    for t in range(n):
        out[:, t + 1] = m["c0"] + m["c1"] * out[:, t] + s * rng.standard_normal(n_paths)
    return out


def collect(fast: bool = False, seed: int = 2027, verbose: bool = True) -> dict:
    say = print if verbose else (lambda *a, **k: None)
    R: dict = {"fast": fast, "seed": seed}
    r = loaders.load_sora()
    sofr = loaders.load_sofr()
    df = loaders.build_dataset(r, sofr, loaders.make_z(loaders.load_usdsgd()))
    R.update(r=r, df=df)
    R["p"] = int(ecm.select_lags(df, 10, 0, True)["bic"].idxmin())

    say("M0 ...")
    R["m0"] = fit_m0(r)
    say("M1 (ECM, level shift) ...")
    gh = C.gregory_hansen(df["r"], df["x"], "C", R["p"], n_sim=0, dates=df.index)
    R["break_date"] = gh["break_date"]
    R["m1"] = fit_m1(df, R["p"])
    R["m1b"] = fit_m1(df, R["p"], R["break_date"])
    R["gap"] = {"M0": float((df["r"] - R["m0"]["theta"]).abs().mean() * 100),
                "M1": float((df["r"] - R["m1"]["anchor"]).abs().mean() * 100),
                "M1 + level shift": float((df["r"] - R["m1b"]["anchor"]).abs().mean() * 100)}

    say("M0-implied paths ...")
    n_paths = 200 if fast else 500
    R["paths"] = simulate_m0(R["m0"], r.iloc[0], len(r) - 1, n_paths, seed)
    R["lags"] = 10
    dr = np.diff(r.to_numpy())
    R["acf_act"] = acf(dr, nlags=R["lags"])[1:]
    sims = np.array([acf(np.diff(pth), nlags=R["lags"])[1:] for pth in R["paths"]])
    R["acf_band"] = np.percentile(sims, [5, 50, 95], axis=0)
    R["hs"] = np.array([1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 60, 90, 120])
    R["vr_act"] = variance_ratio(r.to_numpy(), R["hs"])[0]
    R["vr_band"] = np.percentile(variance_ratio(R["paths"], R["hs"]), [5, 50, 95], axis=0)

    say("Rolling slope ...")
    roll = RollingOLS(df["r"], sm.add_constant(df["x"]), window=500).fit()
    R["roll"] = pd.DataFrame({"slope": roll.params["x"], "se": roll.bse["x"]}).dropna()
    R["slope_full"] = float(sm.OLS(df["r"], sm.add_constant(df["x"])).fit().params["x"])

    say("Forecast race ...")
    R["race"] = race.run_race(df, (1, 20, 60), 500, True, B=99 if fast else 499, null="unit_root", p=R["p"], seed=seed)["summary"]

    # large shocks: M0 residuals more than 4 standard deviations from zero
    z = R["m0"]["resid"] / R["m0"]["resid"].std()
    R["shocks"] = z[z.abs() > 4].index
    return R


# ------------------------------------------------------------------ figures
def fig_timeline(R: dict):
    r, df = R["r"] * 100, R["df"]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(r.index, r, color=GREY, lw=1.2, label="SORA (daily)")
    ax.plot(df.index, df["x"] * 100, color=M0C, lw=0.9, alpha=0.7, label="SOFR (last published)")
    ax.scatter(R["shocks"], r.loc[R["shocks"]], color=RED, s=22, zorder=3, label="Large SORA shocks (|z| > 4)")
    ax.axvline(R["break_date"], color="black", lw=0.9, ls="--", label="Level shift (Gregory-Hansen)")
    ax.set_title("SORA and SOFR, Daily")
    ax.set_xlabel("Year")
    ax.set_ylabel("Rate (% per annum)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def fig_correlation(R: dict):
    df = R["df"]
    d = pd.DataFrame({"sora": df["r"], "sofr": df["x"], "usdsgd_ret": df["z"],
                      "d_sora": df["r"].diff(), "d_sofr": df["x"].diff()}).dropna()
    fig, ax = plt.subplots(figsize=(10, 7.5))
    sns.heatmap(d.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5, ax=ax)
    ax.set_title("Correlation Matrix: Levels and Daily Changes")
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    return fig


def fig_sora_sofr(R: dict):
    """Template 3: scatter with marginal distributions; the line is statsmodels OLS with its 95% band."""
    df = R["df"]
    d = pd.DataFrame({"sofr": df["x"] * 100, "sora": df["r"] * 100})
    g = sns.JointGrid(data=d, x="sofr", y="sora", height=7.5, ratio=5)
    g.plot_joint(sns.scatterplot, s=10, alpha=0.35, color=M0C, edgecolor="none")
    g.plot_marginals(sns.histplot, kde=True, color=M0C)
    res = sm.OLS(d["sora"], sm.add_constant(d["sofr"])).fit()
    grid = pd.DataFrame({"sofr": np.linspace(d["sofr"].min(), d["sofr"].max(), 100)})
    pred = res.get_prediction(sm.add_constant(grid["sofr"])).summary_frame(alpha=0.05)
    g.ax_joint.plot(grid["sofr"], pred["mean"], color=M0C, lw=2)
    g.ax_joint.fill_between(grid["sofr"], pred["mean_ci_lower"], pred["mean_ci_upper"], color=M0C, alpha=0.15)
    g.ax_joint.set_xlabel("SOFR (% per annum)")
    g.ax_joint.set_ylabel("SORA (% per annum)")
    g.figure.suptitle("Relationship: SORA vs SOFR", fontweight="bold", fontsize=13, y=1.0)
    g.figure.tight_layout()
    return g.figure


def fig_anchor(R: dict):
    df, bd = R["df"], R["break_date"]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["r"] * 100, color=GREY, lw=1.2, label="SORA (daily)")
    ax.axhline(R["m0"]["theta"] * 100, color=M0C, lw=2, label=f"M0 anchor (constant, {R['m0']['theta'] * 100:.2f}%)")
    ax.plot(df.index, R["m1"]["anchor"] * 100, color=M1C, lw=1.6, ls=":", label="M1 anchor (moves with SOFR)")
    ax.plot(df.index, R["m1b"]["anchor"] * 100, color=M1C, lw=2, label="M1 anchor + level shift")
    ax.axvline(bd, color="black", lw=0.9, ls="--")
    ax.set_title("Long-Run Anchor: M0 vs M1")
    ax.set_xlabel("Year")
    ax.set_ylabel("Rate (% per annum)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def fig_m0_vs_sora(R: dict):
    """What M0 implies vs what SORA does (2x2): autocorrelation, variance ratio, tails, volatility clustering."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    k = R["lags"]
    lag = np.arange(1, k + 1)

    ax = axes[0, 0]
    lo, mid, hi = R["acf_band"]
    ax.fill_between(lag, lo, hi, color=M0C, alpha=0.25, label="M0 simulated, 90% range")
    ax.plot(lag, mid, color=M0C, lw=1.5)
    ax.plot(lag, R["acf_act"], color="black", marker="o", ms=4, lw=1.2, label="SORA")
    ax.axhline(0, color="black", lw=0.6)
    ax.set(title="ACF of Daily Changes", xlabel="Lag (days)", ylabel="Autocorrelation")
    ax.legend()

    ax = axes[0, 1]
    lo, mid, hi = R["vr_band"]
    ax.fill_between(R["hs"], lo, hi, color=M0C, alpha=0.25, label="M0 simulated, 90% range")
    ax.plot(R["hs"], mid, color=M0C, lw=1.5)
    ax.plot(R["hs"], R["vr_act"], color="black", marker="o", ms=4, lw=1.2, label="SORA")
    ax.axhline(1, color="black", lw=0.6, ls="--")
    ax.set(title="Variance Ratio", xlabel="Horizon h (days)", ylabel="Var(h-day change) / (h x Var(1-day))")
    ax.legend()

    ax = axes[1, 0]
    sm.qqplot(R["m0"]["resid"].to_numpy() * 1e4, line="s", ax=ax, markerfacecolor=M0C, markeredgecolor=M0C, alpha=0.5, ms=4,
              label="M0 residuals")
    sm.qqplot(R["m1"]["resid"].to_numpy() * 1e4, line="s", ax=ax, markerfacecolor=M1C, markeredgecolor=M1C, alpha=0.5, ms=4,
              label="M1 residuals")
    ax.set(title="Normal Q-Q of Residuals", xlabel="Theoretical quantiles", ylabel="Residual (bp)")
    ax.legend(loc="upper left")

    ax = axes[1, 1]
    plot_acf(R["m0"]["resid"].to_numpy() ** 2, lags=k, ax=ax, zero=False, color=M0C, vlines_kwargs={"colors": M0C},
             title=None, alpha=0.05, auto_ylims=True)
    ax.set(title="ACF of Squared M0 Residuals", xlabel="Lag (days)", ylabel="Autocorrelation")
    fig.tight_layout()
    return fig


def fig_rolling_slope(R: dict):
    roll = R["roll"]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(roll.index, roll["slope"], color=M1C, lw=1.8, label="Slope of SORA on SOFR (500-day window)")
    ax.fill_between(roll.index, roll["slope"] - 1.96 * roll["se"], roll["slope"] + 1.96 * roll["se"], color=M1C, alpha=0.2,
                    label="95% band (OLS)")
    ax.axhline(R["slope_full"], color="black", lw=0.9, ls="--", label=f"Full sample ({R['slope_full']:.2f})")
    ax.axhline(1, color=GREY, lw=0.9, ls=":", label="Slope = 1")
    ax.set_title("Rolling Slope of SORA on SOFR")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_xlabel("Year")
    ax.set_ylabel("Slope")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def fig_race(R: dict):
    s = R["race"]
    rows = []
    for h, row in s.iterrows():
        for nm, key in (("Random walk", "rw"), ("M0", "m0"), ("M1", "m1")):
            rows.append({"h": h, "model": nm, "rel": 100 * row[f"rmse_{key}_bp"] / row["rmse_rw_bp"]})
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=d, x="h", y="rel", hue="model", palette={"Random walk": GREY, "M0": M0C, "M1": M1C}, ax=ax)
    ax.axhline(100, color="black", lw=0.8)
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels([f"{h} day{'s' if h > 1 else ''}\n(Clark-West p, M1 vs RW: {s.loc[h, 'p_boot M1 vs RW']:.3f})" for h in s.index])
    ax.set_ylim(min(90, d["rel"].min() - 3), d["rel"].max() + 3)
    ax.set_title("Out-of-Sample RMSE Relative to Random Walk")
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("RMSE (random walk = 100)")
    ax.legend(title=None)
    fig.tight_layout()
    return fig


FIGURES = [("01_timeline", fig_timeline), ("02_correlation", fig_correlation), ("03_sora_vs_sofr", fig_sora_sofr),
           ("04_anchor", fig_anchor), ("05_m0_vs_sora", fig_m0_vs_sora), ("06_rolling_slope", fig_rolling_slope),
           ("07_forecast_race", fig_race)]


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
