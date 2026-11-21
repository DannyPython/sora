"""V1: extreme-value (EVT) validation of the M0 and M1 residuals (Proposal 4.6, Blueprint 6.4).

Question answered
  The VaR backtest (backtest.py) fits a GPD to the top 10% of every rolling window and never looks at the fit.
  Here the tail model is CHECKED once, on the full sample: is a Generalised Pareto distribution a good description
  of the extreme standardised residuals, how heavy is the tail, do M0 and M1 differ, and how much of the answer
  rests on one episode?

What is done (all on the SAME sample, residuals standardised by their own sd so that M1's mechanically smaller
residuals do not favour it, Proposal 4.6)
  mean_excess        mean of (u_i - t | u_i > t) over a grid of thresholds t: roughly linear above the point where the
                     GPD applies, with slope xi / (1 - xi)
  threshold_table    GPD fit above several thresholds (maximum likelihood), n of exceedances, shape xi and its
                     approximate standard error (1 + xi) / sqrt(n), the 99% tail quantile, and the
                     Kolmogorov-Smirnov distance with a PARAMETRIC-BOOTSTRAP p-value (the textbook KS p-value is invalid
                     when the GPD parameters were estimated from the same data)
  choose_threshold   a reproducible rule: the lowest threshold with at least `min_exc` exceedances whose KS p-value is
                     at least `min_p` (a convention, not a theorem)
  compare_models     M0 vs M1: difference in xi and in the 99% quantile, paired moving-block bootstrap (same dates)
  episodes           how many separate clusters hold the exceedances, and what share sits in the densest window
  without_window     the tail fit again with the densest window removed (the threshold stays at its full-sample value)

Both tails are covered: 'up' = rate rises (the S-REIT borrower's risk), 'down' = rate falls.
Limits that stay: ~1,300 daily points with one rate cycle give 30-130 exceedances; the GPD theory assumes independent
exceedances, and volatility clustering breaks that (declustering or a GARCH filter is the usual repair, see README).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import restricted_ecm as E
from .backtest import gpd_fit_mle, gpd_fit_pwm, gpd_quantile
from .nested import block_idx
from .ols import subset

QGRID = (0.85, 0.90, 0.925, 0.95, 0.975)
TAILS = {"up": 1.0, "down": -1.0}
FITTERS = {"mle": gpd_fit_mle, "pwm": gpd_fit_pwm}


# ------------------------------------------------------------------ residuals
def standardised_residuals(df: pd.DataFrame, p: int, with_z: bool = True) -> dict:
    """{'M0': u0, 'M1': u1}: residuals of M1 and of M0 = M1 minus SOFR / FX on the SAME sample, each divided by its
    own residual sd (s = sqrt(RSS / (n - k)))."""
    m1 = E.fit_ecm(df, p, 0, with_z)
    m0 = subset(m1.fit, E.nested_columns(m1.fit.names))
    return {"M0": m0.resid / np.sqrt(m0.s2), "M1": m1.fit.resid / np.sqrt(m1.fit.s2)}


# ------------------------------------------------------------------ GPD fit and goodness of fit
def ks_stat(x, xi: float, beta: float) -> float:
    """Kolmogorov-Smirnov distance between the sample x and GPD(xi, beta)."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    F = stats.genpareto.cdf(x, xi, scale=beta)
    i = np.arange(1, n + 1)
    return float(max((i / n - F).max(), (F - (i - 1) / n).max()))


def ks_pvalue_bootstrap(n: int, xi: float, beta: float, d_obs: float, B: int, rng, fit: str = "mle") -> float:
    """Parametric bootstrap p-value of the KS distance: draw n points from the fitted GPD, REFIT, recompute the
    distance. Draws whose fit fails are skipped."""
    fitter = FITTERS[fit]
    ds = []
    for _ in range(B):
        sim = stats.genpareto.rvs(xi, scale=beta, size=n, random_state=rng)
        try:
            ds.append(ks_stat(sim, *fitter(sim)))
        except Exception:
            continue
    return float((1 + np.sum(np.asarray(ds) >= d_obs)) / (len(ds) + 1))


def fit_tail(u, q: float, B: int, rng, fit: str = "mle", prob: float = 0.01) -> dict:
    """GPD for the exceedances of u over its q-quantile. `prob`: tail probability of the reported quantile."""
    u = np.asarray(u, float)
    thr = float(np.quantile(u, q))
    exc = u[u > thr] - thr
    n = len(exc)
    xi, beta = FITTERS[fit](exc)
    d = ks_stat(exc, xi, beta)
    return {"q": q, "thr": thr, "n_exc": n, "xi": xi, "se_xi": (1 + xi) / np.sqrt(n), "beta": beta, "KS": d,
            "KS_p": ks_pvalue_bootstrap(n, xi, beta, d, B, rng, fit) if B else np.nan,
            f"q{round((1 - prob) * 100):g}": gpd_quantile(thr, xi, beta, n / len(u), prob)}


def threshold_table(resid: dict, qgrid=QGRID, B: int = 200, seed: int = 2027, fit: str = "mle", prob: float = 0.01) -> pd.DataFrame:
    """One row per (model, tail, threshold quantile)."""
    rng = np.random.default_rng(seed)
    rows = []
    for model, s in resid.items():
        for tail, sgn in TAILS.items():
            for q in qgrid:
                rows.append({"model": model, "tail": tail} | fit_tail(sgn * s.to_numpy(), q, B, rng, fit, prob))
    return pd.DataFrame(rows)


def mean_excess(u, qs=np.arange(0.70, 0.985, 0.005)) -> pd.DataFrame:
    """Mean excess function over the thresholds at the quantiles qs (needs at least 5 exceedances)."""
    u = np.asarray(u, float)
    rows = []
    for q in qs:
        t = np.quantile(u, q)
        e = u[u > t] - t
        if len(e) >= 5:
            rows.append({"q": float(q), "thr": float(t), "mean_excess": float(e.mean()), "n_exc": len(e)})
    return pd.DataFrame(rows)


def choose_threshold(table: pd.DataFrame, min_exc: int = 30, min_p: float = 0.10) -> pd.DataFrame:
    """Per (model, tail): the lowest threshold quantile with n_exc >= min_exc and KS_p >= min_p (NaN if none)."""
    rows = []
    for (model, tail), g in table.groupby(["model", "tail"], sort=False):
        ok = g[(g["n_exc"] >= min_exc) & (g["KS_p"] >= min_p)].sort_values("q")
        rows.append({"model": model, "tail": tail, "q": float(ok["q"].iloc[0]) if len(ok) else np.nan,
                     "xi": float(ok["xi"].iloc[0]) if len(ok) else np.nan, "n_exc": int(ok["n_exc"].iloc[0]) if len(ok) else 0})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ M0 vs M1
def compare_models(u0, u1, q: float = 0.90, B: int = 300, block: int = 20, seed: int = 2027, fit: str = "mle",
                   prob: float = 0.01, level: float = 0.90) -> dict:
    """Difference M1 - M0 in the GPD shape and in the `prob` tail quantile, with a percentile interval from a paired
    moving-block bootstrap (the same resampled dates for both models, so their dependence is kept)."""
    u0, u1 = np.asarray(u0, float), np.asarray(u1, float)
    if len(u0) != len(u1):
        raise ValueError("M0 and M1 residuals must be on the same dates")
    fitter = FITTERS[fit]

    def one(u):
        thr = np.quantile(u, q)
        exc = u[u > thr] - thr
        xi, beta = fitter(exc)
        return xi, gpd_quantile(thr, xi, beta, len(exc) / len(u), prob)

    (x0, q0), (x1, q1) = one(u0), one(u1)
    rng = np.random.default_rng(seed)
    dx, dq = [], []
    for _ in range(B):
        i = block_idx(len(u0), block, rng)
        try:
            a, b = one(u0[i]), one(u1[i])
        except Exception:
            continue
        dx.append(b[0] - a[0])
        dq.append(b[1] - a[1])
    lo, hi = 50 * (1 - level), 50 * (1 + level)
    return {"q": q, "xi_M0": x0, "xi_M1": x1, "d_xi": x1 - x0, "d_xi_lo": float(np.percentile(dx, lo)), "d_xi_hi": float(np.percentile(dx, hi)),
            "quantile_M0": q0, "quantile_M1": q1, "d_quantile": q1 - q0, "d_quantile_lo": float(np.percentile(dq, lo)),
            "d_quantile_hi": float(np.percentile(dq, hi)), "n_boot": len(dx), "level": level}


# ------------------------------------------------------------------ how many independent episodes?
def episodes(u: pd.Series, q: float = 0.90, gap_days: int = 7, window_days: int = 120) -> dict:
    """Exceedances of the q-quantile: how many clusters (a gap of more than `gap_days` calendar days starts a new
    one) and how much of the total sits in the densest `window_days` window."""
    thr = u.quantile(q)
    d = u[u > thr].index.sort_values()
    gaps = np.diff(d.values).astype("timedelta64[D]").astype(int)
    counts = [int(((d >= t) & (d < t + pd.Timedelta(days=window_days))).sum()) for t in d]
    k = int(np.argmax(counts))
    return {"n_exc": len(d), "clusters": 1 + int(np.sum(gaps > gap_days)), "densest_window_start": d[k],
            "densest_window_days": window_days, "share_in_densest_window": counts[k] / len(d),
            "first": d[0], "last": d[-1]}


def without_window(u: pd.Series, q: float, start, days: int, B: int = 200, seed: int = 2027, fit: str = "mle") -> dict:
    """GPD fit with the window [start, start + days) removed. The threshold stays at its FULL-sample value, so the fit
    describes the same exceedance level without the episode."""
    thr = float(u.quantile(q))
    keep = ~((u.index >= start) & (u.index < pd.Timestamp(start) + pd.Timedelta(days=days)))
    exc = u[keep][u[keep] > thr].to_numpy() - thr
    xi, beta = FITTERS[fit](exc)
    d = ks_stat(exc, xi, beta)
    rng = np.random.default_rng(seed)
    return {"q": q, "thr": thr, "n_exc": len(exc), "xi": xi, "se_xi": (1 + xi) / np.sqrt(len(exc)), "beta": beta, "KS": d,
            "KS_p": ks_pvalue_bootstrap(len(exc), xi, beta, d, B, rng, fit) if B else np.nan}
