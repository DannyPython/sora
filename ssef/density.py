"""V2: predictive-density validation of RW, M0 and M1 on held-out data (Proposal 4.7; Berkowitz 2001, Diebold-Gunther-
Tay 1998).

Question answered
  The VaR backtest (backtest.py) looks at two quantiles, and at 99% a model is judged on about 8 breaches. Here the
  WHOLE predictive distribution is checked with every held-out point: is it too wide, too narrow, biased, or does it
  miss clustering in the errors?

Method
  The predictive distributions are exactly those of the VaR backtest (backtest.predictive_sets: same rolling window,
  same regressors, same origins). For each held-out value y the probability integral transform
        PIT = F_hat(y)         F_hat = the predictive cdf built from the window
  is uniform on (0, 1) if the distribution is right. Two predictive cdfs, as in the VaR backtest:
        normal      Phi((y - f) / s)                       (the Vasicek Gaussian assumption)
        empirical   rank of (y - f) among the window residuals (no distribution assumed)
  and z = Phi^{-1}(PIT) is iid N(0, 1) if the model is right. Reported per model and predictive cdf:
        KS            Kolmogorov-Smirnov test of PIT ~ U(0,1)
        mean_z, sd_z  bias (mean_z < 0: the forecasts run ABOVE the outcomes) and width (sd_z < 1: the distribution is
                      TOO WIDE, risk overstated; sd_z > 1: too narrow, risk understated), with their own tests
        ar1_z         first-order serial correlation of z
        Berkowitz     likelihood-ratio test of (mean 0, sd 1, no AR(1)) against free values, chi2(3)
        Ljung-Box     on z (serial dependence) and on z^2 (volatility clustering)
        tails         share of PITs in the lower / upper 5% and 1% tails against 5% / 1% (exact binomial test): the same
                      question as the VaR breach count, but read off the whole predictive distribution

For horizons h > 1 the origins are non-overlapping (every h days), as in the VaR backtest; h = 20 leaves ~37 points,
which is too few for the tests to say much.
Limits that stay: the predictive scale is a constant (the window residual sd), so volatility clustering shows up as
a rejection by construction; z is built from estimated, not true, distributions (a small size distortion).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

from . import backtest as bt

KINDS = ("normal", "empirical")
BINS = 10


# ------------------------------------------------------------------ PIT
def pit_values(pred: bt.Predictive, actual) -> dict:
    """{'normal': PIT, 'empirical': PIT} of the held-out values `actual` (same units as pred.f), one per origin.
    The empirical PIT is (number of window residuals <= error + 0.5) / (window + 1), always strictly inside (0, 1)."""
    err = np.asarray(actual, float) - pred.f
    return {"normal": stats.norm.cdf(err / pred.s),
            "empirical": ((pred.e <= err[:, None]).sum(axis=1) + 0.5) / (pred.e.shape[1] + 1)}


# ------------------------------------------------------------------ tests on z
def berkowitz(z) -> dict:
    """Berkowitz (2001) LR test that z is iid N(0,1) against z_t = mu + rho z_{t-1} + sd * e_t (conditional Gaussian
    likelihood, first value dropped), chi2(3). Also the three single-restriction tests on z itself:
    p_mean (t-test of mean 0), p_var (chi2 test of variance 1, two-sided), p_ar1 (rho = 0, z-test with se 1/sqrt(n))."""
    z = np.asarray(z, float)
    y, x = z[1:], z[:-1]
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    s2 = e @ e / n
    ll_u = -0.5 * n * (np.log(2 * np.pi * s2) + 1)
    ll_r = -0.5 * (n * np.log(2 * np.pi) + float(y @ y))
    lr = max(2 * (ll_u - ll_r), 0.0)
    m, v, N = float(z.mean()), float(z.var(ddof=1)), len(z)
    chi = (N - 1) * v
    rho = float(np.corrcoef(z[1:], z[:-1])[0, 1])
    return {"mu": float(b[0]), "rho": float(b[1]), "sd": float(np.sqrt(s2)), "LR": float(lr), "p": float(stats.chi2.sf(lr, 3)),
            "p_mean": float(2 * stats.t.sf(abs(m) / np.sqrt(v / N), N - 1)),
            "p_var": float(2 * min(stats.chi2.cdf(chi, N - 1), stats.chi2.sf(chi, N - 1))),
            "p_ar1": float(2 * stats.norm.sf(abs(rho) * np.sqrt(N)))}


def tail_shares(pit, probs=(0.05, 0.01)) -> dict:
    """Share of PITs in the lower and in the upper `prob` tail and the exact binomial p-value against `prob`. A share
    ABOVE prob means the model puts too little probability in that tail (risk understated); below, too much."""
    pit = np.asarray(pit, float)
    n = len(pit)
    out = {}
    for pr in probs:
        lab = f"{pr:.0%}"
        for side, k in (("lower", int((pit < pr).sum())), ("upper", int((pit > 1 - pr).sum()))):
            out[f"{side}_{lab}"] = k / n
            out[f"p_{side}_{lab}"] = float(stats.binomtest(k, n, pr).pvalue)
    return out


def reading(row: dict, alpha: float = 0.05) -> str:
    """Plain-language verdict from the single-restriction tests."""
    out = []
    if row["p_var"] < alpha:
        out.append("too wide (risk overstated)" if row["sd_z"] < 1 else "too narrow (risk understated)")
    if row["p_mean"] < alpha:
        out.append("forecasts run above outcomes" if row["mean_z"] < 0 else "forecasts run below outcomes")
    if row["p_ar1"] < alpha or row["LjungBox_z_p"] < alpha:
        out.append("serially dependent errors")
    if row["LjungBox_z2_p"] < alpha:
        out.append("volatility clustering")
    return "; ".join(out) if out else "no violation found"


# ------------------------------------------------------------------ the table
def density_table(pits: dict, models=("rw", "m0", "m1"), alpha: float = 0.05) -> pd.DataFrame:
    """One row per (model, predictive cdf) from {(model, kind): PIT array}."""
    rows = []
    for m in models:
        for kind in KINDS:
            pit = np.asarray(pits[(m, kind)], float)
            z = stats.norm.ppf(pit)
            b = berkowitz(z)
            row = {"model": m, "predictive": kind, "N": len(pit), "KS_p": float(stats.kstest(pit, "uniform").pvalue),
                   "mean_z": float(z.mean()), "sd_z": float(z.std(ddof=1)), "excess_kurtosis_z": float(stats.kurtosis(z)),
                   "ar1_z": b["rho"], "p_mean": b["p_mean"], "p_var": b["p_var"], "p_ar1": b["p_ar1"],
                   "Berkowitz_LR": b["LR"], "Berkowitz_p": b["p"],
                   "LjungBox_z_p": float(acorr_ljungbox(z, lags=[5])["lb_pvalue"].iloc[0]),
                   "LjungBox_z2_p": float(acorr_ljungbox(z ** 2, lags=[5])["lb_pvalue"].iloc[0])} | tail_shares(pit)
            row["reading"] = reading(row, alpha)
            rows.append(row)
    return pd.DataFrame(rows)


def pit_histogram(pits: dict, bins: int = BINS) -> pd.DataFrame:
    """Counts of PITs in `bins` equal bins (flat = well calibrated), one column per (model, predictive cdf)."""
    out = {f"{m}_{k}": np.histogram(v, bins=bins, range=(0, 1))[0] for (m, k), v in pits.items()}
    return pd.DataFrame(out, index=pd.Index([f"{i / bins:.1f}-{(i + 1) / bins:.1f}" for i in range(bins)], name="pit_bin"))


def compute_pit(df: pd.DataFrame, h: int = 1, window: int = 500, with_z: bool = True, p: int = 0, q: int = 0,
                step: int | None = None) -> tuple[dict, dict]:
    """(pits, info): PITs {(model, kind): array} for RW, M0, M1 at horizon h, from the VaR backtest's predictive sets."""
    with_z = with_z and "z" in df
    ps = bt.predictive_sets(df["r"].to_numpy(), df["x"].to_numpy(), df["z"].to_numpy() if with_z else None, h, window,
                            with_z, p, q, h if step is None else step)
    pits = {}
    for m, pred in ps["models"].items():
        for kind, v in pit_values(pred, ps["actual"]).items():
            pits[(m, kind)] = v
    idx = df.index[ps["T"] + h]
    return pits, {"dates": idx, "N": len(idx), "h": h}
