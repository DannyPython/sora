"""Block 2 (econometric gate), v2.

Three questions, answered separately instead of one ADF p-value:
  1. unit_root_tests : ADF + KPSS, with and without a trend, on level and difference
                       (backward compatible with report.py: same index/columns).
  2. verdict         : joint ADF/KPSS reading -> I(0) / I(1) / conflicting / low power.
  3. ar1_gate        : the quantity M0 actually needs, c1 in (0,1):
                       OLS c1, Kendall bias-corrected c1, and Hansen (1999) grid-bootstrap CI.
  4. engle_granger   : cointegration of r with x for M1 (correct critical values).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint, kpss


# ------------------------------------------------------------------ 1. unit-root tests
def _maxlag(n: int) -> int:
    return int(np.ceil(12 * (n / 100) ** 0.25))          # Schwert rule


def unit_root_tests(x: pd.Series, alpha: float = 0.05, trend: bool = False) -> pd.DataFrame:
    """ADF (H0: unit root) and KPSS (H0: stationary) on the level and first difference.

    trend=True uses a constant + linear trend in the level (a drifting rate is not a unit root
    by itself, but a trend-stationary reading needs its own test). Lags chosen by BIC, which
    over-fits less than AIC in short samples.
    """
    x = pd.Series(x).dropna().astype(float)
    reg = "ct" if trend else "c"
    rows = []
    for label, s, rg in (("level", x, reg), ("diff", x.diff().dropna(), "c")):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf_stat, adf_p, lags = adfuller(s, maxlag=_maxlag(len(s)), regression=rg, autolag="BIC")[:3]
            kpss_stat, kpss_p, _, _ = kpss(s, regression=rg, nlags="auto")
        rows.append({
            "series": label, "adf_stat": adf_stat, "adf_p": adf_p, "adf_lags": lags,
            "adf_reject_unit_root": adf_p < alpha,
            "kpss_stat": kpss_stat, "kpss_p": kpss_p, "kpss_reject_stationary": kpss_p < alpha,
        })
    return pd.DataFrame(rows).set_index("series")


def verdict(tests: pd.DataFrame) -> str:
    """Joint ADF/KPSS reading of the level. Replaces integration_order()."""
    lvl, dif = tests.loc["level"], tests.loc["diff"]
    adf, kp = lvl.adf_reject_unit_root, lvl.kpss_reject_stationary
    if adf and not kp:
        return "I(0): both tests agree the level is stationary"
    if not adf and kp:
        return ("I(1): both tests agree on a unit root" if dif.adf_reject_unit_root
                else "non-stationary, and the difference is not clearly I(0) either")
    if adf and kp:
        return "conflicting (ADF says stationary, KPSS says not): suspect a structural break or fractional integration"
    return "inconclusive (neither rejects): the sample has too little power to tell I(0) from I(1)"


# ------------------------------------------------------------------ 2. the AR(1) gate for M0
@dataclass(frozen=True)
class AR1Gate:
    n: int
    c1_ols: float
    c1_bc: float                 # Kendall bias-corrected
    ci: tuple[float, float]      # Hansen grid-bootstrap CI for c1
    alpha: float

    def kappa_ols(self, dt: float = 1 / 252) -> float:
        return -np.log(self.c1_ols) / dt if 0 < self.c1_ols < 1 else np.nan

    def kappa_bc(self, dt: float = 1 / 252) -> float:
        return -np.log(self.c1_bc) / dt if 0 < self.c1_bc < 1 else np.nan

    @property
    def reverts(self) -> bool:
        """Mean reversion is supported only if the whole CI sits below 1."""
        return self.ci[1] < 1.0


def _ols_c1(y: np.ndarray):
    """Rowwise OLS of y_t on (1, y_{t-1}); y is (B, n+1). Returns c1, t-stat vs c supplied later."""
    x, z = y[:, :-1], y[:, 1:]
    xd, zd = x - x.mean(1, keepdims=True), z - z.mean(1, keepdims=True)
    sxx = (xd ** 2).sum(1)
    c1 = (xd * zd).sum(1) / sxx
    res = zd - c1[:, None] * xd
    se = np.sqrt((res ** 2).sum(1) / (x.shape[1] - 2) / sxx)
    return c1, se


def ar1_gate(r, alpha: float = 0.05, grid=None, n_boot: int = 499, wild: bool = True,
             seed: int = 2027) -> AR1Gate:
    """Bias-corrected c1 and a grid-bootstrap CI that stays valid near a unit root.

    The delta-method / normal CI is wrong exactly where we are (c1 ~ 0.99): the t-statistic
    is not normal and OLS c1 is biased down by about -(1 + 3 c1)/n (Kendall). Hansen's grid
    bootstrap inverts the bootstrap distribution of the t-statistic over candidate values of c1.
    wild=True multiplies residuals by Rademacher draws so volatility clustering is preserved.
    """
    y = np.asarray(pd.Series(r).dropna(), dtype=float)
    n = len(y) - 1
    c_hat, se_hat = (a[0] for a in _ols_c1(y[None, :]))
    c_bc = c_hat + (1 + 3 * c_hat) / n
    if grid is None:
        grid = np.linspace(min(c_hat, 0.95) - 0.02, 1.02, 80)
    rng = np.random.default_rng(seed)
    accepted = []
    for c0 in grid:
        mu = np.mean(y[1:] - c0 * y[:-1])
        e = y[1:] - mu - c0 * y[:-1]
        e = e - e.mean()
        draws = (e[None, :] * rng.choice([-1.0, 1.0], size=(n_boot, n))) if wild \
            else rng.choice(e, size=(n_boot, n))
        yb = np.empty((n_boot, n + 1))
        yb[:, 0] = y[0]
        for t in range(n):
            yb[:, t + 1] = mu + c0 * yb[:, t] + draws[:, t]
        cb, seb = _ols_c1(yb)
        tb = (cb - c0) / seb
        lo, hi = np.quantile(tb, [alpha / 2, 1 - alpha / 2])
        t_obs = (c_hat - c0) / se_hat
        if lo <= t_obs <= hi:
            accepted.append(c0)
    ci = (min(accepted), max(accepted)) if accepted else (np.nan, np.nan)
    return AR1Gate(n=n, c1_ols=float(c_hat), c1_bc=float(c_bc), ci=ci, alpha=alpha)


# ------------------------------------------------------------------ 3. cointegration for M1
def engle_granger(r: pd.Series, x: pd.Series, alpha: float = 0.05) -> dict:
    """Engle-Granger two-step with MacKinnon critical values (not plain ADF tables).

    Returns the test, the cointegrating slope gamma, and gamma re-estimated on each half of
    the sample: a cointegrating vector that moves between halves is a break, not a tether.
    """
    df = pd.concat([r, x], axis=1, keys=["r", "x"]).dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p, crit = coint(df["r"], df["x"], trend="c", autolag="bic")
    gamma, a = np.polyfit(df["x"], df["r"], 1)
    h = len(df) // 2
    g1 = np.polyfit(df["x"].iloc[:h], df["r"].iloc[:h], 1)[0]
    g2 = np.polyfit(df["x"].iloc[h:], df["r"].iloc[h:], 1)[0]
    return {"stat": stat, "p": p, "reject_no_coint": p < alpha, "gamma": gamma,
            "gamma_first_half": g1, "gamma_second_half": g2}
