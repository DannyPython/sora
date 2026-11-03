"""The noise term of M1 (Proposal 4.3.3): estimate, check, standardise, generate.

The Proposal ASSUMES eps (= sigma dW over one step) is approximately Gaussian with constant variance.
noise_diagnostics TESTS that. Standardised residuals are the input of the EVT step (a conditional,
GARCH-filtered version is warranted when ARCH is strong); this module only supplies them.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera

from .ols import OLSFit


def std_resid(fit: OLSFit) -> pd.Series:
    return (fit.resid / np.sqrt(fit.s2)).rename("std_resid")


def noise_diagnostics(resid: pd.Series, lags_lb: int = 10, lags_arch: int = 5, alpha: float = 0.05) -> pd.DataFrame:
    """Gaussian: Jarque-Bera / kurtosis / tails. White: Ljung-Box. Constant variance: ARCH-LM."""
    e = pd.Series(resid).dropna().to_numpy()
    z = (e - e.mean()) / e.std(ddof=1)
    jb, jb_p, skew, kurt = jarque_bera(e)
    lb_p = float(acorr_ljungbox(e, lags=[lags_lb])["lb_pvalue"].iloc[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arch_p = float(het_arch(e, nlags=lags_arch)[1])
    tail = float(np.mean(np.abs(z) > 3))
    rows = [
        ("excess kurtosis", kurt - 3.0, np.nan, abs(kurt - 3.0) > 1.0),
        ("skewness", skew, np.nan, abs(skew) > 0.5),
        ("share beyond 3 sd", tail, np.nan, tail > 0.0081),
        ("Jarque-Bera", jb, jb_p, jb_p < alpha),
        (f"Ljung-Box ({lags_lb} lags)", np.nan, lb_p, lb_p < alpha),
        (f"ARCH-LM ({lags_arch} lags)", np.nan, arch_p, arch_p < alpha),
    ]
    return pd.DataFrame(rows, columns=["check", "value", "p_value", "violated"]).set_index("check")


def compare_noise(full: OLSFit, restricted: OLSFit) -> dict:
    """M1 vs its nested M0 on the SAME sample. M1's extra regressors shrink the residual variance
    mechanically, so tails must be compared on STANDARDISED residuals (Blueprint 6.4)."""
    return {"n": full.n, "s_eps_m0_bp": float(np.sqrt(restricted.s2) * 1e4), "s_eps_m1_bp": float(np.sqrt(full.s2) * 1e4),
            "variance_reduction": float(1 - full.s2 / restricted.s2)}


def draw_noise(s_eps: float, n_steps: int, n_paths: int, seed: int, method: str = "gaussian", std_res=None) -> np.ndarray:
    """(n_paths, n_steps). 'gaussian' N(0, s_eps^2); 'bootstrap' resamples standardised residuals (keeps the
    fat tails; iid, so volatility clustering is not reproduced)."""
    rng = np.random.default_rng(seed)
    if method == "gaussian":
        return s_eps * rng.standard_normal((n_paths, n_steps))
    if method == "bootstrap":
        z = np.asarray(std_res, float)
        z = (z - z.mean()) / z.std(ddof=1)
        return s_eps * rng.choice(z, size=(n_paths, n_steps), replace=True)
    raise ValueError("method must be 'gaussian' or 'bootstrap'")
