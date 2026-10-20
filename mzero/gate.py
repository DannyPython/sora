"""Block 2 (econometric gate): unit-root / stationarity tests for one series.

Blueprint 6.2: results are interpreted only if these checks hold. For M0 this
decides whether the AR(1) coefficient c1 is meaningfully below 1.
"""
from __future__ import annotations

import warnings

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def unit_root_tests(x: pd.Series, alpha: float = 0.05) -> pd.DataFrame:
    """ADF (H0: unit root) and KPSS (H0: level-stationary) on the level and first difference."""
    x = pd.Series(x).dropna().astype(float)
    rows = []
    for label, s in (("level", x), ("diff", x.diff().dropna())):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # adfuller return-type FutureWarning; kpss p-value table warning
            adf_stat, adf_p, lags = adfuller(s, regression="c", autolag="AIC")[:3]
            kpss_stat, kpss_p, _, _ = kpss(s, regression="c", nlags="auto")
        rows.append({
            "series": label, "adf_stat": adf_stat, "adf_p": adf_p, "adf_lags": lags,
            "adf_reject_unit_root": adf_p < alpha,
            "kpss_stat": kpss_stat, "kpss_p": kpss_p, "kpss_reject_stationary": kpss_p < alpha,
        })
    return pd.DataFrame(rows).set_index("series")


def integration_order(tests: pd.DataFrame) -> str:
    """Coarse reading: I(0) if level is ADF-stationary, I(1) if only the difference is."""
    lvl, dif = tests.loc["level"], tests.loc["diff"]
    if lvl.adf_reject_unit_root and not lvl.kpss_reject_stationary:
        return "I(0)"
    if dif.adf_reject_unit_root and not lvl.adf_reject_unit_root:
        return "I(1)"
    return "inconclusive (tests disagree)"