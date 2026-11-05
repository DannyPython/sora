"""M1 gate (Blueprint Sec. 6.2): is a regression of SORA on SOFR/FX meaningful?

  unit_root_table    ADF (BIC lags) and KPSS on r, x, z and the spread x - r
  engle_granger      r ~ x residual ADF, MacKinnon critical values, over a LAG GRID (the verdict is lag-sensitive)
  gregory_hansen     cointegration allowing ONE break (level shift 'C' or regime shift 'C/S'), critical values
                     simulated under 'no cointegration' with the same n, trimming, lags and break grid
  slope_stability    cointegrating slope of r on x by sub-period and rolling window (gamma drift)
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint, kpss


# ------------------------------------------------------------------ unit roots
def unit_root_table(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    series = {"r (SORA)": df["r"], "x (SOFR)": df["x"], "x - r (spread)": df["x"] - df["r"]}
    if "z" in df:
        series["z (FX return)"] = df["z"]
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, s in series.items():
            for form, v in (("level", s), ("diff", s.diff().dropna())):
                st, p, lag = adfuller(v, regression="c", autolag="BIC")[:3]
                kp = kpss(v, regression="c", nlags="auto")[1]
                rows.append({"series": name, "form": form, "adf_lags_bic": lag, "adf_p": p, "kpss_p": kp,
                             "reading": ("stationary" if p < alpha and kp >= alpha else
                                         "unit root / persistent" if p >= alpha and kp < alpha else "tests disagree")})
    return pd.DataFrame(rows).set_index(["series", "form"])


# ------------------------------------------------------------------ Engle-Granger
def engle_granger(r, x, lags=range(0, 11), trends=("c", "ct"), maxlag_bic: int = 10) -> pd.DataFrame:
    """Residual-based test of H0 'no cointegration' (r ~ x), MacKinnon critical values.
    trend 'c' = constant only; 'ct' = constant + linear trend in the long-run relation.
    Rows: fixed ADF lags plus the BIC-selected lag. The verdict is LAG- and TREND-sensitive: read the whole table."""
    r, x = np.asarray(r, float), np.asarray(x, float)
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr in trends:
            for p in lags:
                t, pv, cv = coint(r, x, trend=tr, maxlag=p, autolag=None)
                rows.append({"trend": tr, "rule": "fixed", "adf_lags": p, "t_stat": t, "p_value": pv, "cv_5pct": cv[1]})
            t, pv, cv = coint(r, x, trend=tr, maxlag=maxlag_bic, autolag="bic")
            cols = [np.ones(len(x)), x] + ([np.arange(len(x), dtype=float)] if tr == "ct" else [])
            X = np.column_stack(cols)
            u = r - X @ np.linalg.lstsq(X, r, rcond=None)[0]
            used = adfuller(u, regression="n", maxlag=maxlag_bic, autolag="BIC")[2]
            rows.append({"trend": tr, "rule": "BIC", "adf_lags": used, "t_stat": t, "p_value": pv, "cv_5pct": cv[1]})
    return pd.DataFrame(rows).set_index(["trend", "rule", "adf_lags"])


# ------------------------------------------------------------------ Gregory-Hansen
def _adf_t(u: np.ndarray, p: int) -> float:
    """t-stat of rho in  du_t = rho u_{t-1} + sum_{k=1..p} phi_k du_{t-k} + e  (no constant)."""
    du = np.diff(u)
    m = len(du) - p
    cols = [u[p:len(u) - 1]] + [du[p - k:len(du) - k] for k in range(1, p + 1)]
    X = np.column_stack(cols)
    y = du[p:]
    XtX = X.T @ X
    b = np.linalg.solve(XtX, X.T @ y)
    e = y - X @ b
    s2 = e @ e / (m - X.shape[1])
    return float(b[0] / np.sqrt(s2 * np.linalg.inv(XtX)[0, 0]))


def _gh_stat(y, x, model: str, lags: int, trim: float, step: int):
    n = len(y)
    lo, hi = int(np.ceil(trim * n)), int(np.floor((1 - trim) * n))
    best = (np.inf, None)
    one = np.ones(n)
    for tau in range(lo, hi, step):
        D = (np.arange(n) >= tau).astype(float)
        cols = [one, D, x] + ([D * x] if model == "C/S" else [])
        X = np.column_stack(cols)
        u = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        t = _adf_t(u, lags)
        if t < best[0]:
            best = (t, tau)
    return best


def _block_idx(n: int, block: int, rng) -> np.ndarray:
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    return (starts[:, None] + np.arange(block)).ravel()[:n]


def gregory_hansen(r, x, model: str = "C", lags: int = 5, trim: float = 0.15, step: int = 1, n_sim: int = 200,
                   seed: int = 2027, dates=None, step_sim: int = 5, null: str = "bootstrap", block: int = 10,
                   return_sims: bool = False) -> dict:
    """Gregory-Hansen (1996) ADF* statistic = most negative residual-ADF t over candidate break dates.
    model 'C'  : level shift   r = a + a2 D + b x
    model 'C/S': regime shift  r = a + a2 D + b x + b2 D x
    H0: no cointegration (any break). Reject when the statistic is BELOW the critical value.
    Critical values are simulated under 'no cointegration' with the same n, lags and trimming:
      null='bootstrap': two random walks built from block-resampled (dr, dx) PAIRS of the actual data, so fat
                        tails, volatility clustering and the contemporaneous correlation are preserved (default);
      null='iid'      : independent Gaussian random walks (reproduces the published Gregory-Hansen values).
    The break grid in the simulation is coarser (step_sim) to keep it fast.
    return_sims=True adds the simulated null statistics under 'sims' (for plotting; off by default so the
    result stays a flat dict of scalars)."""
    if model not in ("C", "C/S"):
        raise ValueError("model must be 'C' or 'C/S'")
    r, x = np.asarray(r, float), np.asarray(x, float)
    stat, tau = _gh_stat(r, x, model, lags, trim, step)
    out = {"model": model, "lags": lags, "stat": stat, "break_index": tau,
           "break_date": (pd.Timestamp(dates[tau]) if dates is not None else None)}
    if n_sim and n_sim > 0:
        rng = np.random.default_rng(seed)
        n = len(r)
        sims = np.empty(n_sim)
        d = np.column_stack([np.diff(r), np.diff(x)])
        d = d - d.mean(axis=0)
        for i in range(n_sim):
            if null == "iid":
                a, b = np.cumsum(rng.standard_normal(n)), np.cumsum(rng.standard_normal(n))
            elif null == "bootstrap":
                e = d[_block_idx(n - 1, block, rng)]
                a, b = np.r_[0.0, np.cumsum(e[:, 0])], np.r_[0.0, np.cumsum(e[:, 1])]
            else:
                raise ValueError("null must be 'bootstrap' or 'iid'")
            sims[i] = _gh_stat(a, b, model, lags, trim, step_sim)[0]
        out.update({"cv_1pct": float(np.quantile(sims, 0.01)), "cv_5pct": float(np.quantile(sims, 0.05)),
                    "cv_10pct": float(np.quantile(sims, 0.10)), "p_value": float((1 + np.sum(sims <= stat)) / (n_sim + 1)),
                    "n_sim": n_sim, "null": null, "reject_5pct": bool(stat < np.quantile(sims, 0.05))})
        if return_sims:
            out["sims"] = sims
    return out


# ------------------------------------------------------------------ gamma stability
def slope_stability(df: pd.DataFrame, windows=(500,), step: int = 50, periods=None) -> dict:
    """OLS slope of r on x (the cointegrating 'gamma'), by sub-period and over rolling windows."""
    def slope(d):
        X = np.column_stack([np.ones(len(d)), d["x"].to_numpy()])
        return float(np.linalg.lstsq(X, d["r"].to_numpy(), rcond=None)[0][1])
    if periods is None:
        edges = pd.date_range(df.index[0], df.index[-1], periods=5)
        periods = [(edges[i], edges[i + 1]) for i in range(4)]
    sub = pd.DataFrame([{"from": a.date(), "to": b.date(), "n": len(d), "slope": slope(d)}
                        for a, b in periods for d in [df[(df.index >= a) & (df.index <= b)]]])
    rolling = {w: pd.Series([slope(df.iloc[i - w:i]) for i in range(w, len(df) + 1, step)],
                            index=[df.index[i - 1] for i in range(w, len(df) + 1, step)]) for w in windows if w < len(df)}
    return {"full": slope(df), "sub_periods": sub, "rolling": rolling}
