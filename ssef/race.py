"""Out-of-sample race at several horizons (Blueprint Sec. 6.6; Proposal 4.5.1).

Direct h-step forecasts of r_{T+h} from information at origin T, rolling window of `window` training pairs:
  RW : r_T                                   (random-walk benchmark - the third competitor)
  M0 : r_{T+h} = c + c1 r_T + e               (classical level reversion, fixed anchor)
  M1 : r_{T+h} = c + c1 r_T + c2 x_T [+ c3 z_T] + e   (anchor moves with SOFR [and FX])
The anchor shows up in slow moves, so horizons 1, 20 and 60 days are used by default.
Training pairs at origin T only use targets already observed (t + h <= T): no look-ahead.

Clark-West (2007) adjusted loss  f = e_s^2 - [e_l^2 - (yhat_s - yhat_l)^2]  for nested (small s inside large l)
models, HAC (Bartlett, h-1 lags) t-statistic, one-sided. Its N(0,1) critical values assume stationary predictors:
with I(1) SOFR they are not valid, so p-values are ALSO bootstrapped: pseudo r series are generated under the
null (see nested.NullDGP), x and z held at their actual values, the whole rolling race is re-run on each.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .nested import NullDGP

PAIRS = (("M0 vs RW", "rw", "m0"), ("M1 vs RW", "rw", "m1"), ("M1 vs M0", "m0", "m1"))


def _rolling(R, X, h, window):
    """Rolling-window OLS forecasts for all origins at once (cumulative sums of X'X and X'y)."""
    n, k = X.shape
    m = n - h
    y = R[h:]
    A = np.einsum("ti,tj->tij", X[:m], X[:m])
    b = X[:m] * y[:, None]
    CA = np.concatenate([np.zeros((1, k, k)), np.cumsum(A, axis=0)])
    Cb = np.concatenate([np.zeros((1, k)), np.cumsum(b, axis=0)])
    T = np.arange(window + h - 1, n - h)
    hi, lo = T - h, T - h - window + 1
    beta = np.linalg.solve(CA[hi + 1] - CA[lo], (Cb[hi + 1] - Cb[lo])[..., None])[..., 0]
    return T, np.einsum("ti,ti->t", X[T], beta)


def direct_forecasts(r, x, z, h: int, window: int = 500, with_z: bool = True) -> dict:
    """Arrays for one horizon (decimals). Computation is done in percent for conditioning."""
    R, XS = np.asarray(r) * 100.0, np.asarray(x) * 100.0
    one = np.ones(len(R))
    X0 = np.column_stack([one, R])
    cols = [one, R, XS] + ([np.asarray(z) * 100.0] if with_z and z is not None else [])
    T, f0 = _rolling(R, X0, h, window)
    _, f1 = _rolling(R, np.column_stack(cols), h, window)
    return {"T": T, "actual": R[T + h] / 100.0, "rw": R[T] / 100.0, "m0": f0 / 100.0, "m1": f1 / 100.0}


def _nw_t(f: np.ndarray, lags: int) -> float:
    N = len(f)
    d = f - f.mean()
    v = d @ d / N
    for l in range(1, lags + 1):
        v += 2 * (1 - l / (lags + 1)) * (d[l:] @ d[:-l]) / N
    return float(f.mean() / np.sqrt(v / N))


def clark_west(actual, f_small, f_large, h: int) -> float:
    e_s, e_l = actual - f_small, actual - f_large
    return _nw_t(e_s ** 2 - (e_l ** 2 - (f_small - f_large) ** 2), max(h - 1, 0))


def _race_stats(r, x, z, hs, window, with_z):
    out = {}
    for h in hs:
        fc = direct_forecasts(r, x, z, h, window, with_z)
        out[h] = {name: clark_west(fc["actual"], fc[s], fc[l], h) for name, s, l in PAIRS}
    return out


def run_race(df: pd.DataFrame, horizons=(1, 20, 60), window: int = 500, with_z: bool = True, B: int = 0,
             null: str = "unit_root", p: int = 2, block: int = 10, seed: int = 2027) -> dict:
    """Returns {'summary': DataFrame per horizon, 'forecasts': {h: DataFrame}, 'boot_t': {...}}.
    B = number of bootstrap replications (0 = skip; the normal p-values are then NOT trustworthy)."""
    with_z = with_z and "z" in df
    r, x = df["r"].to_numpy(), df["x"].to_numpy()
    z = df["z"].to_numpy() if with_z else None
    idx = df.index
    rows, fcs = [], {}
    for h in horizons:
        fc = direct_forecasts(r, x, z, h, window, with_z)
        fcs[h] = pd.DataFrame({"origin": idx[fc["T"]], "actual": fc["actual"], "rw": fc["rw"], "m0": fc["m0"],
                               "m1": fc["m1"]}, index=idx[fc["T"] + h])
        row = {"h": h, "n_oos": len(fc["T"])}
        for nm in ("rw", "m0", "m1"):
            row[f"rmse_{nm}_bp"] = float(np.sqrt(np.mean((fc["actual"] - fc[nm]) ** 2)) * 1e4)
        for name, s, l in PAIRS:
            t = clark_west(fc["actual"], fc[s], fc[l], h)
            row[f"cw_t {name}"] = t
            row[f"p_normal {name}"] = float(stats.norm.sf(t))
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("h")
    for name, _, _ in PAIRS:
        summary[f"p_boot {name}"] = np.nan
    boot = {}
    if B and B > 0:
        # pure-M0 null: drop every exogenous column; r evolves as an AR(p) in differences (unit root imposed if asked)
        q = 0
        restrict = ["x_lag", "dx0"] + (["z"] if with_z else [])
        dgp = NullDGP(df, p, q, with_z, restrict, null, block)
        rng = np.random.default_rng(seed)
        acc = {h: {name: np.empty(B) for name, _, _ in PAIRS} for h in horizons}
        for b in range(B):
            st = _race_stats(dgp.draw(rng), x, z, horizons, window, with_z)
            for h in horizons:
                for name, _, _ in PAIRS:
                    acc[h][name][b] = st[h][name]
        for h in horizons:
            for name, _, _ in PAIRS:
                summary.loc[h, f"p_boot {name}"] = float((1 + np.sum(acc[h][name] >= summary.loc[h, f"cw_t {name}"])) / (B + 1))
        boot = acc
    return {"summary": summary, "forecasts": fcs, "boot_t": boot, "null": null, "B": B, "window": window}


def m0_vs_rw(r, horizons=(1, 20, 60), window: int = 250) -> pd.DataFrame:
    """M0 alone against the random walk at several horizons (direct h-step forecasts, rolling window).
    The anchor of a mean-reverting model only shows up in slow moves, so one day is not enough."""
    R = np.asarray(pd.Series(r).dropna(), dtype=float) * 100.0
    X0 = np.column_stack([np.ones(len(R)), R])
    rows = []
    for h in horizons:
        if window + h + 20 >= len(R):
            continue
        T, f0 = _rolling(R, X0, h, window)
        actual = R[T + h]
        rows.append({"h": h, "n_oos": len(T),
                     "rmse_rw_bp": float(np.sqrt(np.mean((actual - R[T]) ** 2)) * 100.0),
                     "rmse_m0_bp": float(np.sqrt(np.mean((actual - f0) ** 2)) * 100.0)})
    out = pd.DataFrame(rows).set_index("h")
    out["m0_vs_rw_pct"] = (out["rmse_m0_bp"] / out["rmse_rw_bp"] - 1.0) * 100.0
    return out
