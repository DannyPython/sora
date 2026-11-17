"""Out-of-sample race at several horizons (Blueprint Sec. 6.6; Proposal 4.5.1).

Direct h-step forecasts of r_{T+h} from information at origin T, rolling window of `window` training pairs.
M0 and M1 are the estimated ECM (restricted_ecm), with regressors dated at the origin (restricted_ecm.origin_design):
  RW : r_T                                                  (random-walk benchmark - the third competitor)
  M0 : r_{T+h} on [1, r_T, dr_T .. dr_{T-p+1}]              (classical Vasicek on SORA alone)
  M1 : M0 + [x_T, dx_T .. dx_{T-q}, z_T]                     (anchor moves with SOFR and FX)
M0 = M1 minus restricted_ecm.nested_columns, so the two cannot drift apart. p = 0 gives the Proposal's AR(1).
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
from .restricted_ecm import nested_columns, origin_design

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


def direct_forecasts(r, x, z, h: int, window: int = 500, with_z: bool = True, p: int = 0, q: int = 0) -> dict:
    """RW, M0 and M1 forecasts for one horizon (decimals); x=None gives RW and M0 only. "T" holds the origins
    as positions in r. Computation is done in percent for conditioning."""
    pct = lambda v: None if v is None else np.asarray(v, float) * 100.0
    R = pct(r)
    X1, names, s = origin_design(R, pct(x), pct(z) if with_z else None, p, q, with_z)
    m0_cols = [i for i, nm in enumerate(names) if nm not in nested_columns(names)]
    Rs = R[s:]
    T, f0 = _rolling(Rs, X1[:, m0_cols], h, window)
    out = {"T": T + s, "actual": Rs[T + h] / 100.0, "rw": Rs[T] / 100.0, "m0": f0 / 100.0}
    if x is not None:
        out["m1"] = _rolling(Rs, X1, h, window)[1] / 100.0
    return out


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


def _race_stats(r, x, z, hs, window, with_z, p, q):
    out = {}
    for h in hs:
        fc = direct_forecasts(r, x, z, h, window, with_z, p, q)
        out[h] = {name: clark_west(fc["actual"], fc[s], fc[l], h) for name, s, l in PAIRS}
    return out


def run_race(df: pd.DataFrame, horizons=(1, 20, 60), window: int = 500, with_z: bool = True, B: int = 0,
             null: str = "unit_root", p: int = 2, q: int = 0, block: int = 10, seed: int = 2027) -> dict:
    """Returns {'summary': DataFrame per horizon, 'forecasts': {h: DataFrame}, 'boot_t': {...}}.
    p, q: lags of dr and dx, the same as in the estimated ECM.
    B = number of bootstrap replications (0 = skip; the normal p-values are then NOT trustworthy)."""
    with_z = with_z and "z" in df
    r, x = df["r"].to_numpy(), df["x"].to_numpy()
    z = df["z"].to_numpy() if with_z else None
    idx = df.index
    rows, fcs = [], {}
    for h in horizons:
        fc = direct_forecasts(r, x, z, h, window, with_z, p, q)
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
        # M0 null: every SOFR/FX column dropped; r evolves as an AR(p) in differences (unit root imposed if asked)
        dgp = NullDGP(df, p, q, with_z, None, null, block)
        rng = np.random.default_rng(seed)
        acc = {h: {name: np.empty(B) for name, _, _ in PAIRS} for h in horizons}
        for b in range(B):
            st = _race_stats(dgp.draw(rng), x, z, horizons, window, with_z, p, q)
            for h in horizons:
                for name, _, _ in PAIRS:
                    acc[h][name][b] = st[h][name]
        for h in horizons:
            for name, _, _ in PAIRS:
                summary.loc[h, f"p_boot {name}"] = float((1 + np.sum(acc[h][name] >= summary.loc[h, f"cw_t {name}"])) / (B + 1))
        boot = acc
    return {"summary": summary, "forecasts": fcs, "boot_t": boot, "null": null, "B": B, "window": window}


def m0_vs_rw(r, horizons=(1, 20, 60), window: int = 500, p: int = 0) -> pd.DataFrame:
    """M0 alone against the random walk at several horizons (direct_forecasts without SOFR, rolling window).
    The anchor of a mean-reverting model only shows up in slow moves, so one day is not enough."""
    r = np.asarray(pd.Series(r).dropna(), dtype=float)
    rows = []
    for h in horizons:
        if window + p + h + 20 >= len(r):
            continue
        fc = direct_forecasts(r, None, None, h, window, False, p)
        rows.append({"h": h, "n_oos": len(fc["T"]),
                     "rmse_rw_bp": float(np.sqrt(np.mean((fc["actual"] - fc["rw"]) ** 2)) * 1e4),
                     "rmse_m0_bp": float(np.sqrt(np.mean((fc["actual"] - fc["m0"]) ** 2)) * 1e4)})
    out = pd.DataFrame(rows).set_index("h")
    out["m0_vs_rw_pct"] = (out["rmse_m0_bp"] / out["rmse_rw_bp"] - 1.0) * 100.0
    return out
