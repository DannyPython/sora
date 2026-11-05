"""Valid tests when SOFR is I(1) (Blueprint Sec. 6.2-6.3, Proposal 4.4-4.5).

The classical / HAC F-test of H0: beta1 = beta2 = 0 does not follow an F (or chi2) distribution when a
regressor is I(1) and the long-run relation is absent, so its p-value is not a valid inference. Here the
reference distribution is BUILT: data are re-generated under the null, the same statistic is recomputed, and
the p-value is the share of re-generated statistics at least as extreme.

  anchor_test               H0: x_lag = z = 0 in the ECM (M0 = M1 with those columns dropped, same sample)
  cointegration_ecm_test    H0: r_lag = x_lag = 0 (no levels relation at all): F (Pesaran-Shin-Smith type) and
                            the t-test on r_lag (Banerjee-Dolado-Mestre type), bootstrap critical values

Null DGP: restricted ECM fitted on the data, residuals resampled in moving blocks (keeps volatility clustering),
r re-generated recursively; x, z and the dummy are held at their actual values (I(1) path kept).
null='unit_root' imposes a=0 (r is a random walk under H0: the spurious-regression case);
null='fitted' keeps the estimated mean reversion of the restricted model (only differs for anchor_test).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ecm import design_arrays, shift_dummy


def _block_idx(n: int, block: int, rng) -> np.ndarray:
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    return (starts[:, None] + np.arange(block)).ravel()[:n]


def _stats(y, X, names, restrict, tcol="r_lag"):
    """(F for dropping `restrict`, t-stat of tcol) from raw arrays."""
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ (X.T @ y)
    e = y - X @ b
    rss = float(e @ e)
    keep = [i for i, nm in enumerate(names) if nm not in restrict]
    Xr = X[:, keep]
    br = np.linalg.solve(Xr.T @ Xr, Xr.T @ y)
    er = y - Xr @ br
    q = k - len(keep)
    F = ((float(er @ er) - rss) / q) / (rss / (n - k))
    i = names.index(tcol)
    t = b[i] / np.sqrt(rss / (n - k) * XtXi[i, i])
    return F, float(t)


class NullDGP:
    """Restricted ECM + block-resampled residuals -> pseudo r paths that satisfy H0."""

    def __init__(self, df: pd.DataFrame, p: int, q: int, with_z: bool, restrict, null: str = "unit_root",
                 block: int = 10, break_date=None):
        if null not in ("unit_root", "fitted"):
            raise ValueError("null must be 'unit_root' or 'fitted'")
        self.p, self.q, self.with_z, self.block = p, q, with_z and "z" in df, block
        self.r = df["r"].to_numpy()
        self.x = df["x"].to_numpy()
        self.z = df["z"].to_numpy() if self.with_z else None
        self.shift = shift_dummy(df, break_date)
        y, X, names, s = design_arrays(self.r, self.x, self.z, p, q, self.with_z, self.shift)
        self.y, self.X, self.names, self.s = y, X, names, s
        drop = set(restrict) | ({"r_lag"} if null == "unit_root" else set())
        self.null_names = [nm for nm in names if nm not in drop]
        cols = [names.index(nm) for nm in self.null_names]
        Xn = X[:, cols]
        coef = np.linalg.lstsq(Xn, y, rcond=None)[0]
        self.resid = y - Xn @ coef
        self.resid = self.resid - self.resid.mean() * ("const" in self.null_names)
        cd = dict(zip(self.null_names, coef))
        self.a0 = cd.get("r_lag", 0.0)
        self.phi = np.array([cd.get(f"dr_{k}", 0.0) for k in range(1, p + 1)])
        fixed = [nm for nm in self.null_names if nm != "r_lag" and not nm.startswith("dr_")]
        self.fixed = (X[:, [names.index(nm) for nm in fixed]] @ np.array([cd[nm] for nm in fixed])
                      if fixed else np.zeros(len(y)))
        dr = np.r_[np.nan, np.diff(self.r)]
        self.dr_pre = [dr[s - k] for k in range(1, p + 1)]       # dr_{s-1}, ..., dr_{s-p}

    def draw(self, rng) -> np.ndarray:
        """One pseudo r series of the same length as the data (first s values are the actual ones)."""
        m = len(self.y)
        e = self.resid[_block_idx(m, self.block, rng)]
        r_prev = self.r[self.s - 1]
        hist = list(self.dr_pre)
        out = np.empty(len(self.r))
        out[:self.s] = self.r[:self.s]
        a0, phi, fixed = self.a0, self.phi, self.fixed
        for i in range(m):
            dri = fixed[i] + a0 * r_prev + e[i]
            for k in range(self.p):
                dri += phi[k] * hist[k]
            r_prev += dri
            out[self.s + i] = r_prev
            if self.p:
                hist = [dri] + hist[:-1]
        return out

    def stats(self, r_star, restrict, tcol="r_lag"):
        y, X, names, _ = design_arrays(r_star, self.x, self.z, self.p, self.q, self.with_z, self.shift)
        return _stats(y, X, names, restrict, tcol)


def _bootstrap(df, p, q, with_z, restrict, null, B, block, seed, break_date, tcol="r_lag") -> dict:
    dgp = NullDGP(df, p, q, with_z, restrict, null, block, break_date)
    F_obs, t_obs = _stats(dgp.y, dgp.X, dgp.names, restrict, tcol)
    rng = np.random.default_rng(seed)
    Fb, tb = np.empty(B), np.empty(B)
    for i in range(B):
        Fb[i], tb[i] = dgp.stats(dgp.draw(rng), restrict, tcol)
    return {"F": F_obs, "t": t_obs, "B": B, "null": null, "block": block,
            "p_boot_F": float((1 + np.sum(Fb >= F_obs)) / (B + 1)),
            "p_boot_t_left": float((1 + np.sum(tb <= t_obs)) / (B + 1)),       # t on r_lag: more negative = evidence
            "p_boot_t_abs": float((1 + np.sum(np.abs(tb) >= abs(t_obs))) / (B + 1)),
            "cv5_F": float(np.quantile(Fb, 0.95)), "cv5_t_left": float(np.quantile(tb, 0.05)),
            "F_boot": Fb, "t_boot": tb}


def anchor_test(df, p: int = 2, q: int = 0, with_z: bool = True, null: str = "unit_root", B: int = 499,
                block: int = 10, seed: int = 2027, break_date=None) -> dict:
    """H0: beta1 = beta2 = 0 (x_lag and z excluded). Returns the bootstrap F p-value (the valid one)."""
    restrict = ["x_lag"] + (["z"] if with_z and "z" in df else [])
    out = _bootstrap(df, p, q, with_z, restrict, null, B, block, seed, break_date, tcol="x_lag")
    out["restrict"] = restrict
    return out


def cointegration_ecm_test(df, p: int = 2, q: int = 0, with_z: bool = True, B: int = 499, block: int = 10,
                           seed: int = 2027, break_date=None) -> dict:
    """H0: no levels relation (r_lag = x_lag = 0, plus the shift dummy if one is included).
    F on the excluded levels (PSS-type) and left-tail t on r_lag (BDM-type), bootstrap p-values."""
    restrict = ["r_lag", "x_lag"] + (["shift"] if break_date is not None else [])
    out = _bootstrap(df, p, q, with_z, restrict, "unit_root", B, block, seed, break_date, tcol="r_lag")
    out["restrict"] = restrict
    return out
