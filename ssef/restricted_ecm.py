"""M1 in error-correction form (Blueprint v8c Sec. 5.2, eq. 8), estimated by OLS.

    dr_i = c + a r_{i-1} + b x_{i-1} + sum_{j=0..q} d_j dx_{i-j} + g z_i + sum_{k=1..p} phi_k dr_{i-k} + e_i

  a = lambda = -(1 - e^{-kappa dt}) ~ -kappa dt   (speed of adjustment, must lie in (-1, 0))
  long-run anchor  r* = alpha + beta1 x,   alpha = -c/a,  beta1 = -b/a
  z (FX return, I(0)) enters as a short-run term: g.  Levels-equivalent beta2 = -g/a  (see Blueprint 5.3)
  p lags of dr absorb the negative daily autocorrelation (the ADF-style augmentation).

M0 is the SAME regression with every SOFR and FX column dropped, on the SAME sample (see nested_columns):
    dr_i = c + a r_{i-1} + sum_{k=1..p} phi_k dr_{i-k} + e_i       (classical Vasicek on SORA alone, p lags)
so H0 "M0 is enough" sets x_{i-1}, every dx and z to zero. p = 0 is the Proposal's AR(1).

x_i is SOFR already published before SORA date i, so dx_i is known before r_i (no look-ahead).
For forecasting from origin T, origin_design gives the same columns using only data known at T.
The linear Gaussian model allows negative rates (Blueprint 5.5).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .loaders import DT
from .ols import OLSFit, delta_se, fit_ols



def _columns(r, x, z, p: int, q: int, with_z: bool, shift, idx, c: int):
    """ECM regressors for target rows idx. The same-day terms (dx, z) are taken c rows earlier: c = 0 in the
    estimated ECM, c = 1 when forecasting (they are not known yet at the origin). x=None gives M0's columns."""
    dr = np.r_[np.nan, np.diff(r)]
    cols, names = [np.ones(len(idx))], ["const"]
    cols.append(r[idx - 1]); names.append("r_lag")
    if x is not None:
        dx = np.r_[np.nan, np.diff(x)]
        cols.append(x[idx - 1]); names.append("x_lag")
    if shift is not None:
        cols.append(np.asarray(shift, float)[idx - 1]); names.append("shift")
    if x is not None:
        for j in range(q + 1):
            cols.append(dx[idx - j - c]); names.append(f"dx{j}")
    if with_z:
        cols.append(z[idx - c]); names.append("z")
    for k in range(1, p + 1):
        cols.append(dr[idx - k]); names.append(f"dr_{k}")
    return np.column_stack(cols), names


def design_arrays(r, x, z, p: int, q: int, with_z: bool, shift=None):
    """(y, X, names, s): y = dr over rows s..n-1; X columns in the order below. Pure arrays (fast).
    shift: optional 0/1 array (level-shift dummy D); enters the long-run relation as D_{i-1}."""
    s = max(p, q) + 1
    idx = np.arange(s, len(r))
    X, names = _columns(r, x, z, p, q, with_z, shift, idx, 0)
    return np.diff(r)[idx - 1], X, names, s


def origin_design(r, x, z, p: int, q: int, with_z: bool):
    """(X, names, s): row j holds the regressors known at forecast origin T = s + j (T = s..n-1):
    r_T, x_T, dx_{T-j}, z_T and dr_T..dr_{T-p+1}. Same column names as design_arrays, so nested_columns
    gives M0. No level-shift column: a break date chosen from the full sample would leak the future."""
    s = max(p, q + 1 if x is not None else 0)
    X, names = _columns(r, x, z, p, q, with_z and z is not None, None, np.arange(s, len(r)) + 1, 1)
    return X, names, s


def shift_dummy(df: pd.DataFrame, break_date):
    return None if break_date is None else (df.index >= pd.Timestamp(break_date)).astype(float)


def design(df: pd.DataFrame, p: int = 2, q: int = 0, with_z: bool = True, break_date=None):
    with_z = with_z and "z" in df
    z = df["z"].to_numpy() if with_z else None
    y, X, names, s = design_arrays(df["r"].to_numpy(), df["x"].to_numpy(), z, p, q, with_z, shift_dummy(df, break_date))
    idx = df.index[s:]
    return pd.Series(y, index=idx, name="dr"), pd.DataFrame(X, index=idx, columns=names)


def nested_columns(names):
    """Columns removed to get M0 from M1: every SOFR (level and changes) and FX column. The level-shift column
    'shift' is never removed: an M0 allowed one regime change in its mean is the fair null once a break is admitted."""
    return [c for c in names if c in ("x_lag", "z") or c.startswith("dx")]


@dataclass(frozen=True)
class M1ECM:
    fit: OLSFit
    p: int
    q: int
    with_z: bool
    dt: float
    break_date: object = None

    @property
    def has_shift(self) -> bool:
        return "shift" in self.fit.names

    @property
    def level_shift(self) -> float:        # long-run shift of the anchor after the break: -d/a
        return -self.fit.c("shift") / self.a if (self.has_shift and self.a < 0) else np.nan

    @property
    def a(self) -> float:
        return self.fit.c("r_lag")

    @property
    def c1(self) -> float:                 # persistence implied by the ECM: c1 = 1 + a
        return 1.0 + self.a

    @property
    def valid(self) -> bool:               # Proposal 4.3.4 / Blueprint 5.3: c1 in (0, 1)
        return 0.0 < self.c1 < 1.0

    @property
    def kappa(self) -> float:
        return -np.log(self.c1) / self.dt if self.valid else np.nan

    @property
    def alpha(self) -> float:
        return -self.fit.c("const") / self.a if self.a < 0 else np.nan

    @property
    def beta1(self) -> float:              # long-run anchor slope on SOFR
        return -self.fit.c("x_lag") / self.a if self.a < 0 else np.nan

    @property
    def gamma_z(self) -> float:
        return self.fit.c("z") if self.with_z else np.nan

    @property
    def beta2_equiv(self) -> float:
        return -self.gamma_z / self.a if (self.with_z and self.a < 0) else np.nan

    @property
    def s_eps(self) -> float:
        return float(np.sqrt(self.fit.s2))

    @property
    def sigma(self) -> float:              # sigma^2 = 2 kappa Var(eps) / (1 - c1^2)
        return float(np.sqrt(2 * self.kappa * self.fit.s2 / (1 - self.c1 ** 2))) if self.valid else np.nan

    def params_table(self, hac_lags: int | None = None) -> pd.DataFrame:
        """Coefficients with classical and HAC standard errors; HAC t-stats are descriptive only for the
        level terms (non-standard distribution when x is I(1): use the bootstrap tests)."""
        f = self.fit
        out = pd.DataFrame({"coef": f.coef, "se": f.se, "se_hac": f.hac_se(hac_lags)}, index=f.names)
        out["t"] = out["coef"] / out["se"]
        out["t_hac"] = out["coef"] / out["se_hac"]
        return out

    def structural_table(self, hac_lags: int | None = None) -> pd.DataFrame:
        """kappa, alpha, beta1 (and beta2_equiv) with delta-method SEs on the HAC covariance. beta1 = -b/a
        divides by a number close to zero: treat these intervals as unreliable (Fieller problem)."""
        f, names = self.fit, list(self.fit.names)
        V = f.hac_cov(hac_lags)
        ia, ib, ic = names.index("r_lag"), names.index("x_lag"), names.index("const")
        fns = {"kappa": lambda c: -np.log(1 + c[ia]) / self.dt,
               "alpha": lambda c: -c[ic] / c[ia],
               "beta1": lambda c: -c[ib] / c[ia]}
        if self.with_z:
            ig = names.index("z")
            fns["beta2_equiv"] = lambda c: -c[ig] / c[ia]
        if self.has_shift:
            isf = names.index("shift")
            fns["level_shift"] = lambda c: -c[isf] / c[ia]
        vals = {"kappa": self.kappa, "alpha": self.alpha, "beta1": self.beta1, "beta2_equiv": self.beta2_equiv,
                "level_shift": self.level_shift}
        rows = {}
        for k, fn in fns.items():
            se = delta_se(fn, f.coef, V) if self.valid else np.nan
            rows[k] = (vals[k], se, vals[k] - 1.96 * se, vals[k] + 1.96 * se)
        return pd.DataFrame(rows, index=["estimate", "se_delta", "ci_low", "ci_high"]).T


def fit_ecm(df: pd.DataFrame, p: int = 2, q: int = 0, with_z: bool = True, dt: float = DT,
            break_date=None) -> M1ECM:
    with_z = with_z and "z" in df
    y, X = design(df, p, q, with_z, break_date)
    return M1ECM(fit_ols(y, X), p, q, with_z, dt, break_date)


def select_lags(df: pd.DataFrame, pmax: int = 10, q: int = 0, with_z: bool = True, break_date=None) -> pd.DataFrame:
    """AIC / BIC and a residual Ljung-Box p-value for p = 0..pmax lags of dr, all on the SAME sample
    (rows from pmax+1), so the criteria are comparable. The gate and the tests should use enough lags for
    the residuals to be white; the BIC pick is the default suggestion."""
    from statsmodels.stats.diagnostic import acorr_ljungbox

    with_z = with_z and "z" in df
    y, X = design(df, pmax, q, with_z, break_date)
    rows = []
    for p in range(pmax + 1):
        keep = [c for c in X.columns if not (c.startswith("dr_") and int(c.split("_")[1]) > p)]
        f = fit_ols(y, X[keep])
        n, k = f.n, f.k
        ll = n * np.log(f.rss / n)
        lb = float(acorr_ljungbox(f.resid.to_numpy(), lags=[10])["lb_pvalue"].iloc[0])
        rows.append({"p": p, "k": k, "aic": ll + 2 * k, "bic": ll + k * np.log(n), "ljung_box_p": lb})
    out = pd.DataFrame(rows).set_index("p")
    out["best"] = ""
    out.loc[out["bic"].idxmin(), "best"] = "BIC"
    return out
