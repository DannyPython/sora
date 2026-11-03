"""One OLS routine for the whole project. M0 is M1 with columns dropped, on the SAME sample:
build the full design once, then call fit_ols on the full matrix and on a column subset."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class OLSFit:
    names: tuple
    coef: np.ndarray
    cov: np.ndarray            # classical covariance, s2 (X'X)^-1
    s2: float                  # RSS / (n - k)
    rss: float
    resid: pd.Series
    y: np.ndarray = None
    X: np.ndarray = None

    @property
    def n(self) -> int:
        return len(self.y)

    @property
    def k(self) -> int:
        return len(self.names)

    def c(self, name: str) -> float:
        return float(self.coef[self.names.index(name)])

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    def tstat(self, name: str) -> float:
        i = self.names.index(name)
        return float(self.coef[i] / self.se[i])

    def hac_cov(self, lags: int | None = None) -> np.ndarray:
        """Newey-West (Bartlett) covariance, small-sample scaled by n/(n-k)."""
        n, k = self.n, self.k
        L = int(np.floor(4 * (n / 100.0) ** (2 / 9))) if lags is None else lags
        e = self.resid.to_numpy()
        Xe = self.X * e[:, None]
        S = Xe.T @ Xe
        for l in range(1, L + 1):
            G = Xe[l:].T @ Xe[:-l]
            S += (1 - l / (L + 1)) * (G + G.T)
        XtXi = np.linalg.inv(self.X.T @ self.X)
        return XtXi @ S @ XtXi * n / (n - k)

    def hac_se(self, lags: int | None = None) -> np.ndarray:
        return np.sqrt(np.diag(self.hac_cov(lags)))


def fit_ols(y, X, names=None) -> OLSFit:
    """OLS of y on X. X may be a DataFrame (names and index are taken from it) or an array."""
    index = None
    if isinstance(X, pd.DataFrame):
        names = list(X.columns) if names is None else list(names)
        index = X.index
        Xa = X.to_numpy(float)
    else:
        Xa = np.asarray(X, float)
        names = [f"x{i}" for i in range(Xa.shape[1])] if names is None else list(names)
    if isinstance(y, pd.Series) and index is None:
        index = y.index
    ya = np.asarray(y, float)
    beta, *_ = np.linalg.lstsq(Xa, ya, rcond=None)
    e = ya - Xa @ beta
    n, k = Xa.shape
    rss = float(e @ e)
    s2 = rss / (n - k)
    cov = s2 * np.linalg.inv(Xa.T @ Xa)
    return OLSFit(tuple(names), beta, cov, s2, rss, pd.Series(e, index=index, name="resid"), ya, Xa)


def subset(fit: OLSFit, drop) -> OLSFit:
    """Refit on the same sample with the named columns removed (M1 -> M0)."""
    keep = [i for i, nm in enumerate(fit.names) if nm not in set(drop)]
    return fit_ols(pd.Series(fit.y, index=fit.resid.index), pd.DataFrame(fit.X[:, keep], index=fit.resid.index,
                                                                         columns=[fit.names[i] for i in keep]))


def exclusion_f(full: OLSFit, restricted: OLSFit) -> dict:
    """Classical nested F. NOT valid when a regressor is I(1) without cointegration - see nested.py."""
    q = full.k - restricted.k
    F = ((restricted.rss - full.rss) / q) / (full.rss / (full.n - full.k))
    return {"F": float(F), "q": q, "p_classical": float(stats.f.sf(F, q, full.n - full.k))}


def exclusion_wald_hac(full: OLSFit, drop, lags: int | None = None) -> dict:
    """HAC Wald test that the named coefficients are zero. Reported for comparison only: with an
    I(1) regressor the chi2/F reference distribution is not valid."""
    idx = [full.names.index(d) for d in drop]
    b = full.coef[idx]
    V = full.hac_cov(lags)[np.ix_(idx, idx)]
    W = float(b @ np.linalg.solve(V, b))
    q = len(idx)
    return {"W": W, "q": q, "p_hac_chi2": float(stats.chi2.sf(W, q)),
            "p_hac_F": float(stats.f.sf(W / q, q, full.n - full.k))}


def delta_se(fn, coef: np.ndarray, cov: np.ndarray, eps: float = 1e-6) -> float:
    """Delta-method standard error of fn(coef) with a numerical gradient."""
    g = np.zeros(len(coef))
    for i in range(len(coef)):
        h = eps * max(abs(coef[i]), 1e-8)
        up, dn = coef.copy(), coef.copy()
        up[i] += h
        dn[i] -= h
        g[i] = (fn(up) - fn(dn)) / (2 * h)
    return float(np.sqrt(g @ cov @ g))
