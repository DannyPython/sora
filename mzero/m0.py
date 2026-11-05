"""M0: classical Vasicek model, the null (Blueprint v8c, Sec. 4.2, 5.3, 6.1).

    dr = kappa (theta - r) dt + sigma dW,   theta constant.

Exact discretisation over one step dt, with a = exp(-kappa dt):

    r_{t+1} = c0 + c1 r_t + eps,  c0 = (1 - a) theta,  c1 = a,
    Var(eps) = sigma^2 (1 - a^2) / (2 kappa).

M0 is calibrated by OLS on this AR(1). Structural parameters (Sec. 5.3):

    kappa = -ln(c1)/dt,  theta = c0/(1 - c1),  sigma^2 = 2 kappa Var(eps)/(1 - c1^2).

Required check: c1 in (0, 1); otherwise the series does not revert and the
structural parameters are undefined.

Time convention: one observation is one step of dt = 1/252 year. Weekends and
holidays are NOT stretched, so a Friday -> Monday step counts as one step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

DT_DEFAULT = 1.0 / 252.0


# ---------------------------------------------------------------- calibration
def _ols_ar1(values: np.ndarray):
    """OLS of r[1:] on (1, r[:-1]). Returns c0, c1, s2 (RSS/(n-2)), cov(c0,c1), resid."""
    y, x = values[1:], values[:-1]
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n = len(y)
    s2 = float(resid @ resid / (n - 2))
    cov = s2 * np.linalg.inv(X.T @ X)
    return float(beta[0]), float(beta[1]), s2, cov, resid


def structural_params(c0: float, c1: float, s2: float, dt: float = DT_DEFAULT):
    """(kappa, theta, sigma) from AR(1) coefficients; NaNs when c1 is not in (0, 1)."""
    if not 0.0 < c1 < 1.0:
        return np.nan, np.nan, np.nan
    kappa = -np.log(c1) / dt
    theta = c0 / (1.0 - c1)
    sigma = np.sqrt(2.0 * kappa * s2 / (1.0 - c1 ** 2))
    return kappa, theta, sigma


def structural_jacobian(c0: float, c1: float, s2: float, dt: float = DT_DEFAULT) -> np.ndarray:
    """d(kappa, theta, sigma)/d(c0, c1, s2); rows kappa, theta, sigma (delta method)."""
    kappa, _, sigma = structural_params(c0, c1, s2, dt)
    dkappa_dc1 = -1.0 / (dt * c1)
    one_m = 1.0 - c1
    one_m2 = 1.0 - c1 ** 2
    # sigma^2 = f = 2 kappa s2 / (1 - c1^2); sigma = sqrt(f)
    df_dc1 = 2.0 * s2 * (dkappa_dc1 * one_m2 + kappa * 2.0 * c1) / one_m2 ** 2
    df_ds2 = 2.0 * kappa / one_m2
    return np.array([
        [0.0, dkappa_dc1, 0.0],
        [1.0 / one_m, c0 / one_m ** 2, 0.0],
        [0.0, df_dc1 / (2.0 * sigma), df_ds2 / (2.0 * sigma)],
    ])


@dataclass(frozen=True)
class M0Fit:
    dt: float
    c0: float
    c1: float
    s2: float                 # unbiased residual variance, RSS/(n-2)
    cov_c: np.ndarray         # 2x2 covariance of (c0, c1)
    resid: pd.Series          # eps_hat, indexed by the date of r_{t+1}

    @property
    def n(self) -> int:
        return len(self.resid)

    @property
    def valid(self) -> bool:
        """Blueprint 5.3 check: c1 in (0, 1)."""
        return 0.0 < self.c1 < 1.0

    @property
    def kappa(self) -> float:
        return structural_params(self.c0, self.c1, self.s2, self.dt)[0]

    @property
    def theta(self) -> float:
        return structural_params(self.c0, self.c1, self.s2, self.dt)[1]

    @property
    def sigma(self) -> float:
        return structural_params(self.c0, self.c1, self.s2, self.dt)[2]

    @property
    def half_life_steps(self) -> float:
        """Half-life of a shock, in observation steps (ln 2 / kappa, in units of dt)."""
        return np.log(2.0) / self.kappa / self.dt if self.valid else np.nan

    @property
    def std_resid(self) -> pd.Series:
        """Residuals standardised by their own sd (input to Block 5 EVT)."""
        return self.resid / np.sqrt(self.s2)

    def params_table(self, level: float = 0.95) -> pd.DataFrame:
        """Estimates with delta-method standard errors and normal CIs (Sec. 5.5)."""
        z = stats.norm.ppf(0.5 + level / 2.0)
        s2_var = 2.0 * self.s2 ** 2 / (self.n - 2)       # Var(s2) under Gaussian errors
        V = np.zeros((3, 3))
        V[:2, :2] = self.cov_c                            # s2 is independent of (c0, c1)
        V[2, 2] = s2_var

        est = {"c0": self.c0, "c1": self.c1, "s2": self.s2}
        se = {"c0": np.sqrt(V[0, 0]), "c1": np.sqrt(V[1, 1]), "s2": np.sqrt(V[2, 2])}
        if self.valid:
            J = structural_jacobian(self.c0, self.c1, self.s2, self.dt)
            sd = np.sqrt(np.diag(J @ V @ J.T))
            k, th, sg = structural_params(self.c0, self.c1, self.s2, self.dt)
            est |= {"kappa": k, "theta": th, "sigma": sg}
            se |= {"kappa": sd[0], "theta": sd[1], "sigma": sd[2]}
        else:
            est |= {"kappa": np.nan, "theta": np.nan, "sigma": np.nan}
            se |= {"kappa": np.nan, "theta": np.nan, "sigma": np.nan}

        out = pd.DataFrame({"estimate": est, "se": se})
        out["ci_low"] = out["estimate"] - z * out["se"]
        out["ci_high"] = out["estimate"] + z * out["se"]
        return out

    def predict(self, r_t):
        """One-step conditional mean E[r_{t+1} | r_t] = c0 + c1 r_t."""
        return self.c0 + self.c1 * np.asarray(r_t)

    def simulate(self, r0: float, n_steps: int, n_paths: int, seed: int) -> np.ndarray:
        if not self.valid:
            raise ValueError("c1 not in (0, 1): structural parameters undefined")
        return simulate_vasicek(self.kappa, self.theta, self.sigma, r0, n_steps, n_paths, self.dt, seed)


def calibrate(r: pd.Series, dt: float = DT_DEFAULT) -> M0Fit:
    """Calibrate M0 by OLS on the AR(1) of Sec. 6.1."""
    r = pd.Series(r).dropna().astype(float)
    if len(r) < 10:
        raise ValueError("need at least 10 observations")
    c0, c1, s2, cov, resid = _ols_ar1(r.to_numpy())
    return M0Fit(dt=dt, c0=c0, c1=c1, s2=s2, cov_c=cov,
                 resid=pd.Series(resid, index=r.index[1:], name="eps"))


# ------------------------------------------------------------------ simulation
def simulate_vasicek(kappa, theta, sigma, r0, n_steps, n_paths, dt=DT_DEFAULT, seed=0) -> np.ndarray:
    """Exact Vasicek transition (no Euler error). Returns (n_paths, n_steps + 1), column 0 = r0."""
    rng = np.random.default_rng(seed)
    a = np.exp(-kappa * dt)
    sd = sigma * np.sqrt((1.0 - a ** 2) / (2.0 * kappa))
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = r0
    z = rng.standard_normal((n_paths, n_steps))
    for t in range(n_steps):
        paths[:, t + 1] = theta + (paths[:, t] - theta) * a + sd * z[:, t]
    return paths


def vasicek_moments(kappa, theta, sigma, r0, horizon_years):
    """Analytic E[r_T], Var[r_T]; used to validate the simulator."""
    e = np.exp(-kappa * horizon_years)
    return theta + (r0 - theta) * e, sigma ** 2 * (1.0 - e ** 2) / (2.0 * kappa)


# --------------------------------------------------------- out-of-sample race
def rolling_forecast(r: pd.Series, window: int = 250, refit_every: int = 1) -> pd.DataFrame:
    """One-step-ahead rolling-window forecasts of r (Blueprint 6.6, M0 side).

    At origin t the model is fitted on the last `window` observations ending at t
    (no later data), then forecasts r_{t+1}. Columns: origin, forecast, actual,
    error, rw_forecast (random-walk benchmark = r_t), sd (predictive sd).
    """
    if window < 20:
        raise ValueError("window must be >= 20")
    v = pd.Series(r).dropna().astype(float)
    vals = v.to_numpy()
    rows = []
    c0 = c1 = s2 = None
    for i in range(window - 1, len(vals) - 1):
        if c0 is None or (i - (window - 1)) % refit_every == 0:
            c0, c1, s2, _, _ = _ols_ar1(vals[i - window + 1: i + 1])
        fc = c0 + c1 * vals[i]
        rows.append((v.index[i + 1], v.index[i], fc, vals[i + 1], vals[i + 1] - fc, vals[i], np.sqrt(s2)))
    out = pd.DataFrame(rows, columns=["date", "origin", "forecast", "actual", "error", "rw_forecast", "sd"])
    return out.set_index("date")


def rmse(forecast: pd.DataFrame, col: str = "forecast") -> float:
    return float(np.sqrt(np.mean((forecast["actual"] - forecast[col]) ** 2)))