"""M1 noise model (Proposal Sec. 4.3.3-4.3.4, Blueprint v8c Sec. 5.2-5.3, 6.4, 6.7).

M1:   dr = kappa (theta_t - r) dt + sigma dW,    theta_t = alpha + beta1 x_t + beta2 z_t
Discretised (exact in theta_t over one step):
      r_{t+1} = c0 + c1 r_t + c2 x_t + c3 z_t + eps,   eps ~ N(0, s_eps^2)
      s_eps^2 = sigma^2 (1 - c1^2) / (2 kappa)

What this file does with the noise term eps (= sigma dW over one step):
  1. estimate it      : OLS residuals, s_eps, and sigma via sigma^2 = 2 kappa s_eps^2 / (1 - c1^2)
  2. check it         : is it really Gaussian / white / homoskedastic? (the Proposal ASSUMES this)
  3. standardise it   : eps_hat / s_eps  (input to Block 5 EVT; M1 vs M0 comparison)
  4. generate it      : Gaussian or bootstrapped draws, used to simulate M1 paths (Block 8)

Note: the same residuals come out of the levels regression (Proposal 4.3.4) and of the unrestricted
error-correction form  d r_{t+1} = c0 + (c1-1) r_t + c2 x_t + c3 z_t  (subtract r_t from both sides),
so s_eps and sigma do not depend on which of the two you estimate.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera

DT_DEFAULT = 1.0 / 252.0
_HERE = Path(__file__).resolve().parent
# works whether this file sits in <repo>/ or in a sub-folder like <repo>/mzero/ : use whichever has a data/ folder
DATA_DIR = next((d / "data" for d in (_HERE, _HERE.parent) if (d / "data").is_dir()), _HERE / "data")


# ============================================================ data: SOFR timing + FX proxy
def load_sofr(path=DATA_DIR / "SOFR_clean.csv", as_decimal=True) -> pd.DataFrame:
    """SOFR with publication dates. Columns: sofr, pub_date (NaT for the latest row)."""
    df = pd.read_csv(path)
    out = pd.DataFrame({
        "sofr": df["SOFR"].astype(float).to_numpy() / (100.0 if as_decimal else 1.0),
        "pub_date": pd.to_datetime(df["SOFR Publication Date"], format="%d/%m/%Y").to_numpy(),   # array: avoid index alignment
    }, index=pd.DatetimeIndex(pd.to_datetime(df["SOFR Effective Date"], format="%d/%m/%Y"), name="effective"))
    return out.sort_index()


def known_sofr(sofr: pd.DataFrame, dates: pd.DatetimeIndex, strict: bool = True) -> pd.Series:
    """x known BEFORE each SORA value date: latest SOFR whose publication date is < date
    (strict=True, conservative: SOFR is published ~8am New York = ~8pm Singapore, after the SORA day)
    or <= date (strict=False, the repo's update_sofr.py convention). Past values only -> no look-ahead.
    """
    s = sofr.dropna(subset=["pub_date"]).sort_values("pub_date")
    pub = s["pub_date"] + (pd.Timedelta(days=1) if strict else pd.Timedelta(0))   # first date it may be used
    avail = pd.Series(s["sofr"].to_numpy(), index=pd.DatetimeIndex(pub, name="usable_from"))
    avail = avail[~avail.index.duplicated(keep="last")].sort_index()
    return avail.reindex(avail.index.union(dates)).ffill().reindex(dates).rename("x")


def load_usdsgd(path=DATA_DIR / "USDSGD_clean.csv") -> pd.Series:
    """USD/SGD end-of-day rate (SGD per USD) from data/USDSGD_clean.csv: columns Date, USD/SGD (d/m/Y)."""
    df = pd.read_csv(path)
    d, v = df.columns[0], df.columns[1]                      # first col = date, second = rate
    idx = pd.to_datetime(df[d], format="%d/%m/%Y")
    return pd.Series(df[v].astype(float).to_numpy(), index=pd.DatetimeIndex(idx, name="date"), name="usdsgd").sort_index()


def make_z(usdsgd: pd.Series) -> pd.Series:
    """z_t = daily spot return of USD/SGD (SGD per USD; z > 0 = SGD weakened). Verify the quote
    convention of your MAS series (Blueprint 5.1)."""
    return usdsgd.sort_index().pct_change().rename("z")


def build_dataset(r: pd.Series, sofr: pd.DataFrame, z: pd.Series | None = None, strict: bool = True) -> pd.DataFrame:
    """Aligned frame indexed by SORA value date. Row i holds y = r_i and the information set known
    before r_i: r_prev (= r_{i-1}), x (SOFR known before date i), z (return on the previous SORA date).
    Blueprint 7: intersect calendars first, then forward-fill only past values."""
    df = pd.DataFrame({"r": r})
    df["r_prev"] = df["r"].shift(1)
    df["x"] = known_sofr(sofr, df.index, strict)
    if z is not None:
        df["z"] = z.reindex(z.index.union(df.index)).ffill().reindex(df.index).shift(1)
    return df.dropna()


# ============================================================ estimation
@dataclass(frozen=True)
class M1Fit:
    dt: float
    names: list                 # regressor names, e.g. ['c0','c1','c2','c3']
    coef: np.ndarray
    cov: np.ndarray             # covariance of coef
    s2: float                   # RSS / (n - k)
    resid: pd.Series            # eps_hat
    k: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "k", len(self.coef))

    def c(self, name):
        return float(self.coef[self.names.index(name)])

    @property
    def n(self):
        return len(self.resid)

    @property
    def valid(self):                       # Proposal 4.3.4 required check: c1 in (0, 1)
        return 0.0 < self.c("c1") < 1.0

    @property
    def s_eps(self):
        return float(np.sqrt(self.s2))

    @property
    def kappa(self):
        return -np.log(self.c("c1")) / self.dt if self.valid else np.nan

    @property
    def sigma(self):                       # sigma^2 = 2 kappa Var(eps) / (1 - c1^2)
        return float(np.sqrt(2 * self.kappa * self.s2 / (1 - self.c("c1") ** 2))) if self.valid else np.nan

    @property
    def structural(self) -> dict:
        if not self.valid:
            return {}
        c1 = self.c("c1")
        out = {"kappa": self.kappa, "alpha": self.c("c0") / (1 - c1), "beta1": self.c("c2") / (1 - c1)}
        if "c3" in self.names:
            out["beta2"] = self.c("c3") / (1 - c1)
        out["sigma"] = self.sigma
        return out

    @property
    def std_resid(self) -> pd.Series:      # standardised residuals (Blueprint 6.4)
        return (self.resid / self.s_eps).rename("std_resid")

    def sigma_ci(self, level=0.95):
        """Delta-method CI for sigma (depends on c1 and s2 only; s2 independent of the coefficients)."""
        if not self.valid:
            return (np.nan, np.nan)
        c1, s2, dt = self.c("c1"), self.s2, self.dt
        kap = self.kappa
        dk = -1.0 / (dt * c1)
        one = 1 - c1 ** 2
        df_dc1 = 2 * s2 * (dk * one + kap * 2 * c1) / one ** 2      # d(sigma^2)/d c1
        df_ds2 = 2 * kap / one
        g = np.array([df_dc1, df_ds2]) / (2 * self.sigma)
        V = np.diag([self.cov[self.names.index("c1"), self.names.index("c1")], 2 * s2 ** 2 / (self.n - self.k)])
        se = float(np.sqrt(g @ V @ g))
        z = stats.norm.ppf(0.5 + level / 2)
        return (self.sigma - z * se, self.sigma + z * se)


def fit_m1(df: pd.DataFrame, dt: float = DT_DEFAULT) -> M1Fit:
    """OLS of r on [1, r_prev, x, (z)]. df comes from build_dataset. Without 'z' this is the
    Blueprint Sec. 10 fallback (M0 + SOFR only)."""
    cols = ["r_prev", "x"] + (["z"] if "z" in df else [])
    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy() for c in cols])
    y = df["r"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    n, k = X.shape
    s2 = float(e @ e / (n - k))
    cov = s2 * np.linalg.inv(X.T @ X)
    return M1Fit(dt=dt, names=[f"c{i}" for i in range(k)], coef=beta, cov=cov, s2=s2,
                 resid=pd.Series(e, index=df.index, name="eps"))


# ============================================================ checking the noise assumptions
def noise_diagnostics(resid: pd.Series, lags_lb: int = 10, lags_arch: int = 5, alpha: float = 0.05) -> pd.DataFrame:
    """The Proposal assumes eps is approximately Gaussian with constant variance. Test that.
    Gaussian: Jarque-Bera. White: Ljung-Box on eps. Constant variance: ARCH-LM (Engle) on eps."""
    e = pd.Series(resid).dropna().to_numpy()
    z = (e - e.mean()) / e.std(ddof=1)
    jb, jb_p, skew, kurt = jarque_bera(e)
    lb_p = float(acorr_ljungbox(e, lags=[lags_lb])["lb_pvalue"].iloc[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")            # statsmodels return-type FutureWarning
        arch_p = float(het_arch(e, nlags=lags_arch)[1])
    rows = [
        ("excess kurtosis (Gaussian = 0)", kurt - 3.0, np.nan, abs(kurt - 3.0) > 1.0),
        ("skewness (Gaussian = 0)", skew, np.nan, abs(skew) > 0.5),
        ("Jarque-Bera normality", jb, jb_p, jb_p < alpha),
        (f"Ljung-Box serial corr., {lags_lb} lags", np.nan, lb_p, lb_p < alpha),
        (f"ARCH-LM vol. clustering, {lags_arch} lags", np.nan, arch_p, arch_p < alpha),
        ("share beyond 3 sd (Gaussian = 0.27%)", float(np.mean(np.abs(z) > 3)), np.nan, np.mean(np.abs(z) > 3) > 0.0081),
    ]
    out = pd.DataFrame(rows, columns=["check", "value", "p_value", "assumption_violated"]).set_index("check")
    return out


def compare_noise(resid0: pd.Series, resid1: pd.Series, k0: int = 2, k1: int = 3) -> pd.Series:
    """M0 vs M1 residuals on their common dates. M1's extra regressors shrink s_eps mechanically,
    so raw tails must NOT be compared (Blueprint 6.4): use the standardised series for EVT."""
    d = pd.concat([resid0.rename("e0"), resid1.rename("e1")], axis=1, sort=True).dropna()
    n = len(d)
    s0 = np.sqrt((d.e0 ** 2).sum() / (n - k0))
    s1 = np.sqrt((d.e1 ** 2).sum() / (n - k1))
    return pd.Series({"n_common": n, "s_eps_M0": s0, "s_eps_M1": s1,
                      "variance_reduction": 1 - (s1 / s0) ** 2})


# ============================================================ generating noise + simulating M1
def draw_noise(fit: M1Fit, n_steps: int, n_paths: int, seed: int, method: str = "gaussian") -> np.ndarray:
    """(n_paths, n_steps) draws of eps. 'gaussian': N(0, s_eps^2). 'bootstrap': resample standardised
    residuals (keeps fat tails / skew if the Gaussian check fails), rescaled to s_eps."""
    rng = np.random.default_rng(seed)
    if method == "gaussian":
        return fit.s_eps * rng.standard_normal((n_paths, n_steps))
    if method == "bootstrap":
        z = fit.std_resid.to_numpy()
        z = (z - z.mean()) / z.std(ddof=1)
        return fit.s_eps * rng.choice(z, size=(n_paths, n_steps), replace=True)
    raise ValueError("method must be 'gaussian' or 'bootstrap'")


def make_scenarios(x_last: float, n_steps: int, shock: float = 0.01, ramp: int = 63) -> dict:
    """Blueprint 6.7: hold / shock up / shock down paths for SOFR (decimals; 0.01 = 100 bp),
    ramping linearly over `ramp` steps and then held."""
    w = np.minimum(np.arange(1, n_steps + 1) / ramp, 1.0)
    return {"hold": np.full(n_steps, x_last), "up": x_last + shock * w, "down": x_last - shock * w}


def simulate_m1(fit: M1Fit, r0: float, x_path, z_path=None, n_paths: int = 10_000, seed: int = 2027,
                method: str = "gaussian") -> np.ndarray:
    """Simulate r forward: r_{t+1} = c0 + c1 r_t + c2 x_t + c3 z_t + eps, with x, z fixed along a scenario.
    Returns (n_paths, n_steps + 1), column 0 = r0. NOTE: exogenous paths are conditioned on, not modelled
    (Blueprint 6.7)."""
    x_path = np.asarray(x_path, float)
    n_steps = len(x_path)
    z_path = np.zeros(n_steps) if z_path is None else np.asarray(z_path, float)
    eps = draw_noise(fit, n_steps, n_paths, seed, method)
    c0, c1, c2 = fit.c("c0"), fit.c("c1"), fit.c("c2")
    c3 = fit.c("c3") if "c3" in fit.names else 0.0
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = r0
    for t in range(n_steps):
        paths[:, t + 1] = c0 + c1 * paths[:, t] + c2 * x_path[t] + c3 * z_path[t] + eps[:, t]
    return paths


# ============================================================ demo
if __name__ == "__main__":
    import sys
    sys.path[:0] = [str(_HERE), str(_HERE.parent)]
    from mzero import data, m0                   # M0 code from the repo (data loader + AR(1) fit)

    r = data.load_sora()
    fx_path = DATA_DIR / "USDSGD_clean.csv"
    z = make_z(load_usdsgd(fx_path)) if fx_path.exists() else None      # no FX file -> fallback: M0 + SOFR only
    df = build_dataset(r, load_sofr(), z)
    fit = fit_m1(df)
    print(f"M1 ({'SOFR + FX' if 'c3' in fit.names else 'SOFR only'}), n = {fit.n}, {df.index[0].date()} -> {df.index[-1].date()}")
    print("coefficients:", dict(zip(fit.names, np.round(fit.coef, 6))))
    print("structural  :", {k: round(v, 4) for k, v in fit.structural.items()})
    print(f"s_eps = {fit.s_eps * 1e4:.2f} bp/day; sigma = {fit.sigma * 1e4:.0f} bp/sqrt(yr), "
          f"95% CI {tuple(round(v * 1e4) for v in fit.sigma_ci())}")
    print("\nNoise checks on M1 residuals:\n", noise_diagnostics(fit.resid).round(4).to_string())

    m0fit = m0.calibrate(df["r"])                 # same sample for a fair comparison
    print("\nM0 vs M1 noise:\n", compare_noise(m0fit.resid, fit.resid, 2, fit.k).round(6).to_string())

    sc = make_scenarios(df["x"].iloc[-1], 252, shock=0.01)
    for name, xp in sc.items():
        p = simulate_m1(fit, df["r"].iloc[-1], xp, None, n_paths=5000)
        print(f"scenario {name:>4}: 1y SORA 5/50/95% = " + " / ".join(f"{q * 100:.2f}" for q in np.percentile(p[:, -1], [5, 50, 95])) + " %")