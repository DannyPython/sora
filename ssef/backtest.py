"""Value-at-Risk backtesting of RW, M0 and M1 on held-out data (Proposal 4.7 and Phase 4; Blueprint Block 9).

What is forecast
  At every origin T the three models of the race (race.py: RW, M0, M1, rolling window of `window` training
  pairs, direct h-step regression, regressors from restricted_ecm.origin_design) give a point forecast f_T of
  r_{T+h}. Around it a predictive distribution is built from the SAME window's residuals, and its quantiles are
  the VaR of the SORA LEVEL h days ahead:
     tail 'up'   : VaR_L = upper L-quantile of r_{T+h}  (rate-rise risk: the S-REIT borrower's loss side)
     tail 'down' : VaR_L = lower L-quantile of r_{T+h}  (rate-fall risk: the lender's / investor's side)
  for L in {95%, 99%}. A breach ("hit") is r_{T+h} above VaR (up) or below it (down).

Three ways to turn the window's residuals into a quantile (`method`):
  normal : f + s * z_L, s the residual sd. For M0 this is the Vasicek conditional distribution of r_{T+h} (Gaussian,
           variance sigma^2 (1 - e^{-2 kappa h dt}) / 2 kappa), with the parameters re-estimated by direct projection.
  hs     : f + empirical quantile of the window's residuals (filtered historical simulation, iid in the window).
  evt    : f + s * GPD quantile of the standardised residuals (Peaks-Over-Threshold, Proposal 4.6, McNeil-Frey 2000
           without the GARCH filter). In a rolling backtest the threshold cannot be read off a mean-excess plot each
           day, so it is the fixed upper `tail_frac` of the window (default 10%, i.e. 50 exceedances out of 500).
  Standardisation by each model's own s keeps the tail comparison fair (Proposal 4.6): M1's extra regressors shrink
  the raw residuals mechanically.

No look-ahead: the window only holds pairs whose target is already observed at T (t + h <= T), exactly as in the
race; the level-shift dummy is not used (a break date picked from the full sample would leak the future).
For h > 1, consecutive h-step forecasts overlap and their hits are autocorrelated by construction, which breaks
the Kupiec/Christoffersen reference distributions; origins are therefore taken every h days (non-overlapping).

Tests
  kupiec          unconditional coverage, LR_pof ~ chi2(1) (Kupiec 1995), plus the exact binomial p-value
  christoffersen  independence LR_ind ~ chi2(1) and conditional coverage LR_cc = LR_pof + LR_ind ~ chi2(2)
                  (Christoffersen 1998), on the first-order Markov transition counts of the hit sequence
  tick_loss_test  head-to-head: quantile (tick) loss of two models, Diebold-Mariano / Giacomini-White t-statistic
                  with a Newey-West variance. Giacomini-White (2006) is valid for NESTED models when the parameters
                  are estimated on a rolling window of fixed size, which is the scheme used here.
  kupiec_power, christoffersen_power
                  Proposal 4.7: with ~770 daily observations a 99% VaR expects ~8 breaches, so a pass is weak
                  evidence. The power tables say how likely each test is to catch a model that is wrong by a given
                  amount, so that a "pass" is not over-read.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy

from .race import _nw_t
from .restricted_ecm import nested_columns, origin_design

LEVELS = (0.95, 0.99)
TAILS = ("up", "down")
METHODS = ("normal", "hs", "evt")
MODELS = ("rw", "m0", "m1")
PAIRS = (("M0 vs RW", "rw", "m0"), ("M1 vs RW", "rw", "m1"), ("M1 vs M0", "m0", "m1"))


# ------------------------------------------------------------------ rolling predictive distributions
@dataclass(frozen=True)
class Predictive:
    """Rolling-window predictive information for one model and one horizon (units of the input R)."""
    T: np.ndarray        # origins, positions in R
    f: np.ndarray        # point forecast of R[T + h]
    s: np.ndarray        # residual sd of the window, RSS / (window - k)
    e: np.ndarray        # (n_origins, window) window residuals, rescaled so that their RMS equals s
    k: int               # number of estimated coefficients

    @property
    def u(self) -> np.ndarray:
        """Standardised window residuals (unit RMS in every row)."""
        return self.e / self.s[:, None]


def origins(n: int, h: int, window: int, step: int = 1) -> np.ndarray:
    """The origins of race._rolling (first one has a full window of observed targets), every `step` days."""
    return np.arange(window + h - 1, n - h, step)


def rolling_predictive(R, X, h: int, window: int, step: int = 1) -> Predictive:
    """OLS of R[t + h] on X[t] over the `window` pairs with t + h <= T, for every origin T. X=None is the random
    walk (forecast R[T], residuals R[t + h] - R[t], nothing estimated). Same windows as race._rolling."""
    R = np.asarray(R, float)
    n = len(R)
    T = origins(n, h, window, step)
    y = R[h:]                                       # y[t] = R[t + h]
    k = 0 if X is None else X.shape[1]
    f, s = np.empty(len(T)), np.empty(len(T))
    E = np.empty((len(T), window))
    for j, t0 in enumerate(T):
        rows = np.arange(t0 - h - window + 1, t0 - h + 1)
        if X is None:
            e, f[j] = y[rows] - R[rows], R[t0]
        else:
            b = np.linalg.lstsq(X[rows], y[rows], rcond=None)[0]
            e, f[j] = y[rows] - X[rows] @ b, X[t0] @ b
        s[j] = np.sqrt(e @ e / (window - k))
        E[j] = e * np.sqrt(window / (window - k))
    return Predictive(T, f, s, E, k)


def predictive_sets(r, x, z, h: int, window: int = 500, with_z: bool = True, p: int = 0, q: int = 0,
                    step: int = 1) -> dict:
    """RW, M0 and M1 predictive information at horizon h (x=None: RW and M0 only). Columns exactly as in
    race.direct_forecasts (M0 = M1 minus nested_columns). Computation in percent; 'scale' converts back."""
    pct = lambda v: None if v is None else np.asarray(v, float) * 100.0
    R = pct(r)
    X1, names, s0 = origin_design(R, pct(x), pct(z) if with_z else None, p, q, with_z)
    m0_cols = [i for i, nm in enumerate(names) if nm not in nested_columns(names)]
    Rs = R[s0:]
    out = {"rw": rolling_predictive(Rs, None, h, window, step),
           "m0": rolling_predictive(Rs, X1[:, m0_cols], h, window, step)}
    if x is not None:
        out["m1"] = rolling_predictive(Rs, X1, h, window, step)
    T = out["m0"].T
    return {"models": out, "T": T + s0, "actual": Rs[T + h], "scale": 0.01}


# ------------------------------------------------------------------ GPD tail (EVT)
def gpd_fit_pwm(y) -> tuple[float, float]:
    """(xi, beta) of a GPD for exceedances y > 0 by probability-weighted moments (Hosking & Wallis 1987).
    Closed form, fast, and better behaved than ML with ~50 exceedances; valid for xi < 1/2."""
    y = np.sort(np.asarray(y, float))
    n = len(y)
    a0 = y.mean()
    a1 = float(np.sum(y * (n - np.arange(1, n + 1)) / (n - 1)) / n)     # E[Y (1 - F(Y))]
    k = a0 / (a0 - 2 * a1) - 2.0                                        # Hosking's k = -xi
    return -k, 2.0 * a0 * a1 / (a0 - 2 * a1)


def gpd_fit_mle(y) -> tuple[float, float]:
    """(xi, beta) by maximum likelihood (scipy), location fixed at 0."""
    xi, _, beta = stats.genpareto.fit(np.asarray(y, float), floc=0)
    return float(xi), float(beta)


def gpd_quantile(u0: float, xi: float, beta: float, zeta: float, prob: float) -> float:
    """Upper `prob`-tail quantile from a POT fit: zeta = share of exceedances over u0 (McNeil-Frey eq. 6)."""
    ratio = prob / zeta
    if abs(xi) < 1e-6:
        return u0 - beta * np.log(ratio)
    return u0 + beta / xi * (ratio ** (-xi) - 1.0)


def evt_quantiles(U: np.ndarray, probs, tail_frac: float = 0.10, fit: str = "pwm") -> dict:
    """Upper-tail quantiles of each row of U (standardised residuals), one GPD fit per row.
    probs: tail probabilities (e.g. 0.05, 0.01), each must be below tail_frac. Returns {prob: array},
    plus 'xi' (the fitted shapes) for diagnostics."""
    fitter = {"pwm": gpd_fit_pwm, "mle": gpd_fit_mle}[fit]
    probs = list(probs)
    if max(probs) >= tail_frac:
        raise ValueError("every tail probability must be below tail_frac")
    out = {pr: np.empty(len(U)) for pr in probs}
    xis = np.empty(len(U))
    for j, row in enumerate(U):
        u0 = np.quantile(row, 1.0 - tail_frac)
        exc = row[row > u0] - u0
        xi, beta = fitter(exc)
        xis[j] = xi
        for pr in probs:
            out[pr][j] = gpd_quantile(u0, xi, beta, len(exc) / len(row), pr)
    out["xi"] = xis
    return out


# ------------------------------------------------------------------ VaR forecasts
def var_table(pred: Predictive, levels=LEVELS, methods=METHODS, tail_frac: float = 0.10,
              gpd: str = "pwm") -> dict:
    """{(method, tail, level): VaR array} for one model (units of the Predictive)."""
    out = {}
    probs = [1.0 - L for L in levels]
    for method in methods:
        for tail in TAILS:
            sgn = 1.0 if tail == "up" else -1.0
            if method == "evt":
                ev = evt_quantiles(sgn * pred.u, probs, tail_frac, gpd)
            for L in levels:
                if method == "normal":
                    q = stats.norm.ppf(L) * pred.s
                elif method == "hs":
                    q = np.quantile(sgn * pred.e, L, axis=1)
                elif method == "evt":
                    q = ev[1.0 - L] * pred.s
                else:
                    raise ValueError(f"unknown method {method}")
                out[(method, tail, L)] = pred.f + sgn * q
    return out


def var_forecasts(df: pd.DataFrame, h: int = 1, window: int = 500, with_z: bool = True, p: int = 0, q: int = 0,
                  levels=LEVELS, methods=METHODS, tail_frac: float = 0.10, gpd: str = "pwm",
                  step: int | None = None) -> pd.DataFrame:
    """Long table, one row per (target date, model, method, tail, level): origin, actual, forecast, var, hit.
    Rates in decimals. step defaults to h (non-overlapping origins)."""
    with_z = with_z and "z" in df
    r, x = df["r"].to_numpy(), df["x"].to_numpy()
    z = df["z"].to_numpy() if with_z else None
    ps = predictive_sets(r, x, z, h, window, with_z, p, q, h if step is None else step)
    sc, T, act = ps["scale"], ps["T"], ps["actual"] * ps["scale"]
    idx = df.index
    frames = []
    for model, pred in ps["models"].items():
        for (method, tail, L), v in var_table(pred, levels, methods, tail_frac, gpd).items():
            v = v * sc
            hit = act > v if tail == "up" else act < v
            frames.append(pd.DataFrame({"date": idx[T + h], "origin": idx[T], "h": h, "model": model,
                                        "method": method, "tail": tail, "level": L, "actual": act,
                                        "forecast": pred.f * sc, "var": v, "hit": hit.astype(int)}))
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------ coverage tests
def kupiec(hits, p: float) -> dict:
    """Unconditional coverage (Kupiec 1995 POF). p = expected breach probability (0.01 for a 99% VaR).
    Also the exact two-sided binomial p-value, more reliable with a handful of breaches."""
    hits = np.asarray(hits, int)
    N, x = len(hits), int(hits.sum())
    ph = x / N
    lr = max(float(-2.0 * (xlogy(N - x, 1 - p) + xlogy(x, p) - xlogy(N - x, 1 - ph) - xlogy(x, ph))), 0.0)
    return {"N": N, "breaches": x, "expected": N * p, "rate": ph, "LR_pof": lr,
            "p_pof": float(stats.chi2.sf(lr, 1)), "p_binom": float(stats.binomtest(x, N, p).pvalue)}


def transition_counts(hits):
    """(n00, n01, n10, n11): n_ij = number of days with hit i followed by a day with hit j.
    hits may be (N,) or (n_sims, N); counts are then arrays."""
    h = np.asarray(hits, int)
    a, b = h[..., :-1], h[..., 1:]
    return tuple(np.sum((a == i) & (b == j), axis=-1) for i, j in ((0, 0), (0, 1), (1, 0), (1, 1)))


def lr_ind(n00, n01, n10, n11):
    """Christoffersen independence LR from transition counts (scalars or arrays). NaN when untestable
    (no breach, or breaches on every day)."""
    n00, n01, n10, n11 = (np.asarray(v, float) for v in (n00, n01, n10, n11))
    with np.errstate(invalid="ignore", divide="ignore"):
        pi0 = np.where(n00 + n01 > 0, n01 / np.maximum(n00 + n01, 1), 0.0)
        pi1 = np.where(n10 + n11 > 0, n11 / np.maximum(n10 + n11, 1), 0.0)
        pi = (n01 + n11) / (n00 + n01 + n10 + n11)
        lr = -2.0 * (xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
                     - xlogy(n00, 1 - pi0) - xlogy(n01, pi0) - xlogy(n10, 1 - pi1) - xlogy(n11, pi1))
    lr = np.maximum(lr, 0.0)
    return np.where((n01 + n11 == 0) | (n00 + n10 == 0), np.nan, lr)


def christoffersen(hits, p: float) -> dict:
    """Independence (H0: a breach today does not change tomorrow's breach probability) and conditional coverage
    (independence AND the right rate). NaN when there is no breach (nothing to test)."""
    n00, n01, n10, n11 = (int(v) for v in transition_counts(hits))
    out = {"n00": n00, "n01": n01, "n10": n10, "n11": n11,
           "pi01": n01 / (n00 + n01) if n00 + n01 else 0.0, "pi11": n11 / (n10 + n11) if n10 + n11 else 0.0}
    li = float(lr_ind(n00, n01, n10, n11))
    if not np.isfinite(li):
        return out | {"LR_ind": np.nan, "p_ind": np.nan, "LR_cc": np.nan, "p_cc": np.nan}
    # POF on the same N-1 transitions so that LR_cc = LR_pof + LR_ind exactly (Christoffersen 1998, eq. 3.3)
    lr_cc = kupiec(np.asarray(hits, int)[1:], p)["LR_pof"] + li
    return out | {"LR_ind": li, "p_ind": float(stats.chi2.sf(li, 1)), "LR_cc": lr_cc, "p_cc": float(stats.chi2.sf(lr_cc, 2))}


def coverage_table(fc: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Kupiec + Christoffersen for every (h, model, method, tail, level) in a var_forecasts table."""
    rows = []
    for key, g in fc.sort_values("date").groupby(["h", "model", "method", "tail", "level"], sort=False):
        p = 1.0 - key[4]
        k, c = kupiec(g["hit"], p), christoffersen(g["hit"], p)
        if k["p_pof"] < alpha:
            verdict = "fail: breach rate"
        elif np.isfinite(c["p_ind"]) and c["p_ind"] < alpha:
            verdict = "fail: clustering"
        else:
            verdict = "not rejected"
        rows.append(dict(zip(("h", "model", "method", "tail", "level"), key)) | k
                    | {k_: c[k_] for k_ in ("n11", "pi01", "pi11", "LR_ind", "p_ind", "LR_cc", "p_cc")}
                    | {"verdict": verdict})
    out = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(MODELS)}
    return out.sort_values(["h", "tail", "level", "method", "model"], key=lambda s: s.map(order) if s.name == "model"
                           else s).reset_index(drop=True)


# ------------------------------------------------------------------ head-to-head: quantile (tick) loss
def tick_loss(actual, var, tau: float) -> np.ndarray:
    """Quantile loss of a tau-quantile forecast: (tau - 1{y < q}) (y - q). Strictly consistent for quantiles,
    so the model with the lower expected tick loss gives the better VaR (rewards both coverage and sharpness)."""
    d = np.asarray(actual, float) - np.asarray(var, float)
    return (tau - (d < 0)) * d


def nw_lags(n: int) -> int:
    return int(np.floor(4 * (n / 100.0) ** (2 / 9)))


def tick_loss_test(fc: pd.DataFrame) -> pd.DataFrame:
    """Mean tick loss of each model and the Giacomini-White (unconditional) t-statistic for each pair
    (small, large): d = loss_small - loss_large, H1: E[d] > 0 (the large model's VaR is better), one-sided p."""
    rows = []
    for (h, method, tail, L), g in fc.groupby(["h", "method", "tail", "level"], sort=False):
        tau = L if tail == "up" else 1.0 - L
        w = g.pivot(index="date", columns="model", values="var")
        a = g.drop_duplicates("date").set_index("date")["actual"].reindex(w.index)
        loss = {m: tick_loss(a, w[m], tau) for m in w.columns}
        row = {"h": h, "method": method, "tail": tail, "level": L, "N": len(w)}
        row |= {f"loss_{m}_bp": float(np.mean(loss[m]) * 1e4) for m in MODELS if m in loss}
        for name, s_, l_ in PAIRS:
            if s_ in loss and l_ in loss:
                d = np.asarray(loss[s_] - loss[l_])
                t = _nw_t(d, nw_lags(len(d))) if np.any(d != 0) else 0.0
                row[f"gw_t {name}"] = t
                row[f"p {name}"] = float(stats.norm.sf(t))
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ power (Proposal 4.7)
def kupiec_power(N: int, p: float, multiples=(1.0, 1.5, 2.0, 3.0), alpha: float = 0.05) -> pd.DataFrame:
    """Exact probability that the Kupiec test rejects at `alpha` when the true breach rate is m * p.
    m = 1 is the actual size of the test. Also the breach-count acceptance region."""
    xs = np.arange(N + 1)
    lr = np.array([kupiec(np.r_[np.ones(x, int), np.zeros(N - x, int)], p)["LR_pof"] for x in xs])
    rej = lr > stats.chi2.ppf(1 - alpha, 1)
    acc = xs[~rej]
    rows = [{"N": N, "level": 1 - p, "true_rate_multiple": m, "true_rate": m * p,
             "accept_breaches_min": int(acc.min()), "accept_breaches_max": int(acc.max()),
             "power": float(stats.binom.pmf(xs, N, min(m * p, 1.0))[rej].sum())} for m in multiples]
    return pd.DataFrame(rows)


def markov_hits(N: int, p: float, pi11: float, n_sims: int, rng) -> np.ndarray:
    """(n_sims, N) clustered breach sequences with unconditional rate p: P(hit | hit yesterday) = pi11,
    P(hit | no hit yesterday) = pi01 = p (1 - pi11) / (1 - p). pi11 = p gives iid hits."""
    pi01 = p * (1 - pi11) / (1 - p)
    out = np.zeros((n_sims, N), int)
    out[:, 0] = rng.random(n_sims) < p
    U = rng.random((n_sims, N))
    for t in range(1, N):
        out[:, t] = U[:, t] < np.where(out[:, t - 1] == 1, pi11, pi01)
    return out


def christoffersen_power(N: int, p: float, pi11s=(None, 0.1, 0.2, 0.4), n_sims: int = 2000, alpha: float = 0.05,
                         seed: int = 2027) -> pd.DataFrame:
    """Simulated rejection rate of the independence test when breaches cluster (P(hit | hit) = pi11, right
    unconditional rate). pi11=None means iid hits (the test's actual size). Sequences without a breach count as
    'not rejected', which is what happens in practice."""
    rng = np.random.default_rng(seed)
    rows = []
    for pi11 in pi11s:
        H = markov_hits(N, p, p if pi11 is None else pi11, n_sims, rng)
        pv = stats.chi2.sf(lr_ind(*transition_counts(H)), 1)
        rows.append({"N": N, "level": 1 - p, "pi11": p if pi11 is None else pi11,
                     "share_no_breach": float(np.mean(H.sum(axis=1) == 0)),
                     "power_ind": float(np.mean(np.nan_to_num(pv, nan=1.0) < alpha))})
    return pd.DataFrame(rows)
