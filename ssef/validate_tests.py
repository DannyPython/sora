"""Validation of the validation: size and power of the project's tests on data with KNOWN truth
(Proposal 4.4, 4.5.2 and 4.7: "report this limitation explicitly rather than treating a pass as conclusive").

Question answered
  The pipeline rejects or fails to reject H0 (M0 is enough) and passes or fails the cointegration gate. Do those
  verdicts mean what they say at THIS sample size (about 1,300 daily points, one rate cycle) and at the speed of
  reversion SORA actually has? A test is checked the only way it can be: simulate data where the answer is known,
  run the test, count how often it says "reject".

Method
  1. DGPs (`make_dgps`): processes with a known answer, all built from the fitted pipeline models
        spurious      r is a random walk (a = 0), unrelated to SOFR               H0 true
        m0            r reverts to a constant (the classical Vasicek)             H0 true
        m0_break      r reverts to a constant that shifts once (no SOFR link)     H0 true  (a level shift alone)
        m1_*          r reverts to alpha + beta1 * SOFR, for chosen beta1 and a   H0 false (power)
     SOFR and FX stay at their ACTUAL paths (so SOFR is the real I(1) series) and the noise is resampled from the
     fitted residuals in moving blocks (fat tails and volatility clustering kept).
  2. For every simulated dataset the pipeline runs (`one_sim`):
        classical F, HAC Wald       the tests the Proposal's wording would give (NOT valid with an I(1) regressor)
        bootstrap F                 nested.anchor_test, the valid one
        bootstrap F, true break     the same with the (known) break date in the model, break DGP only
        Engle-Granger               the proposal's gate (BIC lags, constant)
        ECM bootstrap t             nested.cointegration_ecm_test, the gate that handles the I(1) regressor
        CUSUM                       parameter stability of the M1 residuals (statsmodels, Ploberger-Kramer)
  3. `rejection_table`: share of datasets with p < alpha and its Wilson interval. Where the test's null is true it is
     the SIZE (should be alpha); where it is false it is the POWER. The null differs by test: for the F tests it is
     "M0 is enough"; for the two gates "no SOFR link" (a rejection there is a FALSE PASS); for CUSUM "parameters are
     stable", which is false in the level-shift process.

Limits that stay: the answer depends on the DGPs chosen (in particular the speed of adjustment); SOFR and FX are
conditioned on, not modelled; one run has Monte Carlo error of about 1 pp at 5% with 400 datasets.
"""
from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import breaks_cusumolsresid

from . import cointegration as C
from . import nested
from . import restricted_ecm as E
from .ols import OLSFit, exclusion_f, exclusion_wald_hac, subset

TESTS = ("classical", "hac", "boot", "boot_break", "eg", "ecm_gate", "cusum")
LABELS = {"classical": "Classical F", "hac": "HAC Wald", "boot": "Bootstrap F", "boot_break": "Bootstrap F, true break",
          "eg": "Engle-Granger", "ecm_gate": "ECM bootstrap gate", "cusum": "CUSUM stability"}


# ------------------------------------------------------------------ simulation engine
def simulate_ecm(coef: dict, df: pd.DataFrame, p: int, shocks, shift=None) -> np.ndarray:
    """Re-generate r from an ECM with the GIVEN coefficients, in the column convention of restricted_ecm:

        dr_i = const + r_lag r_{i-1} + x_lag x_{i-1} + shift D_{i-1} + dx0 dx_i + z z_i + sum_k dr_k dr_{i-k} + shock_i

    x, z and D are taken from `df` / `shift` as they are (SOFR keeps its actual I(1) path); the first p + 1
    values of r are the actual ones. `shocks` has length n - (p + 1). Missing coefficients count as 0."""
    n, s = len(df), p + 1
    shocks = np.asarray(shocks, float)
    if len(shocks) != n - s:
        raise ValueError(f"need {n - s} shocks, got {len(shocks)}")
    x, z = df["x"].to_numpy(), df["z"].to_numpy()
    D = np.zeros(n) if shift is None else np.asarray(shift, float)
    dx = np.r_[np.nan, np.diff(x)]
    r = df["r"].to_numpy().copy()
    dr = np.r_[np.nan, np.diff(r)]
    phi = [coef.get(f"dr_{k}", 0.0) for k in range(1, p + 1)]
    for i in range(s, n):
        d = (coef.get("const", 0.0) + coef["r_lag"] * r[i - 1] + coef.get("x_lag", 0.0) * x[i - 1]
             + coef.get("shift", 0.0) * D[i - 1] + coef.get("dx0", 0.0) * dx[i] + coef.get("z", 0.0) * z[i]
             + shocks[i - s])
        for k in range(p):
            d += phi[k] * dr[i - 1 - k]
        r[i] = r[i - 1] + d
        dr[i] = d
    return r


# ------------------------------------------------------------------ data-generating processes
@dataclass(frozen=True)
class DGP:
    name: str
    coef: dict                      # ECM coefficients by column name
    pool: str                       # residuals to resample: "m0" or "m1"
    h0_true: bool                   # True: M0 is the truth (rejection rate = size); False: power
    break_date: pd.Timestamp | None = None     # true level-shift date, if the DGP has one
    note: str = ""


def coefficients(fit: OLSFit) -> dict:
    return dict(zip(fit.names, (float(c) for c in fit.coef)))


def make_dgps(df: pd.DataFrame, m1: E.M1ECM, m0: OLSFit, anchors=((0.25, None), (1.0, None), (1.0, -0.03)),
              break_date=None, level_shift: float = 0.0) -> list[DGP]:
    """DGPs from the fitted pipeline models. anchors = ((beta1, a), ...) for the M1-true cases, a=None meaning the
    estimated speed of adjustment. break_date / level_shift (decimal, e.g. -0.015) add the 'm0_break' case."""
    c1, c0 = coefficients(m1.fit), coefficients(m0)
    mr, mx = float(df["r"].mean()), float(df["x"].mean())
    a1, a0 = c1["r_lag"], c0["r_lag"]
    phi0 = {k: v for k, v in c0.items() if k.startswith("dr_")}

    def m1_coef(a, beta1):                                 # long-run mean of r kept at its sample mean
        return c1 | {"r_lag": a, "x_lag": -a * beta1, "const": -a * (mr - beta1 * mx)}

    out = [DGP("spurious", phi0 | {"r_lag": 0.0, "const": 0.0}, "m0", True, note="r random walk, SOFR unrelated"),
           DGP("m0", c0 | {"const": -a0 * mr}, "m0", True, note=f"r reverts to a constant (a={a0:.4f})")]
    if break_date is not None:
        bd = pd.Timestamp(break_date)
        out.append(DGP("m0_break", c0 | {"const": -a0 * mr, "shift": -a0 * level_shift}, "m0", True, bd,
                       f"constant mean shifts by {level_shift * 100:+.2f} pp on {bd:%d %b %Y}, no SOFR link"))
    for beta1, a in anchors:
        a_use = a1 if a is None else float(a)
        out.append(DGP(f"m1_b{beta1:g}_a{a_use:.4f}", m1_coef(a_use, beta1), "m1", False,
                       note=f"r reverts to alpha + {beta1:g} SOFR, a={a_use:.4f}" + (" (estimated)" if a is None else "")))
    return out


# ------------------------------------------------------------------ one simulated dataset
@dataclass
class Context:
    df: pd.DataFrame
    p: int
    dgps: list
    pools: dict
    B: int
    null: str = "unit_root"
    ecm_gate: bool = True
    block: int = 10


_CTX: Context | None = None


def _init(ctx: Context) -> None:
    global _CTX
    _CTX = ctx


def one_sim(task: tuple) -> dict:
    """Simulate one dataset from DGP number j with seed (seed, j, i) and run every test on it."""
    j, i, seed = task
    ctx = _CTX
    dgp, df, p = ctx.dgps[j], ctx.df, ctx.p
    rng = np.random.default_rng(np.random.SeedSequence([seed, j, i]))
    pool = ctx.pools[dgp.pool]
    shocks = pool[nested.block_idx(len(df) - (p + 1), ctx.block, rng)]
    D = None if dgp.break_date is None else (df.index >= dgp.break_date).astype(float)
    d = df.copy()
    d["r"] = simulate_ecm(dgp.coef, df, p, shocks, D)

    seeds = rng.integers(1 << 31, size=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = E.fit_ecm(d, p, 0, True)
        drop = E.nested_columns(m.fit.names)
        row = {"dgp": dgp.name, "i": i, "h0_true": dgp.h0_true, "c1": m.c1,
               "classical": exclusion_f(m.fit, subset(m.fit, drop))["p_classical"],
               "hac": exclusion_wald_hac(m.fit, drop)["p_hac_F"],
               "boot": nested.anchor_test(d, p, 0, True, ctx.null, ctx.B, ctx.block, int(seeds[0]))["p_boot_F"],
               "boot_break": np.nan,
               "eg": float(C.engle_granger(d["r"], d["x"], lags=[], trends=("c",)).iloc[0]["p_value"]),
               "ecm_gate": np.nan}
        if dgp.break_date is not None:
            row["boot_break"] = nested.anchor_test(d, p, 0, True, ctx.null, ctx.B, ctx.block, int(seeds[1]),
                                                   break_date=dgp.break_date)["p_boot_F"]
        if ctx.ecm_gate:
            row["ecm_gate"] = nested.cointegration_ecm_test(d, p, 0, True, ctx.B, ctx.block, int(seeds[2]))["p_boot_t_left"]
        try:
            row["cusum"] = float(breaks_cusumolsresid(m.fit.resid.to_numpy(), ddof=m.fit.k)[1])
        except Exception:
            row["cusum"] = np.nan
    return row


def run_mc(df: pd.DataFrame, p: int, dgps: list, m1: E.M1ECM, m0: OLSFit, n_sims: int = 400, B: int = 99,
           seed: int = 2027, workers: int = 1, null: str = "unit_root", ecm_gate: bool = True, block: int = 10) -> pd.DataFrame:
    """One row per simulated dataset (p-values of every test). Reproducible for a given seed, whatever `workers`."""
    ctx = Context(df, p, dgps, {"m0": m0.resid.to_numpy(), "m1": m1.fit.resid.to_numpy()}, B, null, ecm_gate, block)
    tasks = [(j, i, seed) for j in range(len(dgps)) for i in range(n_sims)]
    if workers <= 1:
        _init(ctx)
        rows = [one_sim(t) for t in tasks]
    else:
        with ProcessPoolExecutor(workers, initializer=_init, initargs=(ctx,)) as ex:
            rows = list(ex.map(one_sim, tasks, chunksize=8))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ summary
def wilson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    ci = stats.binomtest(int(k), int(n)).proportion_ci(confidence_level=conf, method="wilson")
    return float(ci.low), float(ci.high)


def null_is_true(test: str, h0_true: bool, has_break: bool) -> bool:
    """Is the null of `test` true in this process? CUSUM's null is stability, false when the process has a break;
    every other test's null is "M0 is enough" / "no SOFR link", true exactly when the process is an H0 process."""
    return (not has_break) if test == "cusum" else h0_true


def rejection_table(raw: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Long table: dgp, truth, test, n, rejections, rate, Wilson 95% interval, reading.
    Null true -> 'oversized' (the gates: 'false pass') if the interval lies above alpha, 'conservative' if below,
    else 'ok'; null false -> 'power'. Tests that were not run on a process (all NaN) are left out; a NaN p-value
    counts as no rejection."""
    rows = []
    for name, g in raw.groupby("dgp", sort=False):
        h0 = bool(g["h0_true"].iloc[0])
        has_break = bool(g["boot_break"].notna().any())
        for t in TESTS:
            if t not in g or g[t].isna().all():
                continue
            n = len(g)
            k = int((g[t] < alpha).sum())
            lo, hi = wilson(k, n)
            null = null_is_true(t, h0, has_break)
            over = "false pass" if t in ("eg", "ecm_gate") else "oversized"
            reading = "power" if not null else (over if lo > alpha else "conservative" if hi < alpha else "ok")
            rows.append({"dgp": name, "truth": "null true (size)" if null else "null false (power)", "test": t, "n": n,
                         "rejections": k, "rate": k / n, "ci_low": lo, "ci_high": hi, "reading": reading})
        rows.append({"dgp": name, "truth": "", "test": "c1_in_(0,1)", "n": len(g), "rejections": int(((g["c1"] > 0) & (g["c1"] < 1)).sum()),
                     "rate": float(((g["c1"] > 0) & (g["c1"] < 1)).mean()), "ci_low": np.nan, "ci_high": np.nan,
                     "reading": "share of fitted c1 inside (0,1)"})
    return pd.DataFrame(rows)


def wide(table: pd.DataFrame) -> pd.DataFrame:
    """dgp x test matrix of rejection rates, for printing."""
    t = table[table["test"] != "c1_in_(0,1)"]
    w = t.pivot(index="dgp", columns="test", values="rate")
    order = list(dict.fromkeys(t["dgp"]))
    return w.reindex(order)[[c for c in TESTS if c in w.columns]]
