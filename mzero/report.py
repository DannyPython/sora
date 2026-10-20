"""Plain-language console report for M0 (ASCII only, so it prints on any terminal)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import m0

W = 78


def _p_adf(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def _p_kpss(p: float) -> str:
    """statsmodels truncates KPSS p-values to the table range [0.01, 0.10]."""
    return "<= 0.01" if p <= 0.01 else ">= 0.10" if p >= 0.10 else f"{p:.3f}"


def _head(title: str) -> str:
    return f"\n{title}\n{'-' * W}"


def format_report(r: pd.Series, tests: pd.DataFrame, fit: m0.M0Fit, fc: pd.DataFrame,
                  window: int, sim_q=None, seed=None) -> str:
    """Build the summary. `tests` from gate.unit_root_tests, `fc` from m0.rolling_forecast."""
    lvl, dif = tests.loc["level"], tests.loc["diff"]
    gate_ok = bool(lvl.adf_reject_unit_root)
    tab = fit.params_table()
    out = ["=" * W, " M0 (classical Vasicek) on SORA - summary", "=" * W]

    out.append(f"Data : {len(r)} daily SORA values, {r.index[0]:%d %b %Y} to {r.index[-1]:%d %b %Y}")
    out.append(f"Model: dr = kappa (theta - r) dt + sigma dW   (theta constant)   | 1 step = 1/252 year")

    # ---- 1. Gate
    out.append(_head("1. Does SORA revert to a mean at all?  (Block 2 gate)"))
    out.append(f"{'Test':<22}{'p-value':<11}Reading")
    out.append(f"{'ADF, SORA level':<22}{_p_adf(lvl.adf_p):<11}"
               + ("rejects unit root -> stationary" if lvl.adf_reject_unit_root
                  else "cannot reject unit root -> looks NON-stationary"))
    out.append(f"{'KPSS, SORA level':<22}{_p_kpss(lvl.kpss_p):<11}"
               + ("rejects stationarity -> looks NON-stationary" if lvl.kpss_reject_stationary
                  else "cannot reject stationarity"))
    out.append(f"{'ADF, daily change':<22}{_p_adf(dif.adf_p):<11}"
               + ("stationary (as expected after differencing)" if dif.adf_reject_unit_root
                  else "still non-stationary"))
    out.append("GATE: " + ("PASSED - SORA level is stationary; M0 parameters are interpretable."
                           if gate_ok else
                           "NOT PASSED - SORA behaves like a random walk (I(1)) in this sample."))

    # ---- 2. Parameters
    out.append(_head("2. Calibrated M0 parameters  (OLS on AR(1), 95% confidence interval)"))
    if fit.valid:
        def row(label, est, name, scale=1.0, nd=2, unit=""):
            lo, hi = tab.loc[name, "ci_low"] * scale, tab.loc[name, "ci_high"] * scale
            return (f"{label:<38}{f'{est * scale:.{nd}f}{unit}':<12}"
                    f"{lo:.{nd}f}{unit} to {hi:.{nd}f}{unit}")
        hl = np.log(2) / fit.kappa
        out.append(f"{'Parameter':<38}{'Estimate':<12}95% CI")
        out.append(row("kappa  reversion speed (per year)", fit.kappa, "kappa"))
        out.append(row("theta  long-run mean of SORA", fit.theta, "theta", 100, 2, "%"))
        out.append(row("sigma  volatility (bp per sqrt year)", fit.sigma, "sigma", 1e4, 0))
        out.append(row("c1     daily persistence (AR(1))", fit.c1, "c1", 1, 4))
        out.append(f"Half-life of a shock: {fit.half_life_steps:.0f} business days ({hl:.2f} year)")
        notes = []
        if tab.loc["c1", "ci_high"] >= 1.0:
            notes.append("c1 CI reaches 1: the data cannot rule out NO mean reversion.")
        if tab.loc["kappa", "ci_low"] <= 0.0:
            notes.append("kappa CI includes 0: reversion speed is not statistically different from zero.")
        if not gate_ok or fit.c1 > 0.98:
            notes.append("c1 is close to 1 / the series fails the gate: these CIs rely on a normal\n"
                         "  approximation that is unreliable near a unit root. Read kappa and theta as descriptive only.")
        out += [f"! {n}" for n in notes]
    else:
        out.append("c1 is outside (0, 1): SORA does not revert; kappa, theta, sigma are undefined (Blueprint 5.3).")

    # ---- 3. Forecast
    e_m0, e_rw = m0.rmse(fc), m0.rmse(fc, "rw_forecast")
    rel = (e_m0 - e_rw) / e_rw
    verdict = ("about equal: M0 adds no forecasting value over 'tomorrow = today'" if abs(rel) < 0.01
               else "M0 is better than 'tomorrow = today'" if rel < 0
               else "M0 is worse than 'tomorrow = today'")
    out.append(_head("3. Out-of-sample forecast, 1 day ahead  (M0 side of Block 7)"))
    out.append(f"Rolling window of {window} days, {len(fc)} forecasts. Error = typical miss in basis points (bp).")
    out.append(f"{'M0 (Vasicek AR(1))':<26}{e_m0 * 1e4:>7.2f} bp")
    out.append(f"{'Random walk benchmark':<26}{e_rw * 1e4:>7.2f} bp   ({rel * 100:+.1f}% vs random walk)")
    out.append(f"Reading: {verdict}.")

    # ---- 4. Simulation
    if sim_q is not None:
        out.append(_head("4. Where could SORA be in 1 year under M0?  (simulation, 10,000 paths)"))
        out.append(f"Start {r.iloc[-1] * 100:.2f}%  ->  5th pct {sim_q[0] * 100:.2f}% | median {sim_q[1] * 100:.2f}% "
                   f"| 95th pct {sim_q[2] * 100:.2f}%   (seed {seed})")

    # ---- Bottom line
    out.append("\n" + "=" * W)
    out.append("Bottom line")
    out.append("=" * W)
    out.append(("- Gate passed." if gate_ok else "- Gate NOT passed: do not over-interpret kappa and theta."))
    out.append("- M0 forecasts " + ("no better than a random walk." if abs(rel) < 0.01
                                    else "better than a random walk." if rel < 0 else "worse than a random walk."))
    if not gate_ok:
        out.append(f"- The sample is short ({len(r) / 252:.1f} years) and SORA trends down within it; "
                   "a longer window may change the gate result.")
    return "\n".join(out)