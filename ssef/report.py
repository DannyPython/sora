"""Plain-language console report for M0 (ASCII only, so it prints on any terminal)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import gate, m0

W = 78


def _p_adf(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def _p_kpss(p: float) -> str:
    """statsmodels truncates KPSS p-values to the table range [0.01, 0.10]."""
    return "<= 0.01" if p <= 0.01 else ">= 0.10" if p >= 0.10 else f"{p:.3f}"


def _head(title: str) -> str:
    return f"\n{title}\n{'-' * W}"


def _kappa_range(ag, dt: float = 1 / 252):
    """Reversion-speed range implied by the bootstrap CI of c1. If the CI reaches 1, the lower end is 0."""
    lo_c, hi_c = ag.ci
    k_hi = -np.log(lo_c) / dt if 0 < lo_c < 1 else np.nan
    k_lo = 0.0 if hi_c >= 1 else -np.log(hi_c) / dt
    return k_lo, k_hi


def format_report(r: pd.Series, tests: pd.DataFrame, fit: m0.M0Fit, fc: pd.DataFrame,
                  window: int, sim_q=None, seed=None, ag=None, mh: pd.DataFrame | None = None) -> str:
    """Build the summary. `tests` from gate.unit_root_tests, `fc` from m0.rolling_forecast, `ag` the gate
    result (gate.ar1_gate / gate.adf_gate), `mh` the multi-horizon table from race.m0_vs_rw."""
    lvl, dif = tests.loc["level"], tests.loc["diff"]
    gate_ok = bool(ag.reverts) if ag is not None else bool(lvl.adf_reject_unit_root)
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
    if ag is not None:
        out.append(f"{'c1, bias-corrected':<22}{ag.c1_bc:<11.4f}95% bootstrap CI {ag.ci[0]:.4f} to {ag.ci[1]:.4f}"
                   f"   ({ag.lags} lagged difference{'s' if ag.lags != 1 else ''} in the regression)")
        out.append(f"Reading: {gate.verdict(tests)}.")
    out.append("GATE: " + ("PASSED - the bootstrap CI for c1 lies entirely below 1; M0 parameters are interpretable."
                           if gate_ok else
                        "FAILED - the bootstrap CI for c1 reaches 1; M0 parameters are not interpretable."))



    # ---- 2. Parameters
    if ag is not None and not gate_ok:
        # The gate failed: the headline is what the data CAN say (bias-corrected c1 and its bootstrap CI),
        # and the OLS point estimates are demoted to a descriptive line, without confidence intervals.
        out.append(_head("2. What M0 can say about SORA's persistence  (gate failed: bootstrap view first)"))
        k_lo, k_hi = _kappa_range(ag)
        hl = lambda k: "unbounded" if not k > 0 else f"{np.log(2) / k:.2f} yr"
        out.append(f"{'Quantity':<38}{'Estimate':<14}95% range")
        out.append(f"{'c1  daily persistence (bias-corr.)':<38}{ag.c1_bc:<14.4f}{ag.ci[0]:.4f} to {ag.ci[1]:.4f}")
        out.append(f"{'kappa  reversion speed (per year)':<38}{ag.kappa_bc():<14.2f}{k_lo:.2f} to "
                   + (f"{k_hi:.2f}" if np.isfinite(k_hi) else "n/a"))
        out.append(f"{'half-life of a shock':<38}{hl(ag.kappa_bc()):<14}{hl(k_hi)} to {hl(k_lo)}")
        if fit.valid:
            out.append(f"{'sigma  volatility (bp per sqrt year)':<38}{fit.sigma * 1e4:<14.0f}"
                       f"{tab.loc['sigma', 'ci_low'] * 1e4:.0f} to {tab.loc['sigma', 'ci_high'] * 1e4:.0f}"
                       "   (well determined: it is the daily noise x sqrt(252))")
        out.append(f"{'theta  long-run mean':<38}{'not identified':<15}theta = c0/(1 - c1) divides by a number close to zero")
        out.append("")
        if ag.ci[1] >= 1.0:
            out.append("! The CI for c1 reaches 1: 'no mean reversion' cannot be ruled out, so kappa may be zero.")
        out.append("! Do not quote kappa or theta from this table as findings; they are bounds, not estimates.")
        if fit.valid:
            out.append(f"For reference only (descriptive, upward-biased kappa, no confidence interval): OLS kappa = {fit.kappa:.2f} "
                       f"per year, theta = {fit.theta * 100:.2f}%, c1 = {fit.c1:.4f}.")
        else:
            out.append("OLS c1 is outside (0, 1): kappa, theta and sigma are undefined (Blueprint 5.3).")
    else:
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
            if ag is not None:
                out.append(f"kappa from bias-corrected c1: {ag.kappa_bc():.2f} per year (OLS estimate above is biased upward)")
            notes = []
            if tab.loc["c1", "ci_high"] >= 1.0:
                notes.append("c1 CI reaches 1: the data cannot rule out NO mean reversion.")
            if tab.loc["kappa", "ci_low"] <= 0.0:
                notes.append("kappa CI includes 0: reversion speed is not statistically different from zero.")
            if fit.c1 > 0.98:
                notes.append("c1 is close to 1: these CIs rely on a normal approximation that is unreliable near a unit root.")
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
    mh_note = None
    if mh is not None and len(mh):
        out.append("\nSlower horizons (direct h-day-ahead forecasts, same rolling window): a pull towards a mean accumulates")
        out.append("over days, so one day is the hardest place for M0 to win.")
        out.append(f"{'horizon':<12}{'forecasts':<11}{'RMSE bp: random walk':<24}{'M0':<9}{'M0 vs random walk'}")
        for h, rw_ in mh.iterrows():
            out.append(f"{str(h) + ' day(s)':<12}{int(rw_['n_oos']):<11}{rw_['rmse_rw_bp']:<24.1f}{rw_['rmse_m0_bp']:<9.1f}{rw_['m0_vs_rw_pct']:+.1f}%")
        mh_note = bool((mh["m0_vs_rw_pct"] < -1.0).any())
        out.append("Reading: " + ("M0 beats the random walk by more than 1% at some horizon." if mh_note
                                  else "M0 does not beat the random walk by more than 1% at any horizon."))

    # ---- 4. Simulation
    if sim_q is not None:
        out.append(_head("4. Where could SORA be in 1 year under M0?  (simulation, 10,000 paths)"))
        out.append(f"Start {r.iloc[-1] * 100:.2f}%  ->  5th pct {sim_q[0] * 100:.2f}% | median {sim_q[1] * 100:.2f}% "
                   f"| 95th pct {sim_q[2] * 100:.2f}%   (seed {seed})")
        if not gate_ok:
            out.append("! These paths use the OLS parameters, which the gate says are not interpretable: treat as an illustration.")
        out.append("! The linear Gaussian model allows negative rates" + (" (the 5th percentile above is negative)." if sim_q[0] < 0 else "."))

    # ---- Bottom line
    out.append("\n" + "=" * W)
    out.append("Bottom line")
    out.append("=" * W)
    out.append(("- Gate passed." if gate_ok else "- Gate NOT passed: do not over-interpret kappa and theta."))
    if ag is not None and not gate_ok:
        k_lo, k_hi = _kappa_range(ag)
        out.append(f"- Persistence: c1 = {ag.c1_bc:.4f} (bias-corrected), 95% CI {ag.ci[0]:.4f} to {ag.ci[1]:.4f}; "
                   f"implied reversion speed between {k_lo:.2f} and {k_hi:.2f} per year." if np.isfinite(k_hi) else
                   f"- Persistence: c1 = {ag.c1_bc:.4f} (bias-corrected), 95% CI {ag.ci[0]:.4f} to {ag.ci[1]:.4f}.")
    out.append("- M0 forecasts " + ("no better than a random walk at one day." if abs(rel) < 0.01
                                    else "better than a random walk at one day." if rel < 0 else "worse than a random walk at one day.")
               + ("" if mh_note is None else (" It beats it at a slower horizon." if mh_note
                                              else " It does not at the slower horizons either.")))
    if not gate_ok:
        out.append(f"- The sample covers about one rate cycle ({len(r) / 252:.1f} years): the gate result and the estimates "
                   "can change with the start date.")
    return "\n".join(out)
