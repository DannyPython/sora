"""Console output for M0 and M1 (ASCII only, so it prints on any terminal).

All formatting lives here; run_m0.py / run_m1.py only compute and save. `format_headline` prints a headline
table; `format_report` (M0) and `format_m1_report` (M1) are the long plain-language reports (--report)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import gate, m0
from .loaders import DT

W = 78


def _fmt(v) -> str:
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, (float, np.floating)):
        return "n/a" if not np.isfinite(v) else f"{v:.4g}"
    return str(v)


def format_headline(h: pd.DataFrame, title: str) -> str:
    """Print a (section, metric, value) table, grouped by section."""
    out = ["=" * W, f" {title}", "=" * W]
    for sec, g in h.groupby("section", sort=False):
        out.append(f"[{sec}]")
        out += [f"  {m:<38}{_fmt(v)}" for m, v in zip(g["metric"], g["value"])]
    return "\n".join(out)


def _pv(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _p_adf(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def _p_kpss(p: float) -> str:
    """statsmodels truncates KPSS p-values to the table range [0.01, 0.10]."""
    return "<= 0.01" if p <= 0.01 else ">= 0.10" if p >= 0.10 else f"{p:.3f}"


def _head(title: str) -> str:
    return f"\n{title}\n{'-' * W}"


def _kappa_range(ag, dt: float = DT):
    """Reversion-speed range implied by the bootstrap CI of c1. If the CI reaches 1, the lower end is 0."""
    lo_c, hi_c = ag.ci
    k_hi = -np.log(lo_c) / dt if 0 < lo_c < 1 else np.nan
    k_lo = 0.0 if hi_c >= 1 else -np.log(hi_c) / dt
    return k_lo, k_hi


def format_report(r: pd.Series, tests: pd.DataFrame, fit: m0.M0Fit, mh: pd.DataFrame,
                  window: int, sim_q=None, seed=None, ag=None) -> str:
    """Build the summary. `tests` from gate.unit_root_tests, `mh` the race.m0_vs_rw table (must include h = 1),
    `ag` the gate result (gate.adf_gate)."""
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
        out.append(_head(f"2. Calibrated M0 parameters  (OLS, {fit.lags} lagged daily changes, 95% confidence interval)"))
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
            out.append(row("c1     daily persistence", fit.c1, "c1", 1, 4))
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
    one = mh.loc[1]
    rel = one["m0_vs_rw_pct"] / 100.0
    verdict = ("about equal: M0 adds no forecasting value over 'tomorrow = today'" if abs(rel) < 0.01
               else "M0 is better than 'tomorrow = today'" if rel < 0
               else "M0 is worse than 'tomorrow = today'")
    out.append(_head("3. Out-of-sample forecast, 1 day ahead  (M0 side of Block 7)"))
    out.append(f"Rolling window of {window} days, {int(one['n_oos'])} forecasts. Error = typical miss in basis points (bp).")
    out.append(f"{'M0 (classical Vasicek)':<26}{one['rmse_m0_bp']:>7.2f} bp")
    out.append(f"{'Random walk benchmark':<26}{one['rmse_rw_bp']:>7.2f} bp   ({rel * 100:+.1f}% vs random walk)")
    out.append(f"Reading: {verdict}.")
    slow = mh.drop(index=1)
    mh_note = None
    if len(slow):
        out.append("\nSlower horizons (direct h-day-ahead forecasts, same rolling window): a pull towards a mean accumulates")
        out.append("over days, so one day is the hardest place for M0 to win.")
        out.append(f"{'horizon':<12}{'forecasts':<11}{'RMSE bp: random walk':<24}{'M0':<9}{'M0 vs random walk'}")
        for h, rw_ in slow.iterrows():
            out.append(f"{str(h) + ' day(s)':<12}{int(rw_['n_oos']):<11}{rw_['rmse_rw_bp']:<24.1f}{rw_['rmse_m0_bp']:<9.1f}{rw_['m0_vs_rw_pct']:+.1f}%")
        mh_note = bool((slow["m0_vs_rw_pct"] < -1.0).any())
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


# ====================================================================== M1
def format_m1_report(R: dict, a) -> str:
    """Long plain-language M1 report. `R` is the dict from run_m1.compute, `a` the parsed arguments."""
    pv, head = _pv, _head
    df, p, with_z, chk, lag_tab = R["df"], R["p"], R["with_z"], R["chk"], R["lag_tab"]
    break_date, ghC, ghS, coint, nst = R["break_date"], R["ghC"], R["ghS"], R["coint"], R["nested"]
    m_nb, m_br, st_nb, st_br, S = R["m_nb"], R["m_br"], R["st_nb"], R["st_br"], R["S"]

    L = ["=" * W, " M1 (policy-anchored Vasicek, error-correction form) on SORA", "=" * W]
    L.append(f"Data : {len(df)} daily obs, {df.index[0]:%d %b %Y} to {df.index[-1]:%d %b %Y}; SOFR timing: "
             + ("published BEFORE the SORA date" if a.sofr_lag == "strict" else "published on/before the SORA date"))
    L.append("Model: d r(t) = c + a r(t-1) + b SOFR(t-1) + d dSOFR(t)" + (" + g FX(t)" if with_z else "")
             + f" + {p} lags of d r + noise")
    L.append(f"Lags : {p} lags of d r ({'BIC choice' if str(a.p).lower() == 'bic' else 'user choice'}; residual "
             f"Ljung-Box p = {lag_tab.loc[p, 'ljung_box_p']:.3f}).  Level reversion a = -(speed x dt).")

    # ------------------------------------------------------------------ 1. data checks
    L.append(head("1. Data checks"))
    sp = chk["spread_pp"]
    L.append("SOFR minus SORA spread (pp), by half-year: " + "  ".join(f"{y}{h}:{v:.1f}" for (y, h), v in sp.items()))
    L.append(f"Daily SORA change: sd {chk['dr_sd_bp']:.1f} bp; autocorrelation lags 1-3 = "
             + ", ".join(f"{v:.2f}" for v in chk["dr_autocorr"]) + f"; {chk['n_jump_40bp']} days with |change| > 40 bp "
             f"(largest {chk['max_abs_jump_bp']:.0f} bp)")
    L.append(f"! The spread is not stable: {sp.iloc[0]:.1f} pp at the start, peak {sp.max():.1f} pp, {sp.iloc[-1]:.1f} pp at the end."
             " A constant-coefficient anchor cannot fit a drifting gap.")
    L.append("! The proposal says 'compounded SORA'. Daily jumps this large look like OVERNIGHT SORA. Check the field in")
    L.append("  updaters/update_sora.py (run it with --inspect) before relying on the dependent variable.")

    # ------------------------------------------------------------------ 2. gate
    L.append(head("2. Gate: is a regression of SORA on SOFR meaningful?"))
    ur, eg = R["ur"], R["eg"]
    L.append(f"{'Series':<18}{'level: ADF p / KPSS p':<28}{'reading':<26}{'difference reading'}")
    for sname in ur.index.get_level_values(0).unique():
        lv, df_ = ur.loc[(sname, "level")], ur.loc[(sname, "diff")]
        L.append(f"{sname:<18}{pv(lv.adf_p) + ' / ' + pv(lv.kpss_p):<28}{lv.reading:<26}{df_.reading}")
    L.append("\nEngle-Granger (H0: no cointegration; p-value by residual-ADF lag and by deterministic trend)")
    L.append(f"{'trend':<8}" + "".join(f"{('lags=' + str(l)):<11}" for l in (0, 2, 5, 8)) + f"{'BIC rule':<20}")
    for tr, label in (("c", "const"), ("ct", "+trend")):
        cells = [eg.loc[(tr, "fixed", l), "p_value"] for l in (0, 2, 5, 8)]
        bic = eg.loc[tr].xs("BIC", level="rule").iloc[0]
        bic_lag = eg.loc[tr].xs("BIC", level="rule").index[0]
        L.append(f"{label:<8}" + "".join(f"{pv(c):<11}" for c in cells) + f"p={pv(bic['p_value'])} (lags={bic_lag})")
    L.append("Reading: the verdict flips with the lag choice and with a trend term; no lag count is 'the' answer.")

    L.append("\nGregory-Hansen (cointegration allowing ONE break; most negative ADF over candidate break dates)")
    L.append(f"{'model':<22}{'statistic':<11}{'5% crit.':<10}{'1% crit.':<10}{'p-value':<9}{'break date'}")
    for g, lab in ((ghC, "level shift (C)"), (ghS, "regime shift (C/S)")):
        L.append(f"{lab:<22}{g['stat']:<11.2f}{g['cv_5pct']:<10.2f}{g['cv_1pct']:<10.2f}{pv(g['p_value']):<9}{g['break_date']:%d %b %Y}")
    L.append(f"(critical values simulated by block bootstrap of the actual (dr, dx) pairs, {a.gh_sim} draws, ADF lags = {p})")
    L.append("Reading: " + ("the no-cointegration null is REJECTED once one break is allowed (cointegration holds)." if ghC["reject_5pct"]
                            else "no cointegration even with one break."))

    ss = R["ss"]
    L.append("\nCointegrating slope of SORA on SOFR (the 'gamma' of the long-run relation)")
    subs = "  ".join(f"{pd.Timestamp(r_['from']):%b%y}-{pd.Timestamp(r_['to']):%b%y}:{r_['slope']:.2f}" for _, r_ in ss["sub_periods"].iterrows())
    L.append(f"full sample {ss['full']:.2f};  by sub-period: {subs}")
    roll = ss["rolling"][500]
    L.append(f"rolling 500-day slope ranges from {roll.min():.2f} to {roll.max():.2f}: not a stable coefficient.")

    L.append("\nError-correction cointegration test (bootstrap critical values; handles the I(1) regressor)")
    L.append(f"{'specification':<36}{'t on a':<9}{'p (t)':<9}{'F(a,b)':<9}{'p (F)':<9}")
    for lab, key in (("no break", "no_break"), (f"level shift from {break_date:%d %b %Y}", "break")):
        c_ = coint.loc[key]
        L.append(f"{lab:<36}{c_['t']:<9.2f}{pv(c_['p_boot_t_left']):<9}{c_['F']:<9.2f}{pv(c_['p_boot_F']):<9}")
    L.append("GATE (no break): " + ("PASSED" if R["gate_nobreak"] else "NOT PASSED - no stable levels relation between SORA and SOFR."))
    L.append("GATE (one break): " + ("PASSED" if ghC["reject_5pct"] else "NOT PASSED")
             + f" - the break date ({break_date:%d %b %Y}) is estimated from these data: treat conditional results as exploratory.")

    # ------------------------------------------------------------------ 3. ECM estimates
    L.append(head("3. M1 error-correction estimates"))
    L.append(f"{'':<34}{'no break':<26}{'level shift (exploratory)'}")

    def cell(st, k, scale=1.0, nd=2):
        if k not in st.index or not np.isfinite(st.loc[k, "estimate"]):
            return "n/a"
        e, s = st.loc[k, "estimate"] * scale, st.loc[k, "se_delta"] * scale
        return f"{e:.{nd}f} (se {s:.{nd}f})"
    L.append(f"{'a  (adjustment per day)':<34}{m_nb.a:<26.4f}{m_br.a:.4f}")
    L.append(f"{'kappa  reversion speed (/yr)':<34}{cell(st_nb, 'kappa'):<26}{cell(st_br, 'kappa')}")
    L.append(f"{'half-life (business days)':<34}{(np.log(2) / m_nb.kappa / DT if m_nb.valid else float('nan')):<26.0f}"
             f"{(np.log(2) / m_br.kappa / DT if m_br.valid else float('nan')):.0f}")
    L.append(f"{'beta1  long-run slope on SOFR':<34}{cell(st_nb, 'beta1'):<26}{cell(st_br, 'beta1')}")
    L.append(f"{'level shift of anchor (pp)':<34}{'-':<26}{cell(st_br, 'level_shift', 100)}")
    if with_z:
        L.append(f"{'FX short-run coefficient g':<34}{m_nb.gamma_z:<26.4f}{m_br.gamma_z:.4f}")
    L.append(f"{'daily noise s_eps (bp)':<34}{m_nb.s_eps * 1e4:<26.2f}{m_br.s_eps * 1e4:.2f}")
    L.append("! beta1 = -b/a divides by a number close to zero: its interval is unreliable (Fieller problem).")
    L.append("! Level terms use a non-standard distribution while SOFR is I(1): see the bootstrap tests below.")

    # ------------------------------------------------------------------ 4. nested test
    L.append(head("4. Does M1 beat M0?  H0: no SOFR or FX term matters  (M0 = M1 minus every SOFR/FX column)"))
    L.append(f"{'specification':<26}{'F':<8}{'classical p':<13}{'HAC p':<9}{'bootstrap p':<14}{'(null: fitted a)'}")
    for lab, key in (("no break", "no_break"), (f"level shift {break_date:%b %Y}", "break")):
        n_ = nst.loc[key]
        L.append(f"{lab:<26}{n_['F']:<8.2f}{pv(n_['p_classical']):<13}{pv(n_['p_hac']):<9}{pv(n_['p_boot']):<14}{pv(n_['p_boot_fitted_a'])}")
    L.append("Only the BOOTSTRAP p is valid (classical and HAC assume a stationary regressor). Valid test of the M0 vs M1")
    L.append("question: " + ("H0 NOT rejected without a break (no anchor detected; this is NOT evidence that SORA reverts to a fixed mean)"
                             if nst.loc["no_break", "p_boot"] >= 0.05 else "H0 rejected without a break")
             + ("; REJECTED once a level shift is allowed (exploratory: break date chosen from the data)."
                if nst.loc["break", "p_boot"] < 0.05 else "; not rejected with a level shift either."))

    # ------------------------------------------------------------------ 5. race
    L.append(head(f"5. Out-of-sample race: random walk vs M0 vs M1  (direct forecasts, {a.window}-day rolling window)"))
    L.append(f"M0 and M1 are the ECMs of sections 3-4 ({p} lags of d r, no break), regressors dated at the forecast origin.")
    L.append(f"{'horizon':<10}{'n':<6}{'RMSE bp: RW':<14}{'M0':<9}{'M1':<9}{'best'}")
    for h in a.horizons:
        v = {k: S.loc[h, f'rmse_{k}_bp'] for k in ('rw', 'm0', 'm1')}
        L.append(f"{str(h) + ' day(s)':<10}{int(S.loc[h, 'n_oos']):<6}{v['rw']:<14.1f}{v['m0']:<9.1f}{v['m1']:<9.1f}{min(v, key=v.get).upper()}")
    L.append(f"\nClark-West test that M1 adds information beyond the smaller model (one-sided; bootstrap, {a.race_B} draws)")
    L.append(f"{'comparison':<12}{'h':<5}{'CW t':<8}{'normal p (invalid)':<20}{'bootstrap p'}")
    for name in ("M1 vs M0", "M1 vs RW", "M0 vs RW"):
        for h in a.horizons:
            L.append(f"{name:<12}{h:<5}{S.loc[h, 'cw_t ' + name]:<8.2f}{pv(S.loc[h, 'p_normal ' + name]):<20}{pv(S.loc[h, 'p_boot ' + name])}")
    L.append("Reading: " + ("the random walk has the lowest RMSE at EVERY horizon; " if R["rw_best_all"] else "")
             + "Clark-West asks a different question (does x add anything beyond M0?), so both can be true.")

    # ------------------------------------------------------------------ 6. noise
    L.append(head("6. The noise term (Proposal 4.3.3 assumes it is Gaussian)"))
    nd, cmp_ = R["nd"], R["noise_cmp"]
    L.append(f"Daily noise: M0 {cmp_['s_eps_m0_bp']:.2f} bp, M1 {cmp_['s_eps_m1_bp']:.2f} bp "
             f"(degrees-of-freedom-adjusted variance reduction from adding SOFR/FX: {cmp_['variance_reduction'] * 100:.1f}%)")
    for nm, r_ in nd.iterrows():
        val = f"{r_['value']:.3f}" if np.isfinite(r_["value"]) else f"p={pv(r_['p_value'])}"
        L.append(f"  {nm:<22}{val:<12}{'VIOLATED' if r_['violated'] else 'ok'}")
    L.append("Reading: fat tails and volatility clustering -> use bootstrapped noise in simulation and a conditional")
    L.append("(GARCH-filtered) EVT step; standardised residuals are saved for it.")

    # ------------------------------------------------------------------ 7. scenarios
    L.append(head(f"7. Scenarios: SORA in 1 year (5th / median / 95th pct, %), SOFR shock {a.shock_bp:.0f} bp, bootstrapped noise"))
    sc = R["scenarios"]
    L.append(f"{'scenario':<10}{'no break':<26}{'level shift (exploratory)'}")
    for name in sc["scenario"].unique():
        cells = [f"{r_.p5_pct:5.2f} / {r_.p50_pct:5.2f} / {r_.p95_pct:5.2f}" for r_ in sc[sc["scenario"] == name].itertuples()]
        L.append(f"{name:<10}{cells[0]:<26}{cells[1]}")
    L.append(f"Start SORA {df['r'].iloc[-1] * 100:.2f}%, SOFR {df['x'].iloc[-1] * 100:.2f}%. SOFR and FX are conditioned on, not modelled.")
    L.append("! The linear Gaussian model allows negative rates"
             + (" (some percentiles above are negative)." if (sc["p5_pct"] < 0).any() else "."))

    # ------------------------------------------------------------------ 8. robustness
    L.append(head("8. Robustness"))
    grid, sd, rw_ = R["grid"], R["start_dates"], R["race_windows"]
    L.append("(a) M0-vs-M1 exclusion test vs lag choice - classical/HAC p-values move a lot with the lags:")
    L.append(f"{'dr lags':<9}{'dx lags':<9}{'a':<10}{'kappa':<8}{'classical p':<13}{'HAC p'}")
    for _, r_ in grid.iterrows():
        L.append(f"{int(r_['p_dr_lags']):<9}{int(r_['q_dx_lags']):<9}{r_['a']:<10.4f}{r_['kappa']:<8.1f}{pv(r_['p_classical']):<13}{pv(r_['p_HAC'])}")
    L.append("\n(b) Start-date sensitivity (one rate cycle in the sample, so the window matters):")
    L.append(f"{'start':<12}{'n':<6}{'EG p (const)':<14}{'EG p (trend)':<14}{'GH stat':<9}{'break':<13}{'slope':<7}{'kappa'}")
    for _, r_ in sd.iterrows():
        L.append(f"{str(r_['start']):<12}{int(r_['n']):<6}{pv(r_['EG_p_const(BIC)']):<14}{pv(r_['EG_p_trend(BIC)']):<14}"
                 f"{r_['GH_C_stat']:<9.2f}{str(r_['GH_break']):<13}{r_['slope_r_on_x']:<7.2f}{r_['kappa']:.1f}")
    L.append("\n(c) Race vs window length (RMSE, bp):")
    L.append(f"{'window':<9}{'h':<5}{'RW':<8}{'M0':<8}{'M1'}")
    for _, r_ in rw_.iterrows():
        L.append(f"{int(r_['window']):<9}{int(r_['h']):<5}{r_['rmse_rw_bp']:<8.1f}{r_['rmse_m0_bp']:<8.1f}{r_['rmse_m1_bp']:.1f}")

    # ------------------------------------------------------------------ bottom line
    L += ["\n" + "=" * W, "Bottom line", "=" * W]
    L.append(f"- No break: SORA and SOFR are {'cointegrated' if R['gate_nobreak'] else 'NOT shown to be cointegrated'} "
             f"(Engle-Granger BIC p = {pv(R['egc_bic'])}; ECM bootstrap p = {pv(coint.loc['no_break', 'p_boot_t_left'])}); "
             f"the anchor test does not reject (bootstrap p = {pv(nst.loc['no_break', 'p_boot'])}).")
    L.append(f"- Allowing one level shift ({break_date:%d %b %Y}): cointegration {'holds' if ghC['reject_5pct'] else 'does not hold'} "
             f"(Gregory-Hansen {ghC['stat']:.2f} vs 5% value {ghC['cv_5pct']:.2f}). Date is data-chosen: check it against MAS policy dates.")
    L.append(f"- Sensitive to the start date: EG p (const) from {pv(sd['EG_p_const(BIC)'].max())} to {pv(sd['EG_p_const(BIC)'].min())} across starts; "
             f"slope {sd['slope_r_on_x'].min():.2f} to {sd['slope_r_on_x'].max():.2f}.")
    L.append("- Race: " + ("random walk is best at every horizon." if R["rw_best_all"] else "see table 5.")
             + f" Bootstrap Clark-West M1 vs M0 at {a.horizons[0]} day: p = {pv(S.loc[a.horizons[0], 'p_boot M1 vs M0'])}"
             + (f"; at {a.horizons[-1]} days: p = {pv(S.loc[a.horizons[-1], 'p_boot M1 vs M0'])}." if len(a.horizons) > 1 else "."))
    L.append("- Headline the data support: no STABLE level anchor over the full window. A SOFR anchor appears only when a regime")
    L.append("  change is allowed (data-chosen date) or the sample starts later, and its slope then varies a lot.")
    L.append("- Open decisions: anchor variable (forward points would make CIP hold by construction), SORA field definition.")
    return "\n".join(L)


def format_backtest_report(R: dict, a) -> str:
    """Plain-language VaR backtest report. `R` is the dict from run_backtest.compute, `a` the parsed arguments."""
    pv, head = _pv, _head
    df, cov, tick, kp, cp = R["df"], R["coverage"], R["tick"], R["power_kupiec"], R["power_christoffersen"]
    names = {"rw": "RW", "m0": "M0", "m1": "M1"}
    L = ["=" * W, " Value-at-Risk backtest: random walk vs M0 vs M1 on held-out SORA", "=" * W]
    L.append(f"Data : {len(df)} daily obs, {df.index[0]:%d %b %Y} to {df.index[-1]:%d %b %Y}; {R['p']} lags of d r; "
             f"FX term {'in' if R['with_z'] else 'out'}; rolling window {a.window} days.")
    L.append("VaR  : quantile of the SORA level h days ahead. 'up' = rate rises (borrower / S-REIT risk), 'down' = rate falls.")
    L.append(f"       normal = Gaussian (the Vasicek assumption); hs = window residual quantile; evt = GPD tail above the "
             f"top {a.tail_frac:.0%} ({a.gpd.upper()} fit).")
    L.append("Tests: Kupiec (right number of breaches), Christoffersen independence (no clustering) and conditional coverage.")

    for h in sorted(cov["h"].unique()):
        c = cov[cov["h"] == h]
        N = int(c["N"].iloc[0])
        L.append(head(f"Coverage at h = {h} day{'s' if h > 1 else ''}: {N} {'daily' if h == 1 else 'non-overlapping'} "
                      "out-of-sample points"))
        L.append(f"{'tail':<6}{'level':<7}{'method':<8}{'model':<6}{'breaches':<10}{'expected':<10}{'p Kupiec':<10}"
                 f"{'p exact':<9}{'p indep':<9}{'p cc':<8}{'verdict'}")
        for _, s in c.iterrows():
            L.append(f"{s['tail']:<6}{s['level']:<7.0%}{s['method']:<8}{names[s['model']]:<6}{s['breaches']:<10}"
                     f"{s['expected']:<10.1f}{pv(s['p_pof']):<10}{pv(s['p_binom']):<9}"
                     f"{('n/a' if not np.isfinite(s['p_ind']) else pv(s['p_ind'])):<9}"
                     f"{('n/a' if not np.isfinite(s['p_cc']) else pv(s['p_cc'])):<8}{s['verdict']}")
        nrej = int((c["verdict"] != "not rejected").sum())
        L.append(f"Reading: {nrej} of {len(c)} model/method/tail/level combinations are rejected at 5%.")
        too_few = c[(c["p_pof"] < 0.05) & (c["breaches"] < c["expected"])]
        if len(too_few):
            L.append(f"  {len(too_few)} rejections are TOO FEW breaches (VaR too wide: costly capital buffer, not under-statement).")
        clus = c[c["verdict"] == "fail: clustering"]
        if len(clus):
            L.append(f"  {len(clus)} fail on clustering only: breaches come in runs, the model is slow to adapt to regime "
                     "changes (constant variance in the window; a GARCH-filtered EVT is the next step).")

    L.append(head("Head-to-head: quantile (tick) loss, Giacomini-White test (valid for nested models with a rolling window)"))
    L.append(f"{'h':<4}{'method':<8}{'tail':<6}{'level':<7}{'loss RW':<9}{'loss M0':<9}{'loss M1':<9}"
             f"{'p M0>RW':<9}{'p M1>RW':<9}{'p M1>M0'}")
    for _, s in tick.iterrows():
        L.append(f"{int(s['h']):<4}{s['method']:<8}{s['tail']:<6}{s['level']:<7.0%}{s['loss_rw_bp']:<9.3f}{s['loss_m0_bp']:<9.3f}"
                 f"{s.get('loss_m1_bp', np.nan):<9.3f}{pv(s['p M0 vs RW']):<9}{pv(s.get('p M1 vs RW', np.nan)):<9}"
                 f"{pv(s.get('p M1 vs M0', np.nan))}")
    L.append("Loss in bp (lower is better). 'p M1>M0' small = M1's VaR is significantly better than M0's.")
    if (tick["N"] < 100).any():
        L.append("! Rows with fewer than 100 points (long horizons) rest on a handful of independent outcomes: descriptive only.")

    L.append(head("Power: how much can a 'pass' tell us? (Proposal 4.7)"))
    L.append(f"{'h':<4}{'level':<7}{'N':<6}{'Kupiec accepts':<17}{'size':<7}{'power x1.5':<12}{'power x2':<10}{'power x3'}")
    for (h, lev), g in kp.groupby(["h", "level"]):
        pw = dict(zip(g["true_rate_multiple"], g["power"]))
        g0 = g.iloc[0]
        acc = f"{int(g0['accept_breaches_min'])}-{int(g0['accept_breaches_max'])} breaches"
        L.append(f"{int(h):<4}{lev:<7.0%}{int(g0['N']):<6}{acc:<17}"
                 f"{pw.get(1.0, np.nan):<7.2f}{pw.get(1.5, np.nan):<12.2f}{pw.get(2.0, np.nan):<10.2f}{pw.get(3.0, np.nan):.2f}")
    L.append("(power = probability the Kupiec test rejects when the true breach rate is 1.5x / 2x / 3x the nominal one)")
    L.append(f"{'h':<4}{'level':<7}{'P(hit|hit)':<12}{'power of independence test':<28}{'share of samples with no breach'}")
    for _, s in cp.iterrows():
        L.append(f"{int(s['h']):<4}{s['level']:<7.0%}{s['pi11']:<12.2f}{s['power_ind']:<28.2f}{s['share_no_breach']:.2f}")
    L.append("(first row of each block: iid breaches, i.e. the test's actual size)")

    es = R["evt_shape"]
    L.append(head("Tail shape of the standardised residuals (GPD xi over the rolling windows)"))
    for _, s in es.iterrows():
        L.append(f"{names[s['model']]:<4}{s['tail']:<6}median xi {s['xi_median']:6.2f}  (10-90%: {s['xi_p10']:.2f} to "
                 f"{s['xi_p90']:.2f}); xi > 0 in {s['share_xi_positive']:.0%} of windows")
    L.append("xi > 0 = heavier than Gaussian tails: the normal VaR understates the extreme quantiles.")

    if "robust" in R:
        rb = R["robust"]
        L.append(head(f"Sensitivity (h = {int(rb['h'].iloc[0])}): share of combinations NOT rejected, by window and threshold"))
        for (w, tf), g in rb.groupby(["window", "tail_frac"]):
            L.append(f"window {w:<5} GPD threshold top {tf:<5.0%} methods {','.join(sorted(g['method'].unique())):<15}"
                     f"{(g['verdict'] == 'not rejected').mean():.0%} of {len(g)}")

    L += ["\n" + "=" * W, "Bottom line", "=" * W]
    c1 = cov[cov["h"] == min(cov["h"])]
    for tail in ("up", "down"):
        for lev in sorted(c1["level"].unique(), reverse=True):
            g = c1[(c1["tail"] == tail) & (c1["level"] == lev)]
            ok = g[g["verdict"] == "not rejected"]
            L.append(f"- {tail:<4} {lev:.0%}, h={int(c1['h'].iloc[0])}: not rejected for "
                     + (", ".join(f"{names[m]}/{me}" for m, me in zip(ok["model"], ok["method"])) or "none"))
    L.append("- Small breach counts give low power: a 'not rejected' is weak evidence of adequacy (see the power table).")
    L.append("- The VaR is conditional on the current window only; volatility clustering is not modelled (constant s per window).")
    return "\n".join(L)
