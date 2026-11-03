"""M1 pipeline: data checks -> gate -> ECM -> valid nested test -> out-of-sample race -> noise -> scenarios ->
robustness. Prints a plain-language report and writes tables + run_info.json to results/m1/.

    python run_m1.py                  # full run (about 1-2 minutes)
    python run_m1.py --fast           # fewer bootstrap draws (about 15 seconds), for checking the code
    python run_m1.py --p 2 --q 0      # fix the lag of dr instead of the BIC choice
    python run_m1.py --no-fx          # drop the FX term (Blueprint Sec. 10 fallback)
    python run_m1.py --sofr-lag same-day --break-date 2025-04-14
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ssef import cointegration as C
from ssef import ecm, loaders, nested, noise, provenance, race, robustness, simulate
from ssef.ols import exclusion_f, exclusion_wald_hac, subset

W = 78


def head(t):
    return f"\n{t}\n{'-' * W}"


def pv(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--p", default="bic", help="lags of dr: an integer or 'bic' (default)")
    ap.add_argument("--pmax", type=int, default=10)
    ap.add_argument("--q", type=int, default=0, help="extra lags of dx")
    ap.add_argument("--no-fx", action="store_true")
    ap.add_argument("--sofr-lag", choices=["strict", "same-day"], default="strict")
    ap.add_argument("--break-date", default=None, help="override the Gregory-Hansen break date (YYYY-MM-DD)")
    ap.add_argument("--B", type=int, default=499, help="bootstrap draws for the nested / cointegration tests")
    ap.add_argument("--race-B", type=int, default=499)
    ap.add_argument("--gh-sim", type=int, default=200)
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 20, 60])
    ap.add_argument("--shock-bp", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default="results/m1")
    a = ap.parse_args()
    if a.fast:
        a.B, a.race_B, a.gh_sim = 99, 99, 40
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seed = a.seed

    # ------------------------------------------------------------------ data
    r_all = loaders.load_sora(start=a.start, end=a.end)
    sofr = loaders.load_sofr()
    fx_ok = (not a.no_fx) and loaders.FX_CSV.exists()
    z = loaders.make_z(loaders.load_usdsgd()) if fx_ok else None
    df = loaders.build_dataset(r_all, sofr, z, strict=(a.sofr_lag == "strict"))
    with_z = "z" in df
    chk = loaders.data_checks(df)

    lag_tab = ecm.select_lags(df, a.pmax, a.q, with_z)
    p = int(lag_tab["bic"].idxmin()) if str(a.p).lower() == "bic" else int(a.p)

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
    ur = C.unit_root_table(df)
    L.append(f"{'Series':<18}{'level: ADF p / KPSS p':<28}{'reading':<26}{'difference reading'}")
    for sname in ur.index.get_level_values(0).unique():
        lv, df_ = ur.loc[(sname, "level")], ur.loc[(sname, "diff")]
        L.append(f"{sname:<18}{pv(lv.adf_p) + ' / ' + pv(lv.kpss_p):<28}{lv.reading:<26}{df_.reading}")
    eg = C.engle_granger(df["r"], df["x"], lags=[0, 2, 5, 8])
    L.append("\nEngle-Granger (H0: no cointegration; p-value by residual-ADF lag and by deterministic trend)")
    L.append(f"{'trend':<8}" + "".join(f"{('lags=' + str(l)):<11}" for l in (0, 2, 5, 8)) + f"{'BIC rule':<20}")
    for tr, label in (("c", "const"), ("ct", "+trend")):
        cells = [eg.loc[(tr, "fixed", l), "p_value"] for l in (0, 2, 5, 8)]
        bic = eg.loc[tr].xs("BIC", level="rule").iloc[0]
        bic_lag = eg.loc[tr].xs("BIC", level="rule").index[0]
        L.append(f"{label:<8}" + "".join(f"{pv(c):<11}" for c in cells) + f"p={pv(bic['p_value'])} (lags={bic_lag})")
    egc_bic = float(eg.loc["c"].xs("BIC", level="rule")["p_value"].iloc[0])
    L.append("Reading: the verdict flips with the lag choice and with a trend term; no lag count is 'the' answer.")

    ghC = C.gregory_hansen(df["r"], df["x"], "C", p, n_sim=a.gh_sim, seed=seed, dates=df.index)
    ghS = C.gregory_hansen(df["r"], df["x"], "C/S", p, n_sim=a.gh_sim, seed=seed, dates=df.index)
    L.append("\nGregory-Hansen (cointegration allowing ONE break; most negative ADF over candidate break dates)")
    L.append(f"{'model':<22}{'statistic':<11}{'5% crit.':<10}{'1% crit.':<10}{'p-value':<9}{'break date'}")
    for g, lab in ((ghC, "level shift (C)"), (ghS, "regime shift (C/S)")):
        L.append(f"{lab:<22}{g['stat']:<11.2f}{g['cv_5pct']:<10.2f}{g['cv_1pct']:<10.2f}{pv(g['p_value']):<9}{g['break_date']:%d %b %Y}")
    L.append(f"(critical values simulated by block bootstrap of the actual (dr, dx) pairs, {a.gh_sim} draws, ADF lags = {p})")
    L.append("Reading: " + ("the no-cointegration null is REJECTED once one break is allowed (cointegration holds)." if ghC["reject_5pct"]
                            else "no cointegration even with one break."))
    break_date = pd.Timestamp(a.break_date) if a.break_date else ghC["break_date"]

    ss = C.slope_stability(df, windows=(500,))
    L.append("\nCointegrating slope of SORA on SOFR (the 'gamma' of the long-run relation)")
    subs = "  ".join(f"{pd.Timestamp(r_['from']):%b%y}-{pd.Timestamp(r_['to']):%b%y}:{r_['slope']:.2f}" for _, r_ in ss["sub_periods"].iterrows())
    L.append(f"full sample {ss['full']:.2f};  by sub-period: {subs}")
    roll = ss["rolling"][500]
    L.append(f"rolling 500-day slope ranges from {roll.min():.2f} to {roll.max():.2f}: not a stable coefficient.")

    coint0 = nested.cointegration_ecm_test(df, p, a.q, with_z, a.B, seed=seed)
    coint1 = nested.cointegration_ecm_test(df, p, a.q, with_z, a.B, seed=seed, break_date=break_date)
    L.append("\nError-correction cointegration test (bootstrap critical values; handles the I(1) regressor)")
    L.append(f"{'specification':<36}{'t on a':<9}{'p (t)':<9}{'F(a,b)':<9}{'p (F)':<9}")
    for lab, c_ in (("no break", coint0), (f"level shift from {break_date:%d %b %Y}", coint1)):
        L.append(f"{lab:<36}{c_['t']:<9.2f}{pv(c_['p_boot_t_left']):<9}{c_['F']:<9.2f}{pv(c_['p_boot_F']):<9}")
    gate_nobreak = (egc_bic < 0.05) and (coint0["p_boot_t_left"] < 0.05)
    L.append("GATE (no break): " + ("PASSED" if gate_nobreak else "NOT PASSED - no stable levels relation between SORA and SOFR."))
    L.append("GATE (one break): " + ("PASSED" if ghC["reject_5pct"] else "NOT PASSED")
             + f" - the break date ({break_date:%d %b %Y}) is estimated from these data: treat conditional results as exploratory.")

    # ------------------------------------------------------------------ 3. ECM estimates
    L.append(head("3. M1 error-correction estimates"))
    m_nb = ecm.fit_ecm(df, p, a.q, with_z)
    m_br = ecm.fit_ecm(df, p, a.q, with_z, break_date=break_date)
    st_nb, st_br = m_nb.structural_table(), m_br.structural_table()
    L.append(f"{'':<34}{'no break':<26}{'level shift (exploratory)'}")
    def cell(st, k, scale=1.0, nd=2):
        if k not in st.index or not np.isfinite(st.loc[k, "estimate"]):
            return "n/a"
        e, s = st.loc[k, "estimate"] * scale, st.loc[k, "se_delta"] * scale
        return f"{e:.{nd}f} (se {s:.{nd}f})"
    L.append(f"{'a  (adjustment per day)':<34}{m_nb.a:<26.4f}{m_br.a:.4f}")
    L.append(f"{'kappa  reversion speed (/yr)':<34}{cell(st_nb, 'kappa'):<26}{cell(st_br, 'kappa')}")
    L.append(f"{'half-life (business days)':<34}{(np.log(2) / m_nb.kappa / ecm.DT_DEFAULT if m_nb.valid else float('nan')):<26.0f}"
             f"{(np.log(2) / m_br.kappa / ecm.DT_DEFAULT if m_br.valid else float('nan')):.0f}")
    L.append(f"{'beta1  long-run slope on SOFR':<34}{cell(st_nb, 'beta1'):<26}{cell(st_br, 'beta1')}")
    L.append(f"{'level shift of anchor (pp)':<34}{'-':<26}{cell(st_br, 'level_shift', 100)}")
    if with_z:
        L.append(f"{'FX short-run coefficient g':<34}{m_nb.gamma_z:<26.4f}{m_br.gamma_z:.4f}")
    L.append(f"{'daily noise s_eps (bp)':<34}{m_nb.s_eps * 1e4:<26.2f}{m_br.s_eps * 1e4:.2f}")
    L.append("! beta1 = -b/a divides by a number close to zero: its interval is unreliable (Fieller problem).")
    L.append("! Level terms use a non-standard distribution while SOFR is I(1): see the bootstrap tests below.")

    # ------------------------------------------------------------------ 4. nested test
    L.append(head("4. Does M1 beat M0?  H0: beta1 = beta2 = 0  (SOFR level and FX excluded; M0 = M1 minus columns)"))
    rows4 = []
    for lab, model, bd in (("no break", m_nb, None), (f"level shift {break_date:%b %Y}", m_br, break_date)):
        drop = ecm.nested_columns(model.fit.names, "anchor")
        fF = exclusion_f(model.fit, subset(model.fit, drop))
        hh = exclusion_wald_hac(model.fit, drop)
        bt = nested.anchor_test(df, p, a.q, with_z, "unit_root", a.B, seed=seed, break_date=bd)
        bf = nested.anchor_test(df, p, a.q, with_z, "fitted", a.B, seed=seed, break_date=bd)
        rows4.append((lab, fF, hh, bt, bf))
    L.append(f"{'specification':<26}{'F':<8}{'classical p':<13}{'HAC p':<9}{'bootstrap p':<14}{'(null: fitted a)'}")
    for lab, fF, hh, bt, bf in rows4:
        L.append(f"{lab:<26}{fF['F']:<8.2f}{pv(fF['p_classical']):<13}{pv(hh['p_hac_F']):<9}{pv(bt['p_boot_F']):<14}{pv(bf['p_boot_F'])}")
    L.append("Only the BOOTSTRAP p is valid (classical and HAC assume a stationary regressor). Valid test of the M0 vs M1")
    L.append("question: " + ("H0 NOT rejected without a break (no anchor detected; this is NOT evidence that SORA reverts to a fixed mean)"
                             if rows4[0][3]["p_boot_F"] >= 0.05 else "H0 rejected without a break")
             + ("; REJECTED once a level shift is allowed (exploratory: break date chosen from the data)."
                if rows4[1][3]["p_boot_F"] < 0.05 else "; not rejected with a level shift either."))

    # ------------------------------------------------------------------ 5. race
    res = race.run_race(df, tuple(a.horizons), a.window, with_z, B=a.race_B, null="unit_root", p=p, seed=seed)
    S = res["summary"]
    L.append(head(f"5. Out-of-sample race: random walk vs M0 vs M1  (direct forecasts, {a.window}-day rolling window)"))
    L.append(f"{'horizon':<10}{'n':<6}{'RMSE bp: RW':<14}{'M0':<9}{'M1':<9}{'best'}")
    for h in a.horizons:
        v = {k: S.loc[h, f'rmse_{k}_bp'] for k in ('rw', 'm0', 'm1')}
        L.append(f"{str(h) + ' day(s)':<10}{int(S.loc[h, 'n_oos']):<6}{v['rw']:<14.1f}{v['m0']:<9.1f}{v['m1']:<9.1f}{min(v, key=v.get).upper()}")
    L.append(f"\nClark-West test that M1 adds information beyond the smaller model (one-sided; bootstrap, {a.race_B} draws)")
    L.append(f"{'comparison':<12}{'h':<5}{'CW t':<8}{'normal p (invalid)':<20}{'bootstrap p'}")
    for name in ("M1 vs M0", "M1 vs RW", "M0 vs RW"):
        for h in a.horizons:
            L.append(f"{name:<12}{h:<5}{S.loc[h, 'cw_t ' + name]:<8.2f}{pv(S.loc[h, 'p_normal ' + name]):<20}{pv(S.loc[h, 'p_boot ' + name])}")
    best_all = all(S.loc[h, "rmse_rw_bp"] <= min(S.loc[h, "rmse_m0_bp"], S.loc[h, "rmse_m1_bp"]) for h in a.horizons)
    L.append("Reading: " + ("the random walk has the lowest RMSE at EVERY horizon; " if best_all else "")
             + "Clark-West asks a different question (does x add anything beyond M0?), so both can be true.")

    # ------------------------------------------------------------------ 6. noise
    L.append(head("6. The noise term (Proposal 4.3.3 assumes it is Gaussian)"))
    nd = noise.noise_diagnostics(m_nb.fit.resid)
    cmp_ = noise.compare_noise(m_nb.fit, subset(m_nb.fit, ecm.nested_columns(m_nb.fit.names, "all")))
    L.append(f"Daily noise: M0 {cmp_['s_eps_m0_bp']:.2f} bp, M1 {cmp_['s_eps_m1_bp']:.2f} bp "
             f"(degrees-of-freedom-adjusted variance reduction from adding SOFR/FX: {cmp_['variance_reduction'] * 100:.1f}%)")
    for nm, r_ in nd.iterrows():
        val = f"{r_['value']:.3f}" if np.isfinite(r_["value"]) else f"p={pv(r_['p_value'])}"
        L.append(f"  {nm:<22}{val:<12}{'VIOLATED' if r_['violated'] else 'ok'}")
    L.append("Reading: fat tails and volatility clustering -> use bootstrapped noise in simulation and a conditional")
    L.append("(GARCH-filtered) EVT step; standardised residuals are saved for it.")

    # ------------------------------------------------------------------ 7. scenarios
    L.append(head(f"7. Scenarios: SORA in 1 year (5th / median / 95th pct, %), SOFR shock {a.shock_bp:.0f} bp, bootstrapped noise"))
    sc = simulate.make_scenarios(df["x"].iloc[-1], 252, a.shock_bp / 1e4)
    L.append(f"{'scenario':<10}{'no break':<26}{'level shift (exploratory)'}")
    neg = False
    for name, xp in sc.items():
        cells = []
        for model in (m_nb, m_br):
            q_ = np.percentile(simulate.simulate_ecm(model, df, xp, None, 10_000, seed, "bootstrap")[:, -1], [5, 50, 95]) * 100
            neg |= q_[0] < 0
            cells.append(f"{q_[0]:5.2f} / {q_[1]:5.2f} / {q_[2]:5.2f}")
        L.append(f"{name:<10}{cells[0]:<26}{cells[1]}")
    L.append(f"Start SORA {df['r'].iloc[-1] * 100:.2f}%, SOFR {df['x'].iloc[-1] * 100:.2f}%. SOFR and FX are conditioned on, not modelled.")
    L.append("! The linear Gaussian model allows negative rates" + (" (some percentiles above are negative)." if neg else "."))

    # ------------------------------------------------------------------ 8. robustness
    L.append(head("8. Robustness"))
    grid = robustness.spec_grid(df, tuple(sorted({0, 2, 5, p})), (0, 1), with_z)
    L.append("(a) Anchor exclusion test vs lag choice - classical/HAC p-values move a lot with the lags:")
    L.append(f"{'dr lags':<9}{'dx lags':<9}{'a':<10}{'kappa':<8}{'classical p':<13}{'HAC p'}")
    for _, r_ in grid.iterrows():
        L.append(f"{int(r_['p_dr_lags']):<9}{int(r_['q_dx_lags']):<9}{r_['a']:<10.4f}{r_['kappa']:<8.1f}{pv(r_['p_classical']):<13}{pv(r_['p_HAC'])}")
    starts = [d for d in ("2021-10-01", "2022-10-01", "2023-10-01", "2024-10-01") if pd.Timestamp(d) >= df.index[0] - pd.Timedelta(days=5)]
    sd = robustness.start_date_sensitivity(r_all, sofr, z, starts, p, a.q, gh_lags=p)
    L.append("\n(b) Start-date sensitivity (one rate cycle in the sample, so the window matters):")
    L.append(f"{'start':<12}{'n':<6}{'EG p (const)':<14}{'EG p (trend)':<14}{'GH stat':<9}{'break':<13}{'slope':<7}{'kappa'}")
    for _, r_ in sd.iterrows():
        L.append(f"{str(r_['start']):<12}{int(r_['n']):<6}{pv(r_['EG_p_const(BIC)']):<14}{pv(r_['EG_p_trend(BIC)']):<14}"
                 f"{r_['GH_C_stat']:<9.2f}{str(r_['GH_break']):<13}{r_['slope_r_on_x']:<7.2f}{r_['kappa']:.1f}")
    rw_ = robustness.race_window_grid(df, (250, 500, 750), tuple(a.horizons), with_z)
    L.append("\n(c) Race vs window length (RMSE, bp):")
    L.append(f"{'window':<9}{'h':<5}{'RW':<8}{'M0':<8}{'M1'}")
    for _, r_ in rw_.iterrows():
        L.append(f"{int(r_['window']):<9}{int(r_['h']):<5}{r_['rmse_rw_bp']:<8.1f}{r_['rmse_m0_bp']:<8.1f}{r_['rmse_m1_bp']:.1f}")

    # ------------------------------------------------------------------ bottom line
    L += ["\n" + "=" * W, "Bottom line", "=" * W]
    L.append(f"- No break: SORA and SOFR are {'cointegrated' if gate_nobreak else 'NOT shown to be cointegrated'} "
             f"(Engle-Granger BIC p = {pv(egc_bic)}; ECM bootstrap p = {pv(coint0['p_boot_t_left'])}); the anchor test does not reject "
             f"(bootstrap p = {pv(rows4[0][3]['p_boot_F'])}).")
    L.append(f"- Allowing one level shift ({break_date:%d %b %Y}): cointegration {'holds' if ghC['reject_5pct'] else 'does not hold'} "
             f"(Gregory-Hansen {ghC['stat']:.2f} vs 5% value {ghC['cv_5pct']:.2f}). Date is data-chosen: check it against MAS policy dates.")
    L.append(f"- Sensitive to the start date: EG p (const) from {pv(sd['EG_p_const(BIC)'].max())} to {pv(sd['EG_p_const(BIC)'].min())} across starts; "
             f"slope {sd['slope_r_on_x'].min():.2f} to {sd['slope_r_on_x'].max():.2f}.")
    L.append("- Race: " + ("random walk is best at every horizon." if best_all else "see table 5.")
             + f" Bootstrap Clark-West M1 vs M0 at {a.horizons[0]} day: p = {pv(S.loc[a.horizons[0], 'p_boot M1 vs M0'])}"
             + (f"; at {a.horizons[-1]} days: p = {pv(S.loc[a.horizons[-1], 'p_boot M1 vs M0'])}." if len(a.horizons) > 1 else "."))
    L.append("- Headline the data support: no STABLE level anchor over the full window. A SOFR anchor appears only when a regime")
    L.append("  change is allowed (data-chosen date) or the sample starts later, and its slope then varies a lot.")
    L.append("- Open decisions: anchor variable (forward points would make CIP hold by construction), SORA field definition.")
    print("\n".join(L))

    # ------------------------------------------------------------------ outputs
    m_nb.params_table().to_csv(out / "m1_ecm_params_no_break.csv")
    m_br.params_table().to_csv(out / "m1_ecm_params_break.csv")
    pd.concat({"no_break": st_nb, "break": st_br}).to_csv(out / "m1_structural.csv")
    eg.to_csv(out / "gate_engle_granger.csv")
    ur.to_csv(out / "gate_unit_roots.csv")
    pd.DataFrame([{k: v for k, v in g.items() if k not in ("break_date",)} | {"break_date": g["break_date"]}
                  for g in (ghC, ghS)]).to_csv(out / "gate_gregory_hansen.csv", index=False)
    S.to_csv(out / "race_summary.csv")
    for h, f_ in res["forecasts"].items():
        f_.to_csv(out / f"race_forecasts_h{h}.csv")
    pd.DataFrame({"resid": m_nb.fit.resid, "std_resid": noise.std_resid(m_nb.fit)}).to_csv(out / "m1_residuals.csv")
    nd.to_csv(out / "noise_checks.csv")
    grid.to_csv(out / "robust_spec_grid.csv", index=False)
    sd.to_csv(out / "robust_start_dates.csv", index=False)
    rw_.to_csv(out / "robust_race_windows.csv", index=False)
    provenance.write(out / "run_info.json", provenance.run_info(df, vars(a), {"p_used": p, "break_date": str(break_date.date()),
                                                                               "with_z": with_z}))
    print(f"\nFiles saved in: {out}")


if __name__ == "__main__":
    main()
