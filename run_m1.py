"""M1 pipeline: data checks -> gate -> ECM -> valid nested test -> out-of-sample race -> noise -> scenarios ->
robustness. Computes everything, writes every table + headline.csv + run_info.json to results/m1/, and prints
the headline numbers. The plain-language report is optional (--report); figures are made from the saved files.

    python run_m1.py                  # full run (about 1-2 minutes)
    python run_m1.py --fast           # fewer bootstrap draws (about 15 seconds), for checking the code
    python run_m1.py --report         # also print the long plain-language report
    python run_m1.py --p 2 --q 0      # fix the lag of dr instead of the BIC choice
    python run_m1.py --no-fx          # drop the FX term (Blueprint Sec. 10 fallback)
    python run_m1.py --sofr-lag same-day --break-date 2025-04-14
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ssef import cointegration as C, restricted_ecm
from ssef import loaders, nested, noise, provenance, race, report, robustness, simulate
from ssef.ols import exclusion_f, exclusion_wald_hac, subset


def parse_args(argv=None):
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
    ap.add_argument("--report", action="store_true", help="also print the long plain-language report")
    ap.add_argument("--out", default="results/m1")
    a = ap.parse_args(argv)
    if a.fast:
        a.B, a.race_B, a.gh_sim = 99, 99, 40
    return a


def compute(a) -> dict:
    """Run every step. Returns plain data (DataFrames, Series, scalars, fitted models); no formatting."""
    seed = a.seed
    R = {}

    # ------------------------------------------------------------------ data
    r_all = loaders.load_sora(start=a.start, end=a.end)
    sofr = loaders.load_sofr()
    fx_ok = (not a.no_fx) and loaders.FX_CSV.exists()
    z = loaders.make_z(loaders.load_usdsgd()) if fx_ok else None
    df = loaders.build_dataset(r_all, sofr, z, strict=(a.sofr_lag == "strict"))
    with_z = "z" in df
    lag_tab = restricted_ecm.select_lags(df, a.pmax, a.q, with_z)
    p = int(lag_tab["bic"].idxmin()) if str(a.p).lower() == "bic" else int(a.p)
    R.update(df=df, with_z=with_z, lag_tab=lag_tab, p=p, chk=loaders.data_checks(df))

    # ------------------------------------------------------------------ gate
    R["ur"] = C.unit_root_table(df)
    eg = C.engle_granger(df["r"], df["x"], lags=[0, 2, 5, 8])
    R["eg"] = eg
    R["egc_bic"] = float(eg.loc["c"].xs("BIC", level="rule")["p_value"].iloc[0])
    R["ghC"] = C.gregory_hansen(df["r"], df["x"], "C", p, n_sim=a.gh_sim, seed=seed, dates=df.index)
    R["ghS"] = C.gregory_hansen(df["r"], df["x"], "C/S", p, n_sim=a.gh_sim, seed=seed, dates=df.index)
    break_date = pd.Timestamp(a.break_date) if a.break_date else R["ghC"]["break_date"]
    R["break_date"] = break_date
    R["ss"] = C.slope_stability(df, windows=(500,))
    coint = {lab: nested.cointegration_ecm_test(df, p, a.q, with_z, a.B, seed=seed, break_date=bd)
             for lab, bd in (("no_break", None), ("break", break_date))}
    R["coint"] = pd.DataFrame({lab: {k: c_[k] for k in ("t", "p_boot_t_left", "cv5_t_left", "F", "p_boot_F", "cv5_F")}
                               for lab, c_ in coint.items()}).T.rename_axis("spec")
    R["gate_nobreak"] = bool(R["egc_bic"] < 0.05 and R["coint"].loc["no_break", "p_boot_t_left"] < 0.05)

    # ------------------------------------------------------------------ ECM estimates
    m_nb = restricted_ecm.fit_ecm(df, p, a.q, with_z)
    m_br = restricted_ecm.fit_ecm(df, p, a.q, with_z, break_date=break_date)
    R.update(m_nb=m_nb, m_br=m_br, st_nb=m_nb.structural_table(), st_br=m_br.structural_table())

    # ------------------------------------------------------------------ nested test: does M1 beat M0?
    rows = {}
    for lab, model, bd in (("no_break", m_nb, None), ("break", m_br, break_date)):
        drop = restricted_ecm.nested_columns(model.fit.names)
        fF = exclusion_f(model.fit, subset(model.fit, drop))
        hh = exclusion_wald_hac(model.fit, drop)
        bt = nested.anchor_test(df, p, a.q, with_z, "unit_root", a.B, seed=seed, break_date=bd)
        bf = nested.anchor_test(df, p, a.q, with_z, "fitted", a.B, seed=seed, break_date=bd)
        rows[lab] = {"F": fF["F"], "p_classical": fF["p_classical"], "p_hac": hh["p_hac_F"],
                     "p_boot": bt["p_boot_F"], "p_boot_fitted_a": bf["p_boot_F"]}
    R["nested"] = pd.DataFrame(rows).T.rename_axis("spec")

    # ------------------------------------------------------------------ race
    R["race"] = race.run_race(df, tuple(a.horizons), a.window, with_z, B=a.race_B, null="unit_root", p=p, q=a.q, seed=seed)
    S = R["race"]["summary"]
    R["S"] = S
    R["rw_best_all"] = bool(all(S.loc[h, "rmse_rw_bp"] <= min(S.loc[h, "rmse_m0_bp"], S.loc[h, "rmse_m1_bp"])
                                for h in a.horizons))

    # ------------------------------------------------------------------ noise
    R["nd"] = noise.noise_diagnostics(m_nb.fit.resid)
    R["noise_cmp"] = noise.compare_noise(m_nb.fit, subset(m_nb.fit, restricted_ecm.nested_columns(m_nb.fit.names)))

    # ------------------------------------------------------------------ scenarios (SORA in 1 year, bootstrapped noise)
    sc = []
    for name, xp in simulate.make_scenarios(df["x"].iloc[-1], 252, a.shock_bp / 1e4).items():
        for lab, model in (("no_break", m_nb), ("break", m_br)):
            q_ = np.percentile(simulate.simulate_ecm(model, df, xp, None, 10_000, seed, "bootstrap")[:, -1], [5, 50, 95]) * 100
            sc.append({"scenario": name, "model": lab, "p5_pct": q_[0], "p50_pct": q_[1], "p95_pct": q_[2]})
    R["scenarios"] = pd.DataFrame(sc)

    # ------------------------------------------------------------------ robustness
    R["grid"] = robustness.spec_grid(df, tuple(sorted({0, 2, 5, p})), (0, 1), with_z)
    starts = [d for d in ("2021-10-01", "2022-10-01", "2023-10-01", "2024-10-01")
              if pd.Timestamp(d) >= df.index[0] - pd.Timedelta(days=5)]
    R["start_dates"] = robustness.start_date_sensitivity(r_all, sofr, z, starts, p, a.q, gh_lags=p)
    R["race_windows"] = robustness.race_window_grid(df, (250, 500, 750), tuple(a.horizons), with_z, p, a.q)
    return R


def _est(st: pd.DataFrame, k: str, scale: float = 1.0) -> float:
    return float(st.loc[k, "estimate"]) * scale if k in st.index else float("nan")


def headline(R: dict, a) -> pd.DataFrame:
    """The numbers that matter, one per row (section, metric, value). Units are in the metric name."""
    df, S, ghC, nst, roll = R["df"], R["S"], R["ghC"], R["nested"], R["ss"]["rolling"][500]
    rows = [
        ("data", "n_obs", len(df)),
        ("data", "start", str(df.index[0].date())),
        ("data", "end", str(df.index[-1].date())),
        ("data", "sofr_timing", a.sofr_lag),
        ("data", "fx_term", R["with_z"]),
        ("data", "dr_lags", R["p"]),
        ("gate", "eg_p_const_bic", R["egc_bic"]),
        ("gate", "ecm_coint_p_no_break", R["coint"].loc["no_break", "p_boot_t_left"]),
        ("gate", "passed_no_break", R["gate_nobreak"]),
        ("gate", "gh_stat", ghC["stat"]),
        ("gate", "gh_cv_5pct", ghC["cv_5pct"]),
        ("gate", "gh_p", ghC["p_value"]),
        ("gate", "passed_one_break", bool(ghC["reject_5pct"])),
        ("gate", "break_date", str(R["break_date"].date())),
        ("gate", "break_date_source", "user" if a.break_date else "gregory_hansen"),
        ("gate", "slope_full_sample", R["ss"]["full"]),
        ("gate", "slope_rolling500_min", roll.min()),
        ("gate", "slope_rolling500_max", roll.max()),
        ("ecm", "kappa_per_yr_no_break", _est(R["st_nb"], "kappa")),
        ("ecm", "kappa_per_yr_break", _est(R["st_br"], "kappa")),
        ("ecm", "beta1_no_break", _est(R["st_nb"], "beta1")),
        ("ecm", "beta1_break", _est(R["st_br"], "beta1")),
        ("ecm", "level_shift_pp", _est(R["st_br"], "level_shift", 100)),
        ("ecm", "s_eps_bp_no_break", R["m_nb"].s_eps * 1e4),
        ("m1_vs_m0", "nested_boot_p_no_break", nst.loc["no_break", "p_boot"]),
        ("m1_vs_m0", "nested_boot_p_break", nst.loc["break", "p_boot"]),
        ("m1_vs_m0", "noise_variance_reduction_pct", R["noise_cmp"]["variance_reduction"] * 100),
    ]
    for h in a.horizons:
        rows += [("race", f"rmse_bp_rw_h{h}", S.loc[h, "rmse_rw_bp"]),
                 ("race", f"rmse_bp_m0_h{h}", S.loc[h, "rmse_m0_bp"]),
                 ("race", f"rmse_bp_m1_h{h}", S.loc[h, "rmse_m1_bp"]),
                 ("race", f"cw_boot_p_m1_vs_m0_h{h}", S.loc[h, "p_boot M1 vs M0"])]
    rows.append(("race", "rw_best_at_every_horizon", R["rw_best_all"]))
    return pd.DataFrame(rows, columns=["section", "metric", "value"])


def save(R: dict, a, out: Path) -> pd.DataFrame:
    """Write every result to `out`. Returns the headline table."""
    h = headline(R, a)
    h.to_csv(out / "headline.csv", index=False)
    chk = R["chk"]
    # data
    chk["spread_pp"].rename_axis(["year", "half"]).rename("spread_pp").to_csv(out / "data_spread_halfyear.csv")
    pd.Series({"dr_sd_bp": chk["dr_sd_bp"], **{f"dr_autocorr_lag{k}": v for k, v in enumerate(chk["dr_autocorr"], 1)},
               "n_jump_40bp": chk["n_jump_40bp"], "max_abs_jump_bp": chk["max_abs_jump_bp"]},
              name="value").rename_axis("metric").to_csv(out / "data_checks.csv")
    chk["largest_jumps"].rename("abs_change_bp").to_csv(out / "data_largest_jumps.csv")
    R["lag_tab"].to_csv(out / "m1_lag_selection.csv")
    # gate
    R["ur"].to_csv(out / "gate_unit_roots.csv")
    R["eg"].to_csv(out / "gate_engle_granger.csv")
    pd.DataFrame([{k: v for k, v in g.items() if k not in ("break_date",)} | {"break_date": g["break_date"]}
                  for g in (R["ghC"], R["ghS"])]).to_csv(out / "gate_gregory_hansen.csv", index=False)
    R["coint"].to_csv(out / "gate_ecm_cointegration.csv")
    R["ss"]["sub_periods"].to_csv(out / "gate_slope_subperiods.csv", index=False)
    R["ss"]["rolling"][500].rename("slope").to_csv(out / "gate_slope_rolling500.csv")
    # ECM
    R["m_nb"].params_table().to_csv(out / "m1_ecm_params_no_break.csv")
    R["m_br"].params_table().to_csv(out / "m1_ecm_params_break.csv")
    pd.concat({"no_break": R["st_nb"], "break": R["st_br"]}).to_csv(out / "m1_structural.csv")
    # M1 vs M0
    R["nested"].to_csv(out / "nested_m1_vs_m0.csv")
    # race
    R["S"].to_csv(out / "race_summary.csv")
    for hz, f_ in R["race"]["forecasts"].items():
        f_.to_csv(out / f"race_forecasts_h{hz}.csv")
    # noise
    m_nb = R["m_nb"]
    pd.DataFrame({"resid": m_nb.fit.resid, "std_resid": noise.std_resid(m_nb.fit)}).to_csv(out / "m1_residuals.csv")
    R["nd"].to_csv(out / "noise_checks.csv")
    pd.Series(R["noise_cmp"], name="value").rename_axis("metric").to_csv(out / "noise_m0_vs_m1.csv")
    # scenarios
    R["scenarios"].to_csv(out / "scenarios_1y.csv", index=False)
    # robustness
    R["grid"].to_csv(out / "robust_spec_grid.csv", index=False)
    R["start_dates"].to_csv(out / "robust_start_dates.csv", index=False)
    R["race_windows"].to_csv(out / "robust_race_windows.csv", index=False)
    provenance.write(out / "run_info.json", provenance.run_info(
        R["df"], vars(a), {"p_used": R["p"], "break_date": str(R["break_date"].date()), "with_z": R["with_z"]}))
    return h


def main(argv=None):
    a = parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    R = compute(a)
    h = save(R, a, out)
    print(report.format_headline(h, "M1 (SOFR/FX-anchored Vasicek, ECM form) on SORA - headline numbers"))
    if a.report:
        print("\n" + report.format_m1_report(R, a))
    print(f"\nAll tables saved in: {out}")


if __name__ == "__main__":
    main()
