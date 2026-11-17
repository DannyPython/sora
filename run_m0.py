"""Run M0 (classical Vasicek on SORA): Block 1 data, Block 2 gate, Block 3 calibration,
M0 side of Block 7 (rolling out-of-sample forecast). Computes everything, writes every table + headline.csv +
run_info.json to results/m0/, and prints the headline numbers. The plain-language report is optional (--report).

    python run_m0.py [--start 2024-01-01] [--end 2026-10-08] [--window 500] [--seed 2027]
    python run_m0.py --report          # also print the long plain-language report
    python run_m0.py --lags 0          # the Proposal's plain AR(1) instead of the BIC-augmented M0
    python run_m0.py --n-boot 99       # faster gate while testing
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ssef import gate, loaders, m0, provenance, race, report


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--dt", type=float, default=loaders.DT, help="year fraction per observation")
    ap.add_argument("--window", type=int, default=500, help="rolling training window (observations), same default as run_m1")
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--lags", default="bic", help="lagged differences in M0 (gate, calibration and forecasts): an integer, 'bic' (default) or 0 for the plain AR(1)")
    ap.add_argument("--n-boot", type=int, default=299, help="bootstrap draws per grid point in the gate")
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 20, 60], help="horizons (days) of the M0 vs random-walk table; 1 is always added")
    ap.add_argument("--report", action="store_true", help="also print the long plain-language report")
    ap.add_argument("--out", default="results/m0")
    return ap.parse_args(argv)


def compute(a) -> dict:
    """Run every step. Returns plain data; no formatting."""
    r = loaders.load_sora(start=a.start, end=a.end)                                         # Block 1: data
    tests = gate.unit_root_tests(r)                                                          # Block 2: gate
    ag = gate.adf_gate(r, p=None if str(a.lags).lower() == "bic" else int(a.lags), n_boot=a.n_boot, seed=a.seed)
    fit = m0.calibrate(r, dt=a.dt, p=ag.lags)                                                # Block 3 (same lags as the gate)
    mh = race.m0_vs_rw(r, tuple(sorted({1, *a.horizons})), a.window, p=ag.lags)              # Block 7 (M0 side)
    sim_q = None                                                                             # Block 8 building block
    if fit.valid:
        paths = fit.simulate(r.iloc[-1], n_steps=252, n_paths=10_000, seed=a.seed)
        sim_q = np.percentile(paths[:, -1], [5, 50, 95])
    return {"r": r, "tests": tests, "ag": ag, "fit": fit, "tab": fit.params_table(), "mh": mh, "sim_q": sim_q}


def headline(R: dict, a) -> pd.DataFrame:
    """The numbers that matter, one per row (section, metric, value). Units are in the metric name."""
    r, ag, fit, mh, sim_q = R["r"], R["ag"], R["fit"], R["mh"], R["sim_q"]
    k_lo, k_hi = report._kappa_range(ag, a.dt)
    nan = float("nan")
    rows = [
        ("data", "n_obs", len(r)),
        ("data", "start", str(r.index[0].date())),
        ("data", "end", str(r.index[-1].date())),
        ("gate", "lags", ag.lags),
        ("gate", "c1_ols", ag.c1_ols),
        ("gate", "c1_bias_corrected", ag.c1_bc),
        ("gate", "c1_ci_low", ag.ci[0]),
        ("gate", "c1_ci_high", ag.ci[1]),
        ("gate", "passed", bool(ag.reverts)),
        ("persistence", "kappa_per_yr_bias_corrected", ag.kappa_bc()),
        ("persistence", "kappa_per_yr_ci_low", k_lo),
        ("persistence", "kappa_per_yr_ci_high", k_hi),
        ("ols_descriptive", "kappa_per_yr", fit.kappa if fit.valid else nan),
        ("ols_descriptive", "theta_pct", fit.theta * 100 if fit.valid else nan),
        ("ols_descriptive", "sigma_bp_per_sqrt_yr", fit.sigma * 1e4 if fit.valid else nan),
    ]
    for h, row in mh.iterrows():
        rows += [("race", f"rmse_bp_rw_h{h}", row["rmse_rw_bp"]),
                 ("race", f"rmse_bp_m0_h{h}", row["rmse_m0_bp"]),
                 ("race", f"m0_vs_rw_pct_h{h}", row["m0_vs_rw_pct"])]
    for q, v in zip((5, 50, 95), sim_q if sim_q is not None else (nan,) * 3):
        rows.append(("simulation_1y", f"sora_p{q}_pct", v * 100))
    return pd.DataFrame(rows, columns=["section", "metric", "value"])


def save(R: dict, a, out: Path) -> pd.DataFrame:
    """Write every result to `out`. Returns the headline table."""
    h = headline(R, a)
    h.to_csv(out / "headline.csv", index=False)
    ag, fit = R["ag"], R["fit"]
    R["tab"].to_csv(out / "m0_params.csv")
    R["tests"].to_csv(out / "m0_unit_root.csv")
    pd.DataFrame([{"lags": ag.lags, "c1_ols": ag.c1_ols, "c1_bias_corrected": ag.c1_bc, "ci_low": ag.ci[0],
                   "ci_high": ag.ci[1], "reverts": ag.reverts}]).to_csv(out / "m0_gate.csv", index=False)
    R["mh"].to_csv(out / "m0_multi_horizon.csv")
    pd.DataFrame({"resid": fit.resid, "std_resid": fit.std_resid}).to_csv(out / "m0_residuals.csv")
    if R["sim_q"] is not None:
        pd.DataFrame([dict(zip(("p5_pct", "p50_pct", "p95_pct"), R["sim_q"] * 100))]).to_csv(out / "m0_simulation_1y.csv", index=False)
    provenance.write(out / "run_info.json", provenance.run_info(R["r"].to_frame("r"), vars(a), {"gate_lags": ag.lags}))
    return h


def main(argv=None):
    a = parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    R = compute(a)
    h = save(R, a, out)
    print(report.format_headline(h, "M0 (classical Vasicek) on SORA - headline numbers"))
    if a.report:
        print("\n" + report.format_report(R["r"], R["tests"], R["fit"], R["mh"], a.window, R["sim_q"], a.seed, R["ag"]))
    print(f"\nAll tables saved in: {out}")


if __name__ == "__main__":
    main()
