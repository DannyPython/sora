"""Run M0 (classical Vasicek on SORA): Block 1 data, Block 2 gate, Block 3 calibration,
M0 side of Block 7 (rolling out-of-sample forecast). Every result comes from this script.

    python run_m0.py [--start 2024-01-01] [--end 2026-10-08] [--window 250] [--seed 2027]
    python run_m0.py --lags 0          # pure AR(1) gate (the original), instead of the BIC-augmented one
    python run_m0.py --n-boot 99       # faster gate while testing
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ssef import data, gate, m0, provenance, race
from ssef import report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--dt", type=float, default=m0.DT_DEFAULT, help="year fraction per observation")
    ap.add_argument("--window", type=int, default=250, help="rolling training window (observations)")
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--lags", default="bic", help="lagged differences in the gate regression: an integer, 'bic' (default) or 0 for the pure AR(1)")
    ap.add_argument("--n-boot", type=int, default=299, help="bootstrap draws per grid point in the gate")
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 20, 60], help="horizons (days) of the M0 vs random-walk table")
    ap.add_argument("--out", default="results/m0")
    ap.add_argument("--verbose", action="store_true", help="also print the raw statistical tables")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.float_format", lambda v: f"{v:.6g}")

    # Block 1: data
    r = data.load_sora(start=a.start, end=a.end)

    # Block 2: gate
    tests = gate.unit_root_tests(r)
    if str(a.lags) == "0":
        ag = gate.ar1_gate(r, n_boot=a.n_boot, seed=a.seed)
    else:
        ag = gate.adf_gate(r, p=None if str(a.lags).lower() == "bic" else int(a.lags), n_boot=a.n_boot, seed=a.seed)

    # Block 3: calibration
    fit = m0.calibrate(r, dt=a.dt)
    tab = fit.params_table()

    # Block 7 (M0 side): rolling one-step forecast
    fc = m0.rolling_forecast(r, window=a.window)
    mh = race.m0_vs_rw(r, tuple(a.horizons), a.window)

    # Scenario-MC building block (Block 8): 1-year-ahead simulated SORA under M0
    sim_q = None
    if fit.valid:
        paths = fit.simulate(r.iloc[-1], n_steps=252, n_paths=10_000, seed=a.seed)
        sim_q = np.percentile(paths[:, -1], [5, 50, 95])

    print(report.format_report(r, tests, fit, fc, a.window, sim_q, a.seed, ag, mh))
    if a.verbose:
        print("\n== Raw tables ==\nunit-root tests:\n" + tests.to_string()
              + f"\n\nAR(1) / structural parameters (n = {fit.n}):\n" + tab.to_string())

    # Outputs + provenance (Blueprint 8)
    tab.to_csv(out / "m0_params.csv")
    tests.to_csv(out / "m0_unit_root.csv")
    pd.DataFrame([{"lags": ag.lags, "c1_ols": ag.c1_ols, "c1_bias_corrected": ag.c1_bc, "ci_low": ag.ci[0],
                   "ci_high": ag.ci[1], "reverts": ag.reverts}]).to_csv(out / "m0_gate.csv", index=False)
    mh.to_csv(out / "m0_multi_horizon.csv")
    pd.DataFrame({"resid": fit.resid, "std_resid": fit.std_resid}).to_csv(out / "m0_residuals.csv")
    fc.to_csv(out / "m0_rolling_forecast.csv")
    info = provenance.run_info(r.to_frame("r"), vars(a), {"gate_lags": ag.lags})
    provenance.write(out / "run_info.json", info)
    print(f"\nFiles saved in: {out}")


if __name__ == "__main__":
    main()