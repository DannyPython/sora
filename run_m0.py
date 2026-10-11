"""Run M0 (classical Vasicek on SORA): Block 1 data, Block 2 gate, Block 3 calibration,
M0 side of Block 7 (rolling out-of-sample forecast). Every result comes from this script.

    python run_m0.py [--start 2024-01-01] [--end 2026-10-08] [--window 250] [--seed 2027]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import statsmodels

from ssef import data, gate, m0, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--dt", type=float, default=m0.DT_DEFAULT, help="year fraction per observation")
    ap.add_argument("--window", type=int, default=250, help="rolling training window (observations)")
    ap.add_argument("--seed", type=int, default=2027)
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

    # Block 3: calibration
    fit = m0.calibrate(r, dt=a.dt)
    tab = fit.params_table()

    # Block 7 (M0 side): rolling one-step forecast
    fc = m0.rolling_forecast(r, window=a.window)

    # Scenario-MC building block (Block 8): 1-year-ahead simulated SORA under M0
    sim_q = None
    if fit.valid:
        paths = fit.simulate(r.iloc[-1], n_steps=252, n_paths=10_000, seed=a.seed)
        sim_q = np.percentile(paths[:, -1], [5, 50, 95])

    print(report.format_report(r, tests, fit, fc, a.window, sim_q, a.seed))
    if a.verbose:
        print("\n== Raw tables ==\nunit-root tests:\n" + tests.to_string()
              + f"\n\nAR(1) / structural parameters (n = {fit.n}):\n" + tab.to_string())

    # Outputs + provenance (Blueprint 8)
    tab.to_csv(out / "m0_params.csv")
    tests.to_csv(out / "m0_unit_root.csv")
    pd.DataFrame({"resid": fit.resid, "std_resid": fit.std_resid}).to_csv(out / "m0_residuals.csv")
    fc.to_csv(out / "m0_rolling_forecast.csv")
    info = {
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_file": str(data.SORA_CSV.name),
        "data_sha256": hashlib.sha256(data.SORA_CSV.read_bytes()).hexdigest(),
        "n_obs": len(r), "first_date": str(r.index[0].date()), "last_date": str(r.index[-1].date()),
        "dt": a.dt, "window": a.window, "seed": a.seed,
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "statsmodels": statsmodels.__version__,
    }
    (out / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"\nFiles saved in: {out}")


if __name__ == "__main__":
    main()