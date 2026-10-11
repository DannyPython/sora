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

from ssef import data, gate, m0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--dt", type=float, default=m0.DT_DEFAULT, help="year fraction per observation")
    ap.add_argument("--window", type=int, default=250, help="rolling training window (observations)")
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--out", default="results/m0")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.float_format", lambda v: f"{v:.6g}")

    # Block 1: data
    r = data.load_sora(start=a.start, end=a.end)
    print(f"SORA: {len(r)} obs, {r.index[0].date()} -> {r.index[-1].date()}, dt = {a.dt:.6f} yr/step")

    # Block 2: gate
    tests = gate.unit_root_tests(r)
    print("\n== Block 2: unit-root tests on SORA (level / first difference) ==")
    print(tests.to_string())
    print("order of integration:", gate.integration_order(tests))

    # Block 3: calibration
    fit = m0.calibrate(r, dt=a.dt)
    tab = fit.params_table()
    print("\n== Block 3: M0 calibration (AR(1) by OLS, delta-method 95% CI) ==")
    print(tab.to_string())
    print(f"n = {fit.n}, c1 in (0,1): {fit.valid}")
    if fit.valid:
        print(f"half-life of a shock: {fit.half_life_steps:.0f} steps ({fit.half_life_steps / 252:.2f} yr)"
              f"; theta = {fit.theta * 100:.3f}%; sigma = {fit.sigma * 1e4:.0f} bp/sqrt(yr)")
        if not tests.loc["level", "adf_reject_unit_root"] or fit.c1 > 0.98:
            print("WARNING: near-unit-root series. OLS c1 is biased downward and the Delta-method CI for\n"
                  "kappa/theta is unreliable (non-standard distribution). Treat the point estimates as\n"
                  "descriptive and see Blueprint 5.3 / 10 (fallback).")
    else:
        print("WARNING: c1 >= 1 or <= 0: SORA does not revert; structural parameters undefined (Sec. 5.3).")

    # Block 7 (M0 side): rolling one-step forecast
    fc = m0.rolling_forecast(r, window=a.window)
    print(f"\n== Block 7 (M0 side): rolling-window one-step forecast, window = {a.window}, "
          f"{len(fc)} forecasts ==")
    print(f"RMSE M0 (Vasicek AR1): {m0.rmse(fc) * 1e4:.3f} bp")
    print(f"RMSE random walk     : {m0.rmse(fc, 'rw_forecast') * 1e4:.3f} bp")

    # Scenario-MC building block (Block 8): 1-year-ahead simulated SORA under M0
    if fit.valid:
        paths = fit.simulate(r.iloc[-1], n_steps=252, n_paths=10_000, seed=a.seed)
        q = np.percentile(paths[:, -1], [5, 50, 95])
        print(f"\nM0 simulated SORA in 1y from {r.iloc[-1] * 100:.3f}% (seed {a.seed}): "
              f"5/50/95% = {q[0] * 100:.3f} / {q[1] * 100:.3f} / {q[2] * 100:.3f} %")

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
    print(f"\nwritten to {out}/")


if __name__ == "__main__":
    main()
