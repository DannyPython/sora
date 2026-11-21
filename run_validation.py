"""Size and power of the project's tests on simulated data with known truth (Proposal 4.4, 4.7; ssef/validate_tests.py).
Opt-in: it is NOT part of run_m0.py / run_m1.py. Writes every table + headline.csv + run_info.json to
results/validation/ and prints the headline numbers.

    python run_validation.py                       # 400 datasets per process, 99 bootstrap draws (about 15 minutes on 4 cores)
    python run_validation.py --fast                # 100 datasets, 49 draws (about 3 minutes), for checking the code
    python run_validation.py --workers 1           # no multiprocessing
    python run_validation.py --anchor 0.5:est 1:-0.05   # your own M1-true cases, beta1:speed ('est' = fitted speed)
    python run_validation.py --no-break            # skip the level-shift case
    python run_validation.py --no-ecm-gate         # skip the ECM bootstrap gate (saves about a third of the time)

The processes are built from the fitted M0 / M1 of run_m1.py (same data, lags, break date), so the answer is about
THIS sample size and THIS speed of reversion. Change --anchor to ask how fast SORA would have to adjust for a
real anchor to be detected.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ssef import cointegration as C
from ssef import loaders, provenance
from ssef import restricted_ecm as E
from ssef import validate_tests as vt
from ssef.ols import subset


def parse_anchor(s: str):
    beta, _, speed = s.partition(":")
    return float(beta), (None if speed in ("", "est") else float(speed))


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--p", default="bic", help="lags of dr: an integer or 'bic' (as in run_m1)")
    ap.add_argument("--pmax", type=int, default=10)
    ap.add_argument("--sims", type=int, default=400, help="simulated datasets per process")
    ap.add_argument("--B", type=int, default=99, help="bootstrap draws inside each test")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--anchor", nargs="+", default=["0.25:est", "1:est", "1:-0.03"],
                    help="M1-true cases as beta1:speed, speed 'est' = the fitted speed of adjustment")
    ap.add_argument("--break-date", default=None, help="level-shift date (default: Gregory-Hansen date of the data)")
    ap.add_argument("--no-break", action="store_true")
    ap.add_argument("--no-ecm-gate", action="store_true")
    ap.add_argument("--null", choices=["unit_root", "fitted"], default="unit_root")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default="results/validation")
    a = ap.parse_args(argv)
    if a.fast:
        a.sims, a.B = 100, 49
    return a


def compute(a) -> dict:
    r = loaders.load_sora(start=a.start, end=a.end)
    df = loaders.build_dataset(r, loaders.load_sofr(), loaders.make_z(loaders.load_usdsgd()))
    p = int(E.select_lags(df, a.pmax, 0, True)["bic"].idxmin()) if str(a.p).lower() == "bic" else int(a.p)
    m1 = E.fit_ecm(df, p, 0, True)
    m0 = subset(m1.fit, E.nested_columns(m1.fit.names))                 # M0 = M1 minus SOFR / FX, same sample
    bd, shift = None, 0.0
    if not a.no_break:
        bd = pd.Timestamp(a.break_date) if a.break_date else C.gregory_hansen(df["r"], df["x"], "C", p, n_sim=0, dates=df.index)["break_date"]
        shift = E.fit_ecm(df, p, 0, True, break_date=bd).level_shift
    dgps = vt.make_dgps(df, m1, m0, [parse_anchor(s) for s in a.anchor], bd, shift)
    t0 = time.time()
    raw = vt.run_mc(df, p, dgps, m1, m0, a.sims, a.B, a.seed, a.workers, a.null, not a.no_ecm_gate)
    table = vt.rejection_table(raw, a.alpha)
    return {"df": df, "p": p, "m1": m1, "dgps": dgps, "raw": raw, "table": table, "wide": vt.wide(table),
            "seconds": time.time() - t0, "break_date": bd, "level_shift": shift}


def headline(R: dict, a) -> pd.DataFrame:
    """The numbers that matter, one per row (section, metric, value)."""
    t, df = R["table"], R["df"]
    rows = [("data", "n_obs", len(df)), ("data", "dr_lags", R["p"]), ("data", "sims_per_process", a.sims),
            ("data", "bootstrap_draws", a.B), ("data", "alpha", a.alpha), ("data", "fitted_speed_a", R["m1"].a),
            ("data", "fitted_kappa_per_year", R["m1"].kappa), ("data", "fitted_beta1", R["m1"].beta1)]
    for d in R["dgps"]:
        rows.append(("process", d.name, d.note))
    for _, s in t[t["test"] != "c1_in_(0,1)"].iterrows():
        rows.append((f"{s['truth']}", f"{s['dgp']} | {vt.LABELS[s['test']]}",
                     f"{s['rate']:.3f} [{s['ci_low']:.3f}, {s['ci_high']:.3f}] {s['reading']}"))
    for _, s in t[t["test"] == "c1_in_(0,1)"].iterrows():
        rows.append(("c1_check", f"{s['dgp']} | share of fitted c1 in (0,1)", f"{s['rate']:.3f}"))
    return pd.DataFrame(rows, columns=["section", "metric", "value"])


def main(argv=None):
    a = parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    R = compute(a)
    hl = headline(R, a)
    pd.set_option("display.width", 200, "display.max_columns", 20, "display.float_format", lambda v: f"{v:.3f}")
    print("=" * 78)
    print(" Size and power of the tests on simulated data with known truth")
    print("=" * 78)
    print(f"{len(R['df'])} daily obs, {R['p']} lags of d r; fitted a = {R['m1'].a:.4f} (kappa {R['m1'].kappa:.2f}/yr), "
          f"fitted beta1 = {R['m1'].beta1:.2f}.")
    print(f"{a.sims} datasets per process, {a.B} bootstrap draws per test, alpha = {a.alpha}; {R['seconds']:.0f}s.\n")
    for d in R["dgps"]:
        print(f"  {d.name:<22}{'H0 true ' if d.h0_true else 'H0 false'}  {d.note}")
    print("\nShare of simulated datasets with p < alpha. Where the test's null is true this is its SIZE (should be alpha; for the")
    print("gates a rejection is a FALSE PASS); where it is false it is POWER. CUSUM's null is stability, false only in m0_break.")
    print(R["wide"].rename(columns=vt.LABELS).to_string())
    over = R["table"][R["table"]["reading"].isin(["oversized", "false pass"])]
    if len(over):
        print("\nOversized or false pass (interval above alpha): " + "; ".join(f"{r_['dgp']} | {vt.LABELS[r_['test']]} {r_['rate']:.1%}" for _, r_ in over.iterrows()))
    c1 = R["table"][R["table"]["test"] == "c1_in_(0,1)"]
    print(f"Share of fitted c1 inside (0,1) when r is a random walk: {float(c1[c1['dgp'] == 'spurious']['rate'].iloc[0]):.1%} "
          "(the c1 check cannot flag a unit root by itself).")
    print("Reading: a 'fail to reject' is informative only where the H0-false rows show high power; the speed of reversion matters.")

    R["raw"].to_csv(out / "validation_raw.csv", index=False)
    R["table"].to_csv(out / "validation_rejection.csv", index=False)
    R["wide"].to_csv(out / "validation_rejection_wide.csv")
    hl.to_csv(out / "headline.csv", index=False)
    provenance.write(out / "run_info.json", provenance.run_info(R["df"], vars(a), {"p_used": R["p"], "break_date": str(R["break_date"])}))
    print(f"\nFiles saved in: {out}")
    return R


if __name__ == "__main__":
    main()
