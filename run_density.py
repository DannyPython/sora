"""Predictive-density validation of RW, M0 and M1 on held-out data (Proposal 4.7; ssef/density.py). Writes every table +
headline.csv + run_info.json to results/density/ and prints the headline numbers (the long report is optional).

    python run_density.py                    # 1-day horizon (a few seconds)
    python run_density.py --horizons 1 20    # also 20 days (non-overlapping origins: only ~37 points)
    python run_density.py --report           # also print the plain-language report
    python run_density.py --p 2 --no-fx      # same switches as run_m1.py / run_backtest.py

The predictive distributions are those of run_backtest.py (same models, data, lags, rolling window and origins). Where
the VaR backtest asks "how often is the 99% VaR breached?", this asks "is the whole predictive distribution right?"
using every held-out point: PIT histogram (results/density/pit_histogram_h1.csv), KS, Berkowitz, bias, width, tails.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ssef import density, loaders, provenance, report
from ssef import restricted_ecm as E


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--p", default="bic", help="lags of dr: an integer or 'bic' (as in run_m1)")
    ap.add_argument("--pmax", type=int, default=10)
    ap.add_argument("--q", type=int, default=0, help="extra lags of dx")
    ap.add_argument("--no-fx", action="store_true")
    ap.add_argument("--sofr-lag", choices=["strict", "same-day"], default="strict")
    ap.add_argument("--window", type=int, default=500, help="rolling training window, as in the race")
    ap.add_argument("--horizons", type=int, nargs="+", default=[1])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--report", action="store_true", help="also print the plain-language report")
    ap.add_argument("--out", default="results/density")
    return ap.parse_args(argv)


def compute(a) -> dict:
    r = loaders.load_sora(start=a.start, end=a.end)
    fx_ok = (not a.no_fx) and loaders.FX_CSV.exists()
    z = loaders.make_z(loaders.load_usdsgd()) if fx_ok else None
    df = loaders.build_dataset(r, loaders.load_sofr(), z, strict=(a.sofr_lag == "strict"))
    with_z = "z" in df
    p = int(E.select_lags(df, a.pmax, a.q, with_z)["bic"].idxmin()) if str(a.p).lower() == "bic" else int(a.p)
    R = {"df": df, "p": p, "with_z": with_z, "tables": {}, "hist": {}, "n": {}, "dates": {}}
    for h in a.horizons:
        pits, info = density.compute_pit(df, h, a.window, with_z, p, a.q)
        R["tables"][h] = density.density_table(pits, alpha=a.alpha)
        R["hist"][h] = density.pit_histogram(pits)
        R["n"][h] = info["N"]
        R["dates"][h] = info["dates"]
    return R


def headline(R: dict, a) -> pd.DataFrame:
    """The numbers that matter, one per row (section, metric, value). Units are in the metric name; the full tables
    (every test, every predictive cdf) are in the CSV files."""
    df = R["df"]
    rows = [("data", "n_obs", len(df)), ("data", "start", str(df.index[0].date())), ("data", "end", str(df.index[-1].date())),
            ("data", "dr_lags", R["p"]), ("data", "fx_term", R["with_z"]), ("data", "window", a.window)]
    for h, t in R["tables"].items():
        d = R["dates"][h]
        rows += [(f"density_h{h}", "oos_points (non-overlapping)", R["n"][h]),
                 (f"density_h{h}", "oos_start", str(d[0].date())), (f"density_h{h}", "oos_end", str(d[-1].date()))]
        for kind, cols in (("normal", (("ks_p", "KS_p", 1), ("berkowitz_p", "Berkowitz_p", 1), ("mean_z", "mean_z", 1),
                                       ("sd_z", "sd_z", 1), ("ar1_z", "ar1_z", 1), ("ljungbox_z2_p", "LjungBox_z2_p", 1),
                                       ("share_lower5_pct", "lower_5%", 100), ("share_upper5_pct", "upper_5%", 100),
                                       ("share_lower1_pct", "lower_1%", 100), ("share_upper1_pct", "upper_1%", 100))),
                           ("empirical", (("ks_p", "KS_p", 1), ("berkowitz_p", "Berkowitz_p", 1), ("mean_z", "mean_z", 1),
                                          ("sd_z", "sd_z", 1)))):
            g = t[t["predictive"] == kind].set_index("model")
            for name, col, scale in cols:
                for m in g.index:
                    rows.append((f"density_h{h}_{kind}", f"{name}_{m}", float(g.loc[m, col]) * scale))
        for _, s in t.iterrows():
            rows.append((f"reading_h{h}", f"{s['model']}_{s['predictive']}", s["reading"]))
    return pd.DataFrame(rows, columns=["section", "metric", "value"])


def print_report(R: dict, a) -> None:
    """The long plain-language report (--report)."""
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.float_format", lambda v: f"{v:.3f}")
    print(f"{len(R['df'])} daily obs, {R['p']} lags of d r, rolling window {a.window}. PIT uniform <=> distribution right.")
    print("mean_z < 0: forecasts run above outcomes. sd_z < 1: distribution too wide (risk overstated); > 1: too narrow.\n")
    for h, t in R["tables"].items():
        print(f"Horizon {h} day(s): {R['n'][h]} non-overlapping held-out points")
        print(t[["model", "predictive", "N", "KS_p", "mean_z", "sd_z", "ar1_z", "Berkowitz_p", "LjungBox_z2_p"]].to_string(index=False))
        print("\nShare of the normal PIT in the lower / upper 5% and 1% tails (expected 5% / 1%; above = risk understated in that tail):")
        n = t[t["predictive"] == "normal"]
        print(n[["model", "lower_5%", "upper_5%", "lower_1%", "upper_1%", "p_lower_5%", "p_upper_5%"]].to_string(index=False))
        print("\nReading:")
        for _, s in t.iterrows():
            print(f"  {s['model']:<3} {s['predictive']:<9} {s['reading']}")
        if R["n"][h] < 100:
            print(f"  ! only {R['n'][h]} points: these tests have little power at this horizon.")
        print()


def save(R: dict, a, out: Path) -> pd.DataFrame:
    h = headline(R, a)
    h.to_csv(out / "headline.csv", index=False)
    for hz, t in R["tables"].items():
        t.to_csv(out / f"density_table_h{hz}.csv", index=False)
        R["hist"][hz].to_csv(out / f"pit_histogram_h{hz}.csv")
    provenance.write(out / "run_info.json", provenance.run_info(R["df"], vars(a), {"p_used": R["p"]}))
    return h


def main(argv=None):
    a = parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    R = compute(a)
    h = save(R, a, out)
    print(report.format_headline(h, "Predictive-density validation: RW vs M0 vs M1 on held-out SORA - headline numbers"))
    if a.report:
        print("\n" + "=" * 78 + "\n")
        print_report(R, a)
    print(f"\nAll tables saved in: {out}")
    return R


if __name__ == "__main__":
    main()
