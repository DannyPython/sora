"""VaR backtest of RW, M0 and M1 on held-out data (Proposal 4.7 and Phase 4). Computes everything, writes every
table + headline.csv + run_info.json to results/backtest/, and prints the headline numbers.

    python run_backtest.py                    # 95% / 99% VaR at 1 and 20 days, all three methods
    python run_backtest.py --report           # also print the plain-language report
    python run_backtest.py --p 2 --no-fx      # same switches as run_m1.py
    python run_backtest.py --gpd mle          # ML instead of PWM for the GPD tail
    python run_backtest.py --horizons 1       # 1-day only

The models are those of the out-of-sample race (run_m1.py): same data, same lags, same rolling window, same
regressors. Only the predictive distribution around the point forecast is new (see ssef/backtest.py).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ssef import backtest as bt
from ssef import loaders, provenance, report, restricted_ecm


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--p", default="bic", help="lags of dr: an integer or 'bic' (default, as in run_m1)")
    ap.add_argument("--pmax", type=int, default=10)
    ap.add_argument("--q", type=int, default=0, help="extra lags of dx")
    ap.add_argument("--no-fx", action="store_true")
    ap.add_argument("--sofr-lag", choices=["strict", "same-day"], default="strict")
    ap.add_argument("--window", type=int, default=500, help="rolling training window, as in the race")
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 20])
    ap.add_argument("--levels", type=float, nargs="+", default=list(bt.LEVELS))
    ap.add_argument("--tail-frac", type=float, default=0.10, help="share of the window above the GPD threshold")
    ap.add_argument("--gpd", choices=["pwm", "mle"], default="pwm")
    ap.add_argument("--power-sims", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--no-robust", action="store_true", help="skip the window / threshold sensitivity grid")
    ap.add_argument("--report", action="store_true", help="also print the plain-language report")
    ap.add_argument("--out", default="results/backtest")
    return ap.parse_args(argv)


def compute(a) -> dict:
    """Run every step. Returns plain data; no formatting."""
    R = {}
    r = loaders.load_sora(start=a.start, end=a.end)
    fx_ok = (not a.no_fx) and loaders.FX_CSV.exists()
    z = loaders.make_z(loaders.load_usdsgd()) if fx_ok else None
    df = loaders.build_dataset(r, loaders.load_sofr(), z, strict=(a.sofr_lag == "strict"))
    with_z = "z" in df
    lag_tab = restricted_ecm.select_lags(df, a.pmax, a.q, with_z)
    p = int(lag_tab["bic"].idxmin()) if str(a.p).lower() == "bic" else int(a.p)
    R.update(df=df, with_z=with_z, p=p)

    kw = dict(window=a.window, with_z=with_z, p=p, q=a.q, levels=tuple(a.levels), tail_frac=a.tail_frac, gpd=a.gpd)
    fc = pd.concat([bt.var_forecasts(df, h, **kw) for h in a.horizons], ignore_index=True)
    R["fc"] = fc
    R["coverage"] = bt.coverage_table(fc)
    R["tick"] = bt.tick_loss_test(fc)

    # power at the sample sizes actually tested
    kp, cp = [], []
    for h in a.horizons:
        N = int(R["coverage"].loc[R["coverage"]["h"] == h, "N"].iloc[0])
        for L in a.levels:
            kp.append(bt.kupiec_power(N, 1 - L).assign(h=h))
            cp.append(bt.christoffersen_power(N, 1 - L, n_sims=a.power_sims, seed=a.seed).assign(h=h))
    R["power_kupiec"] = pd.concat(kp, ignore_index=True)
    R["power_christoffersen"] = pd.concat(cp, ignore_index=True)

    # GPD shape in the rolling windows (h = first horizon): a heavy tail has xi > 0
    h0 = a.horizons[0]
    ps = bt.predictive_sets(df["r"].to_numpy(), df["x"].to_numpy(), df["z"].to_numpy() if with_z else None,
                            h0, a.window, with_z, p, a.q, h0)
    rows = []
    for m, pred in ps["models"].items():
        for tail, sgn in (("up", 1.0), ("down", -1.0)):
            xi = bt.evt_quantiles(sgn * pred.u, [min(1 - L for L in a.levels)], a.tail_frac, a.gpd)["xi"]
            rows.append({"h": h0, "model": m, "tail": tail, "xi_median": float(np.median(xi)),
                         "xi_p10": float(np.quantile(xi, 0.1)), "xi_p90": float(np.quantile(xi, 0.9)),
                         "share_xi_positive": float(np.mean(xi > 0))})
    R["evt_shape"] = pd.DataFrame(rows)

    # sensitivity: rolling window and GPD threshold (h = first horizon)
    if not a.no_robust:
        rows = []
        for w in (250, 500, 750):
            if w + p + h0 + 50 >= len(df):
                continue
            for tf in (0.08, 0.10, 0.15):
                if tf <= max(1 - L for L in a.levels):
                    continue
                kw_ = kw | {"window": w, "tail_frac": tf}
                methods = bt.METHODS if tf == a.tail_frac else ("evt",)
                ct = bt.coverage_table(bt.var_forecasts(df, h0, methods=methods, **kw_))
                rows.append(ct.assign(window=w, tail_frac=tf)[["window", "tail_frac", "h", "model", "method", "tail",
                                                               "level", "N", "breaches", "expected", "p_pof", "p_cc",
                                                               "verdict"]])
        R["robust"] = pd.concat(rows, ignore_index=True)
    return R


def headline(R: dict, a) -> pd.DataFrame:
    """The numbers that matter, one per row (section, metric, value). Full tables are in the CSV files."""
    df, cov, tick = R["df"], R["coverage"], R["tick"]
    rows = [("data", "n_obs", len(df)), ("data", "start", str(df.index[0].date())),
            ("data", "end", str(df.index[-1].date())), ("data", "dr_lags", R["p"]), ("data", "fx_term", R["with_z"]),
            ("data", "window", a.window), ("data", "gpd_fit", a.gpd), ("data", "tail_frac", a.tail_frac)]
    for h in a.horizons:
        c = cov[cov["h"] == h]
        rows.append((f"var_h{h}", "oos_points (non-overlapping)", int(c["N"].iloc[0])))
        for tail in ("up", "down"):
            for L in sorted(a.levels, reverse=True):
                for method in ("normal", "evt"):
                    for m in bt.MODELS:
                        s = c[(c["tail"] == tail) & (c["level"] == L) & (c["method"] == method) & (c["model"] == m)]
                        if s.empty:
                            continue
                        s = s.iloc[0]
                        rows.append((f"var_h{h}", f"{tail} {L:.0%} {method:<6} {m}",
                                     f"{s['breaches']}/{s['expected']:.1f}  p_pof {s['p_pof']:.3f}  "
                                     f"p_cc {s['p_cc']:.3f}  {s['verdict']}"))
        t = tick[(tick["h"] == h) & (tick["method"] == "evt")]
        for _, s in t.iterrows():
            rows.append((f"head_to_head_h{h}", f"gw_p_m1_beats_m0 {s['tail']} {s['level']:.0%} evt",
                         s.get("p M1 vs M0", np.nan)))
    kp, cp = R["power_kupiec"], R["power_christoffersen"]
    for h in a.horizons:
        for L in sorted(a.levels, reverse=True):
            s = kp[(kp["h"] == h) & (np.isclose(kp["level"], L)) & (kp["true_rate_multiple"] == 2.0)]
            if not s.empty:
                rows.append((f"power_h{h}", f"kupiec_power {L:.0%} if rate x2", float(s["power"].iloc[0])))
            s = cp[(cp["h"] == h) & (np.isclose(cp["level"], L)) & (np.isclose(cp["pi11"], 0.2))]
            if not s.empty:
                rows.append((f"power_h{h}", f"christoffersen_power {L:.0%} pi11=0.2", float(s["power_ind"].iloc[0])))
    return pd.DataFrame(rows, columns=["section", "metric", "value"])


def save(R: dict, a, out: Path) -> pd.DataFrame:
    h = headline(R, a)
    h.to_csv(out / "headline.csv", index=False)
    for hz, g in R["fc"].groupby("h"):
        g.to_csv(out / f"var_forecasts_h{hz}.csv", index=False)
    R["coverage"].to_csv(out / "backtest_coverage.csv", index=False)
    R["tick"].to_csv(out / "backtest_tick_loss.csv", index=False)
    R["power_kupiec"].to_csv(out / "power_kupiec.csv", index=False)
    R["power_christoffersen"].to_csv(out / "power_christoffersen.csv", index=False)
    R["evt_shape"].to_csv(out / "evt_gpd_shape.csv", index=False)
    if "robust" in R:
        R["robust"].to_csv(out / "robust_window_threshold.csv", index=False)
    provenance.write(out / "run_info.json", provenance.run_info(R["df"], vars(a), {"p_used": R["p"], "with_z": R["with_z"]}))
    return h


def main(argv=None):
    a = parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    R = compute(a)
    h = save(R, a, out)
    print(report.format_headline(h, "VaR backtest: RW vs M0 vs M1 on held-out SORA - headline numbers"))
    if a.report:
        print("\n" + report.format_backtest_report(R, a))
    print(f"\nAll tables saved in: {out}")


if __name__ == "__main__":
    main()
