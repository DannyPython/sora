"""EVT validation of the M0 and M1 residuals (Proposal 4.6, Blueprint 6.4; ssef/evt.py). Writes every table +
headline.csv + run_info.json to results/evt/ and prints the headline numbers.

    python run_evt.py                  # 200 bootstrap draws per fit (about 1 minute)
    python run_evt.py --fast           # 50 draws (about 15 seconds), for checking the code
    python run_evt.py --q-compare 0.95 # threshold quantile of the M0 vs M1 comparison and the episode check
    python run_evt.py --fit pwm        # probability-weighted moments instead of maximum likelihood

Same data, lags and sample as run_m1.py; M0 is the ECM without SOFR and FX on the SAME sample as M1, and both
residual series are standardised by their own sd before any tail is compared (Proposal 4.6).
Plot data: results/evt/mean_excess.csv (threshold vs mean excess) and threshold_table.csv (xi, KS p-value by threshold).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ssef import evt, loaders, provenance
from ssef import restricted_ecm as E


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--p", default="bic", help="lags of dr: an integer or 'bic' (as in run_m1)")
    ap.add_argument("--pmax", type=int, default=10)
    ap.add_argument("--no-fx", action="store_true")
    ap.add_argument("--sofr-lag", choices=["strict", "same-day"], default="strict")
    ap.add_argument("--B", type=int, default=200, help="parametric-bootstrap draws for each KS p-value")
    ap.add_argument("--B-compare", type=int, default=300, help="paired block-bootstrap draws for M0 vs M1")
    ap.add_argument("--fit", choices=["mle", "pwm"], default="mle")
    ap.add_argument("--q-compare", type=float, default=0.90, help="threshold quantile for the M0 vs M1 comparison and episodes")
    ap.add_argument("--window-days", type=int, default=120, help="length of the 'one episode' window")
    ap.add_argument("--gap-days", type=int, default=7, help="a gap longer than this starts a new cluster of exceedances")
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default="results/evt")
    a = ap.parse_args(argv)
    if a.fast:
        a.B, a.B_compare = 50, 100
    return a


def compute(a) -> dict:
    r = loaders.load_sora(start=a.start, end=a.end)
    fx_ok = (not a.no_fx) and loaders.FX_CSV.exists()
    z = loaders.make_z(loaders.load_usdsgd()) if fx_ok else None
    df = loaders.build_dataset(r, loaders.load_sofr(), z, strict=(a.sofr_lag == "strict"))
    with_z = "z" in df
    p = int(E.select_lags(df, a.pmax, 0, with_z)["bic"].idxmin()) if str(a.p).lower() == "bic" else int(a.p)
    u = evt.standardised_residuals(df, p, with_z)
    R = {"df": df, "p": p, "with_z": with_z, "u": u}
    R["table"] = evt.threshold_table(u, B=a.B, seed=a.seed, fit=a.fit)
    R["chosen"] = evt.choose_threshold(R["table"])
    R["mean_excess"] = pd.concat({(m, t): evt.mean_excess(sgn * s.to_numpy()) for m, s in u.items()
                                  for t, sgn in evt.TAILS.items()}, names=["model", "tail"]).reset_index(level=[0, 1])
    R["compare"] = pd.DataFrame([{"tail": t} | evt.compare_models(sgn * u["M0"].to_numpy(), sgn * u["M1"].to_numpy(), a.q_compare,
                                                                   a.B_compare, seed=a.seed, fit=a.fit)
                                 for t, sgn in evt.TAILS.items()])
    ep, wo = [], []
    for m, s in u.items():
        for t, sgn in evt.TAILS.items():
            e = evt.episodes(sgn * s, a.q_compare, a.gap_days, a.window_days)
            ep.append({"model": m, "tail": t} | e)
            wo.append({"model": m, "tail": t, "window_start": e["densest_window_start"]}
                      | evt.without_window(sgn * s, a.q_compare, e["densest_window_start"], a.window_days, B=a.B, seed=a.seed, fit=a.fit))
    R["episodes"], R["without_window"] = pd.DataFrame(ep), pd.DataFrame(wo)
    return R


def headline(R: dict, a) -> pd.DataFrame:
    """The numbers that matter, one per row (section, metric, value)."""
    df, t, cmp_ = R["df"], R["table"], R["compare"]
    rows = [("data", "n_obs", len(df)), ("data", "dr_lags", R["p"]), ("data", "fx_term", R["with_z"]),
            ("data", "fit", a.fit), ("data", "q_compare", a.q_compare), ("data", "bootstrap_draws_KS", a.B)]
    for tail in evt.TAILS:
        for m in ("M0", "M1"):
            s = t[(t["model"] == m) & (t["tail"] == tail) & np.isclose(t["q"], a.q_compare)]
            if s.empty:
                continue
            s = s.iloc[0]
            rows.append((f"tail_{tail}", f"{m} xi at q={a.q_compare:g}",
                         f"{s['xi']:.3f} (se {s['se_xi']:.3f}), {int(s['n_exc'])} exceedances, KS p {s['KS_p']:.3f}"))
        c = cmp_[cmp_["tail"] == tail].iloc[0]
        rows.append((f"tail_{tail}", "M1 - M0 xi", f"{c['d_xi']:+.3f} ({c['level']:.0%} CI {c['d_xi_lo']:+.3f} to {c['d_xi_hi']:+.3f})"))
        rows.append((f"tail_{tail}", "M1 - M0 99% quantile (sd units)",
                     f"{c['d_quantile']:+.3f} ({c['level']:.0%} CI {c['d_quantile_lo']:+.3f} to {c['d_quantile_hi']:+.3f})"))
    for _, e in R["episodes"].iterrows():
        rows.append((f"episodes_{e['tail']}", f"{e['model']} exceedances / clusters / share in densest {e['densest_window_days']}d",
                     f"{e['n_exc']} / {e['clusters']} / {e['share_in_densest_window']:.1%} (from {e['densest_window_start']:%d %b %Y})"))
    for _, w in R["without_window"].iterrows():
        full = t[(t["model"] == w["model"]) & (t["tail"] == w["tail"]) & np.isclose(t["q"], w["q"])]
        rows.append((f"without_episode_{w['tail']}", f"{w['model']} xi with the densest window removed",
                     f"{w['xi']:.3f} (se {w['se_xi']:.3f}), {int(w['n_exc'])} exceedances, KS p {w['KS_p']:.3f}"
                     + (f"; full sample {full['xi'].iloc[0]:.3f}" if len(full) else "")))
    for _, c in R["chosen"].iterrows():
        rows.append(("threshold_rule", f"{c['model']} {c['tail']}",
                     f"q = {c['q']:.3f}, xi {c['xi']:.3f}, {c['n_exc']} exceedances" if np.isfinite(c["q"]) else "no threshold passes the rule"))
    return pd.DataFrame(rows, columns=["section", "metric", "value"])


def main(argv=None):
    a = parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    R = compute(a)
    hl = headline(R, a)
    pd.set_option("display.width", 200, "display.max_columns", 20, "display.float_format", lambda v: f"{v:.3f}")
    print("=" * 78)
    print(" EVT validation: GPD tail of the standardised M0 and M1 residuals")
    print("=" * 78)
    print(f"{len(R['df'])} daily obs, {R['p']} lags of d r, {a.fit.upper()} fit; KS p-values by parametric bootstrap ({a.B} draws).")
    print("'up' = rate rises (S-REIT borrower's risk), 'down' = rate falls. xi > 0: heavier than exponential.\n")
    t = R["table"]
    print(t[["model", "tail", "q", "thr", "n_exc", "xi", "se_xi", "KS_p", "q99"]].to_string(index=False))
    print("\nM0 vs M1 (paired block bootstrap, same dates):")
    print(R["compare"][["tail", "q", "xi_M0", "xi_M1", "d_xi", "d_xi_lo", "d_xi_hi", "d_quantile", "d_quantile_lo", "d_quantile_hi"]].to_string(index=False))
    print("\nHow many separate episodes? (densest window; clusters = runs separated by more than the gap)")
    print(R["episodes"][["model", "tail", "n_exc", "clusters", "share_in_densest_window", "densest_window_start"]].to_string(index=False))
    print("\nTail shape without the densest window (threshold kept at its full-sample value):")
    print(R["without_window"][["model", "tail", "window_start", "n_exc", "xi", "se_xi", "KS_p"]].to_string(index=False))
    print("\nReading:")
    cmp_ = R["compare"]
    for _, c in cmp_.iterrows():
        inc = c["d_xi_lo"] <= 0 <= c["d_xi_hi"]
        print(f"  {c['tail']:>4}: M1 - M0 shape difference {'includes' if inc else 'excludes'} 0 -> "
              + ("the tails of M0 and M1 cannot be told apart." if inc else "the tails differ."))
    sel = t[np.isclose(t["q"], a.q_compare)]
    flat = sel[(sel["xi"] - 1.96 * sel["se_xi"]) <= 0]
    if len(flat):
        print(f"  {len(flat)} of {len(sel)} fits at q={a.q_compare:g} have a 95% band for xi that includes 0: a heavy tail is not established there.")
    print("  See the 'without the densest window' table: a large change in xi means the tail rests on one episode.")
    R["table"].to_csv(out / "threshold_table.csv", index=False)
    R["chosen"].to_csv(out / "threshold_rule.csv", index=False)
    R["mean_excess"].to_csv(out / "mean_excess.csv", index=False)
    R["compare"].to_csv(out / "m0_vs_m1.csv", index=False)
    R["episodes"].to_csv(out / "episodes.csv", index=False)
    R["without_window"].to_csv(out / "without_densest_window.csv", index=False)
    pd.DataFrame({m: s for m, s in R["u"].items()}).to_csv(out / "std_residuals.csv")
    hl.to_csv(out / "headline.csv", index=False)
    provenance.write(out / "run_info.json", provenance.run_info(R["df"], vars(a), {"p_used": R["p"]}))
    print(f"\nFiles saved in: {out}")
    return R


if __name__ == "__main__":
    main()
