"""Run the M1 noise model: estimate, check, compare with M0, simulate. Prints a plain-language report.

    python run_m1_noise.py                          # full M1: SOFR + USD/SGD (data/USDSGD_clean.csv)
    python run_m1_noise.py --no-fx                  # SOFR only (Blueprint Sec. 10 fallback)
    python run_m1_noise.py --sofr-lag same-day      # repo convention instead of the conservative default
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import m1_noise as m1
from mzero import data, m0

W = 78


def head(t):
    return f"\n{t}\n{'-' * W}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fx", default=str(m1.DATA_DIR / "USDSGD_clean.csv"), help="USD/SGD CSV (columns Date, USD/SGD; d/m/Y)")
    ap.add_argument("--no-fx", action="store_true", help="drop z_t: SOFR-only fallback model")
    ap.add_argument("--sofr-lag", choices=["strict", "same-day"], default="strict")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--shock-bp", type=float, default=100.0, help="SOFR shock for up/down scenarios")
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--out", default="results/m1_noise")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    r = data.load_sora(start=a.start, end=a.end)
    z = None
    if not a.no_fx and Path(a.fx).exists():
        z = m1.make_z(m1.load_usdsgd(a.fx))
    df = m1.build_dataset(r, m1.load_sofr(), z, strict=(a.sofr_lag == "strict"))
    fit = m1.fit_m1(df)
    m0fit = m0.calibrate(df["r"])                                   # same sample as M1
    diag = m1.noise_diagnostics(fit.resid)
    cmp = m1.compare_noise(m0fit.resid, fit.resid, 2, fit.k)
    lo, hi = fit.sigma_ci()
    gauss_bad = bool(diag.loc["Jarque-Bera normality", "assumption_violated"])

    L = ["=" * W, " M1 noise model on SORA - summary", "=" * W]
    L.append(f"Data : {fit.n} daily observations, {df.index[0]:%d %b %Y} to {df.index[-1]:%d %b %Y}")
    L.append("Model: r(t+1) = c0 + c1 r(t) + c2 SOFR(t)" + (" + c3 FX(t)" if "c3" in fit.names else "")
             + " + noise   | " + ("SOFR + FX" if "c3" in fit.names else "SOFR only (FX not used)"))
    L.append(f"SOFR timing: {'only values published BEFORE the SORA date' if a.sofr_lag == 'strict' else 'values published on or before the SORA date'}")

    L.append(head("1. How big is the noise?"))
    L.append(f"Daily noise  s_eps : {fit.s_eps * 1e4:6.2f} bp   (typical unexplained daily move in SORA)")
    if fit.valid:
        L.append(f"Annualised   sigma : {fit.sigma * 1e4:6.0f} bp per sqrt(year)   95% CI {lo * 1e4:.0f} to {hi * 1e4:.0f}")
        L.append(f"Reversion speed kappa: {fit.kappa:.2f} /yr  (c1 = {fit.c('c1'):.4f})")
    else:
        L.append("c1 is outside (0,1): the fitted process does not revert, sigma is undefined (Proposal 4.3.4).")

    L.append(head("2. Does M1 reduce the noise compared with M0?  (same dates)"))
    L.append(f"{'M0 daily noise':<28}{cmp['s_eps_M0'] * 1e4:6.2f} bp")
    L.append(f"{'M1 daily noise':<28}{cmp['s_eps_M1'] * 1e4:6.2f} bp")
    L.append(f"{'Variance explained by M1':<28}{cmp['variance_reduction'] * 100:6.1f} %   (extra regressors always lower this a little)")
    L.append("Reading: " + ("SOFR/FX explain very little of the DAILY noise. M1's case must rest on the slow level."
                            if cmp["variance_reduction"] < 0.10 else "M1 explains a meaningful part of the daily noise."))

    L.append(head("3. Is the noise really Gaussian, white and constant?  (Proposal 4.3.3 assumes so)"))
    def v(name, nd=2):
        return diag.loc[name, "value"]
    ek = v("excess kurtosis (Gaussian = 0)"); sk = v("skewness (Gaussian = 0)")
    tail = v("share beyond 3 sd (Gaussian = 0.27%)")
    def verdict(name):
        return "VIOLATED" if diag.loc[name, "assumption_violated"] else "ok"
    rows = [
        ("Fat tails (excess kurtosis)", f"{ek:.2f}", verdict("excess kurtosis (Gaussian = 0)"), "0 for a Gaussian; large = extreme days are too frequent"),
        ("Skew", f"{sk:.2f}", verdict("skewness (Gaussian = 0)"), "0 = symmetric"),
        ("Moves beyond 3 sd", f"{tail * 100:.2f}%", verdict("share beyond 3 sd (Gaussian = 0.27%)"), "Gaussian gives 0.27%"),
        ("Jarque-Bera normality", f"p={diag.loc['Jarque-Bera normality', 'p_value']:.3f}", verdict("Jarque-Bera normality"), "p < 0.05 = not Gaussian"),
        ("Ljung-Box autocorrelation", f"p={diag.iloc[3]['p_value']:.3f}", verdict(diag.index[3]), "p < 0.05 = noise has memory"),
        ("ARCH-LM vol. clustering", f"p={diag.iloc[4]['p_value']:.3f}", verdict(diag.index[4]), "p < 0.05 = calm/turbulent spells"),
    ]
    L.append(f"{'Check':<30}{'Value':<10}{'Result':<10}Meaning")
    L += [f"{n:<30}{val:<10}{res:<10}{m}" for n, val, res, m in rows]
    L.append("Reading: " + ("the Gaussian assumption is REJECTED (fat tails). Use method='bootstrap' when simulating, "
                            "and keep the generalised Pareto step (Block 5)." if gauss_bad
                            else "no strong evidence against the Gaussian assumption."))

    L.append(head(f"4. Simulated SORA in 1 year (5th / median / 95th percentile, %), SOFR shock {a.shock_bp:.0f} bp"))
    sc = m1.make_scenarios(df["x"].iloc[-1], 252, shock=a.shock_bp / 1e4)
    zp = None if "c3" not in fit.names else np.zeros(252)
    L.append(f"{'Scenario':<10}{'Gaussian noise':<30}{'Bootstrapped noise':<30}")
    neg = False
    for name, xp in sc.items():
        cells = []
        for method in ("gaussian", "bootstrap"):
            p = m1.simulate_m1(fit, df["r"].iloc[-1], xp, zp, 10_000, a.seed, method)
            q = np.percentile(p[:, -1], [5, 50, 95]) * 100
            neg |= q[0] < 0
            cells.append(f"{q[0]:6.2f} / {q[1]:6.2f} / {q[2]:6.2f}")
        L.append(f"{name:<10}{cells[0]:<30}{cells[1]:<30}")
    L.append(f"Start SORA {df['r'].iloc[-1] * 100:.2f}%, SOFR {df['x'].iloc[-1] * 100:.2f}%; 10,000 paths, seed {a.seed}; FX held at 0.")
    if neg:
        L.append("! Some percentiles are negative: the linear Gaussian form allows negative rates (Blueprint 5.5).")

    L += ["\n" + "=" * W, "Bottom line", "=" * W]
    L.append(f"- Daily noise is {fit.s_eps * 1e4:.1f} bp; M1 reduces M0's noise variance by only {cmp['variance_reduction'] * 100:.1f}%.")
    L.append("- Gaussian noise assumption: " + ("REJECTED, use bootstrapped noise." if gauss_bad else "not rejected."))
    L.append("- Coefficients are NOT interpretable until the r ~ SOFR cointegration gate is run (Blueprint 6.2).")
    if "c3" not in fit.names:
        L.append("- FX not used: this is the SOFR-only fallback, not the full M1.")
    print("\n".join(L))

    pd.DataFrame({"resid": fit.resid, "std_resid": fit.std_resid}).to_csv(out / "m1_residuals.csv")
    diag.to_csv(out / "m1_noise_checks.csv")
    pd.Series({**{k: float(v_) for k, v_ in fit.structural.items()}, "s_eps": fit.s_eps}).to_csv(out / "m1_params.csv", header=["value"])
    print(f"\nFiles saved in: {out}")


if __name__ == "__main__":
    main()