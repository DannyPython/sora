"""Figures comparing M0 (classical Vasicek) and M1 (policy-anchored, error-correction form).

    python plot_results.py            # all figures -> results/figures/ (PNG + PDF)
    python plot_results.py --fast     # fewer bootstrap draws (for checking the code)

Everything is recomputed from the data with the same code and seeds as run_m0.py / run_m1.py, so the numbers in
the figures are the numbers in those reports.

Design rules (data-viz method): colour follows the model, never the rank - M0 blue, M1 orange, SOFR aqua; raw SORA
and the random walk stay neutral. One axis per panel (no dual axes), thin marks, solid recessive grid, text in
neutral ink (never in the series colour), a legend wherever two or more series are drawn, and direct labels on the
few things that matter. Aqua is under 3:1 contrast on the light surface, so SOFR is always labelled directly.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                       # draw to files; no window needed
import matplotlib.dates as mdates          # noqa: E402
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402
import pandas as pd                        # noqa: E402
from scipy import stats                    # noqa: E402

from . import cointegration as C           # noqa: E402
from . import ecm, gate, loaders, m0, nested, noise, race, simulate  # noqa: E402
from .ols import subset  # noqa: E402

# ------------------------------------------------------------------ style
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"
M0C, M1C, SOFRC = "#2a78d6", "#eb6834", "#1baf7a"          # categorical slots 1-3 (validated, incl. all-pairs)
RWC = "#8a8985"                                             # random walk: neutral grey
N_DAYS = 252


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",                      # ships with matplotlib: identical look on every machine
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.axisbelow": True, "lines.linewidth": 1.8, "legend.frameon": False, "legend.fontsize": 9,
        "axes.unicode_minus": True,
    })


def _title(fig, main: str, sub: str | None = None, note: str | None = None, top: float = 0.88) -> None:
    fig.suptitle(main, x=0.012, y=0.985, ha="left", va="top", fontsize=14.5, fontweight="bold", color=INK)
    if sub:
        fig.text(0.012, 0.985 - 0.058, sub, ha="left", va="top", fontsize=10.2, color=INK2)
    if note:
        fig.text(0.012, 0.008, note, ha="left", va="bottom", fontsize=8.2, color=MUTED)
    fig.subplots_adjust(top=top)


def _years(ax) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def _pp(ax, label: str = "% per annum") -> None:
    ax.set_ylabel(label)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.0f"))


# ------------------------------------------------------------------ computations
def _acf(d: np.ndarray, lags: int = 10) -> np.ndarray:
    """Autocorrelation of the rows of d at lags 1..lags (vectorised over paths)."""
    d = np.atleast_2d(d)
    d = d - d.mean(axis=1, keepdims=True)
    den = (d * d).sum(axis=1)
    return np.stack([(d[:, k:] * d[:, :-k]).sum(axis=1) / den for k in range(1, lags + 1)], axis=1)


def _vr(x: np.ndarray, hs) -> np.ndarray:
    """Lo-MacKinlay variance ratios Var(r_{t+h} - r_t) / (h Var(r_{t+1} - r_t)); rows are paths."""
    x = np.atleast_2d(x)
    v1 = np.diff(x, axis=1).var(axis=1)
    return np.stack([(x[:, h:] - x[:, :-h]).var(axis=1) / (h * v1) for h in hs], axis=1)


def collect(fast: bool = False, seed: int = 2027, verbose: bool = True) -> dict:
    """Run everything the figures need. Same data, settings and seeds as run_m0.py / run_m1.py."""
    say = print if verbose else (lambda *a, **k: None)
    B = 99 if fast else 499
    R: dict = {"fast": fast, "seed": seed}
    r = loaders.load_sora()
    sofr = loaders.load_sofr()
    df = loaders.build_dataset(r, sofr, loaders.make_z(loaders.load_usdsgd()))
    R.update(r=r, df=df)
    p = int(ecm.select_lags(df, 10, 0, True)["bic"].idxmin())
    R["p"] = p

    say("M0: calibration and gate ...")
    fit0 = m0.calibrate(r)
    R["fit0"] = fit0
    nb_boot = 99 if fast else 299
    R["gate_ar1"] = gate.ar1_gate(r, n_boot=nb_boot, seed=seed)
    R["gate_adf"] = gate.adf_gate(r, n_boot=nb_boot, seed=seed)
    R["m0_tab"] = fit0.params_table()
    R["ur"] = gate.unit_root_tests(r)

    say("M1: gate, break, ECM, tests ...")
    ghC = C.gregory_hansen(df["r"], df["x"], "C", p, n_sim=40 if fast else 200, seed=seed, dates=df.index)
    bd = ghC["break_date"]
    R.update(ghC=ghC, break_date=bd, nb=ecm.fit_ecm(df, p, 0, True), br=ecm.fit_ecm(df, p, 0, True, break_date=bd))
    R["coint0"] = nested.cointegration_ecm_test(df, p, 0, True, B, seed=seed)
    R["coint1"] = nested.cointegration_ecm_test(df, p, 0, True, B, seed=seed, break_date=bd)
    R["anchor0"] = nested.anchor_test(df, p, 0, True, "unit_root", B, seed=seed)
    R["anchor1"] = nested.anchor_test(df, p, 0, True, "unit_root", B, seed=seed, break_date=bd)

    # long-run anchors: M0 constant theta; M1 theta_t = alpha + beta1 * SOFR (+ level shift after the break)
    x, rr = df["x"].to_numpy(), df["r"].to_numpy()
    D = (df.index >= bd).astype(float)
    nb, br = R["nb"], R["br"]
    R["anchor"] = {"m0": np.full(len(df), fit0.theta), "m1_nobreak": nb.alpha + nb.beta1 * x,
                   "m1_break": br.alpha + br.beta1 * x + br.level_shift * D}
    R["gap"] = {k: float(np.abs(rr - a).mean() * 100) for k, a in R["anchor"].items()}
    R["corr"] = {k: (float(np.corrcoef(a, rr)[0, 1]) if a.std() > 0 else 0.0) for k, a in R["anchor"].items()}

    say("SORA's own behaviour vs M0-implied paths ...")
    n_paths = 150 if fast else 400
    paths = m0.simulate_vasicek(fit0.kappa, fit0.theta, fit0.sigma, r.iloc[0], len(r) - 1, n_paths, m0.DT_DEFAULT, seed)
    R["paths"] = paths
    lags = 10
    R["acf_act"] = _acf(np.diff(r.to_numpy()), lags)[0]
    sim = _acf(np.diff(paths, axis=1), lags)
    R["acf_m0"] = (sim.mean(0), np.percentile(sim, 5, 0), np.percentile(sim, 95, 0))
    hs = np.array([1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 60, 90, 120])
    R["vr_h"] = hs
    R["vr_act"] = _vr(r.to_numpy(), hs)[0]
    vs = _vr(paths, hs)
    R["vr_m0"] = (vs.mean(0), np.percentile(vs, 5, 0), np.percentile(vs, 95, 0))

    say("Forecast race ...")
    horizons = (1, 20, 60)
    R["race"] = race.run_race(df, horizons, 500, True, B=B, null="unit_root", p=p, seed=seed)

    say("One-year scenarios ...")
    n_sc = 3000 if fast else 6000
    last = r.iloc[-1]
    R["last"] = last
    R["future"] = pd.bdate_range(r.index[-1] + pd.offsets.BDay(1), periods=N_DAYS)
    q = lambda P: np.percentile(P[:, 1:], [5, 50, 95], axis=0)
    R["fan_m0"] = q(fit0.simulate(last, N_DAYS, n_sc, seed))
    sc = simulate.make_scenarios(df["x"].iloc[-1], N_DAYS, 0.01)
    R["fan_m1"] = {k: q(simulate.simulate_ecm(br, df, sc[k], None, n_sc, seed, "bootstrap")) for k in sc}
    R["fan_m1_nobreak"] = q(simulate.simulate_ecm(nb, df, sc["hold"], None, n_sc, seed, "bootstrap"))

    say("Noise and the SORA-SOFR link ...")
    R["res0"], R["res1"] = fit0.resid, nb.fit.resid
    R["z0"], R["z1"] = fit0.std_resid.to_numpy(), noise.std_resid(nb.fit).to_numpy()
    R["noise_diag"] = noise.noise_diagnostics(nb.fit.resid)
    R["roll_slope"] = C.slope_stability(df, windows=(500,), step=20)["rolling"][500]
    R["spread"] = (df["x"] - df["r"]) * 100
    R["noise_cmp"] = noise.compare_noise(nb.fit, subset(nb.fit, ecm.nested_columns(nb.fit.names, "all")))
    return R


# ------------------------------------------------------------------ figures
def _break_line(ax, bd, label: bool = True, ypos: float = 0.97, va: str = "top") -> None:
    ax.axvline(bd, color=INK2, lw=0.9)
    if label:
        ax.text(bd, ypos, f"  level shift\n  {bd:%d %b %Y}", transform=ax.get_xaxis_transform(), ha="left", va=va,
                fontsize=8.5, color=INK2)


def _dodge(ys, gap: float) -> list:
    """Spread label positions so that neighbours are at least `gap` apart (keeps the order, stays near the data)."""
    order = np.argsort(ys)
    pos = np.array(ys, float)[order]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + gap)
    pos -= (pos.mean() - np.array(ys, float).mean())               # re-centre on the original cluster
    out = np.empty(len(ys))
    out[order] = pos
    return list(out)


def fig_overview(R: dict):
    """Same data, two long-run anchors: a constant (M0) vs one that moves with SOFR (M1)."""
    df, bd = R["df"], R["break_date"]
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.6), sharex=True, sharey=True)
    _title(fig, "Same data, two anchors: M0 holds one level, M1's level moves with SOFR",
           "Each model says SORA is pulled towards a long-run anchor. Shaded area = how far SORA sits from that anchor.",
           "M1 with a level shift: the break date is chosen from the data (Gregory-Hansen), so its fit is in-sample and exploratory. "
           "Anchors use the full-sample estimates.", top=0.86)
    r = df["r"] * 100
    a0, a1, a1n = (R["anchor"][k] * 100 for k in ("m0", "m1_break", "m1_nobreak"))
    ax = axes[0]
    ax.fill_between(df.index, r, a0, color=M0C, alpha=0.13, lw=0)
    ax.plot(df.index, r, color=INK, lw=1.7, label="SORA (actual)")
    ax.plot(df.index, a0, color=M0C, lw=2.2, label=f"M0 anchor: constant theta = {R['fit0'].theta * 100:.2f}%")
    ax.set_title("M0 (classical Vasicek)")
    ax.set_title(f"average distance of SORA from its anchor: {R['gap']['m0']:.2f} pp    |    correlation anchor-SORA: {R['corr']['m0']:.2f}",
                 loc="right", fontsize=9.3, fontweight="normal", color=INK2)
    ax.legend(loc="upper left", ncol=2, bbox_to_anchor=(0.0, 1.0))
    ax = axes[1]
    ax.fill_between(df.index, r, a1, color=M1C, alpha=0.13, lw=0)
    ax.plot(df.index, df["x"] * 100, color=SOFRC, lw=1.3, label="SOFR")
    ax.plot(df.index, r, color=INK, lw=1.7, label="SORA (actual)")
    ax.plot(df.index, a1n, color=M1C, lw=1.3, ls=(0, (4, 2)), label="M1 anchor, no break")
    ax.plot(df.index, a1, color=M1C, lw=2.2, label="M1 anchor, with level shift")
    ax.set_title("M1 (anchor moves with SOFR)")
    _break_line(ax, bd, ypos=0.04, va="bottom")
    ax.set_title(f"average distance: {R['gap']['m1_break']:.2f} pp (level shift), {R['gap']['m1_nobreak']:.2f} pp (no break)    |    "
                 f"correlation: {R['corr']['m1_break']:.2f} / {R['corr']['m1_nobreak']:.2f}",
                 loc="right", fontsize=9.3, fontweight="normal", color=INK2)
    ax.legend(loc="upper left", ncol=2, bbox_to_anchor=(0.0, 1.0))
    sofr_end = df["x"].iloc[-1] * 100
    ax.text(df.index[-1], sofr_end + 0.12, "SOFR", ha="right", va="bottom", fontsize=9, color=INK2)
    for a in axes:
        _pp(a)
        a.set_ylim(-0.1, 7.2)
    _years(axes[1])
    return fig


def fig_link(R: dict):
    """Why a fixed anchor fails: the SORA-SOFR link is not stable and the gap keeps moving."""
    df, bd, br, fit0 = R["df"], R["break_date"], R["br"], R["fit0"]
    fig = plt.figure(figsize=(12, 7.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], hspace=0.32, wspace=0.22)
    _title(fig, "A fixed anchor cannot follow SORA: its link to SOFR shifts and the gap keeps widening",
           "Left: SORA against SOFR, each dot a day. M0 says SOFR is irrelevant (flat line); M1 says SORA follows SOFR, with a step down after the level shift.",
           "Right panels share one time axis. Rolling slope = regression of SORA on SOFR over the previous 500 days.", top=0.86)
    ax = fig.add_subplot(gs[:, 0])
    pre, post = df[df.index < bd], df[df.index >= bd]
    ax.scatter(pre["x"] * 100, pre["r"] * 100, s=11, color="#b4b3ae", lw=0, label=f"before {bd:%d %b %Y}", zorder=2)
    ax.scatter(post["x"] * 100, post["r"] * 100, s=11, color=INK, lw=0, label=f"from {bd:%d %b %Y}", zorder=3)
    xs = np.linspace(0, 5.6, 50)
    ax.plot(xs, np.full_like(xs, fit0.theta * 100), color=M0C, lw=2.2, label="M0: SOFR irrelevant", zorder=4)
    b1, a_, sh = br.beta1, br.alpha, br.level_shift
    xr = np.linspace(pre["x"].min() * 100, pre["x"].max() * 100, 30)
    ax.plot(xr, 100 * a_ + b1 * xr, color=M1C, lw=2.2, zorder=4, label=f"M1: slope {b1:.2f}, before")
    xr2 = np.linspace(post["x"].min() * 100, post["x"].max() * 100, 30)
    ax.plot(xr2, 100 * (a_ + sh) + b1 * xr2, color=M1C, lw=2.2, ls=(0, (4, 2)), zorder=4,
            label=f"M1: same slope, {sh * 100:.1f} pp lower after")
    ax.set_xlabel("SOFR, % per annum")
    ax.set_ylabel("SORA, % per annum")
    ax.set_xlim(0, 5.7)
    ax.set_ylim(-0.1, 5.5)
    ax.set_title("SORA vs SOFR")
    ax.legend(loc="upper left", ncol=2, fontsize=8.6, markerscale=1.6, columnspacing=1.2)
    ax1 = fig.add_subplot(gs[0, 1])
    rs = R["roll_slope"]
    ax1.plot(rs.index, rs.values, color=INK, lw=1.8)
    ax1.axhline(0, color=GRID, lw=1)
    ax1.set_title("Rolling slope of SORA on SOFR")
    ax1.text(0.02, 0.95, f"ranges from {rs.min():.2f} to {rs.max():.2f}:\nnot a stable coefficient", transform=ax1.transAxes,
             va="top", fontsize=9, color=INK2)
    ax1.set_ylabel("slope")
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
    sp = R["spread"]
    ax2.plot(sp.index, sp.values, color=INK, lw=1.4)
    ax2.axhline(0, color=GRID, lw=1)
    ax2.set_title("Gap: SOFR minus SORA")
    ax2.set_ylabel("percentage points")
    ax2.text(0.02, 0.95, f"from {sp.iloc[:60].mean():.1f} pp (2021) to {sp.iloc[-60:].mean():.1f} pp (now)", transform=ax2.transAxes,
             va="top", fontsize=9, color=INK2)
    for a in (ax1, ax2):
        _break_line(a, bd, label=a is ax1, ypos=0.28)
        _years(a)
    return fig


def fig_m0_vs_sora(R: dict):
    """SORA's own behaviour compared with what M0 implies."""
    fit0, ag0, agp, tab = R["fit0"], R["gate_ar1"], R["gate_adf"], R["m0_tab"]
    r = R["r"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
    _title(fig, "M0 vs SORA's own behaviour: the data are more persistent, and reverse faster, than M0 allows",
           "Blue = what M0 implies (simulated from its own fitted parameters). Black = what SORA actually did.",
           "M0 simulated from the OLS-fitted parameters (same sample length). Bands = 5th to 95th percentile of those simulations. Gaussian M0 paths can go negative.",
           top=0.865)
    fig.subplots_adjust(hspace=0.42, wspace=0.24, left=0.1)
    # (a) persistence
    ax = axes[0, 0]
    rows = [("Normal approx.", fit0.c1, tab.loc["c1", "ci_low"], tab.loc["c1", "ci_high"]),
            ("Bootstrap, AR(1)", ag0.c1_bc, *ag0.ci),
            (f"Bootstrap, {agp.lags} lags", agp.c1_bc, *agp.ci)]
    for i, (lab, est, lo, hi) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([lo, hi], [y, y], color=M0C, lw=2.4, solid_capstyle="butt")
        ax.plot([est], [y], "o", color=M0C, ms=7, mec=SURFACE, mew=1.6, zorder=3)
        ax.text(est - 0.0003, y + 0.16, f"{est:.4f}", fontsize=8.6, color=INK2, va="bottom", ha="right")
    ax.axvline(1.0, color=INK, lw=1.2)
    ax.text(1.0005, 3.7, "c1 = 1: no mean\nreversion", fontsize=9, color=INK, va="top")
    ax.set_xlim(0.972, 1.012)
    ax.set_ylim(-0.5, 3.75)
    ax.set_yticks([2, 1, 0], [r[0] for r in rows], fontsize=8.8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("daily persistence c1 (estimate and 95% interval)")
    ax.set_title("(a) Can SORA be told apart from a unit root?")
    ax.text(0.9735, 3.7, "The normal-approximation interval excludes 1\n(too narrow near a unit root); the valid\n"
            f"bootstrap intervals reach 1 -> gate {'passed' if agp.reverts else 'failed'}.", fontsize=8.8, color=INK2, va="top")
    # (b) ACF
    ax = axes[0, 1]
    lg = np.arange(1, len(R["acf_act"]) + 1)
    m, lo, hi = R["acf_m0"]
    ax.fill_between(lg, lo, hi, color=M0C, alpha=0.16, lw=0, label="M0-implied (5-95%)")
    ax.plot(lg, m, color=M0C, lw=2, label="M0-implied mean")
    ax.bar(lg, R["acf_act"], width=0.42, color=INK, label="SORA (actual)", zorder=3)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(lg)
    ax.set_xlabel("lag (business days)")
    ax.set_ylabel("autocorrelation of daily changes")
    ax.set_title("(b) Day-to-day memory: SORA reverses, M0 has none")
    ax.legend(loc="upper right", ncol=1)
    # (c) variance ratio
    ax = axes[1, 0]
    hs = R["vr_h"]
    m, lo, hi = R["vr_m0"]
    ax.fill_between(hs, lo, hi, color=M0C, alpha=0.16, lw=0)
    ax.plot(hs, m, color=M0C, lw=2, label="M0-implied (band = 5-95%)")
    ax.plot(hs, R["vr_act"], color=INK, lw=2, marker="o", ms=4.5, mec=SURFACE, mew=1, label="SORA (actual)")
    ax.axhline(1.0, color=RWC, lw=1.2)
    ax.text(1.05, 1.03, "random walk = 1", ha="left", va="bottom", fontsize=8.8, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 60, 120])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(0, 1.25)
    ax.set_xlabel("horizon h (business days)")
    ax.set_ylabel("variance ratio")
    ax.set_title("(c) Short-run swings fade much faster than in M0")
    ax.legend(loc="lower left")
    i20 = list(hs).index(20)
    ax.plot([20], [R["vr_act"][i20]], "o", color=INK, ms=7, mec=SURFACE, mew=1.2, zorder=4)
    ax.text(17.5, R["vr_act"][i20] - 0.035, f"SORA {R['vr_act'][i20]:.2f} at 20 days", fontsize=9, color=INK2, va="center", ha="right")
    ax.text(17.5, m[i20] - 0.07, f"M0 {m[i20]:.2f} at 20 days", fontsize=9, color=INK2, va="top", ha="right")
    # (d) sample paths
    ax = axes[1, 1]
    for P in R["paths"][:18]:
        ax.plot(r.index, P * 100, color=M0C, lw=0.7, alpha=0.28)
    ax.plot(r.index, r * 100, color=INK, lw=1.8, label="SORA (actual)")
    ax.axhline(fit0.theta * 100, color=M0C, lw=2.2, label=f"M0 anchor {fit0.theta * 100:.2f}%")
    ax.plot([], [], color=M0C, lw=0.9, alpha=0.5, label="18 paths simulated from M0")
    ax.set_title("(d) A world generated by M0 stays near its anchor")
    ax.set_ylim(-2.6, 9.4)
    ax.legend(loc="upper left", ncol=1)
    _years(ax)
    _pp(ax)
    return fig


def fig_outlook(R: dict):
    """What each model expects for the next 12 months."""
    r, fut, last = R["r"], R["future"], R["last"]
    f0, f1, f1n = R["fan_m0"] * 100, R["fan_m1"], R["fan_m1_nobreak"] * 100
    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    hold, up, down = (f1[k] * 100 for k in ("hold", "up", "down"))
    _title(fig, f"Next 12 months: M0 expects SORA to drift up to {f0[1, -1]:.1f}%; M1 stays near {hold[1, -1]:.1f}% and moves with SOFR",
           f"Starting from SORA = {last * 100:.2f}% on {r.index[-1]:%d %b %Y}. Lines = median path, bands = 5th to 95th percentile.",
           "Illustration, not a forecast: M0's gate failed and M1's level shift is exploratory. SOFR paths for M1 are assumed (held, or +/-100 bp over 3 months); "
           "the random walk is 'no change'.", top=0.87)
    hist = r[r.index >= r.index[-1] - pd.Timedelta(days=365)]
    ax.plot(hist.index, hist * 100, color=INK, lw=1.9, label="SORA (actual, last 12 months)")
    ax.fill_between(fut, f0[0], f0[2], color=M0C, alpha=0.14, lw=0)
    ax.fill_between(fut, hold[0], hold[2], color=M1C, alpha=0.16, lw=0)
    ax.plot(fut, f0[1], color=M0C, lw=2.4, label="M0 median")
    ax.plot(fut, np.full(len(fut), last * 100), color=RWC, lw=1.6, label="Random walk (no change)")
    ax.plot(fut, f1n[1], color=M1C, lw=1.3, ls=(0, (4, 2)), label="M1 without level shift (median)")
    ax.plot(fut, hold[1], color=M1C, lw=2.4, label="M1 median, SOFR held")
    ax.plot(fut, up[1], color=M1C, lw=1.2, alpha=0.75)
    ax.plot(fut, down[1], color=M1C, lw=1.2, alpha=0.75)
    ax.axvline(r.index[-1], color=INK2, lw=0.9)
    xe = fut[-1] + pd.Timedelta(days=6)
    labs = [(f0[1, -1], f"M0  {f0[1, -1]:.1f}%"), (up[1, -1], f"M1, SOFR +100 bp  {up[1, -1]:.1f}%"),
            (hold[1, -1], f"M1, SOFR held  {hold[1, -1]:.1f}%"), (down[1, -1], f"M1, SOFR -100 bp  {down[1, -1]:.1f}%"),
            (f1n[1, -1], f"M1 without shift  {f1n[1, -1]:.1f}%")]
    for yy, (y0, t) in zip(_dodge([l[0] for l in labs], 0.17), labs):
        ax.plot([fut[-1], xe - pd.Timedelta(days=1)], [y0, yy], color=GRID, lw=0.8)
        ax.text(xe, yy, t, fontsize=9, va="center", color=INK2)
    ax.text(r.index[-1], 0.15, "  today", va="bottom", fontsize=9, color=INK2)
    ax.set_xlim(hist.index[0], fut[-1] + pd.Timedelta(days=110))
    ax.set_ylim(0, 4.5)
    _pp(ax)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1.0))      # whole-number ticks match the "%.0f" labels
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(loc="upper left", ncol=1, fontsize=8.8)
    return fig


def fig_race(R: dict):
    """Forecast race: is either model better than 'tomorrow = today'?"""
    S, fcs = R["race"]["summary"], R["race"]["forecasts"]
    hz = list(S.index)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.4), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.24})
    rw_best = all(S.loc[h, "rmse_rw_bp"] <= min(S.loc[h, "rmse_m0_bp"], S.loc[h, "rmse_m1_bp"]) for h in hz)
    p1 = S.loc[hz[0], "p_boot M1 vs M0"]
    later = [h for h in hz[1:] if S.loc[h, "p_boot M1 vs M0"] < 0.05]
    verdict = ("SOFR adds a little at one day, nothing beyond" if p1 < 0.05 and not later else
               "SOFR adds nothing we can detect" if p1 >= 0.05 and not later else "SOFR helps at some horizons")
    _title(fig, ("Neither model beats 'tomorrow = today'; " if rw_best else "The random walk is not always best; ") + verdict,
           "Out-of-sample errors from rolling-window forecasts (500-day window). Lower is better.",
           f"Bootstrap Clark-West p-values test whether M1 improves on M0 ({R['race']['B']} draws; normal-theory p-values are not valid with an I(1) regressor).",
           top=0.865)
    ax = axes[0]
    w, x = 0.26, np.arange(len(hz))
    series = (("rw", "Random walk", RWC, -w), ("m0", "M0", M0C, 0.0), ("m1", "M1", M1C, w))
    for key, lab, col, off in series:
        v = [S.loc[h, f"rmse_{key}_bp"] for h in hz]
        ax.bar(x + off, v, width=w - 0.025, color=col, label=lab, zorder=3)
        for xi, vi in zip(x + off, v):
            ax.text(xi, vi + 1.0, f"{vi:.0f}", ha="center", va="bottom", fontsize=8.6, color=INK2)
    ax.set_xticks(x, [f"{h} day{'s' if h > 1 else ''}\nM1 vs M0: p = {S.loc[h, 'p_boot M1 vs M0']:.3f}" for h in hz])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("typical forecast error, bp (RMSE)")
    ax.set_title("Typical error by horizon")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    h = 20 if 20 in fcs else hz[-1]
    f = fcs[h]
    e_rw = (f["actual"] - f["rw"]) ** 2
    for key, lab, col in (("m0", "M0", M0C), ("m1", "M1", M1C)):
        cum = (((f["actual"] - f[key]) ** 2 - e_rw) * 1e8).cumsum() / 1e3
        ax.plot(cum.index, cum.values, color=col, lw=2, label=lab)
        ax.text(cum.index[-1], cum.iloc[-1], f"  {lab}", va="center", fontsize=9, color=INK2)
    ax.axhline(0, color=INK, lw=1.1)
    _break_line(ax, R["break_date"], ypos=0.45, va="top")
    ax.set_title(f"Running score vs random walk, {h}-day forecasts")
    ax.set_ylabel("cumulative excess squared error (thousand bp$^2$)")
    ax.set_xlabel("above the zero line = worse than the random walk")
    ax.set_xlim(f.index[0], f.index[-1] + pd.Timedelta(days=150))
    _years(ax)
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.62))
    fig.subplots_adjust(bottom=0.16)
    return fig


def fig_noise(R: dict):
    """The noise is the same in both models - the models differ in the anchor, not in the shocks."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), gridspec_kw={"wspace": 0.22})
    kurt = R["noise_diag"].loc["excess kurtosis", "value"]
    _title(fig, "The noise does not separate the models: both leave the same fat tails and volatile spells",
           "Standardised one-day residuals. Adding SOFR and FX explains almost none of the daily variance, so M1's case rests on the level, not the noise.",
           "Residual of the ECM without a break (M1) vs the AR(1) in levels (M0), each divided by its own standard deviation.", top=0.85)
    ax = axes[0]
    for key, lab, col, z in (("m0", "M0", M0C, R["z0"]), ("m1", "M1", M1C, R["z1"])):
        zs = np.sort(z)
        th = stats.norm.ppf((np.arange(1, len(zs) + 1) - 0.5) / len(zs))
        ax.plot(th, zs, "o", ms=3.4, color=col, mew=0, alpha=0.8, label=lab)
    lim = 6.2
    ax.plot([-lim, lim], [-lim, lim], color=INK, lw=1.1)
    ax.text(3.2, 2.2, "Gaussian", fontsize=9, color=INK2, ha="left", rotation=0)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-8, 9)
    ax.set_xlabel("quantile of a normal distribution")
    ax.set_ylabel("quantile of the residuals (standard deviations)")
    ax.set_title("QQ plot: tails far beyond Gaussian")
    ax.text(0.03, 0.95, f"excess kurtosis {kurt:.1f} (Gaussian = 0)", transform=ax.transAxes, va="top", fontsize=9.2, color=INK2)
    ax.legend(loc="lower right", markerscale=2.2)
    ax = axes[1]
    for lab, col, res in (("M0", M0C, R["res0"]), ("M1", M1C, R["res1"])):
        s = res.rolling(60).std() * 1e4
        ax.plot(s.index, s.values, color=col, lw=1.9, label=lab)
    ax.set_title("Rolling 60-day volatility of the residuals")
    ax.set_ylabel("basis points per day")
    ax.legend(loc="upper right")
    _years(ax)
    fig.subplots_adjust(bottom=0.14)
    return fig


def fig_summary(R: dict):
    """One-page scorecard: key characteristics of M0 and M1 side by side."""
    S = R["race"]["summary"]
    ag, gh, nb, br = R["gate_adf"], R["ghC"], R["nb"], R["br"]
    f0 = R["fit0"]
    cmp_ = R["noise_cmp"]
    fig = plt.figure(figsize=(12.5, 7.4))
    _title(fig, "M0 vs M1 at a glance", "What each model assumes, what the data say about it, and how it performs.",
           f"Recomputed with the code and seeds of run_m0.py / run_m1.py ({R['race']['B']} bootstrap draws; p-values move slightly with the draw count). "
           "pp = percentage points; bp = basis points.",
           top=0.87)
    ax = fig.add_axes([0.012, 0.045, 0.976, 0.80])
    ax.axis("off")
    ax.set_xlim(0, 1)
    pv = lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
    hl = np.log(2) / ag.kappa_bc() if np.isfinite(ag.kappa_bc()) and ag.kappa_bc() > 0 else np.inf
    rows = [
        ("Long-run level", "constant  theta", "moves:  theta_t = alpha + b1 SOFR + b2 FX"),
        ("Parameters", "kappa, theta, sigma", "kappa, alpha, b1, b2, sigma (+ one level shift)"),
        ("Persistence (c1)", f"{ag.c1_bc:.4f}   [{ag.ci[0]:.4f}, {ag.ci[1]:.4f}]", f"adjustment a = {nb.a:.3f} per day (no break), {br.a:.3f} (level shift)"),
        ("Gate: is the model's premise supported?", f"FAILED  (c1 interval reaches 1)" if not ag.reverts else "passed",
         f"no break: not cointegrated (p = {pv(R['coint0']['p_boot_t_left'])})\nlevel shift: cointegrated (Gregory-Hansen {gh['stat']:.2f} vs {gh['cv_5pct']:.2f})"),
        ("Reversion speed kappa (per year)", f"{ag.kappa_bc():.2f}   (range 0 to {np.nan_to_num(-np.log(ag.ci[0]) * 252):.1f})",
         f"{nb.kappa:.1f} (no break), {br.kappa:.1f} (level shift)"),
        ("Distance of SORA from the anchor", f"{R['gap']['m0']:.2f} pp on average", f"{R['gap']['m1_nobreak']:.2f} pp (no break), {R['gap']['m1_break']:.2f} pp (level shift, in-sample)"),
        ("Does SOFR/FX matter? (valid test)", "-", f"no break: p = {pv(R['anchor0']['p_boot_F'])}   |   level shift: p = {pv(R['anchor1']['p_boot_F'])}"),
        ("Daily noise (bp)", f"{cmp_['s_eps_m0_bp']:.1f}", f"{cmp_['s_eps_m1_bp']:.1f}   (SOFR/FX explain ~{max(cmp_['variance_reduction'], 0) * 100:.0f}% of the daily variance)"),
        ("Forecast error vs random walk", "  ".join(f"{h}d: {S.loc[h, 'rmse_m0_bp'] / S.loc[h, 'rmse_rw_bp'] - 1:+.0%}" for h in S.index),
         "  ".join(f"{h}d: {S.loc[h, 'rmse_m1_bp'] / S.loc[h, 'rmse_rw_bp'] - 1:+.0%}" for h in S.index)),
        ("Does M1 beat M0 out of sample?", "-", "  ".join(f"{h}d: p = {pv(S.loc[h, 'p_boot M1 vs M0'])}" for h in S.index)),
    ]
    top, rh = 0.985, 0.0915
    ax.add_patch(plt.Rectangle((0.0, top - 0.058), 1.0, 0.058, color="#efeeea", lw=0, transform=ax.transAxes))
    ax.text(0.01, top - 0.029, "", va="center")
    xs = (0.012, 0.275, 0.57)
    for x, lab, col in ((xs[1], "M0  classical Vasicek", M0C), (xs[2], "M1  policy-anchored (error-correction form)", M1C)):
        ax.add_patch(plt.Rectangle((x - 0.012, top - 0.047), 0.007, 0.036, color=col, lw=0, transform=ax.transAxes))
        ax.text(x, top - 0.029, lab, va="center", fontsize=11, fontweight="bold", color=INK, transform=ax.transAxes)
    y = top - 0.058
    for i, (a, b, c) in enumerate(rows):
        y -= rh
        ax.plot([0, 1], [y + rh, y + rh], color=GRID, lw=0.9, transform=ax.transAxes)
        ax.text(xs[0], y + rh / 2, a, va="center", fontsize=9.6, color=INK2, transform=ax.transAxes, wrap=True)
        ax.text(xs[1], y + rh / 2, b, va="center", fontsize=9.6, color=INK, transform=ax.transAxes)
        ax.text(xs[2], y + rh / 2, c, va="center", fontsize=9.6, color=INK, transform=ax.transAxes, linespacing=1.4)
    return fig


FIGURES = (("00_summary", fig_summary), ("01_two_anchors", fig_overview), ("02_sora_sofr_link", fig_link),
           ("03_m0_vs_sora", fig_m0_vs_sora), ("04_next_year", fig_outlook), ("05_forecast_race", fig_race),
           ("06_noise", fig_noise))


def make_all(out_dir="results/figures", fast: bool = False, formats=("png", "pdf"), seed: int = 2027, dpi: int = 200,
             verbose: bool = True, R: dict | None = None) -> list:
    """Draw every figure and save it. Returns the list of written files."""
    _style()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        R = collect(fast, seed, verbose) if R is None else R
        written = []
        for name, fn in FIGURES:
            fig = fn(R)
            for fmt in formats:
                path = out / f"{name}.{fmt}"
                fig.savefig(path, dpi=dpi)
                written.append(path)
            plt.close(fig)
            if verbose:
                print(f"  drawn: {name}")
    return written
