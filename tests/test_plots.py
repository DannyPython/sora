"""The figures must show the pipeline's numbers: collect() is checked against real runs of run_m0.py / run_m1.py."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ssef import ecm, loaders, plots

ROOT = Path(__file__).resolve().parent.parent
needs_data = pytest.mark.skipif(not (loaders.SORA_CSV.exists() and loaders.SOFR_CSV.exists() and loaders.FX_CSV.exists()),
                                reason="data CSVs not present")


# ------------------------------------------------------------------ helpers (synthetic data)
def test_variance_ratio_is_one_for_a_random_walk_and_follows_the_ar1_formula():
    rng = np.random.default_rng(1)
    rw = np.cumsum(rng.standard_normal((200, 2000)), axis=1)
    assert plots.variance_ratio(rw, [5, 20, 60]).mean(axis=0) == pytest.approx([1, 1, 1], abs=0.08)
    ou = np.zeros((200, 2000))
    e = rng.standard_normal((200, 2000))
    for t in range(1, 2000):
        ou[:, t] = 0.9 * ou[:, t - 1] + e[:, t]
    hs = np.array([5, 20, 60])
    assert plots.variance_ratio(ou, hs).mean(axis=0) == pytest.approx((1 - 0.9 ** hs) / (hs * 0.1), abs=0.05)


def _link(n=4000, seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n)
    x = np.cumsum(rng.standard_normal(n)) * 0.02 + 3.0
    r = np.zeros(n)
    for t in range(1, n):                                            # dr = 0.1 (1 + 0.5 x - r) + noise
        r[t] = r[t - 1] + 0.1 * (1.0 + 0.5 * x[t - 1] - r[t - 1]) + 0.05 * rng.standard_normal()
    return pd.DataFrame({"r": r, "x": x, "z": rng.standard_normal(n) * 0.001}, index=idx)


def test_anchor_path_tracks_a_known_sofr_link():
    df = _link()
    m = ecm.fit_ecm(df, 1, 0, True)
    a = plots.anchor_path(m, df)
    assert a.iloc[-1] == pytest.approx(1.0 + 0.5 * df["x"].iloc[-1], abs=0.4)
    assert np.corrcoef(a, df["x"])[0, 1] == pytest.approx(1.0)       # no shift: the anchor is a line in SOFR


def test_anchor_path_applies_the_level_shift_only_from_the_break_date():
    df = _link()
    bd = df.index[2500]
    m = ecm.fit_ecm(df, 1, 0, True, break_date=bd)
    a, a0 = plots.anchor_path(m, df), m.alpha + m.beta1 * df["x"]
    assert np.allclose((a - a0).iloc[:2500], 0.0)
    assert np.allclose((a - a0).iloc[2500:], m.level_shift)


def test_start_dates_are_quarterly_and_keep_enough_observations():
    idx = pd.bdate_range("2021-10-05", "2026-11-04")                                # data starting mid-quarter
    s = plots.start_dates(idx, 400)
    assert s[0] == pd.Timestamp("2021-10-01")
    assert all((idx >= d).sum() >= 400 for d in s)
    assert (idx >= s[-1] + pd.offsets.QuarterBegin(startingMonth=1)).sum() < 400   # the next quarter would be too short
    assert all(b - a >= pd.Timedelta(days=89) for a, b in zip(s[:-1], s[1:]))


# ------------------------------------------------------------------ real data: figures == console reports
@pytest.fixture(scope="module")
def R():
    return plots.collect(fast=True, verbose=False)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    out = tmp_path_factory.mktemp("runs")
    run = lambda *a: subprocess.run([sys.executable, *a], capture_output=True, text=True, cwd=ROOT, check=True)
    m1 = run("run_m1.py", "--fast", "--out", str(out / "m1"))
    run("run_m0.py", "--n-boot", "99", "--out", str(out / "m0"))
    return out, m1.stdout


@needs_data
def test_m0_gate_matches_run_m0(R, runs):
    g = pd.read_csv(runs[0] / "m0" / "m0_gate.csv").iloc[0]
    ag = R["gate0"]
    assert (ag.c1_ols, ag.c1_bc, ag.ci[0], ag.ci[1], ag.lags) == pytest.approx(
        (g.c1_ols, g.c1_bias_corrected, g.ci_low, g.ci_high, g.lags))
    prm = pd.read_csv(runs[0] / "m0" / "m0_params.csv", index_col=0)
    assert R["fit0"].c1 == pytest.approx(prm.loc["c1", "estimate"])               # the AR(1) calibration, a different regression
    assert (R["tab0"].loc["c1", "ci_low"], R["tab0"].loc["c1", "ci_high"]) == pytest.approx(
        (prm.loc["c1", "ci_low"], prm.loc["c1", "ci_high"]))


@needs_data
def test_m1_gate_matches_run_m1(R, runs):
    out = runs[0] / "m1"
    gh = pd.read_csv(out / "gate_gregory_hansen.csv").iloc[0]
    assert R["gh"]["stat"] == pytest.approx(gh["stat"])
    assert (R["gh"]["cv_5pct"], R["gh"]["p_value"]) == pytest.approx((gh["cv_5pct"], gh["p_value"]))
    assert str(R["break_date"].date()) == str(gh["break_date"])[:10]
    assert len(R["gh"]["sims"]) == R["gh"]["n_sim"]
    eg = pd.read_csv(out / "gate_engle_granger.csv").set_index(["trend", "rule", "adf_lags"])
    for idx, row in eg.iterrows():
        assert R["eg"].loc[idx, "p_value"] == pytest.approx(row["p_value"])
    ur = pd.read_csv(out / "gate_unit_roots.csv").set_index(["series", "form"])
    assert R["ur"]["adf_p"].to_numpy() == pytest.approx(ur["adf_p"].to_numpy())


@needs_data
def test_estimates_race_and_start_dates_match_run_m1(R, runs):
    out = runs[0] / "m1"
    race = pd.read_csv(out / "race_summary.csv").set_index("h")
    for col in ("rmse_rw_bp", "rmse_m0_bp", "rmse_m1_bp", "p_boot M1 vs M0", "p_boot M1 vs RW"):
        assert R["race"][col].to_numpy() == pytest.approx(race[col].to_numpy())
    prm = pd.read_csv(out / "m1_ecm_params_no_break.csv", index_col=0)
    assert R["m1"].a == pytest.approx(prm.loc["r_lag", "coef"])
    assert R["m1"].fit.tstat("x_lag") == pytest.approx(prm.loc["x_lag", "t"])
    sd = pd.read_csv(out / "robust_start_dates.csv")
    mine = R["starts"].assign(start=lambda d: d["start"].astype(str)).set_index("start")
    for _, row in sd.iterrows():
        assert mine.loc[str(row["start"]), "slope_r_on_x"] == pytest.approx(row["slope_r_on_x"])
        assert mine.loc[str(row["start"]), "EG_p_const(BIC)"] == pytest.approx(row["EG_p_const(BIC)"])


@needs_data
def test_bootstrap_tests_match_the_console_report(R, runs):
    text = runs[1]
    p = lambda v: "<0.001" if v < 0.001 else f"{v:.3f}"
    for key, label in (("no break", "no break"), ("level shift", "level shift")):
        t = R["anchor_tests"][key]
        row = next(l for l in text.splitlines() if l.startswith(label) and f"{t['classical']['F']:.2f}" in l)
        assert p(t["classical"]["p_classical"]) in row and p(t["hac"]["p_hac_F"]) in row and p(t["boot"]["p_boot_F"]) in row
    c = R["coint0"]
    row = next(l for l in text.splitlines() if l.startswith("no break") and f"{c['t']:.2f}" in l)
    assert p(c["p_boot_t_left"]) in row


@needs_data
def test_collected_results_are_consistent(R):
    a = R["anchor"]
    gap = {k: float((R["df"]["r"] - v).abs().mean()) for k, v in a.items()}
    assert gap["M1 + level shift"] < gap["M1"] < gap["M0"]          # distance of SORA from each model's long-run level
    assert R["roll"]["se"].gt(0).all()
    assert R["paths"] is not None and R["acf_band"].shape == (3, R["lags"])
    assert len(R["starts"]) >= 5


@needs_data
def test_rolling_slope_matches_the_package_estimate(R):
    from ssef import cointegration as C
    ref = C.slope_stability(R["df"], windows=(500,), step=1)["rolling"][500]
    both = R["roll"]["slope"].reindex(ref.index).dropna()
    assert len(both) > 100
    assert both.to_numpy() == pytest.approx(ref.loc[both.index].to_numpy(), rel=1e-6)


@needs_data
def test_all_figures_are_written_as_valid_files(R, tmp_path):
    files = plots.make_all(tmp_path, formats=("png", "pdf"), dpi=50, R=R)
    assert len(files) == 2 * len(plots.FIGURES)
    for f in files:
        head = f.read_bytes()[:8]
        assert head.startswith(b"\x89PNG") if f.suffix == ".png" else head.startswith(b"%PDF")


def test_gregory_hansen_returns_scalars_only_unless_sims_are_requested():
    from ssef import cointegration as C
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal(300))
    r = 0.5 * x + rng.standard_normal(300)
    plain = C.gregory_hansen(r, x, "C", 2, n_sim=10)
    assert "sims" not in plain
    withs = C.gregory_hansen(r, x, "C", 2, n_sim=10, return_sims=True)
    assert len(withs["sims"]) == 10 and withs["stat"] == plain["stat"] and withs["p_value"] == plain["p_value"]


def test_command_line_help_runs():
    out = subprocess.run([sys.executable, str(ROOT / "plot_results.py"), "--help"], capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0 and "--fast" in out.stdout
