import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ssef import ecm, loaders, m0, plots

ROOT = Path(__file__).resolve().parent.parent
needs_data = pytest.mark.skipif(not (loaders.SORA_CSV.exists() and loaders.SOFR_CSV.exists() and loaders.FX_CSV.exists()),
                                reason="data CSVs not present")


def _ar1(n, rho, c=0.0, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = c + rho * x[t - 1] + rng.standard_normal()
    return x


# ------------------------------------------------------------------ numbers
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


def test_fit_m0_recovers_known_ar1_parameters():
    r = pd.Series(_ar1(20000, 0.98, c=0.04, seed=3), index=pd.bdate_range("2000-01-03", periods=20000))
    m = plots.fit_m0(r)
    assert m["c1"] == pytest.approx(0.98, abs=0.01)
    assert m["theta"] == pytest.approx(0.04 / 0.02, rel=0.15)
    assert m["valid"]
    assert m["kappa"] == pytest.approx(-np.log(m["c1"]) * 252)


def test_fit_m0_falls_back_to_the_sample_mean_when_there_is_no_reversion():
    rw = pd.Series(np.cumsum(np.random.default_rng(5).standard_normal(300)) + 500, index=pd.bdate_range("2000-01-03", periods=300))
    m = plots.fit_m0(rw)
    if not m["valid"]:                                                   # c1 >= 1 in this draw
        assert m["theta"] == pytest.approx(rw.mean())
        assert np.isnan(m["kappa"])


def test_simulated_m0_paths_have_the_fitted_persistence():
    r = pd.Series(_ar1(3000, 0.97, c=0.05, seed=7), index=pd.bdate_range("2000-01-03", periods=3000))
    m = plots.fit_m0(r)
    paths = plots.simulate_m0(m, r.iloc[0], 2999, 20, seed=1)              # n steps -> n + 1 points
    c1 = [plots.fit_m0(pd.Series(p, index=r.index))["c1"] for p in paths]
    assert np.mean(c1) == pytest.approx(m["c1"], abs=0.01)


@needs_data
def test_statsmodels_fits_agree_with_the_package_estimators():
    r = loaders.load_sora()
    df = loaders.build_dataset(r, loaders.load_sofr(), loaders.make_z(loaders.load_usdsgd()))
    a = plots.fit_m0(r)
    b = m0.calibrate(r)
    assert (a["c0"], a["c1"]) == pytest.approx((b.c0, b.c1), rel=1e-8)
    assert a["kappa"] == pytest.approx(b.kappa, rel=1e-8)
    for bd in (None, df.index[len(df) // 2]):
        f = plots.fit_m1(df, 5, bd)
        g = ecm.fit_ecm(df, 5, 0, True, break_date=bd)
        assert f["alpha"] == pytest.approx(g.alpha, rel=1e-8)
        assert f["beta1"] == pytest.approx(g.beta1, rel=1e-8)
        assert f["c1"] == pytest.approx(g.c1, rel=1e-8)
        if bd is not None:
            assert f["shift"] == pytest.approx(g.level_shift, rel=1e-8)


def test_m1_anchor_tracks_a_known_sofr_link():
    rng = np.random.default_rng(11)
    n = 4000
    idx = pd.bdate_range("2000-01-03", periods=n)
    x = np.cumsum(rng.standard_normal(n)) * 0.02 + 3.0
    r = np.zeros(n)
    for t in range(1, n):                                                # dr = 0.1 (1 + 0.5 x - r) + noise
        r[t] = r[t - 1] + 0.1 * (1.0 + 0.5 * x[t - 1] - r[t - 1]) + 0.05 * rng.standard_normal()
    df = pd.DataFrame({"r": r, "x": x, "z": rng.standard_normal(n) * 0.001}, index=idx)
    f = plots.fit_m1(df, 1)
    assert f["alpha"] == pytest.approx(1.0, abs=0.3)
    assert f["beta1"] == pytest.approx(0.5, abs=0.1)
    assert f["anchor"].iloc[-1] == pytest.approx(1.0 + 0.5 * x[-1], abs=0.4)


# ------------------------------------------------------------------ whole pipeline on the real data
@pytest.fixture(scope="module")
def R():
    return plots.collect(fast=True, verbose=False)


@needs_data
def test_collected_results_are_consistent(R):
    g = R["gap"]
    assert g["M1 + level shift"] < g["M1"] < g["M0"]                    # distance of SORA from each model's anchor
    assert set(R["shocks"]).issubset(set(R["r"].index))
    assert R["roll"]["se"].gt(0).all()
    assert len(R["acf_act"]) == R["lags"] == R["acf_band"].shape[1]
    s = R["race"]
    assert list(s.index) == [1, 20, 60]
    assert s["p_boot M1 vs RW"].between(0, 1).all()
    assert (s["rmse_rw_bp"] > 0).all()


@needs_data
def test_all_figures_are_written_as_valid_files(R, tmp_path):
    files = plots.make_all(tmp_path, formats=("png", "pdf"), dpi=50, R=R)
    assert len(files) == 2 * len(plots.FIGURES)
    for f in files:
        head = f.read_bytes()[:8]
        assert head.startswith(b"\x89PNG") if f.suffix == ".png" else head.startswith(b"%PDF")


def test_command_line_help_runs():
    out = subprocess.run([sys.executable, str(ROOT / "plot_results.py"), "--help"], capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0 and "--fast" in out.stdout
