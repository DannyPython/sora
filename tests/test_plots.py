import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ssef import loaders, plots

ROOT = Path(__file__).resolve().parent.parent
needs_data = pytest.mark.skipif(not (loaders.SORA_CSV.exists() and loaders.SOFR_CSV.exists() and loaders.FX_CSV.exists()),
                                reason="data CSVs not present")


# ------------------------------------------------------------------ helpers
def test_acf_matches_the_definition_and_white_noise_is_near_zero():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(5000)
    a = plots._acf(x, 5)[0]
    xc = x - x.mean()
    assert a[0] == pytest.approx((xc[1:] @ xc[:-1]) / (xc @ xc))
    assert np.abs(a).max() < 0.06
    ar = np.zeros(5000)
    for t in range(1, 5000):
        ar[t] = 0.6 * ar[t - 1] + x[t]
    assert plots._acf(ar, 3)[0][0] == pytest.approx(0.6, abs=0.05)


def test_variance_ratio_is_one_for_a_random_walk_and_below_one_for_mean_reversion():
    rng = np.random.default_rng(1)
    rw = np.cumsum(rng.standard_normal((200, 2000)), axis=1)
    assert plots._vr(rw, [5, 20, 60]).mean(axis=0) == pytest.approx([1.0, 1.0, 1.0], abs=0.08)
    ou = np.zeros((200, 2000))
    e = rng.standard_normal((200, 2000))
    for t in range(1, 2000):
        ou[:, t] = 0.9 * ou[:, t - 1] + e[:, t]
    hs = np.array([5, 20, 60])
    exact = (1 - 0.9 ** hs) / (hs * (1 - 0.9))                 # AR(1): VR(h) = (1 - rho^h) / (h (1 - rho))
    assert plots._vr(ou, hs).mean(axis=0) == pytest.approx(exact, abs=0.05)


def test_label_dodging_separates_neighbours_keeps_order_and_stays_centred():
    ys = [2.26, 2.24, 1.93, 1.33, 0.74]
    out = plots._dodge(ys, 0.17)
    assert all(b - a >= 0.17 - 1e-9 for a, b in zip(sorted(out), sorted(out)[1:]))
    assert np.argsort(out).tolist() == np.argsort(ys).tolist()
    assert np.mean(out) == pytest.approx(np.mean(ys))
    assert plots._dodge([1.0, 2.0, 3.0], 0.1) == pytest.approx([1.0, 2.0, 3.0])          # already apart: unchanged


def test_palette_is_the_validated_one():
    assert (plots.M0C, plots.M1C, plots.SOFRC) == ("#2a78d6", "#eb6834", "#1baf7a")


def test_cli_help_runs():
    out = subprocess.run([sys.executable, str(ROOT / "plot_results.py"), "--help"], capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert out.returncode == 0 and "--fast" in out.stdout


# ------------------------------------------------------------------ the figures
@pytest.fixture(scope="module")
def results():
    return plots.collect(fast=True, verbose=False)


@needs_data
def test_collected_numbers_have_the_expected_structure_and_ordering(results):
    R = results
    n = len(R["df"])
    assert all(len(a) == n for a in R["anchor"].values())
    assert R["df"].index[0] < R["break_date"] < R["df"].index[-1]
    # the point of the comparison: M1's moving anchor sits closer to SORA than M0's constant, with or without a break
    assert R["gap"]["m1_break"] < R["gap"]["m1_nobreak"] < R["gap"]["m0"]
    assert R["corr"]["m0"] == pytest.approx(0.0, abs=1e-6) and R["corr"]["m1_break"] > 0.9
    assert R["fan_m0"].shape == (3, plots.N_DAYS) and all(f.shape == (3, plots.N_DAYS) for f in R["fan_m1"].values())
    assert (R["fan_m0"][0] <= R["fan_m0"][1]).all() and (R["fan_m0"][1] <= R["fan_m0"][2]).all()
    assert list(R["race"]["summary"].index) == [1, 20, 60]
    assert R["acf_act"].shape == (10,) and R["acf_m0"][0].shape == (10,)
    # SORA reverses at short lags and swings fade faster than M0 implies (the claims the M0 figure makes)
    assert R["acf_act"][0] < R["acf_m0"][1][0] and R["vr_act"][R["vr_h"].tolist().index(20)] < R["vr_m0"][1][R["vr_h"].tolist().index(20)]


@needs_data
def test_all_figures_are_written_as_valid_images(results, tmp_path):
    files = plots.make_all(tmp_path, fast=True, formats=("png", "pdf"), dpi=60, verbose=False, R=results)
    assert len(files) == 2 * len(plots.FIGURES)
    for f in files:
        head = f.read_bytes()[:8]
        assert f.stat().st_size > 5_000
        assert head.startswith(b"\x89PNG") if f.suffix == ".png" else head.startswith(b"%PDF")


@needs_data
def test_the_race_headline_follows_the_p_value(results):
    plots._style()
    R = copy.deepcopy({k: results[k] for k in ("race", "break_date")})
    S = R["race"]["summary"]
    for p1, expected in ((0.01, "adds a little at one day"), (0.40, "adds nothing we can detect")):
        S["p_boot M1 vs M0"] = [p1, 0.5, 0.6]
        fig = plots.fig_race(R)
        assert expected in fig._suptitle.get_text()
        plots.plt.close(fig)
