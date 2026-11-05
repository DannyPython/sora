import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from ssef import gate, loaders, m0, race, report

ROOT = Path(__file__).resolve().parent.parent
HAVE_DATA = loaders.SORA_CSV.exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="SORA CSV not present")
DT = m0.DT_DEFAULT


def ar1(c=0.97, n=600, phi=(), sigma=0.001, seed=0, mean=0.02):
    """AR(1) in levels with optional lagged-difference terms (phi_k on dy_{t-k}); returns a Series."""
    rng = np.random.default_rng(seed)
    p = len(phi)
    y = np.full(n + 60, mean)
    e = rng.standard_normal(n + 60) * sigma
    for t in range(p + 1, n + 60):
        dyl = sum(phi[k] * (y[t - 1 - k] - y[t - 2 - k]) for k in range(p))
        y[t] = mean * (1 - c) + c * y[t - 1] + dyl + e[t]
    return pd.Series(y[60:], index=pd.bdate_range("2015-01-01", periods=n))


# ------------------------------------------------------------------ calibration
def test_calibrate_matches_statsmodels_and_rolling_helper():
    r = ar1(seed=1)
    fit = m0.calibrate(r)
    res = sm.OLS(r.iloc[1:].to_numpy(), sm.add_constant(r.iloc[:-1].to_numpy())).fit()
    assert (fit.c0, fit.c1) == pytest.approx(tuple(res.params))
    assert fit.s2 == pytest.approx(res.mse_resid) and fit.cov_c == pytest.approx(res.cov_params())
    c0, c1, s2, _, e = m0._ols_ar1(r.to_numpy())              # the array version used in rolling loops
    assert (c0, c1, s2) == pytest.approx((fit.c0, fit.c1, fit.s2)) and e == pytest.approx(fit.resid.to_numpy())
    assert fit.resid.index.equals(r.index[1:])


def test_structural_round_trip_and_invalid_c1():
    kappa, theta, sigma = 3.0, 0.025, 0.012
    a = np.exp(-kappa * DT)
    k, th, sg = m0.structural_params((1 - a) * theta, a, sigma ** 2 * (1 - a ** 2) / (2 * kappa), DT)
    assert (k, th, sg) == pytest.approx((kappa, theta, sigma))
    for c1 in (1.0, 1.02, 0.0, -0.3):
        assert all(np.isnan(m0.structural_params(0.001, c1, 1e-6, DT)))


def test_jacobian_matches_numeric():
    c0, c1, s2 = 1.5e-4, 0.9917, 1.66e-6
    J = m0.structural_jacobian(c0, c1, s2, DT)
    x = np.array([c0, c1, s2])
    num = np.zeros((3, 3))
    for j in range(3):
        h = np.zeros(3)
        h[j] = 1e-7 * abs(x[j])
        num[:, j] = (np.array(m0.structural_params(*(x + h), DT)) - np.array(m0.structural_params(*(x - h), DT))) / (2 * h[j])
    assert J == pytest.approx(num, rel=1e-5, abs=1e-8)


def test_simulator_matches_analytic_moments_and_is_seeded():
    kappa, theta, sigma, r0 = 2.0, 0.03, 0.015, 0.01
    paths = m0.simulate_vasicek(kappa, theta, sigma, r0, 252, 100_000, DT, seed=7)
    mean, var = m0.vasicek_moments(kappa, theta, sigma, r0, 252 * DT)
    assert paths[:, -1].mean() == pytest.approx(mean, abs=3e-4) and paths[:, -1].var() == pytest.approx(var, rel=0.03)
    assert (m0.simulate_vasicek(2, 0.03, 0.01, 0.01, 10, 5, DT, 3) == m0.simulate_vasicek(2, 0.03, 0.01, 0.01, 10, 5, DT, 3)).all()


def test_rolling_forecast_has_no_lookahead():
    r = ar1(n=400, seed=2)
    base = m0.rolling_forecast(r, window=100)
    r2 = r.copy()
    r2.iloc[251:] += 0.05                                      # change the future only
    pert = m0.rolling_forecast(r2, window=100)
    same = base.index[base["origin"] <= r.index[250]]
    assert pert.loc[same, "forecast"].to_numpy() == pytest.approx(base.loc[same, "forecast"].to_numpy())


def test_m0_vs_rw_rmse_and_shape():
    r = ar1(n=500, seed=3)
    out = race.m0_vs_rw(r, (1, 5), window=100)
    assert list(out.index) == [1, 5] and (out["n_oos"] > 300).all()
    R = r.to_numpy() * 100
    T = np.arange(100, len(R) - 1)                              # h = 1, window = 100: origins window+h-1 .. n-1-h
    assert out.loc[1, "rmse_rw_bp"] == pytest.approx(np.sqrt(np.mean((R[T + 1] - R[T]) ** 2)) * 100)
    assert out.loc[1, "m0_vs_rw_pct"] == pytest.approx((out.loc[1, "rmse_m0_bp"] / out.loc[1, "rmse_rw_bp"] - 1) * 100)


# ------------------------------------------------------------------ the gate
def test_select_ar_lags_finds_the_true_lags():
    assert gate.select_ar_lags(ar1(c=0.9, n=3000, phi=(0.4, -0.2), seed=4), pmax=6) == 2


def test_adf_gate_with_zero_lags_matches_the_pure_ar1_gate():
    r = ar1(c=0.985, n=600, seed=5)
    a, b = gate.ar1_gate(r, n_boot=199, seed=1), gate.adf_gate(r, p=0, n_boot=199, seed=1)
    assert b.lags == 0 and b.c1_ols == pytest.approx(a.c1_ols, abs=1e-9)
    assert b.c1_bc == pytest.approx(a.c1_bc, abs=0.004)
    assert b.ci[0] == pytest.approx(a.ci[0], abs=0.012) and b.ci[1] == pytest.approx(a.ci[1], abs=0.012)


def test_bias_correction_moves_c1_up_towards_the_truth():
    ols, bc = [], []
    for s in range(30):
        g = gate.adf_gate(ar1(c=0.97, n=300, seed=100 + s), p=0, n_boot=99, grid=np.array([0.97]), seed=s)
        ols.append(g.c1_ols), bc.append(g.c1_bc)
    assert np.mean(ols) < 0.97 and abs(np.mean(bc) - 0.97) < abs(np.mean(ols) - 0.97)


def test_adf_gate_interval_covers_the_truth_in_simulation():
    hits, R, grid = 0, 24, np.linspace(0.93, 1.01, 20)
    for s in range(R):
        g = gate.adf_gate(ar1(c=0.97, n=400, phi=(-0.15, -0.1), seed=200 + s), p=2, n_boot=49, grid=grid, seed=s)
        hits += g.ci[0] <= 0.97 <= g.ci[1]
    assert hits / R >= 0.75                                     # nominal 95%; loose bound for a small, fast simulation


def test_adf_gate_rarely_declares_reversion_on_a_random_walk():
    rng = np.random.default_rng(6)
    declared = 0
    for s in range(12):
        rw = pd.Series(0.02 + np.cumsum(rng.standard_normal(500) * 0.001))
        declared += gate.adf_gate(rw, p=1, n_boot=49, grid=np.linspace(0.93, 1.01, 15), seed=s).reverts
    assert declared <= 3


# ------------------------------------------------------------------ the report
def _gate_stub(ci, c1_bc=0.995):
    return gate.AR1Gate(n=1000, c1_ols=0.99, c1_bc=c1_bc, ci=ci, alpha=0.05, lags=2)


def test_kappa_range_from_the_bootstrap_interval():
    k_lo, k_hi = report._kappa_range(_gate_stub((0.98, 1.005)))
    assert k_lo == 0.0 and k_hi == pytest.approx(-np.log(0.98) / DT)
    k_lo, k_hi = report._kappa_range(_gate_stub((0.95, 0.99)))
    assert k_lo == pytest.approx(-np.log(0.99) / DT) and k_hi == pytest.approx(-np.log(0.95) / DT)


def _report(r, ag):
    fit = m0.calibrate(r)
    return report.format_report(r, gate.unit_root_tests(r), fit, m0.rolling_forecast(r, 250), 250, None, 1, ag,
                                race.m0_vs_rw(r, (1, 20), 250))


def test_report_headline_follows_the_gate():
    rw = pd.Series(0.02 + np.cumsum(np.random.default_rng(7).standard_normal(1000) * 0.001), index=pd.bdate_range("2019-01-01", periods=1000))
    failed = _report(rw, _gate_stub((0.98, 1.005)))
    assert "What M0 can say about SORA's persistence" in failed and "not identified" in failed
    assert "Calibrated M0 parameters" not in failed and "For reference only" in failed          # OLS kappa demoted
    mr = ar1(c=0.9, n=1000, seed=8)
    passed = _report(mr, _gate_stub((0.88, 0.93), c1_bc=0.905))
    assert "Calibrated M0 parameters" in passed and "What M0 can say" not in passed
    assert all(ord(ch) < 128 for ch in failed + passed)


# ------------------------------------------------------------------ end to end
@needs_data
def test_run_m0_driver(tmp_path):
    cmd = [sys.executable, str(ROOT / "run_m0.py"), "--lags", "2", "--n-boot", "29", "--out", str(tmp_path)]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "Bottom line" in out.stdout and all(ord(ch) < 128 for ch in out.stdout)
    info = json.loads((tmp_path / "run_info.json").read_text())
    assert {"git_commit", "data_sha256", "args", "gate_lags"} <= set(info) and info["gate_lags"] == 2
    assert (tmp_path / "m0_multi_horizon.csv").exists() and (tmp_path / "m0_gate.csv").exists()
