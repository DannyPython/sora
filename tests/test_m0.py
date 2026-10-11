import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from ssef import data, gate, m0

DT = m0.DT_DEFAULT


def _ou_series(kappa=5.0, theta=0.02, sigma=0.01, n=20000, seed=1):
    paths = m0.simulate_vasicek(kappa, theta, sigma, theta, n, 1, DT, seed)
    return pd.Series(paths[0], index=pd.bdate_range("2000-01-03", periods=n + 1))


def test_structural_round_trip():
    kappa, theta, sigma = 3.0, 0.025, 0.012
    a = np.exp(-kappa * DT)
    c0, c1 = (1 - a) * theta, a
    s2 = sigma ** 2 * (1 - a ** 2) / (2 * kappa)
    k, th, sg = m0.structural_params(c0, c1, s2, DT)
    assert (k, th, sg) == pytest.approx((kappa, theta, sigma))


@pytest.mark.parametrize("c1", [1.0, 1.02, 0.0, -0.3])
def test_invalid_c1_gives_nan(c1):
    assert all(np.isnan(m0.structural_params(0.001, c1, 1e-6, DT)))


def test_jacobian_matches_numeric():
    c0, c1, s2 = 1.5e-4, 0.9917, 1.66e-6
    J = m0.structural_jacobian(c0, c1, s2, DT)
    x = np.array([c0, c1, s2])
    num = np.zeros((3, 3))
    for j in range(3):
        h = np.zeros(3)
        h[j] = 1e-7 * abs(x[j])
        up = np.array(m0.structural_params(*(x + h), DT))
        dn = np.array(m0.structural_params(*(x - h), DT))
        num[:, j] = (up - dn) / (2 * h[j])
    assert J == pytest.approx(num, rel=1e-5, abs=1e-8)


def test_ols_matches_statsmodels():
    r = _ou_series(n=500)
    fit = m0.calibrate(r)
    res = sm.OLS(r.iloc[1:].to_numpy(), sm.add_constant(r.iloc[:-1].to_numpy())).fit()
    assert (fit.c0, fit.c1) == pytest.approx(tuple(res.params))
    assert fit.s2 == pytest.approx(res.mse_resid)
    assert fit.cov_c == pytest.approx(res.cov_params())


def test_recovers_known_parameters_and_ci_coverage():
    true = {"kappa": 5.0, "theta": 0.02, "sigma": 0.01}
    reps, hits = 200, dict.fromkeys(true, 0)
    for s in range(reps):
        fit = m0.calibrate(_ou_series(n=3000, seed=100 + s))
        assert fit.valid
        tab = fit.params_table()
        for k, v in true.items():
            hits[k] += tab.loc[k, "ci_low"] <= v <= tab.loc[k, "ci_high"]
    for k in true:                                  # nominal 95%, MC noise ~1.5pp
        assert 0.88 <= hits[k] / reps <= 0.99, (k, hits[k] / reps)
    big = m0.calibrate(_ou_series(n=50000, seed=1))
    assert big.sigma == pytest.approx(true["sigma"], rel=0.02)


def test_std_resid_unit_variance():
    fit = m0.calibrate(_ou_series(n=2000))
    assert fit.std_resid.std(ddof=2) == pytest.approx(1.0, rel=1e-3)
    assert fit.std_resid.mean() == pytest.approx(0.0, abs=1e-9)


def test_simulator_matches_analytic_moments():
    kappa, theta, sigma, r0, T = 2.0, 0.03, 0.015, 0.01, 1.0
    paths = m0.simulate_vasicek(kappa, theta, sigma, r0, 252, 200000, DT, seed=7)
    mean, var = m0.vasicek_moments(kappa, theta, sigma, r0, 252 * DT)
    assert paths[:, -1].mean() == pytest.approx(mean, abs=2e-4)
    assert paths[:, -1].var() == pytest.approx(var, rel=0.02)


def test_simulation_is_seeded():
    a = m0.simulate_vasicek(2, 0.03, 0.01, 0.01, 10, 5, DT, seed=3)
    b = m0.simulate_vasicek(2, 0.03, 0.01, 0.01, 10, 5, DT, seed=3)
    assert (a == b).all()


def test_rolling_forecast_has_no_lookahead():
    r = _ou_series(n=400)
    base = m0.rolling_forecast(r, window=100)
    cut = 250
    r2 = r.copy()
    r2.iloc[cut + 1:] += 0.05                      # perturb the future only
    pert = m0.rolling_forecast(r2, window=100)
    # forecasts whose origin is <= cut use only r[:cut+1] and must be unchanged
    same = base.index[base["origin"] <= r.index[cut]]
    assert pert.loc[same, "forecast"].to_numpy() == pytest.approx(base.loc[same, "forecast"].to_numpy())
    assert len(base) == len(r) - 100


def test_rolling_forecast_errors_consistent():
    fc = m0.rolling_forecast(_ou_series(n=300), window=60)
    assert (fc["actual"] - fc["forecast"]).to_numpy() == pytest.approx(fc["error"].to_numpy())
    assert m0.rmse(fc) == pytest.approx(np.sqrt((fc["error"] ** 2).mean()))


def test_gate_flags_random_walk_and_ou():
    rng = np.random.default_rng(0)
    rw = pd.Series(np.cumsum(rng.standard_normal(1500)))
    assert gate.integration_order(gate.unit_root_tests(rw)) == "I(1)"
    assert gate.integration_order(gate.unit_root_tests(_ou_series(n=1500, kappa=30))) == "I(0)"


def test_real_data_loads():
    r = data.load_sora()
    assert r.index.is_monotonic_increasing and not r.index.has_duplicates
    assert r.notna().all() and 0.0 < r.min() < r.max() < 0.10     # decimals, not percent
    assert len(r) > 600
    sub = data.load_sora(start="2025-01-01", end="2025-12-31")
    assert sub.index.min() >= pd.Timestamp("2025-01-01") and sub.index.max() <= pd.Timestamp("2025-12-31")


def test_real_data_calibration_runs():
    fit = m0.calibrate(data.load_sora())
    assert 0.9 < fit.c1 < 1.01
    assert fit.params_table().shape == (6, 4)
