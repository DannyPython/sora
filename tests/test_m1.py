import dataclasses
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

from ssef import cointegration as C
from ssef import ecm, loaders, nested, noise, race, simulate
from ssef.ols import exclusion_f, fit_ols, subset

ROOT = Path(__file__).resolve().parent.parent
HAVE_DATA = loaders.SORA_CSV.exists() and loaders.SOFR_CSV.exists() and loaders.FX_CSV.exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="data CSVs not present")


# ------------------------------------------------------------------ helpers
def sim_system(n=600, a=-0.05, b1=0.8, alpha=0.002, sigma=0.0008, seed=0, cointegrated=True, shift=None):
    """x: I(1) random walk. r follows an ECM around alpha + b1 x (cointegrated) or is an independent random walk."""
    rng = np.random.default_rng(seed)
    x = 0.02 + np.cumsum(rng.normal(0, 0.0006, n))
    z = rng.normal(0, 0.002, n)
    r = np.empty(n)
    r[0] = alpha + b1 * x[0]
    for i in range(1, n):
        if cointegrated:
            anchor = alpha + b1 * x[i - 1] + (shift[i - 1] if shift is not None else 0.0)
            r[i] = r[i - 1] + a * (r[i - 1] - anchor) + sigma * rng.standard_normal()   # a < 0 pulls r to the anchor
        else:
            r[i] = r[i - 1] + sigma * rng.standard_normal()
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame({"r": r, "x": x, "z": z}, index=idx)


# ------------------------------------------------------------------ OLS layer
def test_fit_ols_matches_statsmodels():
    df = sim_system(400, seed=1)
    y, X = ecm.design(df, 2, 0, True)
    f = fit_ols(y, X)
    s = sm.OLS(y.to_numpy(), X.to_numpy()).fit()
    assert f.coef == pytest.approx(s.params)
    assert f.s2 == pytest.approx(s.mse_resid)
    assert f.cov == pytest.approx(s.cov_params())
    assert f.hac_se(5) == pytest.approx(s.get_robustcov_results("HAC", maxlags=5, use_correction=True).bse, rel=1e-6)


def test_nested_f_matches_statsmodels_and_m0_is_columns_dropped():
    df = sim_system(400, seed=2)
    y, X = ecm.design(df, 2, 0, True)
    full = fit_ols(y, X)
    drop = ecm.nested_columns(full.names, "anchor")
    red = subset(full, drop)
    assert red.n == full.n and red.names == tuple(c for c in full.names if c not in drop)   # same sample
    ours = exclusion_f(full, red)
    F, p, q = sm.OLS(y.to_numpy(), X.to_numpy()).fit().compare_f_test(sm.OLS(y.to_numpy(), X[list(red.names)].to_numpy()).fit())
    assert ours["F"] == pytest.approx(F) and ours["p_classical"] == pytest.approx(p) and ours["q"] == q


def test_ecm_equals_levels_regression_residuals():
    """Levels r_i on (1, r_{i-1}, x_{i-1}, x_i) spans the same space as the ECM with dx0: same residuals."""
    df = sim_system(300, seed=3)
    y, X = ecm.design(df, 0, 0, False)
    e_ecm = fit_ols(y, X).resid.to_numpy()
    lv = pd.DataFrame({"c": 1.0, "r1": df.r.shift(1), "x1": df.x.shift(1), "x0": df.x}, index=df.index).iloc[1:]
    e_lv = fit_ols(df.r.iloc[1:], lv).resid.to_numpy()
    assert e_ecm == pytest.approx(e_lv)


# ------------------------------------------------------------------ ECM recovery
def test_ecm_recovers_structural_parameters():
    a, b1, alpha = -0.05, 0.8, 0.002
    ests = [ecm.fit_ecm(sim_system(3000, a, b1, alpha, seed=s), 1, 0, True) for s in range(5)]
    assert np.mean([m.a for m in ests]) == pytest.approx(a, abs=0.012)
    assert np.mean([m.beta1 for m in ests]) == pytest.approx(b1, abs=0.1)
    m = ests[0]
    assert m.valid and 0 < m.c1 < 1 and m.kappa == pytest.approx(-np.log(1 + m.a) * 252)


def test_level_shift_recovered():
    n = 1500
    shift = np.r_[np.zeros(n // 2), np.full(n - n // 2, -0.01)]
    df = sim_system(n, a=-0.1, b1=0.8, seed=4, shift=shift)
    m = ecm.fit_ecm(df, 1, 0, True, break_date=df.index[n // 2])
    assert m.has_shift and m.level_shift == pytest.approx(-0.01, abs=0.003)


def test_simulate_ecm_reproduces_fitted_one_step_with_zero_noise():
    """Catches any lag / dx / dr indexing mistake: one simulated step must equal the fitted value for the
    next actual row when fed the actual next x and z."""
    df = sim_system(500, seed=5)
    full = ecm.fit_ecm(df, 3, 1, True)
    head = df.iloc[:-1]
    m = ecm.fit_ecm(head, 3, 1, True)
    m0 = dataclasses.replace(m, fit=dataclasses.replace(m.fit, s2=0.0))
    path = simulate.simulate_ecm(m0, head, [df["x"].iloc[-1]], [df["z"].iloc[-1]], n_paths=3, seed=1)
    y, X = ecm.design(df, 3, 1, True)
    expected = df["r"].iloc[-2] + float(X.iloc[-1].to_numpy() @ m.fit.coef)
    assert path[:, 1] == pytest.approx(expected)
    assert full.fit.n == m.fit.n + 1


# ------------------------------------------------------------------ data alignment
def test_known_sofr_has_no_lookahead():
    eff = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"])
    sofr = pd.DataFrame({"sofr": [0.05, 0.051, 0.052, 0.053, 0.054],
                         "pub_date": pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", pd.NaT])},
                        index=pd.DatetimeIndex(eff, name="effective"))
    dates = pd.DatetimeIndex(["2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"])
    strict = loaders.known_sofr(sofr, dates, strict=True)
    same = loaders.known_sofr(sofr, dates, strict=False)
    # strict: SOFR published on d is usable from d+1; same-day: from d
    assert np.isnan(strict.iloc[0]) and strict.iloc[1] == 0.05 and strict.iloc[2] == 0.051
    assert strict.iloc[3] == 0.052 and strict.iloc[4] == 0.053            # 2024-01-05 published value, Mon 8th row gets Fri's
    assert same.iloc[0] == 0.05 and same.iloc[1] == 0.051 and same.iloc[3] == 0.053
    for v, x in strict.dropna().items():
        pub = sofr.loc[sofr.sofr == x, "pub_date"].iloc[0]
        assert pub < v


@needs_data
def test_real_data_loads_clean():
    r = loaders.load_sora()
    assert r.index.is_monotonic_increasing and not r.index.has_duplicates and 0 < r.min() < r.max() < 0.2
    df = loaders.build_dataset(r, loaders.load_sofr(), loaders.make_z(loaders.load_usdsgd()))
    assert df.notna().all().all() and {"r", "x", "z"} <= set(df.columns)
    sub = loaders.load_sora(start="2024-01-01", end="2024-12-31")
    assert sub.index.min() >= pd.Timestamp("2024-01-01") and sub.index.max() <= pd.Timestamp("2024-12-31")


# ------------------------------------------------------------------ gate
def test_engle_granger_matches_statsmodels():
    df = sim_system(500, seed=6)
    eg = C.engle_granger(df.r, df.x, lags=[2], trends=("c",))
    t, p, _ = coint(df.r, df.x, trend="c", maxlag=2, autolag=None)
    assert eg.loc[("c", "fixed", 2), "t_stat"] == pytest.approx(t)
    assert eg.loc[("c", "fixed", 2), "p_value"] == pytest.approx(p)


def test_engle_granger_power_and_size():
    def rate(coint_flag):
        hits = 0
        for s in range(30):
            d = sim_system(500, a=-0.15, seed=300 + s, cointegrated=coint_flag)
            hits += C.engle_granger(d.r, d.x, lags=[2], trends=("c",)).iloc[0]["p_value"] < 0.05
        return hits / 30
    assert rate(True) >= 0.8 and rate(False) <= 0.25


def test_gregory_hansen_finds_level_shift_that_engle_granger_misses():
    n = 600
    shift = np.r_[np.zeros(n // 2), np.full(n - n // 2, -0.012)]
    df = sim_system(n, a=-0.1, b1=0.8, sigma=0.0004, seed=7, shift=shift)
    eg_p = C.engle_granger(df.r, df.x, lags=[2], trends=("c",)).iloc[0]["p_value"]
    gh = C.gregory_hansen(df.r, df.x, "C", lags=2, n_sim=60, seed=1, dates=df.index, null="iid")
    assert gh["reject_5pct"] and abs(gh["break_index"] - n // 2) < 40
    assert eg_p > gh["p_value"]


def test_gregory_hansen_size_under_null():
    rej = 0
    for s in range(20):
        ind = sim_system(300, seed=100 + s, cointegrated=False)
        rej += C.gregory_hansen(ind.r, ind.x, "C", lags=2, n_sim=40, seed=s, null="iid")["reject_5pct"]
    assert rej <= 5            # nominal 5% of 20 = 1; generous bound against Monte Carlo noise


# ------------------------------------------------------------------ bootstrap tests: size and power
def test_bootstrap_anchor_test_has_size_and_classical_f_does_not():
    boot_rej = classical_rej = 0
    R = 100
    for s in range(R):
        df = sim_system(300, seed=200 + s, cointegrated=False)
        m = ecm.fit_ecm(df, 1, 0, True)
        classical_rej += exclusion_f(m.fit, subset(m.fit, ecm.nested_columns(m.fit.names, "anchor")))["p_classical"] < 0.05
        boot_rej += nested.anchor_test(df, 1, 0, True, B=99, seed=s)["p_boot_F"] < 0.05
    assert boot_rej / R <= 0.10                       # close to the nominal 5%
    assert classical_rej >= boot_rej + 3              # the classical test over-rejects with an I(1) regressor


def test_bootstrap_statistic_equals_classical_f_on_actual_data():
    """The F computed inside the bootstrap must be the ordinary nested F on the actual sample."""
    df = sim_system(400, seed=15)
    for brk in (None, df.index[200]):
        dgp = nested.NullDGP(df, 2, 0, True, ["x_lag", "z"], "unit_root", break_date=brk)
        F_boot_code, t = nested._stats(dgp.y, dgp.X, dgp.names, ["x_lag", "z"], "x_lag")
        m = ecm.fit_ecm(df, 2, 0, True, break_date=brk)
        ours = exclusion_f(m.fit, subset(m.fit, ecm.nested_columns(m.fit.names, "anchor")))
        assert F_boot_code == pytest.approx(ours["F"]) and t == pytest.approx(m.fit.tstat("x_lag"))


def test_bootstrap_cointegration_test_power():
    df = sim_system(500, a=-0.05, seed=8)
    out = nested.cointegration_ecm_test(df, 1, 0, True, B=199, seed=1)
    assert out["p_boot_t_left"] < 0.05 and out["p_boot_F"] < 0.05
    ind = sim_system(500, seed=8, cointegrated=False)
    assert nested.cointegration_ecm_test(ind, 1, 0, True, B=199, seed=1)["p_boot_t_left"] > 0.05


def test_null_dgp_keeps_exogenous_series_and_is_seeded():
    df = sim_system(300, seed=9)
    dgp = nested.NullDGP(df, 2, 0, True, ["x_lag", "z"], "unit_root")
    r1, r2 = dgp.draw(np.random.default_rng(3)), dgp.draw(np.random.default_rng(3))
    assert len(r1) == len(df) and (r1 == r2).all()
    assert r1[:dgp.s] == pytest.approx(df.r.to_numpy()[:dgp.s])
    assert not np.allclose(r1, df.r.to_numpy())


# ------------------------------------------------------------------ race
def test_rolling_matches_bruteforce_ols():
    df = sim_system(260, seed=10)
    R, XS = df.r.to_numpy() * 100, df.x.to_numpy() * 100
    X = np.column_stack([np.ones(len(R)), R, XS])
    h, w = 5, 120
    T, f = race._rolling(R, X, h, w)
    for k in (0, 7, len(T) - 1):
        t = T[k]
        pairs = np.arange(t - h - w + 1, t - h + 1)
        beta = np.linalg.lstsq(X[pairs], R[pairs + h], rcond=None)[0]
        assert f[k] == pytest.approx(X[t] @ beta, rel=1e-8)


def test_race_forecasts_do_not_look_ahead():
    df = sim_system(400, seed=11)
    r, x, z = df.r.to_numpy(), df.x.to_numpy(), df.z.to_numpy()
    base = race.direct_forecasts(r, x, z, 5, 100, True)
    k = 300
    r2 = r.copy()
    r2[k:] += 0.5
    pert = race.direct_forecasts(r2, x, z, 5, 100, True)
    ok = base["T"] < k                                   # origins before the perturbation
    for key in ("rw", "m0", "m1"):
        assert pert[key][ok] == pytest.approx(base[key][ok])


def test_clark_west_formula_and_sign():
    rng = np.random.default_rng(0)
    y = rng.normal(size=500)
    fs = np.zeros(500)
    fl = 0.7 * y + rng.normal(scale=0.5, size=500)       # a genuinely informative larger model
    f = (y - fs) ** 2 - ((y - fl) ** 2 - (fs - fl) ** 2)
    assert race.clark_west(y, fs, fl, 1) == pytest.approx(race._nw_t(f, 0))
    assert race.clark_west(y, fs, fl, 1) > 5
    assert race.clark_west(y, fs, fs + rng.normal(scale=0.5, size=500), 1) < 3   # pure noise: no gain


def test_race_summary_shape_and_bootstrap_columns():
    df = sim_system(500, seed=12)
    res = race.run_race(df, (1, 10), window=150, B=9, p=1, seed=1)
    s = res["summary"]
    assert list(s.index) == [1, 10] and {"rmse_rw_bp", "cw_t M1 vs M0", "p_boot M1 vs M0"} <= set(s.columns)
    assert s["p_boot M1 vs M0"].between(0, 1).all()


# ------------------------------------------------------------------ noise
def test_noise_helpers():
    df = sim_system(500, seed=13)
    m = ecm.fit_ecm(df, 1, 0, True)
    z = noise.std_resid(m.fit)
    assert z.mean() == pytest.approx(0, abs=1e-8) and z.std(ddof=m.fit.k) == pytest.approx(1, rel=0.05)
    g = noise.draw_noise(m.s_eps, 50, 2000, 1)
    bt = noise.draw_noise(m.s_eps, 50, 2000, 1, "bootstrap", z.to_numpy())
    assert g.std() == pytest.approx(m.s_eps, rel=0.02) and bt.std() == pytest.approx(m.s_eps, rel=0.02)
    assert (noise.draw_noise(m.s_eps, 5, 3, 7) == noise.draw_noise(m.s_eps, 5, 3, 7)).all()
    assert {"excess kurtosis", "Jarque-Bera"} <= set(noise.noise_diagnostics(m.fit.resid).index)


def test_lag_selection_prefers_true_lags():
    rng = np.random.default_rng(14)
    n = 3000
    e = rng.normal(size=n) * 0.001
    dr = np.zeros(n)
    for i in range(2, n):
        dr[i] = 0.4 * dr[i - 1] - 0.2 * dr[i - 2] + e[i]
    df = pd.DataFrame({"r": 0.02 + np.cumsum(dr), "x": 0.02 + np.cumsum(rng.normal(0, 0.0005, n))},
                      index=pd.bdate_range("2010-01-01", periods=n))
    sel = ecm.select_lags(df, 6, 0, False)
    assert sel["bic"].idxmin() == 2


# ------------------------------------------------------------------ end to end
@needs_data
def test_driver_runs_and_is_reproducible(tmp_path):
    cmd = [sys.executable, str(ROOT / "run_m1.py"), "--fast", "--seed", "5", "--out", str(tmp_path)]
    a = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=300)
    assert a.returncode == 0, a.stderr[-2000:]
    for key in ("1. Data checks", "2. Gate", "Gregory-Hansen", "4. Does M1 beat M0?", "5. Out-of-sample race", "Bottom line"):
        assert key in a.stdout
    assert all(ord(ch) < 128 for ch in a.stdout)
    assert (tmp_path / "run_info.json").exists() and (tmp_path / "race_summary.csv").exists()
    b = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=300)
    assert a.stdout == b.stdout                          # fixed seeds -> identical report
