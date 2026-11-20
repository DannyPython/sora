import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ssef import backtest as bt
from ssef import race

from test_m1 import HAVE_DATA, sim_system

ROOT = Path(__file__).resolve().parent.parent
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="data CSVs not present")


# ------------------------------------------------------------------ coverage tests
def test_kupiec_textbook_value():
    # 250 days, 10 breaches of a 99% VaR: LR_pof = 12.96 (Jorion's textbook example), rejected
    k = bt.kupiec(np.r_[np.ones(10), np.zeros(240)], 0.01)
    assert k["LR_pof"] == pytest.approx(12.955, abs=1e-3)
    assert k["p_pof"] < 0.001 and k["breaches"] == 10 and k["expected"] == pytest.approx(2.5)
    # exactly the expected rate: LR = 0
    assert bt.kupiec(np.r_[np.ones(5), np.zeros(495)], 0.01)["LR_pof"] == pytest.approx(0.0, abs=1e-12)
    # no breach at all is still defined (0 log 0 = 0)
    k0 = bt.kupiec(np.zeros(500), 0.01)
    assert np.isfinite(k0["LR_pof"]) and k0["LR_pof"] == pytest.approx(-2 * 500 * np.log(0.99))


def test_christoffersen_counts_and_decomposition():
    hits = np.array([0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0])
    n00, n01, n10, n11 = (int(v) for v in bt.transition_counts(hits))
    assert (n00, n01, n10, n11) == (4, 3, 3, 3) and n00 + n01 + n10 + n11 == len(hits) - 1
    c = bt.christoffersen(hits, 0.05)
    # hand computation of LR_ind
    pi0, pi1, pi = 3 / 7, 3 / 6, 6 / 13
    ll1 = 4 * np.log(1 - pi0) + 3 * np.log(pi0) + 3 * np.log(1 - pi1) + 3 * np.log(pi1)
    ll0 = 7 * np.log(1 - pi) + 6 * np.log(pi)
    assert c["LR_ind"] == pytest.approx(-2 * (ll0 - ll1))
    assert c["LR_cc"] == pytest.approx(bt.kupiec(hits[1:], 0.05)["LR_pof"] + c["LR_ind"])
    # no breach: nothing to test
    assert np.isnan(bt.christoffersen(np.zeros(100), 0.01)["p_ind"])


def test_christoffersen_detects_clustering_but_not_iid():
    rng = np.random.default_rng(3)
    clustered = bt.markov_hits(1000, 0.05, 0.5, 1, rng)[0]
    assert bt.christoffersen(clustered, 0.05)["p_ind"] < 0.01
    assert clustered.mean() == pytest.approx(0.05, abs=0.03)          # right unconditional rate by construction
    iid = bt.markov_hits(1000, 0.05, 0.05, 400, rng)
    pv = stats.chi2.sf(bt.lr_ind(*bt.transition_counts(iid)), 1)
    assert 0.01 < np.mean(pv < 0.05) < 0.10                           # size close to 5%
    # the vectorised LR agrees with the scalar one
    assert bt.lr_ind(*bt.transition_counts(iid[0])) == pytest.approx(bt.christoffersen(iid[0], 0.05)["LR_ind"])


def test_kupiec_power_is_exact_binomial():
    pw = bt.kupiec_power(500, 0.01, (1.0, 2.0, 4.0)).set_index("true_rate_multiple")
    assert pw.loc[1.0, "power"] < 0.08                                 # size near the nominal 5%
    assert pw.loc[1.0, "power"] < pw.loc[2.0, "power"] < pw.loc[4.0, "power"] <= 1.0
    # brute force: simulate the rejection rate when the true rate is doubled
    rng = np.random.default_rng(0)
    x = rng.binomial(500, 0.02, 20000)
    crit = stats.chi2.ppf(0.95, 1)
    lr = np.array([bt.kupiec(np.r_[np.ones(k), np.zeros(500 - k)], 0.01)["LR_pof"] for k in range(501)])
    assert np.mean(lr[x] > crit) == pytest.approx(pw.loc[2.0, "power"], abs=0.01)
    lo, hi = pw.loc[1.0, ["accept_breaches_min", "accept_breaches_max"]]
    assert lo <= 5 <= hi


# ------------------------------------------------------------------ EVT tail
@pytest.mark.parametrize("fit", ["pwm", "mle"])
def test_gpd_fit_recovers_parameters(fit):
    xi, beta = 0.2, 0.7
    y = stats.genpareto.rvs(xi, scale=beta, size=20000, random_state=1)
    f = {"pwm": bt.gpd_fit_pwm, "mle": bt.gpd_fit_mle}[fit]
    xh, bh = f(y)
    assert xh == pytest.approx(xi, abs=0.03) and bh == pytest.approx(beta, rel=0.05)


def test_evt_quantile_matches_true_quantile_of_fat_tailed_noise():
    nu = 4
    U = stats.t.rvs(nu, size=(3, 20000), random_state=2)
    for prob in (0.01, 0.001):
        q = bt.evt_quantiles(U, [prob], tail_frac=0.05)[prob]
        assert q == pytest.approx(np.full(3, stats.t.ppf(1 - prob, nu)), rel=0.08)
    # beyond the threshold the GPD quantile is above the Gaussian one for fat tails
    q = bt.evt_quantiles(U / U.std(axis=1, keepdims=True), [0.001], 0.05)[0.001]
    assert np.all(q > stats.norm.ppf(0.999))
    with pytest.raises(ValueError):
        bt.evt_quantiles(U, [0.2], tail_frac=0.10)


def test_gpd_quantile_exponential_limit():
    # xi -> 0 is the exponential tail: u0 - beta log(prob / zeta)
    assert bt.gpd_quantile(1.0, 0.0, 0.5, 0.1, 0.01) == pytest.approx(1.0 + 0.5 * np.log(10))
    assert bt.gpd_quantile(1.0, 1e-4, 0.5, 0.1, 0.01) == pytest.approx(1.0 + 0.5 * np.log(10), rel=1e-3)


# ------------------------------------------------------------------ predictive distributions
@pytest.mark.parametrize("h", [1, 5])
def test_point_forecasts_are_the_race_forecasts(h):
    df = sim_system(500, seed=21)
    r, x, z = df.r.to_numpy(), df.x.to_numpy(), df.z.to_numpy()
    fc = race.direct_forecasts(r, x, z, h, 150, True, 2, 0)
    ps = bt.predictive_sets(r, x, z, h, 150, True, 2, 0, step=1)
    assert np.array_equal(ps["T"], fc["T"])
    assert ps["actual"] * ps["scale"] == pytest.approx(fc["actual"])
    for m in ("rw", "m0", "m1"):
        assert ps["models"][m].f * ps["scale"] == pytest.approx(fc[m], abs=1e-12)


def test_window_residuals_bruteforce():
    df = sim_system(300, seed=22)
    R = df.r.to_numpy() * 100
    X = np.column_stack([np.ones(len(R)), R])
    h, w = 3, 80
    pred = bt.rolling_predictive(R, X, h, w)
    j = 17
    T = pred.T[j]
    t = np.arange(T - h - w + 1, T - h + 1)
    assert np.all(t + h <= T)                                          # only targets observed at the origin
    b = np.linalg.lstsq(X[t], R[t + h], rcond=None)[0]
    e = R[t + h] - X[t] @ b
    assert pred.s[j] == pytest.approx(np.sqrt(e @ e / (w - 2)))
    assert np.sqrt(np.mean(pred.e[j] ** 2)) == pytest.approx(pred.s[j])
    assert np.sqrt(np.mean(pred.u[j] ** 2)) == pytest.approx(1.0)


def test_var_forecasts_do_not_look_ahead():
    df = sim_system(450, seed=23)
    base = bt.var_forecasts(df, 1, 120, True, 2)
    k = 330
    for col in ("r", "x", "z"):
        d2 = df.copy()
        d2.iloc[k:, d2.columns.get_loc(col)] += 0.5
        pert = bt.var_forecasts(d2, 1, 120, True, 2)
        known = base["origin"] < df.index[k]
        assert pert.loc[known, "var"].to_numpy() == pytest.approx(base.loc[known, "var"].to_numpy())


def test_var_forecasts_shape_and_ordering():
    df = sim_system(400, seed=24)
    fc = bt.var_forecasts(df, 1, 120, True, 1)
    n_dates = fc["date"].nunique()
    assert len(fc) == n_dates * 3 * 3 * 2 * 2                          # models x methods x tails x levels
    w = fc.pivot_table(index=["date", "model", "method"], columns=["tail", "level"], values="var")
    assert (w[("up", 0.99)] >= w[("up", 0.95)]).all() and (w[("down", 0.99)] <= w[("down", 0.95)]).all()
    assert (w[("up", 0.95)] > w[("down", 0.95)]).all()
    assert (fc["origin"] < fc["date"]).all()
    # h > 1 uses non-overlapping origins
    f5 = bt.var_forecasts(df, 5, 120, True, 1)
    o = np.sort(f5["origin"].unique())
    pos = df.index.get_indexer(o)
    assert np.all(np.diff(pos) == 5)


def test_correct_model_gives_correct_coverage():
    """Gaussian AR data: the normal VaR of the true model family should not be rejected, and the breach
    rate should be close to nominal."""
    rates, n = [], 2500
    for seed in range(8):                         # one sample of ~2000 days is noisy: average a few
        rng = np.random.default_rng(seed)
        r = np.empty(n)
        r[0] = 0.02
        for t in range(1, n):
            r[t] = 0.02 + 0.97 * (r[t - 1] - 0.02) + 0.001 * rng.standard_normal()
        df = pd.DataFrame({"r": r, "x": 0.02 + np.cumsum(rng.normal(0, 0.0005, n))},
                          index=pd.bdate_range("2010-01-01", periods=n))
        cov = bt.coverage_table(bt.var_forecasts(df, 1, 500, False, 0, methods=("normal",)))
        rates.append(cov[cov["model"] == "m0"].set_index(["tail", "level"])["rate"])
    mean = pd.concat(rates, axis=1).mean(axis=1)
    nominal = 1 - mean.index.get_level_values("level").to_numpy()
    assert np.all(np.abs(mean.to_numpy() / nominal - 1) < 0.15)


def test_fat_tails_break_normal_var_but_not_evt():
    rng = np.random.default_rng(6)
    n = 3000
    e = stats.t.rvs(3, size=n, random_state=rng) * 0.001 / np.sqrt(3)
    r = 0.02 + np.cumsum(e)
    df = pd.DataFrame({"r": r, "x": 0.02 + np.cumsum(rng.normal(0, 0.0005, n))}, index=pd.bdate_range("2010-01-01", periods=n))
    cov = bt.coverage_table(bt.var_forecasts(df, 1, 500, False, 0, levels=(0.99,)))
    rw = cov[(cov["model"] == "rw")].set_index(["method", "tail"])
    # Student-t(3) scaled to the same sd: the 99% normal quantile sits inside the true one -> too many breaches
    assert (rw.loc["normal", "rate"] > 0.012).all()
    assert (abs(rw.loc["evt", "rate"] - 0.01) < abs(rw.loc["normal", "rate"] - 0.01)).all()


def test_tick_loss_prefers_the_true_quantile():
    rng = np.random.default_rng(7)
    y = rng.standard_normal(20000)
    good = bt.tick_loss(y, np.full_like(y, stats.norm.ppf(0.99)), 0.99).mean()
    for q in (1.5, 3.0):
        assert bt.tick_loss(y, np.full_like(y, q), 0.99).mean() > good
    assert np.all(bt.tick_loss(y, np.zeros_like(y), 0.5) >= 0)


def test_tick_loss_test_sign():
    df = sim_system(500, seed=25)
    fc = bt.var_forecasts(df, 1, 150, True, 1, methods=("normal",))
    # make M1's VaR useless (shifted far away): M1 must lose to M0
    bad = fc["model"] == "m1"
    fc.loc[bad, "var"] = fc.loc[bad, "var"] + np.where(fc.loc[bad, "tail"] == "up", -0.01, 0.01)
    t = bt.tick_loss_test(fc)
    assert (t["p M1 vs M0"] > 0.99).all() and (t["loss_m1_bp"] > t["loss_m0_bp"]).all()


# ------------------------------------------------------------------ driver
@needs_data
def test_backtest_driver_runs_and_is_reproducible(tmp_path):
    args = [sys.executable, str(ROOT / "run_backtest.py"), "--horizons", "1", "--power-sims", "200", "--no-robust"]
    for d in ("a", "b"):
        subprocess.run(args + ["--out", str(tmp_path / d)], check=True, capture_output=True, cwd=ROOT)
    for f in ("headline.csv", "backtest_coverage.csv", "backtest_tick_loss.csv", "power_christoffersen.csv"):
        assert (tmp_path / "a" / f).read_text() == (tmp_path / "b" / f).read_text()
    cov = pd.read_csv(tmp_path / "a" / "backtest_coverage.csv")
    assert set(cov["model"]) == {"rw", "m0", "m1"} and set(cov["method"]) == {"normal", "hs", "evt"}
    assert (tmp_path / "a" / "run_info.json").exists() and (tmp_path / "a" / "var_forecasts_h1.csv").exists()
