import numpy as np
import pandas as pd
import pytest
from scipy import stats

import run_evt
from ssef import evt
from ssef import restricted_ecm as E

from test_m1 import HAVE_DATA, sim_system

needs_data = pytest.mark.skipif(not HAVE_DATA, reason="data CSVs not present")


def gpd_sample(n, xi, beta, seed):
    return stats.genpareto.rvs(xi, scale=beta, size=n, random_state=np.random.default_rng(seed))


# ------------------------------------------------------------------ fit and goodness of fit
def test_ks_stat_matches_scipy():
    x = gpd_sample(100, 0.2, 1.0, 1)
    assert evt.ks_stat(x, 0.2, 1.0) == pytest.approx(stats.kstest(x, "genpareto", args=(0.2, 0, 1.0)).statistic)
    assert evt.ks_stat(x, 0.9, 5.0) > evt.ks_stat(x, 0.2, 1.0)                    # a wrong GPD fits worse


def test_fit_tail_recovers_the_shape_of_a_known_gpd():
    """Exceedances of a GPD over any threshold are GPD with the same shape (threshold stability)."""
    u = gpd_sample(20000, 0.25, 1.0, 2)
    r = evt.fit_tail(u, 0.5, B=0, rng=np.random.default_rng(0))
    assert r["xi"] == pytest.approx(0.25, abs=0.06) and r["n_exc"] == 10000 and np.isnan(r["KS_p"])
    assert r["thr"] == pytest.approx(np.quantile(u, 0.5))
    # the reported quantile is the one asked for: P(U > q99) = 1%
    assert np.mean(u > r["q99"]) == pytest.approx(0.01, abs=0.004)


def test_bootstrap_ks_p_value_is_calibrated_where_the_textbook_one_is_not():
    """With the parameters estimated from the same data the textbook KS p-value is too large (conservative);
    the parametric bootstrap p-value is roughly uniform when the GPD is true."""
    rng = np.random.default_rng(3)
    boot, naive = [], []
    for i in range(40):
        x = gpd_sample(60, 0.2, 1.0, 100 + i)
        xi, beta = evt.gpd_fit_mle(x)
        d = evt.ks_stat(x, xi, beta)
        boot.append(evt.ks_pvalue_bootstrap(60, xi, beta, d, 60, rng))
        naive.append(stats.kstest(x, "genpareto", args=(xi, 0, beta)).pvalue)
    boot, naive = np.array(boot), np.array(naive)
    assert 0.3 < boot.mean() < 0.7 and np.mean(boot < 0.05) <= 0.15
    assert naive.mean() > boot.mean() + 0.1


def test_bootstrap_ks_p_value_rejects_a_tail_that_is_not_a_gpd():
    """Exponential values capped at 2 (with a hair of jitter): above the median the exceedances pile up at the cap,
    which no GPD describes."""
    rng = np.random.default_rng(7)
    capped = np.minimum(rng.exponential(size=2000), 2.0) + rng.uniform(size=2000) * 1e-6
    r = evt.fit_tail(capped, 0.5, B=60, rng=np.random.default_rng(9))
    assert r["KS_p"] < 0.05


def test_mean_excess_of_a_gpd_has_slope_xi_over_one_minus_xi():
    xi = 0.3
    u = gpd_sample(300000, xi, 1.0, 10)
    me = evt.mean_excess(u, qs=np.arange(0.5, 0.95, 0.02))
    slope = np.polyfit(me["thr"], me["mean_excess"], 1)[0]
    assert slope == pytest.approx(xi / (1 - xi), abs=0.04)
    assert (me["n_exc"] >= 5).all() and list(me.columns) == ["q", "thr", "mean_excess", "n_exc"]


def test_pwm_and_mle_agree_on_a_large_sample():
    u = gpd_sample(20000, 0.2, 1.0, 11)
    a = evt.fit_tail(u, 0.9, B=0, rng=np.random.default_rng(0), fit="mle")
    b = evt.fit_tail(u, 0.9, B=0, rng=np.random.default_rng(0), fit="pwm")
    assert a["xi"] == pytest.approx(b["xi"], abs=0.05)


# ------------------------------------------------------------------ threshold rule, comparison, episodes
def test_choose_threshold_takes_the_lowest_threshold_that_passes_both_conditions():
    t = pd.DataFrame({"model": "M0", "tail": "up", "q": [0.85, 0.90, 0.95, 0.975], "n_exc": [200, 120, 60, 10],
                      "KS_p": [0.02, 0.30, 0.90, 0.95], "xi": [0.1, 0.2, 0.3, 0.4]})
    c = evt.choose_threshold(t).iloc[0]
    assert (c["q"], c["xi"], c["n_exc"]) == (0.90, 0.2, 120)                 # 0.85 fails KS; 0.90 and 0.95 pass: the LOWEST wins; 0.975 has too few
    few = evt.choose_threshold(t.assign(n_exc=[200, 20, 60, 10])).iloc[0]
    assert few["q"] == 0.95                                                   # 0.90 now has too few exceedances
    none = evt.choose_threshold(t.assign(KS_p=0.01)).iloc[0]
    assert np.isnan(none["q"]) and none["n_exc"] == 0


def test_compare_models_is_zero_for_identical_series_and_positive_for_a_heavier_tail():
    rng = np.random.default_rng(12)
    u = rng.standard_normal(1500)
    same = evt.compare_models(u, u, B=60, seed=1)
    assert same["d_xi"] == 0 and same["d_xi_lo"] == 0 and same["d_xi_hi"] == 0 and same["d_quantile"] == 0
    heavy = stats.t.rvs(3, size=1500, random_state=np.random.default_rng(13))
    c = evt.compare_models(u, heavy / heavy.std(), B=100, seed=1)
    assert c["d_xi"] > 0 and c["d_quantile"] > 0 and c["d_xi_lo"] > -0.05
    with pytest.raises(ValueError):
        evt.compare_models(u, u[:-1])


def _series(dates_hot, n=150, start="2020-01-01"):
    idx = pd.date_range(start, periods=n, freq="D")
    s = pd.Series(0.0, index=idx)
    s.loc[pd.to_datetime(dates_hot)] = 10.0
    return s


def test_episodes_counts_clusters_and_the_densest_window():
    hot = ["2020-01-05", "2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09",       # cluster 1: 5 in a row
           "2020-02-10", "2020-02-11", "2020-02-12",                                      # cluster 2: 3
           "2020-04-01", "2020-04-20"]                                                    # two singletons, > 7 days apart
    e = evt.episodes(_series(hot), q=0.9, gap_days=7, window_days=10)
    assert e["n_exc"] == 10 and e["clusters"] == 4
    assert e["densest_window_start"] == pd.Timestamp("2020-01-05") and e["share_in_densest_window"] == 0.5
    assert e["first"] == pd.Timestamp("2020-01-05") and e["last"] == pd.Timestamp("2020-04-20")
    assert evt.episodes(_series(hot), 0.9, gap_days=40, window_days=10)["clusters"] == 2      # a longer gap merges clusters


def test_without_window_removes_the_window_and_keeps_the_threshold():
    rng = np.random.default_rng(14)
    idx = pd.date_range("2020-01-01", periods=1000, freq="D")
    u = pd.Series(rng.standard_normal(1000), index=idx)
    q = 0.9
    thr = u.quantile(q)
    start = pd.Timestamp("2020-03-01")
    inside = int(((u.index >= start) & (u.index < start + pd.Timedelta(days=60)) & (u > thr)).sum())
    full = int((u > thr).sum())
    w = evt.without_window(u, q, start, 60, B=0)
    assert w["thr"] == pytest.approx(thr) and w["n_exc"] == full - inside and inside > 0


# ------------------------------------------------------------------ residuals and the drivers
def test_standardised_residuals_are_on_the_same_dates_and_scaled_by_their_own_sd():
    df = sim_system(500, seed=3)
    u = evt.standardised_residuals(df, 2)
    assert u["M0"].index.equals(u["M1"].index) and len(u["M1"]) == len(df) - 3
    m1 = E.fit_ecm(df, 2, 0, True)
    n, k1 = m1.fit.n, m1.fit.k
    k0 = k1 - len(E.nested_columns(m1.fit.names))
    assert np.sqrt(np.mean(u["M1"] ** 2)) == pytest.approx(np.sqrt((n - k1) / n))        # sd s uses n - k
    assert np.sqrt(np.mean(u["M0"] ** 2)) == pytest.approx(np.sqrt((n - k0) / n))


@needs_data
def test_run_evt_fast_writes_every_table(tmp_path):
    R = run_evt.main(["--fast", "--B", "20", "--B-compare", "30", "--out", str(tmp_path)])
    t = R["table"]
    assert len(t) == 2 * 2 * len(evt.QGRID) and t["KS_p"].between(0, 1).all() and (t["n_exc"] >= 20).all()
    assert R["u"]["M0"].index.equals(R["u"]["M1"].index)
    for f in ("threshold_table.csv", "threshold_rule.csv", "mean_excess.csv", "m0_vs_m1.csv", "episodes.csv",
              "without_densest_window.csv", "std_residuals.csv", "headline.csv", "run_info.json"):
        assert (tmp_path / f).exists(), f
    assert set(R["compare"]["tail"]) == {"up", "down"} and (R["episodes"]["clusters"] > 1).all()
    assert (R["without_window"]["n_exc"] < R["episodes"]["n_exc"]).all()
