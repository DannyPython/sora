import numpy as np
import pandas as pd
import pytest
from scipy import stats

import run_density
from ssef import backtest as bt
from ssef import density, loaders
from ssef import restricted_ecm as E

from test_m1 import HAVE_DATA

needs_data = pytest.mark.skipif(not HAVE_DATA, reason="data CSVs not present")


def synthetic_pits(n=800, mean=0.0, sd=1.0, seed=0, window=300):
    """PITs of a predictive N(0,1) built from iid window residuals, against outcomes drawn from N(mean, sd)."""
    rng = np.random.default_rng(seed)
    pred = bt.Predictive(T=np.arange(n), f=np.zeros(n), s=np.ones(n), e=rng.standard_normal((n, window)), k=0)
    actual = rng.normal(mean, sd, n)
    v = density.pit_values(pred, actual)
    return {(m, k): v[k] for m in ("rw", "m0", "m1") for k in density.KINDS}


# ------------------------------------------------------------------ PIT
def test_pit_values_follow_the_definition():
    pred = bt.Predictive(T=np.array([0, 1]), f=np.array([0.0, 1.0]), s=np.array([1.0, 2.0]),
                         e=np.array([[-1.0, 0.0, 0.5, 2.0], [-2.0, 0.0, 2.0, 4.0]]), k=1)
    v = density.pit_values(pred, np.array([0.5, 6.0]))
    assert v["normal"] == pytest.approx([stats.norm.cdf(0.5), stats.norm.cdf(2.5)])
    # row 0: error 0.5 ties with a window residual, which counts (<=): {-1, 0, 0.5} -> (3 + 0.5) / 5 ;
    # row 1: error 5, all four -> 4.5 / 5
    assert v["empirical"] == pytest.approx([0.7, 0.9])
    assert ((v["empirical"] > 0) & (v["empirical"] < 1)).all()


# ------------------------------------------------------------------ tests on z
def test_berkowitz_lr_matches_an_independent_likelihood_computation():
    z = stats.norm.rvs(size=120, random_state=np.random.default_rng(1)) * 1.2 + 0.1
    b = density.berkowitz(z)
    y, x = z[1:], z[:-1]
    X = np.column_stack([np.ones(len(y)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    sd = np.sqrt(np.mean((y - X @ beta) ** 2))
    lr = 2 * (stats.norm.logpdf(y, loc=X @ beta, scale=sd).sum() - stats.norm.logpdf(y).sum())
    assert b["LR"] == pytest.approx(lr) and b["p"] == pytest.approx(stats.chi2.sf(lr, 3))
    assert (b["mu"], b["rho"]) == pytest.approx(tuple(beta)) and b["sd"] == pytest.approx(sd)


def test_single_restriction_tests_match_scipy_and_the_formulas():
    z = stats.norm.rvs(size=200, random_state=np.random.default_rng(2)) * 0.9 - 0.15
    b = density.berkowitz(z)
    assert b["p_mean"] == pytest.approx(stats.ttest_1samp(z, 0).pvalue)
    chi = 199 * z.var(ddof=1)
    assert b["p_var"] == pytest.approx(2 * min(stats.chi2.cdf(chi, 199), stats.chi2.sf(chi, 199)))
    rho = np.corrcoef(z[1:], z[:-1])[0, 1]
    assert b["p_ar1"] == pytest.approx(2 * stats.norm.sf(abs(rho) * np.sqrt(200)))


def test_berkowitz_size_and_power():
    rng = np.random.default_rng(3)
    n, sims = 250, 300

    def rate(make):
        return np.mean([density.berkowitz(make())["p"] < 0.05 for _ in range(sims)])

    assert 0.02 <= rate(lambda: rng.standard_normal(n)) <= 0.10                          # size
    assert rate(lambda: rng.standard_normal(n) + 0.3) > 0.5                             # biased
    assert rate(lambda: rng.standard_normal(n) * 0.7) > 0.8                             # too wide
    assert rate(lambda: rng.standard_normal(n) * 1.4) > 0.8                             # too narrow

    def ar(rho=0.3):
        e = rng.standard_normal(n + 50) * np.sqrt(1 - rho ** 2)
        x = np.zeros(n + 50)
        for t in range(1, n + 50):
            x[t] = rho * x[t - 1] + e[t]
        return x[50:]
    assert rate(ar) > 0.5                                                                # serial dependence


def test_tail_shares_counts_and_binomial_p_values():
    pit = np.r_[np.full(7, 0.01), np.full(3, 0.995), np.full(90, 0.5)]
    t = density.tail_shares(pit)
    assert t["lower_5%"] == 0.07 and t["upper_5%"] == 0.03
    assert t["p_lower_5%"] == pytest.approx(stats.binomtest(7, 100, 0.05).pvalue)
    assert t["lower_1%"] == 0.0 and t["upper_1%"] == 0.03
    assert t["p_upper_1%"] == pytest.approx(stats.binomtest(3, 100, 0.01).pvalue)


# ------------------------------------------------------------------ the table and its reading
def test_a_correct_distribution_is_not_flagged():
    t = density.density_table(synthetic_pits(seed=5)).set_index(["model", "predictive"])
    assert (t["KS_p"] > 0.01).all() and (t["Berkowitz_p"] > 0.01).all()
    assert t.loc[("m0", "normal"), "reading"] == "no violation found"
    assert t.loc[("m0", "empirical"), "reading"] == "no violation found"


def test_width_and_bias_are_read_in_the_right_direction():
    narrow = density.density_table(synthetic_pits(sd=1.8, seed=6)).set_index(["model", "predictive"])
    assert narrow.loc[("m1", "normal"), "sd_z"] > 1.3 and "too narrow" in narrow.loc[("m1", "normal"), "reading"]
    assert narrow.loc[("m1", "normal"), "lower_5%"] > 0.1                                # fat tails: more than 5% in each tail
    wide = density.density_table(synthetic_pits(sd=0.6, seed=7)).set_index(["model", "predictive"])
    assert wide.loc[("m1", "normal"), "sd_z"] < 0.8 and "too wide" in wide.loc[("m1", "normal"), "reading"]
    assert wide.loc[("m1", "normal"), "upper_5%"] < 0.02
    high = density.density_table(synthetic_pits(mean=-0.4, seed=8)).set_index(["model", "predictive"])
    assert high.loc[("m1", "normal"), "mean_z"] < -0.3 and "forecasts run above outcomes" in high.loc[("m1", "normal"), "reading"]
    low = density.density_table(synthetic_pits(mean=0.4, seed=9)).set_index(["model", "predictive"])
    assert "forecasts run below outcomes" in low.loc[("m1", "normal"), "reading"]


def test_volatility_clustering_and_serial_dependence_are_flagged():
    rng = np.random.default_rng(10)
    n = 1500
    h = np.ones(n)
    for t in range(1, n):                                                                # ARCH(1)-type variance
        h[t] = 0.3 + 0.6 * (h[t - 1] ** 0.5 * rng.standard_normal()) ** 2
    pred = bt.Predictive(T=np.arange(n), f=np.zeros(n), s=np.ones(n), e=rng.standard_normal((n, 300)), k=0)
    v = density.pit_values(pred, np.sqrt(h) * rng.standard_normal(n))
    t = density.density_table({(m, k): v[k] for m in ("rw", "m0", "m1") for k in density.KINDS}).iloc[0]
    assert t["LjungBox_z2_p"] < 0.01 and "volatility clustering" in t["reading"]


def test_histogram_counts_sum_to_the_number_of_points():
    pits = synthetic_pits(n=400, seed=11)
    h = density.pit_histogram(pits)
    assert h.shape == (density.BINS, len(pits)) and (h.sum() == 400).all()


# ------------------------------------------------------------------ the real data, and consistency with the VaR backtest
@needs_data
def test_tail_shares_reproduce_the_var_backtest_breach_counts():
    """V2 and the VaR backtest use the same predictive sets: a 95% / 99% breach is exactly a PIT in the extreme tail."""
    df = loaders.build_dataset(loaders.load_sora(), loaders.load_sofr(), loaders.make_z(loaders.load_usdsgd()))
    p = 8
    pits, info = density.compute_pit(df, 1, 500, True, p, 0)
    fc = bt.var_forecasts(df, 1, window=500, with_z=True, p=p, q=0, levels=(0.95, 0.99), methods=("normal",))
    for model in ("rw", "m0", "m1"):
        for tail, side in (("up", lambda v: v > 0.95), ("down", lambda v: v < 0.05)):
            hits = fc[(fc["model"] == model) & (fc["tail"] == tail) & (fc["level"] == 0.95)]["hit"].sum()
            assert int(side(pits[(model, "normal")]).sum()) == hits
        hits99 = fc[(fc["model"] == model) & (fc["tail"] == "up") & (fc["level"] == 0.99)]["hit"].sum()
        assert int((pits[(model, "normal")] > 0.99).sum()) == hits99
    assert info["N"] == len(pits[("m0", "normal")]) == len(info["dates"])


@needs_data
def test_run_density_writes_every_table(tmp_path):
    R = run_density.main(["--horizons", "1", "20", "--out", str(tmp_path)])
    for h in (1, 20):
        t = R["tables"][h]
        assert len(t) == 6 and t["N"].eq(R["n"][h]).all() and t["KS_p"].between(0, 1).all()
        assert (R["hist"][h].sum() == R["n"][h]).all()
        assert (tmp_path / f"density_table_h{h}.csv").exists() and (tmp_path / f"pit_histogram_h{h}.csv").exists()
    assert R["n"][20] < 100 < R["n"][1]
    assert (tmp_path / "headline.csv").exists() and (tmp_path / "run_info.json").exists()
