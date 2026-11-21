import numpy as np
import pandas as pd
import pytest

from ssef import loaders
from ssef import restricted_ecm as E
from ssef import validate_tests as vt
from ssef.ols import subset

from test_m1 import HAVE_DATA, sim_system

needs_data = pytest.mark.skipif(not HAVE_DATA, reason="data CSVs not present")


def _models(df, p=2):
    m1 = E.fit_ecm(df, p, 0, True)
    return m1, subset(m1.fit, E.nested_columns(m1.fit.names))


# ------------------------------------------------------------------ simulation engine
def test_simulate_ecm_reproduces_the_data_when_fed_its_own_residuals():
    """Identity test: fitted coefficients + fitted residuals as shocks must give back the actual r, exactly. Any slip
    in a column, a lag or a timing (dx_i, z_i, x_{i-1}, dr_{i-k}) breaks it."""
    df = sim_system(500, seed=3)
    for p in (0, 3):
        m1 = E.fit_ecm(df, p, 0, True)
        r = vt.simulate_ecm(vt.coefficients(m1.fit), df, p, m1.fit.resid.to_numpy())
        assert r == pytest.approx(df["r"].to_numpy(), abs=1e-10)


def test_simulate_ecm_identity_holds_with_a_level_shift_column():
    df = sim_system(500, seed=4)
    bd = df.index[300]
    m1 = E.fit_ecm(df, 2, 0, True, break_date=bd)
    D = (df.index >= bd).astype(float)
    r = vt.simulate_ecm(vt.coefficients(m1.fit), df, 2, m1.fit.resid.to_numpy(), D)
    assert r == pytest.approx(df["r"].to_numpy(), abs=1e-10)


def test_level_shift_acts_from_the_break_date_and_only_then():
    df = sim_system(300, seed=5)
    coef = {"const": 0.0, "r_lag": -0.1}
    bd = df.index[200]
    D = (df.index >= bd).astype(float)
    shocks = np.zeros(len(df) - 3)
    base = vt.simulate_ecm(coef, df, 2, shocks, D)
    shifted = vt.simulate_ecm(coef | {"shift": 0.01}, df, 2, shocks, D)
    k = 200                                                  # D_{i-1} switches on at i = 201
    assert shifted[: k + 1] == pytest.approx(base[: k + 1])
    assert np.all(shifted[k + 1:] > base[k + 1:])


def test_simulate_ecm_checks_the_shock_length():
    df = sim_system(100, seed=1)
    with pytest.raises(ValueError):
        vt.simulate_ecm({"r_lag": -0.1}, df, 2, np.zeros(10))


# ------------------------------------------------------------------ data-generating processes
def test_make_dgps_sets_the_known_truth():
    df = sim_system(400, a=-0.05, b1=0.8, seed=2)
    m1, m0 = _models(df)
    dg = {d.name: d for d in vt.make_dgps(df, m1, m0, anchors=((0.5, None), (1.0, -0.03)), break_date=df.index[250], level_shift=-0.01)}
    assert list(dg) == ["spurious", "m0", "m0_break", f"m1_b0.5_a{m1.a:.4f}", "m1_b1_a-0.0300"]
    assert dg["spurious"].coef["r_lag"] == 0.0 and dg["spurious"].h0_true
    assert "x_lag" not in dg["m0"].coef and "z" not in dg["m0"].coef            # no SOFR / FX term
    c = dg["m1_b1_a-0.0300"].coef
    assert -c["x_lag"] / c["r_lag"] == pytest.approx(1.0)                          # long-run slope = beta1
    assert c["r_lag"] == -0.03 and not dg["m1_b1_a-0.0300"].h0_true
    # long-run mean of r equals the sample mean: const / -a = mean(r) - beta1 mean(x)
    assert c["const"] / -c["r_lag"] == pytest.approx(df["r"].mean() - df["x"].mean())
    # break DGP: the shift column moves the long-run level by level_shift
    b = dg["m0_break"]
    assert -b.coef["shift"] / b.coef["r_lag"] == pytest.approx(-0.01) and b.break_date == df.index[250]
    est = dg[f"m1_b0.5_a{m1.a:.4f}"].coef
    assert est["r_lag"] == pytest.approx(m1.a)                                      # a=None keeps the estimated speed


def test_make_dgps_without_a_break_has_no_break_process():
    df = sim_system(300, seed=2)
    m1, m0 = _models(df)
    assert "m0_break" not in [d.name for d in vt.make_dgps(df, m1, m0)]


def test_a_simulated_anchor_is_recovered_by_the_estimator():
    df = sim_system(3000, a=-0.05, b1=0.8, sigma=0.0008, seed=7)
    m1, m0 = _models(df, 1)
    d = next(x for x in vt.make_dgps(df, m1, m0, anchors=((0.6, -0.04),)) if x.name.startswith("m1"))
    rng = np.random.default_rng(0)
    shocks = rng.choice(m1.fit.resid.to_numpy(), size=len(df) - 2)
    sim = df.assign(r=vt.simulate_ecm(d.coef, df, 1, shocks))
    f = E.fit_ecm(sim, 1, 0, True)
    assert f.a == pytest.approx(-0.04, abs=0.02)
    assert f.beta1 == pytest.approx(0.6, abs=0.25)


# ------------------------------------------------------------------ summary
def test_wilson_interval_matches_the_formula():
    k, n, z = 5, 100, 1.959963984540054
    ph = k / n
    centre = (ph + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    lo, hi = vt.wilson(k, n)
    assert (lo, hi) == pytest.approx((centre - half, centre + half), abs=1e-6)


def test_rejection_table_counts_rates_and_readings():
    raw = pd.DataFrame({
        "dgp": ["a"] * 200 + ["b"] * 200, "h0_true": [True] * 200 + [False] * 200,
        "c1": 0.99,
        "classical": [0.01] * 40 + [0.5] * 160 + [0.01] * 150 + [0.5] * 50,           # a: 20% (oversized), b: 75%
        "boot": [0.01] * 10 + [0.5] * 190 + [0.01] * 100 + [0.5] * 100,               # a: 5% (ok), b: 50%
        "hac": [0.01] * 2 + [0.5] * 198 + [0.5] * 200,                                 # a: 1% (conservative), b: 0%
        "eg": np.nan, "ecm_gate": np.nan, "boot_break": np.nan, "cusum": np.nan})
    t = vt.rejection_table(raw).set_index(["dgp", "test"])
    assert t.loc[("a", "classical"), ["rejections", "rate", "reading"]].tolist() == [40, 0.2, "oversized"]
    assert t.loc[("a", "classical"), "truth"] == "null true (size)" and t.loc[("b", "classical"), "truth"] == "null false (power)"
    assert t.loc[("a", "boot"), "reading"] == "ok" and t.loc[("a", "hac"), "reading"] == "conservative"
    assert t.loc[("b", "classical"), "reading"] == "power" and t.loc[("b", "classical"), "rate"] == 0.75
    assert ("a", "eg") not in t.index                                                  # not run -> left out
    assert t.loc[("a", "classical"), "ci_low"] < 0.2 < t.loc[("a", "classical"), "ci_high"]
    assert t.loc[("a", "c1_in_(0,1)"), "rate"] == 1.0
    w = vt.wide(vt.rejection_table(raw))
    assert list(w.index) == ["a", "b"] and list(w.columns) == ["classical", "hac", "boot"]


def test_cusum_and_gates_have_their_own_nulls():
    """CUSUM's null (stability) is false in the process with a break -> its rejections there are POWER, not size.
    A gate that rejects where there is no SOFR link is a FALSE PASS."""
    n = 200
    rej = [0.01] * 40 + [0.5] * 160                                                  # 20% rejections
    cols = {"c1": 0.9, "classical": 0.5, "hac": 0.5, "boot": 0.5, "ecm_gate": np.nan}
    raw = pd.concat([pd.DataFrame({"dgp": "plain", "h0_true": True, "boot_break": np.nan, "eg": rej, "cusum": rej, **cols}),
                     pd.DataFrame({"dgp": "brk", "h0_true": True, "boot_break": 0.5, "eg": rej, "cusum": rej, **cols})])
    t = vt.rejection_table(raw).set_index(["dgp", "test"])
    assert t.loc[("plain", "cusum"), ["truth", "reading"]].tolist() == ["null true (size)", "oversized"]
    assert t.loc[("brk", "cusum"), ["truth", "reading"]].tolist() == ["null false (power)", "power"]
    assert t.loc[("brk", "eg"), ["truth", "reading"]].tolist() == ["null true (size)", "false pass"]
    assert vt.null_is_true("cusum", True, True) is False and vt.null_is_true("boot", True, True) is True


def test_nan_p_values_count_as_no_rejection():
    raw = pd.DataFrame({"dgp": "a", "h0_true": True, "c1": 0.5, "classical": [0.01, np.nan, 0.5, 0.5], "hac": 0.5, "boot": 0.5,
                        "eg": np.nan, "ecm_gate": np.nan, "boot_break": np.nan, "cusum": [np.nan] * 4})
    t = vt.rejection_table(raw).set_index("test")
    assert t.loc["classical", "rate"] == 0.25 and "cusum" not in t.index


# ------------------------------------------------------------------ the Monte Carlo itself
def _tiny(df, p=2, **kw):
    m1, m0 = _models(df, p)
    dgps = vt.make_dgps(df, m1, m0, anchors=((1.0, -0.05),), break_date=df.index[300], level_shift=-0.01)
    return m1, m0, dgps


def test_run_mc_is_reproducible_and_independent_of_the_number_of_workers():
    df = sim_system(400, seed=9)
    m1, m0, dgps = _tiny(df)
    kw = dict(n_sims=3, B=9, seed=11)
    a = vt.run_mc(df, 2, dgps, m1, m0, workers=1, **kw)
    b = vt.run_mc(df, 2, dgps, m1, m0, workers=1, **kw)
    c = vt.run_mc(df, 2, dgps, m1, m0, workers=2, **kw)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(a, c)
    assert len(a) == 3 * len(dgps)
    assert a.groupby("dgp")["classical"].nunique().eq(3).all()                    # every dataset in a process is different
    assert a["boot_break"].notna().groupby(a["dgp"]).any().to_dict() == {"spurious": False, "m0": False, "m0_break": True,
                                                                          "m1_b1_a-0.0500": False}
    p_cols = ["classical", "hac", "boot", "eg", "ecm_gate", "cusum"]
    assert a[p_cols].stack().between(0, 1).all()
    d = vt.run_mc(df, 2, dgps, m1, m0, workers=1, n_sims=3, B=9, seed=12)           # another seed -> other draws
    assert not a["boot"].equals(d["boot"])


def test_ecm_gate_can_be_skipped():
    df = sim_system(300, seed=1)
    m1, m0 = _models(df)
    raw = vt.run_mc(df, 2, vt.make_dgps(df, m1, m0, anchors=((1.0, None),)), m1, m0, n_sims=2, B=9, ecm_gate=False)
    assert raw["ecm_gate"].isna().all()


@needs_data
def test_on_the_real_data_a_strong_anchor_is_found_and_a_constant_mean_is_not():
    """Pipeline-level behaviour, small and seeded: strong anchor -> the bootstrap F nearly always rejects;
    M0 true -> it rarely does. (The full size / power numbers come from run_validation.py.)"""
    df = loaders.build_dataset(loaders.load_sora(), loaders.load_sofr(), loaders.make_z(loaders.load_usdsgd()))
    m1 = E.fit_ecm(df, 8, 0, True)
    m0 = subset(m1.fit, E.nested_columns(m1.fit.names))
    dgps = [d for d in vt.make_dgps(df, m1, m0, anchors=((1.0, -0.03),)) if d.name in ("m0", "m1_b1_a-0.0300")]
    raw = vt.run_mc(df, 8, dgps, m1, m0, n_sims=24, B=49, seed=5, workers=2, ecm_gate=False)
    rate = raw.assign(rej=raw["boot"] < 0.05).groupby("dgp")["rej"].mean()
    assert rate["m1_b1_a-0.0300"] >= 0.9
    assert rate["m0"] <= 0.25
