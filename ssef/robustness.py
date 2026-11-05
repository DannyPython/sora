"""Sensitivity of the M1 conclusions to the choices that drive them (Blueprint Sec. 12: one rate cycle in
five years, so start date, window, lag length and break handling must all be varied)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import cointegration as C
from . import ecm
from .loaders import build_dataset
from .ols import exclusion_f, exclusion_wald_hac, subset
from .race import run_race


def spec_grid(df: pd.DataFrame, ps=(0, 1, 2, 5), qs=(0, 1), with_z: bool = True, break_date=None) -> pd.DataFrame:
    """Anchor-exclusion test (x_lag, z) for several lag choices. Classical and HAC p-values are shown ONLY to
    expose how fast the apparent significance moves; the valid p-value is the bootstrap one (nested.anchor_test)."""
    rows = []
    for p in ps:
        for q in qs:
            m = ecm.fit_ecm(df, p, q, with_z, break_date=break_date)
            drop = ecm.nested_columns(m.fit.names, "anchor")
            f = exclusion_f(m.fit, subset(m.fit, drop))
            h = exclusion_wald_hac(m.fit, drop)
            rows.append({"p_dr_lags": p, "q_dx_lags": q, "n": m.fit.n, "a": m.a, "t_a": m.fit.tstat("r_lag"),
                         "kappa": m.kappa, "beta1_longrun": m.beta1, "F_anchor": f["F"],
                         "p_classical": f["p_classical"], "p_HAC": h["p_hac_F"]})
    return pd.DataFrame(rows)


def start_date_sensitivity(r, sofr, z, starts, p: int = 2, q: int = 0, gh_lags: int = 5, gh_sim: int = 0) -> pd.DataFrame:
    """Re-run the gate and the ECM on shorter windows (later start dates)."""
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s in starts:
            d = build_dataset(r[r.index >= pd.Timestamp(s)], sofr, z)
            eg_c = C.engle_granger(d.r, d.x, lags=[], trends=("c",)).iloc[0]
            eg_ct = C.engle_granger(d.r, d.x, lags=[], trends=("ct",)).iloc[0]
            gh = C.gregory_hansen(d.r, d.x, "C", gh_lags, n_sim=gh_sim, dates=d.index, step=2)
            m = ecm.fit_ecm(d, p, q, True)
            rows.append({"start": pd.Timestamp(s).date(), "n": len(d), "EG_p_const(BIC)": eg_c["p_value"],
                         "EG_p_trend(BIC)": eg_ct["p_value"], "GH_C_stat": gh["stat"],
                         "GH_break": gh["break_date"].date(), "GH_p": gh.get("p_value", np.nan),
                         "slope_r_on_x": C.slope_stability(d, windows=())["full"], "a": m.a, "kappa": m.kappa,
                         "beta1_longrun": m.beta1})
    return pd.DataFrame(rows)


def race_window_grid(df, windows=(250, 500, 750), horizons=(1, 20, 60), with_z: bool = True) -> pd.DataFrame:
    rows = []
    for w in windows:
        if w + max(horizons) + 50 >= len(df):
            continue
        s = run_race(df, horizons, w, with_z, B=0)["summary"]
        for h in horizons:
            rows.append({"window": w, "h": h, "n_oos": int(s.loc[h, "n_oos"]), "rmse_rw_bp": s.loc[h, "rmse_rw_bp"],
                         "rmse_m0_bp": s.loc[h, "rmse_m0_bp"], "rmse_m1_bp": s.loc[h, "rmse_m1_bp"]})
    return pd.DataFrame(rows)
