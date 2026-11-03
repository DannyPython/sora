"""Scenario simulation of M1 (Blueprint Sec. 6.7): hold / shock-up / shock-down paths for SOFR.

The exogenous paths are CONDITIONED on, not modelled; a fully stochastic x and z is an extension.
The linear Gaussian model allows negative rates (Blueprint 5.5): `floor` is an ad-hoc truncation, off by default.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ecm import M1ECM
from .noise import draw_noise, std_resid


def make_scenarios(x_last: float, n_steps: int, shock: float = 0.01, ramp: int = 63) -> dict:
    """SOFR paths in decimals (0.01 = 100 bp): linear ramp over `ramp` steps, then held."""
    w = np.minimum(np.arange(1, n_steps + 1) / ramp, 1.0)
    return {"hold": np.full(n_steps, x_last), "up": x_last + shock * w, "down": x_last - shock * w}


def simulate_ecm(model: M1ECM, df: pd.DataFrame, x_path, z_path=None, n_paths: int = 10_000, seed: int = 2027,
                 method: str = "gaussian", floor: float | None = None) -> np.ndarray:
    """(n_paths, n_steps + 1); column 0 is the last observed r. x_path[0] is SOFR for the first simulated day."""
    c = dict(zip(model.fit.names, model.fit.coef))
    p, q = model.p, model.q
    x_path = np.asarray(x_path, float)
    n_steps = len(x_path)
    z_path = np.zeros(n_steps) if z_path is None else np.asarray(z_path, float)
    eps = draw_noise(model.s_eps, n_steps, n_paths, seed, method, std_resid(model.fit).to_numpy())
    off = q + 1
    xs = np.r_[df["x"].to_numpy()[-off:], x_path]                 # x_{-q-1..-1} actual, then the scenario
    dr = np.diff(df["r"].to_numpy())
    hist = [np.full(n_paths, dr[-k]) for k in range(1, p + 1)]    # dr_{-1}, dr_{-2}, ...
    r = np.empty((n_paths, n_steps + 1))
    r[:, 0] = df["r"].iloc[-1]
    shift = 1.0 if model.has_shift else 0.0                       # simulated days are all after the break
    for s in range(n_steps):
        i = off + s
        dri = c["const"] + c["r_lag"] * r[:, s] + c["x_lag"] * xs[i - 1] + eps[:, s]
        if model.has_shift:
            dri = dri + c["shift"] * shift
        for j in range(q + 1):
            dri = dri + c[f"dx{j}"] * (xs[i - j] - xs[i - j - 1])
        if model.with_z:
            dri = dri + c["z"] * z_path[s]
        for k in range(1, p + 1):
            dri = dri + c[f"dr_{k}"] * hist[k - 1]
        r[:, s + 1] = r[:, s] + dri
        if floor is not None:
            r[:, s + 1] = np.maximum(r[:, s + 1], floor)
        hist = [r[:, s + 1] - r[:, s]] + hist[:-1] if p else hist
    return r
