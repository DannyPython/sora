"""Block 1 (data collection) for M0: load the cleaned SORA series.

M0 uses SORA only. SOFR / FX alignment (Blueprint Sec. 7) belongs to M1.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SORA_CSV = Path(__file__).resolve().parent.parent / "data" / "Domestic_Interest_Rates_clean.csv"
DATE_COL = "SORA Value Date"
RATE_COL = "SORA"


def load_sora(path=SORA_CSV, start=None, end=None, as_decimal=True) -> pd.Series:
    """Return daily SORA indexed by value date.

    The CSV holds percent per annum; `as_decimal=True` converts to decimals
    (3.6063 -> 0.036063) so that kappa, theta, sigma are in per-year decimal units.
    """
    df = pd.read_csv(path)
    missing = {DATE_COL, RATE_COL} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="%d/%m/%Y")
    df = df.dropna(subset=[RATE_COL]).sort_values(DATE_COL)
    if df[DATE_COL].duplicated().any():
        raise ValueError(f"{path}: duplicate value dates")

    r = pd.Series(df[RATE_COL].astype(float).to_numpy(),
                  index=pd.DatetimeIndex(df[DATE_COL], name="value_date"), name="SORA")
    if as_decimal:
        r = r / 100.0
    if start is not None:
        r = r[r.index >= pd.Timestamp(start)]
    if end is not None:
        r = r[r.index <= pd.Timestamp(end)]
    return r