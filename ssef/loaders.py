"""All data loaders and the as-of alignment (Blueprint Sec. 7). One place, no look-ahead.

CSV formats (as produced by the updaters, dates d/m/Y, rates in percent):
  data/Domestic_Interest_Rates_clean.csv : SORA Value Date, SORA Publication Date, SORA
  data/SOFR_clean.csv                    : SOFR Effective Date, SOFR Publication Date, SOFR
  data/USDSGD_clean.csv                  : Date, USD/SGD   (SGD per USD)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SORA_CSV = DATA_DIR / "Domestic_Interest_Rates_clean.csv"
SOFR_CSV = DATA_DIR / "SOFR_clean.csv"
FX_CSV = DATA_DIR / "USDSGD_clean.csv"


def _dmy(s) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(s, format="%d/%m/%Y"))


def load_sora(path=SORA_CSV, start=None, end=None, as_decimal=True) -> pd.Series:
    df = pd.read_csv(path)
    d = _dmy(df["SORA Value Date"])
    r = pd.Series(df["SORA"].astype(float).to_numpy() / (100.0 if as_decimal else 1.0),
                  index=pd.DatetimeIndex(d, name="value_date"), name="r").sort_index()
    if r.index.has_duplicates:
        raise ValueError(f"{path}: duplicate SORA value dates")
    r = r.dropna()
    if start is not None:
        r = r[r.index >= pd.Timestamp(start)]
    if end is not None:
        r = r[r.index <= pd.Timestamp(end)]
    return r


def load_sofr(path=SOFR_CSV, as_decimal=True) -> pd.DataFrame:
    """Columns: sofr, pub_date (NaT on the latest row, which has no successor yet)."""
    df = pd.read_csv(path)
    out = pd.DataFrame({
        "sofr": df["SOFR"].astype(float).to_numpy() / (100.0 if as_decimal else 1.0),
        "pub_date": pd.to_datetime(df["SOFR Publication Date"], format="%d/%m/%Y").to_numpy(),  # array: no index alignment
    }, index=pd.DatetimeIndex(_dmy(df["SOFR Effective Date"]), name="effective"))
    return out.sort_index()


def load_usdsgd(path=FX_CSV) -> pd.Series:
    df = pd.read_csv(path)
    return pd.Series(df.iloc[:, 1].astype(float).to_numpy(), index=pd.DatetimeIndex(_dmy(df.iloc[:, 0]), name="date"),
                     name="usdsgd").sort_index()


def known_sofr(sofr: pd.DataFrame, dates: pd.DatetimeIndex, strict: bool = True) -> pd.Series:
    """Latest SOFR already published BEFORE each SORA value date (as-of merge on publication date).
    strict=True : publication date < SORA date (conservative: SOFR appears ~8pm Singapore time).
    strict=False: publication date <= SORA date (the updater's convention)."""
    s = sofr.dropna(subset=["pub_date"]).sort_values("pub_date")
    first_use = s["pub_date"] + (pd.Timedelta(days=1) if strict else pd.Timedelta(0))
    avail = pd.Series(s["sofr"].to_numpy(), index=pd.DatetimeIndex(first_use, name="usable_from"))
    avail = avail[~avail.index.duplicated(keep="last")].sort_index()
    return avail.reindex(avail.index.union(dates)).ffill().reindex(dates).rename("x")


def make_z(usdsgd: pd.Series) -> pd.Series:
    """z = daily spot return of USD/SGD. SGD per USD, so z > 0 means the SGD weakened (verify vs source)."""
    return usdsgd.sort_index().pct_change().rename("z")


def build_dataset(r: pd.Series, sofr: pd.DataFrame, z: pd.Series | None = None, strict: bool = True) -> pd.DataFrame:
    """One row per SORA value date i: r_i, x_i (SOFR known before i), z_i (FX return on the previous
    SORA date, so it is known before r_i). Past values only: no look-ahead."""
    df = pd.DataFrame({"r": r})
    df["x"] = known_sofr(sofr, df.index, strict)
    if z is not None:
        df["z"] = z.reindex(z.index.union(df.index)).ffill().reindex(df.index).shift(1)
    return df.dropna()


def data_checks(df: pd.DataFrame) -> dict:
    """Facts the reader needs before trusting the series (spread drift, jumps, autocorrelation)."""
    d = df["r"].diff().dropna()
    spread = ((df["x"] - df["r"]) * 100).groupby([df.index.year, np.where(df.index.month > 6, "H2", "H1")]).mean()
    return {
        "spread_pp": spread.round(2),
        "dr_autocorr": [float(d.autocorr(k)) for k in (1, 2, 3)],
        "dr_sd_bp": float(d.std() * 1e4),
        "max_abs_jump_bp": float(d.abs().max() * 1e4),
        "n_jump_40bp": int((d.abs() > 0.004).sum()),
        "largest_jumps": (d.abs().nlargest(5) * 1e4).round(0),
    }
