"""Health check of the three data files (the project's 'database').

    python check_data.py            # prints PASS / WARN / FAIL / INFO per check; exit code 1 on any FAIL

FAIL  the file cannot be trusted (bad format, duplicates, unsorted, values out of range, publication before value).
WARN  worth a look but analysis can proceed (stale data, calendar gaps, series that end on different days).
INFO  facts to know (row counts, largest daily move, US-holiday gaps that the as-of merge handles).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .loaders import DATA_DIR

DATE_RE = re.compile(r"^\d{2}/\d{2}/\d{4}$")
SPECS = {
    "SORA": dict(file="Domestic_Interest_Rates_clean.csv", cols=["SORA Value Date", "SORA Publication Date", "SORA"],
                 pub=1, val=2, lo=-1.0, hi=20.0),
    "SOFR": dict(file="SOFR_clean.csv", cols=["SOFR Effective Date", "SOFR Publication Date", "SOFR"],
                 pub=1, val=2, lo=-1.0, hi=20.0, pub_is_next_row=True),
    "USDSGD": dict(file="USDSGD_clean.csv", cols=["Date", "USD/SGD"], pub=None, val=1, lo=0.5, hi=3.0),
}


@dataclass(frozen=True)
class Check:
    scope: str
    name: str
    level: str          # PASS | WARN | FAIL | INFO
    detail: str


def _first(items, n=3) -> str:
    items = list(items)
    return ", ".join(str(x) for x in items[:n]) + (" ..." if len(items) > n else "")


def check_file(name: str, data_dir: Path, asof: pd.Timestamp, stale_days: int = 7, max_gap_days: int = 7):
    """(checks, dates, values) for one file; dates/values are None if the file could not be parsed."""
    spec, out = SPECS[name], []
    add = lambda n, lvl, d: out.append(Check(name, n, lvl, d))
    path = Path(data_dir) / spec["file"]
    if not path.exists():
        add("file", "FAIL", f"{path} not found")
        return out, None, None
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(df.columns) != spec["cols"]:
        add("header", "FAIL", f"columns {list(df.columns)} != expected {spec['cols']}")
        return out, None, None
    add("header", "PASS", "as expected")
    dcol = spec["cols"][0]
    bad = df.index[~df[dcol].str.match(DATE_RE)]
    if len(bad):
        add("date format", "FAIL", f"{len(bad)} value(s) not dd/mm/yyyy, e.g. {_first(df.loc[bad, dcol])}")
        return out, None, None
    dates = pd.to_datetime(df[dcol], format="%d/%m/%Y", errors="coerce")
    if dates.isna().any():
        add("date format", "FAIL", f"{int(dates.isna().sum())} impossible date(s), e.g. {_first(df.loc[dates.isna(), dcol])}")
        return out, None, None
    add("date format", "PASS", f"{len(df)} rows, {dates.min():%d %b %Y} to {dates.max():%d %b %Y}")

    dup = df[dcol][dates.duplicated(keep=False)]
    unsorted = np.where(np.diff(dates.to_numpy().astype("int64")) <= 0)[0]
    if len(dup):                                                  # one stable check name whatever the outcome
        add("dates unique and sorted", "FAIL", f"{len(dup)} duplicated, e.g. {_first(dup.unique())}")
    elif len(unsorted):
        add("dates unique and sorted", "FAIL", f"not strictly increasing at row {unsorted[0] + 1} ({df[dcol].iloc[unsorted[0] + 1]})")
    else:
        add("dates unique and sorted", "PASS", "strictly increasing")

    val = pd.to_numeric(df[spec["cols"][spec["val"]]], errors="coerce")
    if val.isna().any():
        add("values", "FAIL", f"{int(val.isna().sum())} missing / non-numeric, e.g. row {_first(df.index[val.isna()] + 2)} (file line)")
    elif ((val < spec["lo"]) | (val > spec["hi"])).any():
        o = val[(val < spec["lo"]) | (val > spec["hi"])]
        add("values", "FAIL", f"{len(o)} outside [{spec['lo']}, {spec['hi']}], e.g. {_first(df.loc[o.index, dcol])}")
    else:
        add("values", "PASS", f"all numeric, within [{spec['lo']}, {spec['hi']}] (min {val.min():.4f}, max {val.max():.4f})")

    wk = dates[dates.dt.dayofweek >= 5]
    add("weekend dates", "WARN" if len(wk) else "PASS", f"{len(wk)} weekend row(s), e.g. {_first(wk.dt.strftime('%d/%m/%Y'))}" if len(wk) else "none")
    gaps = dates.diff().dt.days
    big = gaps[gaps > max_gap_days]
    add("calendar gaps", "WARN" if len(big) else "PASS",
        f"{len(big)} gap(s) over {max_gap_days} days, e.g. before {_first(dates[big.index].dt.strftime('%d/%m/%Y'))}" if len(big)
        else f"longest gap {int(gaps.max())} days")

    pcol = spec["pub"]
    if pcol is not None:
        pub_s = df[spec["cols"][pcol]]
        blank = pub_s == ""
        mid_blank = blank.iloc[:-1].sum()
        if mid_blank:
            add("publication dates", "FAIL", f"{int(mid_blank)} blank before the last row")
        else:
            bad_pub = ~pub_s[~blank].str.match(DATE_RE)
            if bad_pub.any():
                add("publication dates", "FAIL", f"{int(bad_pub.sum())} not dd/mm/yyyy")
            else:
                pub = pd.to_datetime(pub_s.where(~blank), format="%d/%m/%Y")
                early = (pub < dates).sum()
                late = ((pub - dates).dt.days > 7).sum()
                if early:
                    add("publication dates", "FAIL", f"{int(early)} publication date(s) before the value date")
                elif late:
                    add("publication dates", "WARN", f"{int(late)} published more than 7 days after the value date")
                else:
                    add("publication dates", "PASS", f"never before the value date; {int(blank.sum())} blank (latest row)")
                if spec.get("pub_is_next_row"):
                    nxt = dates.shift(-1)
                    mism = int(((pub != nxt) & pub.notna()).sum())
                    add("publication rule", "WARN" if mism else "PASS",
                        f"{mism} row(s) where publication date != next effective date" if mism
                        else "publication date = next effective date (the updater's convention)")
    chg = val.diff().abs()
    if chg.notna().any():
        i = chg.idxmax()
        add("largest daily move", "INFO", f"{chg.max():.4f} on {df[dcol].iloc[i]}")
    age = (asof.normalize() - dates.max()).days
    add("freshness", "WARN" if age > stale_days else "PASS", f"last observation {age} day(s) before {asof:%d %b %Y}")
    return out, dates, val


def validate_all(data_dir=DATA_DIR, asof=None, stale_days: int = 7) -> list:
    asof = pd.Timestamp.today() if asof is None else pd.Timestamp(asof)
    checks, dates = [], {}
    for name in SPECS:
        c, d, _ = check_file(name, data_dir, asof, stale_days)
        checks += c
        if d is not None:
            dates[name] = d
    add = lambda n, lvl, d: checks.append(Check("cross-file", n, lvl, d))
    if len(dates) == 3:
        S, F, X = (set(dates[k]) for k in ("SORA", "SOFR", "USDSGD"))
        miss = sorted(S - X)
        add("SORA dates in USD/SGD", "WARN" if miss else "PASS",
            f"{len(miss)} SORA date(s) have no FX rate, e.g. {_first(pd.Timestamp(m).strftime('%d/%m/%Y') for m in miss)}" if miss
            else "every SORA date has an FX rate (both are Singapore calendars)")
        add("SORA dates without SOFR", "INFO", f"{len(S - F)} (US holidays; the as-of merge carries the last published SOFR); "
            f"{len(F - S)} SOFR dates are not SORA dates")
        starts = [dates[k].min() for k in dates]
        add("study window start", "WARN" if (max(starts) - min(starts)).days > 3 else "PASS",
            ", ".join(f"{k} {dates[k].min():%d %b %Y}" for k in dates))
        ends = {k: dates[k].max() for k in dates}
        spread = (max(ends.values()) - min(ends.values())).days
        add("series end dates", "WARN" if spread > 5 else "PASS",
            ", ".join(f"{k} {v:%d %b %Y}" for k, v in ends.items()) + f" (spread {spread} days)")
    return checks


def format_checks(checks: list) -> str:
    w = max(len(c.scope) for c in checks), max(len(c.name) for c in checks)
    lines = [f"{c.level:<5} {c.scope:<{w[0]}}  {c.name:<{w[1]}}  {c.detail}" for c in checks]
    n = {k: sum(c.level == k for c in checks) for k in ("FAIL", "WARN", "PASS", "INFO")}
    lines.append("")
    lines.append(f"FAIL {n['FAIL']}   WARN {n['WARN']}   PASS {n['PASS']}   INFO {n['INFO']}")
    return "\n".join(lines)


def has_failures(checks: list) -> bool:
    return any(c.level == "FAIL" for c in checks)


def to_records(checks: list) -> list:
    return [asdict(c) for c in checks]
