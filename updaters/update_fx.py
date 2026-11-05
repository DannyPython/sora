"""Update data/USDSGD_clean.csv from the MAS API (end-of-day USD/SGD, SGD per USD).

    python updaters/update_fx.py                # fetch, validate, write
    python updaters/update_fx.py --dry-run      # fetch + validate, print what would change, write nothing
    python updaters/update_fx.py --inspect      # list the fields the API offers

Needs MAS_API_KEY_FX in the environment or in .env (see .env.example).
"""
from __future__ import annotations

import argparse
import sys

import urllib3

import updater_common as common

URL = ("https://eservices.mas.gov.sg/apimg-gw/server/"
       "monthly_statistical_bulletin_non610ora/exchange_rates_end_of_period_daily/views/"
       "exchange_rates_end_of_period_daily")
KEY_VAR = "MAS_API_KEY_FX"
CSV_PATH = common.DATA_DIR / "USDSGD_clean.csv"

# Field names in the API response: adjust after running --inspect.
# RATE_FIELD = None -> auto-detect: the first field whose name contains "usd".
DATE_FIELD = "end_of_day"
RATE_FIELD = None
START_DATE = "2021-10-01"     # match the study window; keep consistent with the SORA/SOFR files

HEADER = ["Date", "USD/SGD"]
VALUE_RANGE = (0.5, 3.0)      # SGD per USD
http = urllib3.PoolManager()


def fetch_records(key: str) -> list:
    return common.paged_fetch(http, URL, key, DATE_FIELD)


def detect_rate_field(rec: dict) -> str:
    if RATE_FIELD:
        return RATE_FIELD
    for k in rec:
        if "usd" in k.lower() and k != DATE_FIELD:
            return k
    raise RuntimeError(f"No field containing 'usd' in {list(rec)}; set RATE_FIELD")


def merge(existing: dict, records: list, rate_field: str, start: str = START_DATE) -> list:
    """Existing CSV rows + API records -> sorted rows [date, USD/SGD]. Dates are parsed before comparing."""
    start_dt = common.to_dt(start)
    data = {k: list(v) for k, v in existing.items()}               # keyed by date d/m/Y
    for rec in records:
        if rec.get(rate_field) in (None, "") or common.to_dt(rec[DATE_FIELD]) < start_dt:
            continue
        v = common.fmt_dmy(common.to_dt(rec[DATE_FIELD]))
        data[v] = [v, f"{float(rec[rate_field]):.4f}"]
    return common.sort_rows(data.values())


def update(dry_run: bool = False, csv_path=CSV_PATH) -> None:
    records = fetch_records(common.api_key(KEY_VAR))
    if not records:
        sys.exit("The API returned no records.")
    rate_f = detect_rate_field(records[0])
    print(f"using date field '{DATE_FIELD}', rate field '{rate_f}'")
    existing = common.read_rows(csv_path)
    rows = merge(existing, records, rate_f, START_DATE)
    if not rows:
        sys.exit("Nothing kept: check START_DATE and the field names (run with --inspect).")
    common.check_rows(rows, HEADER, 1, *VALUE_RANGE, label="USD/SGD")
    print(common.diff_summary(existing, rows) + f"; latest date {rows[-1][0]}")
    if dry_run:
        print("dry run: nothing written")
        return
    common.write_rows_atomic(csv_path, HEADER, rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.inspect:
        common.run(common.inspect, http, URL, common.api_key(KEY_VAR))
    else:
        common.run(update, a.dry_run)
