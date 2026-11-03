"""Update data/Domestic_Interest_Rates_clean.csv from the MAS API (daily SORA).

    python updaters/update_sora.py              # fetch, validate, write
    python updaters/update_sora.py --dry-run    # fetch + validate, print what would change, write nothing
    python updaters/update_sora.py --inspect    # list the fields the API offers (overnight vs compounded SORA)
    python updaters/update_sora.py --field <name> --csv data/<other>.csv    # another series, in its own file

Needs MAS_API_KEY_SORA in the environment or in .env (see .env.example).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import urllib3

import updater_common as common

URL = ("https://eservices.mas.gov.sg/apimg-gw/server/"
       "monthly_statistical_bulletin_non610mssql/domestic_interest_rates_daily/views/"
       "domestic_interest_rates_daily")
KEY_VAR = "MAS_API_KEY_SORA"
CSV_PATH = common.DATA_DIR / "Domestic_Interest_Rates_clean.csv"

# Field names in the API response: adjust after running --inspect
DATE_FIELD = "end_of_day"
PUB_FIELD = "published_date"
SORA_FIELD = "sora"
START_DATE = "2021-10-01"     # keep the CSV consistent with the study window

HEADER = ["SORA Value Date", "SORA Publication Date", "SORA"]
VALUE_RANGE = (-1.0, 20.0)    # percent per annum: anything outside is a data error, not a rate
http = urllib3.PoolManager()


def fetch_records(key: str) -> list:
    return common.paged_fetch(http, URL, key, DATE_FIELD)


def merge(existing: dict, records: list, start: str = START_DATE, field: str = SORA_FIELD) -> list:
    """Existing CSV rows + API records -> sorted rows [value date, publication date, SORA] (d/m/Y strings).
    Dates are parsed before being compared with `start`."""
    start_dt = common.to_dt(start)
    data = {k: list(v) for k, v in existing.items()}               # keyed by value date d/m/Y
    for rec in records:
        if rec.get(field) in (None, "") or common.to_dt(rec[DATE_FIELD]) < start_dt:
            continue
        v = common.fmt_dmy(common.to_dt(rec[DATE_FIELD]))
        p = common.fmt_dmy(common.to_dt(rec[PUB_FIELD])) if PUB_FIELD and rec.get(PUB_FIELD) else ""
        old_pub = data.get(v, [None, ""])[1]
        data[v] = [v, p or old_pub, f"{float(rec[field]):.4f}"]
    rows = common.sort_rows(data.values())
    # if no publication field: publication date = next business day = next row's value date
    for a, b in zip(rows, rows[1:]):
        if not a[1]:
            a[1] = b[0]
    return rows


def update(dry_run: bool = False, field: str = SORA_FIELD, csv_path=CSV_PATH) -> None:
    if field != SORA_FIELD and Path(csv_path) == CSV_PATH:
        sys.exit(f"--field {field} would overwrite the main SORA file: give a separate --csv path.")
    records = fetch_records(common.api_key(KEY_VAR))
    existing = common.read_rows(csv_path)
    rows = merge(existing, records, START_DATE, field)
    if not rows:
        sys.exit("Nothing kept: check START_DATE and the field names (run with --inspect).")
    common.check_rows(rows, HEADER, 2, *VALUE_RANGE, label=f"SORA ({field})", pub_idx=1)
    print(common.diff_summary(existing, rows) + f"; latest value date {rows[-1][0]}")
    if dry_run:
        print("dry run: nothing written")
        return
    common.write_rows_atomic(csv_path, HEADER, rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--field", default=SORA_FIELD)
    ap.add_argument("--csv", default=str(CSV_PATH))
    a = ap.parse_args()
    if a.inspect:
        common.run(common.inspect, http, URL, common.api_key(KEY_VAR))
    else:
        common.run(update, a.dry_run, a.field, a.csv)
