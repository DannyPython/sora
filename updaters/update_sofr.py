"""Update data/SOFR_clean.csv from the NY Fed Markets Data API (public, no key needed).

    python updaters/update_sofr.py              # fetch, validate, write
    python updaters/update_sofr.py --dry-run    # fetch + validate, print what would change, write nothing
    python updaters/update_sofr.py --inspect    # show the raw start of the API response
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta

import urllib3

import updater_common as common

URL = "https://markets.newyorkfed.org/api/rates/secured/sofr/search.json"
CSV_PATH = common.DATA_DIR / "SOFR_clean.csv"

# Field names in the API response: adjust after running --inspect
DATE_FIELD = "effectiveDate"
SOFR_FIELD = "percentRate"
START_DATE = "2021-10-01"     # match the study window; keep consistent with the SORA file

HEADER = ["SOFR Effective Date", "SOFR Publication Date", "SOFR"]
VALUE_RANGE = (-1.0, 20.0)    # percent per annum
http = urllib3.PoolManager()


def fetch_records() -> list:
    # One call covers the whole range (a few hundred rows per year).
    r = http.request("GET", URL, fields={"startDate": START_DATE, "endDate": date.today().isoformat()})
    if r.status != 200:
        raise RuntimeError(f"HTTP {r.status}: {r.data[:300]}")
    return json.loads(r.data.decode("utf-8"))["refRates"]


def merge(existing: dict, records: list, start: str = START_DATE) -> list:
    """Existing CSV rows + API records -> sorted rows [effective date, publication date, SOFR].
    The NY Fed publishes SOFR on the next US business day (~8am ET), so publication date = next row's
    effective date. The latest row stays blank until the next rate arrives. This is the look-ahead guard
    used by the as-of merge in ssef.loaders. Dates are parsed before being compared with `start`."""
    start_dt = common.to_dt(start)
    data = {k: list(v) for k, v in existing.items()}               # keyed by effective date d/m/Y
    for rec in records:
        if rec.get(SOFR_FIELD) in (None, "") or common.to_dt(rec[DATE_FIELD]) < start_dt:
            continue
        v = common.fmt_dmy(common.to_dt(rec[DATE_FIELD]))
        old_pub = data.get(v, [None, ""])[1]
        data[v] = [v, old_pub, f"{float(rec[SOFR_FIELD]):.4f}"]
    rows = common.sort_rows(data.values())
    for a, b in zip(rows, rows[1:]):
        if not a[1]:
            a[1] = b[0]
    return rows


def update(dry_run: bool = False, csv_path=CSV_PATH) -> None:
    existing = common.read_rows(csv_path)
    rows = merge(existing, fetch_records(), START_DATE)
    if not rows:
        sys.exit("Nothing kept: check START_DATE and the field names (run with --inspect).")
    common.check_rows(rows, HEADER, 2, *VALUE_RANGE, label="SOFR", pub_idx=1)
    print(common.diff_summary(existing, rows) + f"; latest effective date {rows[-1][0]}")
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
        def _inspect():
            r = http.request("GET", URL, fields={"startDate": (date.today() - timedelta(days=30)).isoformat(),
                                                 "endDate": date.today().isoformat()})
            print("status:", r.status)
            print(r.data.decode("utf-8")[:1500])
        common.run(_inspect)
    else:
        common.run(update, a.dry_run)
