import csv
import json
import os
import sys
from datetime import datetime

import urllib3

URL = ("https://eservices.mas.gov.sg/apimg-gw/server/"
       "monthly_statistical_bulletin_non610mssql/domestic_interest_rates_daily/views/"
       "domestic_interest_rates_daily")
API_KEY = "222fca40-7f98-42bd-95e7-6d70e414c912"          
CSV_PATH = "data/Domestic_Interest_Rates_clean.csv"

# Field names in the API response: adjust after running --inspect
DATE_FIELD = "end_of_day"
PUB_FIELD = "published_date"
SORA_FIELD = "sora"
START_DATE = "2024-01-01"     # keep the CSV consistent with range

HEADER = ["SORA Value Date", "SORA Publication Date", "SORA"]
http = urllib3.PoolManager()


def fetch_records(page=1000):
    out, seen, offset = [], set(), 0
    while True:
        r = http.request("GET", URL, fields={"limit": page, "offset": offset},
                         headers={"keyid": API_KEY})
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}: {r.data[:300]}")
        recs = json.loads(r.data.decode("utf-8"))["elements"]
        new = [x for x in recs if x[DATE_FIELD] not in seen]
        if not new:                      # empty page, or the API ignores offset
            return out
        seen.update(x[DATE_FIELD] for x in new)
        out += new
        offset += page
        if len(recs) < page:
            return out


def to_dmy(s):
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%d %b %Y", "%d/%m/%Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(f"Unrecognised date: {s}")


def load_csv():
    try:
        with open(CSV_PATH, newline="") as f:
            return {r[0]: r for r in list(csv.reader(f))[1:]}
    except FileNotFoundError:
        return {}


def update():
    data = load_csv()                                   # keyed by value date d/m/Y
    for rec in fetch_records():
        if rec.get(SORA_FIELD) in (None, "") or rec[DATE_FIELD] < START_DATE:
            continue
        v = to_dmy(rec[DATE_FIELD]).strftime("%d/%m/%Y")
        p = to_dmy(rec[PUB_FIELD]).strftime("%d/%m/%Y") if PUB_FIELD and rec.get(PUB_FIELD) else ""
        old_pub = data.get(v, [None, ""])[1]
        data[v] = [v, p or old_pub, f"{float(rec[SORA_FIELD]):.4f}"]

    rows = sorted(data.values(), key=lambda r: datetime.strptime(r[0], "%d/%m/%Y"))
    # if no publication field: publication date = next business day = next row's value date
    for a, b in zip(rows, rows[1:]):
        if not a[1]:
            a[1] = b[0]

    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    print(f"{len(rows)} rows, latest value date {rows[-1][0]}")


if __name__ == "__main__":
    if "--inspect" in sys.argv:
        r = http.request("GET", URL, fields={"limit": 3, "offset": 0},
                         headers={"keyid": API_KEY})
        print("status:", r.status)
        print(r.data.decode("utf-8")[:1500])
    else:
        update()