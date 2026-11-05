import csv
import json
import sys
from datetime import datetime, date

import urllib3

# NY Fed Markets Data API, public, no key needed.
URL = "https://markets.newyorkfed.org/api/rates/secured/sofr/search.json"
CSV_PATH = "data/SOFR_clean.csv"

# Field names in the API response: adjust after running --inspect
DATE_FIELD = "effectiveDate"
SOFR_FIELD = "percentRate"
START_DATE = "2021-10-01"     # match the study window; keep consistent with the SORA file

HEADER = ["SOFR Effective Date", "SOFR Publication Date", "SOFR"]
http = urllib3.PoolManager()


def fetch_records():
    # One call covers the whole range (a few hundred rows per year).
    r = http.request("GET", URL, fields={"startDate": START_DATE,
                                         "endDate": date.today().isoformat()})
    if r.status != 200:
        raise RuntimeError(f"HTTP {r.status}: {r.data[:300]}")
    return json.loads(r.data.decode("utf-8"))["refRates"]


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
    data = load_csv()                                   # keyed by effective date d/m/Y
    for rec in fetch_records():
        if rec.get(SOFR_FIELD) in (None, "") or rec[DATE_FIELD] < START_DATE:
            continue
        v = to_dmy(rec[DATE_FIELD]).strftime("%d/%m/%Y")
        old_pub = data.get(v, [None, ""])[1]
        data[v] = [v, old_pub, f"{float(rec[SOFR_FIELD]):.4f}"]

    rows = sorted(data.values(), key=lambda r: datetime.strptime(r[0], "%d/%m/%Y"))
    # The NY Fed publishes SOFR on the next US business day (~8am ET), so
    # publication date = next row's effective date. Latest row stays blank
    # until the next rate arrives. This is the look-ahead guard: in the merge,
    # only use a SOFR value on/after its publication date.
    for a, b in zip(rows, rows[1:]):
        if not a[1]:
            a[1] = b[0]

    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    print(f"{len(rows)} rows, latest effective date {rows[-1][0]}")


if __name__ == "__main__":
    if "--inspect" in sys.argv:
        r = http.request("GET", URL, fields={"startDate": "2026-10-01",
                                             "endDate": date.today().isoformat()})
        print("status:", r.status)
        print(r.data.decode("utf-8")[:1500])
    else:
        update()
