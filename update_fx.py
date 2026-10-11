import csv
import json
import sys
from datetime import datetime

import urllib3

URL = ("https://eservices.mas.gov.sg/apimg-gw/server/"
       "monthly_statistical_bulletin_non610ora/exchange_rates_end_of_period_daily/views/"
       "exchange_rates_end_of_period_daily")
API_KEY = "8f4bbd9f-11f8-4ca1-a520-cba4de933eab"          
CSV_PATH = "data/USDSGD_clean.csv"

# Field names in the API response: adjust after running --inspect.
# RATE_FIELD = None -> auto-detect: the first field whose name contains "usd".
DATE_FIELD = "end_of_day"
RATE_FIELD = None
START_DATE = "2021-10-01"     # match the study window; keep consistent with the SORA/SOFR files

HEADER = ["Date", "USD/SGD"]
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
        offset += len(recs)              # the API may cap the page size below `page`


def to_dmy(s):
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%d %b %Y", "%d/%m/%Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(f"Unrecognised date: {s}")


def detect_rate_field(rec):
    if RATE_FIELD:
        return RATE_FIELD
    for k in rec:
        if "usd" in k.lower() and k != DATE_FIELD:
            return k
    raise RuntimeError(f"No field containing 'usd' in {list(rec)}; set RATE_FIELD")


def load_csv():
    try:
        with open(CSV_PATH, newline="") as f:
            return {r[0]: r for r in list(csv.reader(f))[1:]}
    except FileNotFoundError:
        return {}


def update():
    data = load_csv()                                   # keyed by date d/m/Y
    recs = fetch_records()
    rate_f = detect_rate_field(recs[0])
    print(f"using date field '{DATE_FIELD}', rate field '{rate_f}'")
    for rec in recs:
        if rec.get(rate_f) in (None, "") or rec[DATE_FIELD] < START_DATE:
            continue
        v = to_dmy(rec[DATE_FIELD]).strftime("%d/%m/%Y")
        data[v] = [v, f"{float(rec[rate_f]):.4f}"]

    rows = sorted(data.values(), key=lambda r: datetime.strptime(r[0], "%d/%m/%Y"))
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    print(f"{len(rows)} rows" +
          (f", latest date {rows[-1][0]}" if rows else " (nothing kept, check START_DATE / fields)"))


if __name__ == "__main__":
    if "--inspect" in sys.argv:
        r = http.request("GET", URL, fields={"limit": 3, "offset": 0},
                         headers={"keyid": API_KEY})
        print("status:", r.status)
        print(r.data.decode("utf-8")[:1500])
    else:
        update()
