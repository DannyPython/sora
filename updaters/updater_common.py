"""Shared helpers for the three data updaters.

What they fix compared with the copy-pasted versions:
  * API keys come from the environment (or a local, git-ignored .env file), never from the source code;
  * dates are PARSED before being compared with START_DATE (comparing strings only works while the API
    happens to return ISO dates);
  * pagination follows the rows actually returned (a capped page size no longer ends the download early);
  * every update is validated BEFORE anything is written, and the CSV is replaced atomically, so a failed
    or partial download can no longer leave a truncated file behind;
  * file paths are anchored to the repository root, not to the current working directory.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import urllib3

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATE_FORMATS = ("%Y-%m-%d", "%d %b %Y", "%d/%m/%Y", "%Y-%m-%dT%H:%M:%S")
CSV_DATE = "%d/%m/%Y"


# ------------------------------------------------------------------ keys
def load_env(path: Path | None = None) -> None:
    """Read KEY=VALUE lines from .env into os.environ (existing variables win). Comments with '#'."""
    path = Path(path) if path else ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def api_key(var: str) -> str:
    """The MAS API key for `var`, from the environment or .env. Stops with a clear message if it is missing."""
    load_env()
    key = os.environ.get(var, "").strip()
    if not key:
        sys.exit(f"Missing {var}. Copy .env.example to .env and fill it in (or export {var}). "
                 "Keys must never be committed to git; rotate any key that was.")
    return key


# ------------------------------------------------------------------ dates
def to_dt(s) -> datetime:
    """Parse the date formats the APIs and our CSVs use. Raises ValueError for anything else."""
    s = str(s).strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(f"Unrecognised date: {s}")


def fmt_dmy(dt: datetime) -> str:
    return dt.strftime(CSV_DATE)


def start_dt(start: str) -> datetime:
    return to_dt(start)


# ------------------------------------------------------------------ CSV in / out
def read_rows(path) -> dict:
    """Existing CSV as {first column (d/m/Y): row}; empty dict if the file does not exist yet."""
    try:
        with open(path, newline="") as f:
            return {r[0]: r for r in list(csv.reader(f))[1:] if r}
    except FileNotFoundError:
        return {}


def sort_rows(rows) -> list:
    return sorted(rows, key=lambda r: datetime.strptime(r[0], CSV_DATE))


def check_rows(rows: list, header: list, value_idx: int, lo: float, hi: float, label: str,
               pub_idx: int | None = None) -> None:
    """Refuse to write a bad file: right width, strictly increasing dates, numeric values inside [lo, hi],
    publication date (if any) not before the value date. Raises ValueError listing the first problems."""
    problems = []
    prev = None
    for i, r in enumerate(rows):
        if len(r) != len(header):
            problems.append(f"row {i}: {len(r)} columns, expected {len(header)}")
            continue
        try:
            d = datetime.strptime(r[0], CSV_DATE)
        except ValueError:
            problems.append(f"row {i}: bad date {r[0]!r}")
            continue
        if prev is not None and d <= prev:
            problems.append(f"row {i}: date {r[0]} is not after the previous row")
        prev = d
        try:
            v = float(r[value_idx])
        except ValueError:
            problems.append(f"row {i} ({r[0]}): value {r[value_idx]!r} is not a number")
            continue
        if not math.isfinite(v) or not lo <= v <= hi:
            problems.append(f"row {i} ({r[0]}): value {v} outside [{lo}, {hi}]")
        if pub_idx is not None and r[pub_idx]:
            try:
                if datetime.strptime(r[pub_idx], CSV_DATE) < d:
                    problems.append(f"row {i} ({r[0]}): publication date {r[pub_idx]} is before the value date")
            except ValueError:
                problems.append(f"row {i}: bad publication date {r[pub_idx]!r}")
    if problems:
        raise ValueError(f"{label}: refusing to write, {len(problems)} problem(s):\n  " + "\n  ".join(problems[:8]))


def write_rows_atomic(path, header: list, rows: list) -> None:
    """Write to a temporary file in the same folder, then replace the target in one step."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def diff_summary(existing: dict, rows: list) -> str:
    """What an update does: dates added, and dates whose stored VALUE (last column) was revised."""
    added = [r for r in rows if r[0] not in existing]
    revised = [r for r in rows if r[0] in existing and existing[r[0]][-1] != r[-1]]
    msg = f"{len(rows)} rows ({len(added)} new, {len(revised)} revised)"
    if revised:
        msg += "; revised: " + ", ".join(f"{r[0]} {existing[r[0]][-1]}->{r[-1]}" for r in revised[:5])
        msg += " ..." if len(revised) > 5 else ""
    return msg


# ------------------------------------------------------------------ HTTP
def paged_fetch(http, url: str, key: str, date_field: str, page: int = 1000) -> list:
    """All records of an MAS 'elements' endpoint. Pagination follows the rows actually returned, so an API
    that caps the page size below `page` is still read to the end; a page with nothing new stops the loop."""
    out, seen, offset = [], set(), 0
    while True:
        r = http.request("GET", url, fields={"limit": page, "offset": offset}, headers={"keyid": key})
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}: {r.data[:300]}")
        recs = json.loads(r.data.decode("utf-8"))["elements"]
        new = [x for x in recs if x[date_field] not in seen]
        if not new:
            return out
        seen.update(x[date_field] for x in new)
        out += new
        offset += len(recs)


def inspect(http, url: str, key: str) -> None:
    """Print the status and the FIELD NAMES of the first record: the quickest way to see which series an
    endpoint offers (e.g. overnight vs compounded SORA)."""
    r = http.request("GET", url, fields={"limit": 3, "offset": 0}, headers={"keyid": key})
    print("status:", r.status)
    body = r.data.decode("utf-8")
    try:
        els = json.loads(body)["elements"]
        print(f"{len(els)} sample record(s); fields of the first one:")
        for k, v in els[0].items():
            print(f"  {k:<32} {v!r}")
    except (ValueError, KeyError, IndexError):
        print(body[:1500])


def run(fn, *args, **kwargs) -> None:
    """Run an updater action; network, API-shape and validation errors become ONE clear line and exit code 1
    (not a 30-line traceback), so update_all.py can report which updater failed and why."""
    try:
        fn(*args, **kwargs)
    except (urllib3.exceptions.HTTPError, OSError, RuntimeError, ValueError, KeyError) as e:
        sys.exit(f"ERROR ({type(e).__name__}): {str(e)[:400]}")
