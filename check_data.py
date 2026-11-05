"""Check the data files in data/ and report PASS / WARN / FAIL / INFO.

    python check_data.py                    # exit code 1 if any FAIL
    python check_data.py --strict           # exit code 1 on WARN as well
    python check_data.py --asof 2026-10-30  # judge freshness against a given date
    python check_data.py --json results/data_checks.json
"""
import argparse
import json
import sys
from pathlib import Path

from ssef import validate
from ssef.loaders import DATA_DIR

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=str(DATA_DIR))
ap.add_argument("--asof", default=None, help="date used for the freshness check (default: today)")
ap.add_argument("--stale-days", type=int, default=7)
ap.add_argument("--strict", action="store_true")
ap.add_argument("--json", default=None)
a = ap.parse_args()

checks = validate.validate_all(a.data_dir, a.asof, a.stale_days)
print(validate.format_checks(checks))
if a.json:
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(validate.to_records(checks), indent=2))
bad = validate.has_failures(checks) or (a.strict and any(c.level == "WARN" for c in checks))
sys.exit(1 if bad else 0)
