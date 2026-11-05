import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "updaters"))
import update_fx  # noqa: E402
import update_sofr  # noqa: E402
import update_sora  # noqa: E402
import updater_common as common  # noqa: E402

from ssef import loaders, validate  # noqa: E402

HAVE_DATA = loaders.SORA_CSV.exists() and loaders.SOFR_CSV.exists() and loaders.FX_CSV.exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="data CSVs not present")


# ------------------------------------------------------------------ reference: the logic BEFORE the refactor
def old_sora_merge(existing, recs, start="2021-10-01"):
    data = {k: list(v) for k, v in existing.items()}
    for rec in recs:
        if rec.get("sora") in (None, "") or rec["end_of_day"] < start:            # string comparison
            continue
        v = common.to_dt(rec["end_of_day"]).strftime("%d/%m/%Y")
        p = common.to_dt(rec["published_date"]).strftime("%d/%m/%Y") if rec.get("published_date") else ""
        old_pub = data.get(v, [None, ""])[1]
        data[v] = [v, p or old_pub, f"{float(rec['sora']):.4f}"]
    rows = sorted(data.values(), key=lambda r: datetime.strptime(r[0], "%d/%m/%Y"))
    for a, b in zip(rows, rows[1:]):
        if not a[1]:
            a[1] = b[0]
    return rows


def old_sofr_merge(existing, recs, start="2021-10-01"):
    data = {k: list(v) for k, v in existing.items()}
    for rec in recs:
        if rec.get("percentRate") in (None, "") or rec["effectiveDate"] < start:
            continue
        v = common.to_dt(rec["effectiveDate"]).strftime("%d/%m/%Y")
        data[v] = [v, data.get(v, [None, ""])[1], f"{float(rec['percentRate']):.4f}"]
    rows = sorted(data.values(), key=lambda r: datetime.strptime(r[0], "%d/%m/%Y"))
    for a, b in zip(rows, rows[1:]):
        if not a[1]:
            a[1] = b[0]
    return rows


def sora_records(n=40, start="2024-01-02", seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n)
    return [{"end_of_day": d.strftime("%Y-%m-%d"), "published_date": (d + pd.offsets.BDay(1)).strftime("%Y-%m-%d"),
             "sora": f"{3.0 + 0.1 * rng.standard_normal():.4f}"} for d in days]


# ------------------------------------------------------------------ updaters: merge logic
def test_sora_merge_matches_the_previous_logic():
    recs = sora_records(60)
    existing = {r[0]: r for r in old_sora_merge({}, recs[:30])}
    recs[5]["sora"] = "9.1234"                                              # a revision
    recs.append({"end_of_day": "2021-09-29", "published_date": "2021-09-30", "sora": "0.0900"})   # before START_DATE
    recs.append({"end_of_day": "2030-01-02", "published_date": None, "sora": ""})                 # empty value
    assert update_sora.merge(existing, recs) == old_sora_merge(existing, recs)


def test_sofr_merge_matches_the_previous_logic():
    days = pd.bdate_range("2024-01-02", periods=50)
    recs = [{"effectiveDate": d.strftime("%Y-%m-%d"), "percentRate": 5.3 - 0.01 * i} for i, d in enumerate(days)]
    recs += [{"effectiveDate": "2021-09-30", "percentRate": 0.05}, {"effectiveDate": "2025-01-01", "percentRate": None}]
    existing = {r[0]: r for r in old_sofr_merge({}, recs[:20])}
    new = update_sofr.merge(existing, recs)
    assert new == old_sofr_merge(existing, recs)
    assert new[-1][1] == ""                                                 # latest row: publication date still blank
    assert all(a[1] == b[0] for a, b in zip(new, new[1:]))                  # publication = next effective date


def test_fx_merge_sorts_and_formats():
    recs = [{"end_of_day": "2024-01-03", "usd_sgd": "1.3300"}, {"end_of_day": "2024-01-02", "usd_sgd": "1.3200"},
            {"end_of_day": "2021-09-30", "usd_sgd": "1.3600"}]
    assert update_fx.merge({}, recs, "usd_sgd") == [["02/01/2024", "1.3200"], ["03/01/2024", "1.3300"]]


@needs_data
def test_merging_the_existing_files_back_in_changes_nothing():
    sora = pd.read_csv(loaders.SORA_CSV, dtype=str)
    recs = [{"end_of_day": datetime.strptime(r["SORA Value Date"], "%d/%m/%Y").strftime("%Y-%m-%d"),
             "published_date": datetime.strptime(r["SORA Publication Date"], "%d/%m/%Y").strftime("%Y-%m-%d"),
             "sora": r["SORA"]} for _, r in sora.iterrows()]
    existing = common.read_rows(loaders.SORA_CSV)
    assert update_sora.merge(existing, recs) == [existing[k] for k in existing]
    sofr = pd.read_csv(loaders.SOFR_CSV, dtype=str)
    recs = [{"effectiveDate": datetime.strptime(r["SOFR Effective Date"], "%d/%m/%Y").strftime("%Y-%m-%d"), "percentRate": r["SOFR"]}
            for _, r in sofr.iterrows()]
    existing = common.read_rows(loaders.SOFR_CSV)
    assert update_sofr.merge(existing, recs) == [existing[k] for k in existing]


def test_start_date_is_compared_as_a_date_not_as_text():
    recs = [{"end_of_day": "30 Sep 2021", "published_date": "01 Oct 2021", "sora": "0.0900"},      # before the start
            {"end_of_day": "04 Oct 2021", "published_date": "05 Oct 2021", "sora": "0.0802"}]
    assert not ("30 Sep 2021" < "2021-10-01")                    # why the old string comparison kept the first row
    assert [r[0] for r in update_sora.merge({}, recs)] == ["04/10/2021"]


def test_date_parser():
    assert common.to_dt("2024-01-02") == common.to_dt("02 Jan 2024") == common.to_dt("02/01/2024") == datetime(2024, 1, 2)
    assert common.to_dt("2024-01-02T00:00:00") == datetime(2024, 1, 2)
    with pytest.raises(ValueError):
        common.to_dt("yesterday")


# ------------------------------------------------------------------ updaters: keys, writing, pagination
def test_keys_come_from_env_file_and_environment_wins(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("# comment\nMAS_TEST_KEY = 'abc123'\nOTHER=from_file\n\nbadline\n")
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.delenv("MAS_TEST_KEY", raising=False)
    monkeypatch.setenv("OTHER", "from_env")
    assert common.api_key("MAS_TEST_KEY") == "abc123" and os.environ["OTHER"] == "from_env"
    monkeypatch.delenv("MAS_MISSING", raising=False)
    with pytest.raises(SystemExit) as e:
        common.api_key("MAS_MISSING")
    assert "MAS_MISSING" in str(e.value) and ".env.example" in str(e.value)


def test_no_key_is_hard_coded_in_the_updaters():
    for f in ("update_sora.py", "update_fx.py", "update_sofr.py", "updater_common.py"):
        text = (ROOT / "updaters" / f).read_text()
        import re
        assert not re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", text), f


def test_check_rows_rejects_bad_data():
    H = ["Date", "Pub", "V"]
    ok = [["02/01/2024", "03/01/2024", "3.5"], ["03/01/2024", "", "3.6"]]
    common.check_rows(ok, H, 2, 0, 10, "t", pub_idx=1)
    bad = {
        "not a number": [["02/01/2024", "", "x"]],
        "outside": [["02/01/2024", "", "99"]],
        "not after the previous": [["03/01/2024", "", "3"], ["02/01/2024", "", "3"]],
        "before the value date": [["03/01/2024", "02/01/2024", "3"]],
        "bad date": [["2024-01-02", "", "3"]],
        "columns": [["02/01/2024", "3"]],
    }
    for msg, rows in bad.items():
        with pytest.raises(ValueError, match=msg):
            common.check_rows(rows, H, 2, 0, 10, "t", pub_idx=1)


def test_atomic_write_keeps_the_original_if_it_fails(monkeypatch, tmp_path):
    f = tmp_path / "x.csv"
    common.write_rows_atomic(f, ["a", "b"], [["1", "2"]])
    assert f.read_text().splitlines() == ["a,b", "1,2"]
    monkeypatch.setattr(os, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        common.write_rows_atomic(f, ["a", "b"], [["9", "9"]])
    assert f.read_text().splitlines() == ["a,b", "1,2"] and [p.name for p in tmp_path.iterdir()] == ["x.csv"]


class FakeHttp:
    """An API that caps the page size at `cap`, whatever limit is requested."""
    def __init__(self, total, cap):
        self.rows = [{"end_of_day": (pd.Timestamp("2021-01-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d"), "sora": "1"}
                     for i in range(total)]
        self.cap = cap

    def request(self, method, url, fields=None, headers=None):
        off, lim = fields["offset"], min(fields["limit"], self.cap)
        body = json.dumps({"elements": self.rows[off:off + lim]}).encode()
        return type("R", (), {"status": 200, "data": body})()


def test_pagination_follows_the_rows_returned_not_the_page_size_asked():
    http = FakeHttp(total=1300, cap=500)
    assert len(common.paged_fetch(http, "u", "k", "end_of_day", page=1000)) == 1300
    # the loop the SORA updater used before stopped as soon as a page was shorter than the page it asked for:
    out, offset = [], 0
    while True:
        recs = json.loads(http.request("GET", "u", fields={"limit": 1000, "offset": offset}).data)["elements"]
        out += recs
        offset += 1000
        if len(recs) < 1000:
            break
    assert len(out) == 500                                       # silently truncated


def test_update_dry_run_writes_nothing_and_real_run_writes_a_valid_file(monkeypatch, tmp_path, capsys):
    target = tmp_path / "sora.csv"
    monkeypatch.setattr(common, "api_key", lambda v: "k")
    monkeypatch.setattr(update_sora, "fetch_records", lambda key: sora_records(30))
    update_sora.update(dry_run=True, csv_path=target)
    assert not target.exists() and "dry run" in capsys.readouterr().out
    update_sora.update(csv_path=target)
    rows = pd.read_csv(target)
    assert list(rows.columns) == update_sora.HEADER and len(rows) == 30
    assert validate.check_file  # module imported


def test_update_refuses_to_write_implausible_values(monkeypatch, tmp_path):
    target = tmp_path / "sora.csv"
    recs = sora_records(10)
    recs[4]["sora"] = "99.0"
    monkeypatch.setattr(common, "api_key", lambda v: "k")
    monkeypatch.setattr(update_sora, "fetch_records", lambda key: recs)
    with pytest.raises(ValueError, match="outside"):
        update_sora.update(csv_path=target)
    assert not target.exists()


def test_another_sora_field_cannot_overwrite_the_main_file(monkeypatch):
    monkeypatch.setattr(common, "api_key", lambda v: "k")
    with pytest.raises(SystemExit, match="separate --csv"):
        update_sora.update(field="comp_sora_1m")


# ------------------------------------------------------------------ validation
def make_data(tmp_path, n=60):
    days = pd.bdate_range("2024-01-02", periods=n)
    sora = pd.DataFrame({"SORA Value Date": days.strftime("%d/%m/%Y"),
                         "SORA Publication Date": (days + pd.offsets.BDay(1)).strftime("%d/%m/%Y"),
                         "SORA": np.round(3 + 0.01 * np.arange(n), 4)})
    sdays = days.delete(10)                                                   # a US holiday: no SOFR that day
    sofr = pd.DataFrame({"SOFR Effective Date": sdays.strftime("%d/%m/%Y"),
                         "SOFR Publication Date": list(sdays[1:].strftime("%d/%m/%Y")) + [""],
                         "SOFR": np.round(5.3 - 0.001 * np.arange(len(sdays)), 4)})
    fx = pd.DataFrame({"Date": days.strftime("%d/%m/%Y"), "USD/SGD": np.round(1.33 + 0.0005 * np.arange(n), 4)})
    for df, f in ((sora, "Domestic_Interest_Rates_clean.csv"), (sofr, "SOFR_clean.csv"), (fx, "USDSGD_clean.csv")):
        df.to_csv(tmp_path / f, index=False)
    return days[-1]


def levels(checks):
    return {(c.scope, c.name): c.level for c in checks}


def edit(tmp_path, fname, fn):
    df = pd.read_csv(tmp_path / fname, dtype=str, keep_default_na=False)
    fn(df)
    df.to_csv(tmp_path / fname, index=False)


def test_clean_synthetic_data_passes_everything(tmp_path):
    last = make_data(tmp_path)
    checks = validate.validate_all(tmp_path, asof=last)
    assert not [c for c in checks if c.level in ("FAIL", "WARN")], validate.format_checks(checks)


def duplicate_a_row(d):
    d.loc[5] = d.loc[4].tolist()


def swap_two_rows(d):
    d.loc[[1, 0]] = d.loc[[0, 1]].to_numpy()


@pytest.mark.parametrize("fname,mutate,scope,name,expected", [
    ("Domestic_Interest_Rates_clean.csv", duplicate_a_row, "SORA", "dates unique and sorted", "FAIL"),
    ("SOFR_clean.csv", swap_two_rows, "SOFR", "dates unique and sorted", "FAIL"),
    ("Domestic_Interest_Rates_clean.csv", lambda d: d.__setitem__("SORA", d["SORA"].where(d.index != 7, "55")), "SORA", "values", "FAIL"),
    ("Domestic_Interest_Rates_clean.csv", lambda d: d.__setitem__("SORA", d["SORA"].where(d.index != 7, "")), "SORA", "values", "FAIL"),
    ("Domestic_Interest_Rates_clean.csv", lambda d: d.__setitem__("SORA Publication Date", d["SORA Publication Date"].where(d.index != 9, "01/01/2020")), "SORA", "publication dates", "FAIL"),
    ("Domestic_Interest_Rates_clean.csv", lambda d: d.__setitem__("SORA Value Date", d["SORA Value Date"].where(d.index != 3, "2024-01-05")), "SORA", "date format", "FAIL"),
    ("SOFR_clean.csv", lambda d: d.__setitem__("SOFR Publication Date", d["SOFR Publication Date"].where(d.index != 4, "")), "SOFR", "publication dates", "FAIL"),
    ("SOFR_clean.csv", lambda d: d.__setitem__("SOFR Publication Date", d["SOFR Publication Date"].where(d.index != 4, "30/12/2030")), "SOFR", "publication rule", "WARN"),
    ("USDSGD_clean.csv", lambda d: d.__setitem__("USD/SGD", d["USD/SGD"].where(d.index != 2, "13.3")), "USDSGD", "values", "FAIL"),
    ("USDSGD_clean.csv", lambda d: d.drop(index=3, inplace=True), "cross-file", "SORA dates in USD/SGD", "WARN"),
])
def test_validation_catches_each_kind_of_damage(tmp_path, fname, mutate, scope, name, expected):
    last = make_data(tmp_path)
    edit(tmp_path, fname, mutate)
    assert levels(validate.validate_all(tmp_path, asof=last))[(scope, name)] == expected


def test_validation_catches_header_missing_file_and_stale_data(tmp_path):
    last = make_data(tmp_path)
    edit(tmp_path, "USDSGD_clean.csv", lambda d: d.rename(columns={"USD/SGD": "rate"}, inplace=True))
    assert levels(validate.validate_all(tmp_path, asof=last))[("USDSGD", "header")] == "FAIL"
    (tmp_path / "SOFR_clean.csv").unlink()
    assert levels(validate.validate_all(tmp_path, asof=last))[("SOFR", "file")] == "FAIL"
    make_data(tmp_path)
    assert levels(validate.validate_all(tmp_path, asof=last + pd.Timedelta(days=30)))[("SORA", "freshness")] == "WARN"


@needs_data
def test_the_real_data_files_pass_the_checks():
    last = max(pd.read_csv(f, dtype=str).iloc[-1, 0] for f in (loaders.SORA_CSV, loaders.SOFR_CSV, loaders.FX_CSV))
    checks = validate.validate_all(asof=pd.to_datetime(last, format="%d/%m/%Y"))
    assert not [c for c in checks if c.level in ("FAIL", "WARN")], validate.format_checks(checks)


def test_check_data_cli_exit_codes(tmp_path):
    last = make_data(tmp_path)
    cmd = [sys.executable, str(ROOT / "check_data.py"), "--data-dir", str(tmp_path), "--asof", str(last.date())]
    good = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert good.returncode == 0 and "FAIL 0" in good.stdout
    edit(tmp_path, "Domestic_Interest_Rates_clean.csv", lambda d: d.__setitem__("SORA", d["SORA"].where(d.index != 7, "55")))
    bad = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert bad.returncode == 1 and "FAIL" in bad.stdout
