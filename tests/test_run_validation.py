import pandas as pd
import pytest

import run_validation
from test_evt import check_headline_layout
from test_m1 import HAVE_DATA

needs_data = pytest.mark.skipif(not HAVE_DATA, reason="data CSVs not present")
SMALL = ["--sims", "2", "--B", "9", "--workers", "1", "--no-ecm-gate"]


@needs_data
def test_run_validation_prints_the_headline_layout_and_the_report_only_on_request(tmp_path, capsys):
    R = run_validation.main(SMALL + ["--out", str(tmp_path)])
    out = capsys.readouterr().out
    h = check_headline_layout(out, tmp_path, ["data", "fitted_m1", "spurious", "m0", "m0_break", "flags"])
    assert "Share of simulated datasets" not in out
    names = [d.name for d in R["dgps"]]
    assert all(f"[{n}]" in out for n in names) and len(names) == 6
    g = h[h["section"] == "m0_break"].set_index("metric")["value"]
    assert {"classical_f_pct", "bootstrap_f_true_break_pct", "cusum_pct", "fitted_c1_in_0_1_pct"} <= set(g.index)
    assert "bootstrap_f_true_break_pct" not in h[h["section"] == "spurious"]["metric"].tolist()
    assert "ecm_bootstrap_gate_pct" not in set(h["metric"])                               # --no-ecm-gate: not run, not listed
    pct = pd.to_numeric(h[h["metric"].str.endswith("_pct")]["value"])
    assert pct.between(0, 100).all()
    run_validation.main(SMALL + ["--report", "--out", str(tmp_path)])
    out2 = capsys.readouterr().out
    assert out2.index("headline numbers") < out2.index("Share of simulated datasets") < out2.index("All tables saved in:")
