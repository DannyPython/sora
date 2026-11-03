"""Run provenance (Blueprint Sec. 8): data hashes, sample, seeds, library versions, git commit."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import statsmodels

from . import loaders


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5,
                             cwd=Path(__file__).resolve().parent.parent)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def run_info(df: pd.DataFrame, args: dict, extra: dict | None = None) -> dict:
    info = {
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "data_sha256": {p.name: _sha(p) for p in (loaders.SORA_CSV, loaders.SOFR_CSV, loaders.FX_CSV) if p.exists()},
        "n_obs": len(df), "first_date": str(df.index[0].date()), "last_date": str(df.index[-1].date()),
        "args": args,
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "statsmodels": statsmodels.__version__,
    }
    info.update(extra or {})
    return info


def write(path: Path, info: dict) -> None:
    Path(path).write_text(json.dumps(info, indent=2, default=str))
