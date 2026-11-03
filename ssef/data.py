"""Block 1 (data collection) for M0: SORA loader.

Kept for backward compatibility with run_m0.py and report code. The implementation lives in
`ssef.loaders` (one set of loaders for the whole project); M0 uses SORA only, SOFR / FX alignment
(Blueprint Sec. 7) is in `loaders.build_dataset`.
"""
from __future__ import annotations

from .loaders import SORA_CSV, load_sora  # noqa: F401  (re-exported)

DATE_COL = "SORA Value Date"
RATE_COL = "SORA"
