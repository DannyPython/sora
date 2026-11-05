# sora

SSEF 2027: is the classical Vasicek short-rate model misspecified for SORA? (Blueprint v8c)
M0 = classical Vasicek (constant long-run mean). M1 = the long-run level moves with SOFR and USD/SGD (error-correction form).

## Quick start

```
pip install -r requirements.txt
python run_m0.py              # M0: gate, calibration, forecasts      -> results/m0/
python run_m1.py              # M1: gate, ECM, valid tests, race      -> results/m1/   (about 30 s; --fast for 8 s)
python plot_results.py        # eight M0 vs M1 figures from the real pipeline -> results/figures/ (PNG + PDF)
python check_data.py          # health check of the three data files (exit code 1 on any FAIL)
python -m pytest              # all tests
```

Refreshing the data needs MAS API keys: copy `.env.example` to `.env`, fill it in (it is git-ignored), then
`python update_all.py` (add `--dry-run` to fetch and validate without writing). Keys must never be committed.

## Layout

| Path | Role |
|---|---|
| `ssef/loaders.py` | All loaders (SORA, SOFR, USD/SGD), the as-of SOFR merge on publication dates, `build_dataset`, data checks |
| `ssef/data.py` | Backward-compatible re-export of the SORA loader |
| `ssef/ols.py` | The one OLS routine (HAC, nested F, delta method). M0 is M1 with columns dropped |
| `ssef/m0.py` | M0: AR(1) calibration, structural parameters, delta-method CIs, exact simulator, rolling forecast |
| `ssef/gate.py` | M0 gate: ADF/KPSS, `ar1_gate`, `adf_gate` (with lagged differences), Engle-Granger |
| `ssef/report.py` | M0 plain-language report (headline is the bootstrap view when the gate fails) |
| `ssef/ecm.py` | M1 error-correction regression, level shift, lag selection |
| `ssef/cointegration.py` | M1 gate: unit roots, Engle-Granger over lags/trend, Gregory-Hansen, slope stability |
| `ssef/nested.py` | Bootstrap tests valid with an I(1) regressor (anchor test, ECM cointegration test) |
| `ssef/race.py` | Random walk vs M0 vs M1 at 1/20/60 days, bootstrap Clark-West |
| `ssef/noise.py`, `ssef/simulate.py`, `ssef/robustness.py` | M1 noise checks, scenarios, sensitivity |
| `ssef/provenance.py` | Run provenance: data hashes, versions, seeds, git commit -> `run_info.json` |
| `ssef/validate.py`, `check_data.py` | Data validation |
| `run_m0.py`, `run_m1.py` | Reproducible drivers |
| `updaters/`, `update_all.py` | Data refresh from MAS and the NY Fed (see `data/README.md`) |
| `data/` | The three daily series and their data dictionary (`data/README.md`) |
| `ssef/plots.py`, `plot_results.py` | The eight M0 vs M1 figures: data, M0 gate, M1 gate, long-run levels, anchor test, instability, forecast race, M0 vs SORA. Numbers come from the same pipeline calls and seeds as `run_m0.py` / `run_m1.py` (a test compares them); statsmodels for ACF, Q-Q, rolling OLS |
| `tests/` | `test_m0.py`, `test_m1.py`, `test_data.py`, `test_plots.py` |
| `m1_noise.py`, `run_m1_noise.py` | Earlier M1 noise prototype, superseded by `ssef/` + `run_m1.py` (kept until reviewed) |
| `README_M1.md`, `docs/` | M1 notes and sample reports of one full run |

## Conventions

* One observation = one step of `dt = 1/252` year (business days; weekends not stretched).
* SOFR is used only after its publication date (`strict` by default); FX return `z > 0` means the SGD weakened.
* The linear Gaussian models allow negative rates.
* Every random step is seeded; every run writes `run_info.json` with the data hashes and the git commit.
