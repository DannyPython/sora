# M1 package (ssef) - policy-anchored Vasicek in error-correction form

Scope: the M1 work only (ECM, valid nested test, cointegration gate, out-of-sample race, M1 noise, robustness).
Not included: M0 report/headline, EVT, VaR backtests, S-REIT step, updaters (see "Left for others").

This branch is for review only and is purely additive: no existing file is modified, and `mzero/`, `m1_noise.py`,
`run_m1_noise.py` and the updaters are untouched. `docs/m1_sample_report.txt` is the output of one full run on the data
through 29 Oct 2026 (regenerate it with `python run_m1.py`).

## Install and run

    pip install numpy pandas scipy statsmodels pytest      # pytest is only needed for the tests
    python run_m1.py            # full report, about 30 seconds, writes results/m1/
    python run_m1.py --fast     # fewer bootstrap draws, about 8 seconds
    python -m pytest            # 23 tests, about 25 seconds

Useful options: `--p 2` (fix the dr lags; default is the BIC choice), `--no-fx`, `--sofr-lag same-day`,
`--break-date 2025-04-14` (override the Gregory-Hansen date), `--B 999`, `--window 500`, `--horizons 1 20 60`.

## Layout (copy into the repository root; nothing in mzero/ is touched)

| File | Role |
|---|---|
| `ssef/loaders.py` | All loaders, as-of SOFR merge on publication dates, `build_dataset`, data checks |
| `ssef/ols.py` | ONE OLS routine `fit_ols`; HAC covariance; nested F; delta-method SE |
| `ssef/ecm.py` | M1 error-correction regression; M0 = the same regression with columns dropped (`nested_columns`, `ols.subset`); lag selection |
| `ssef/cointegration.py` | Gate: unit roots, Engle-Granger over lags and trend, Gregory-Hansen, slope stability |
| `ssef/nested.py` | Valid tests with an I(1) regressor: bootstrap anchor test and ECM cointegration test (PSS/BDM type) |
| `ssef/race.py` | Random walk vs M0 vs M1, h = 1/20/60, Clark-West with bootstrap p-values |
| `ssef/noise.py` | Noise diagnostics, standardised residuals, Gaussian/bootstrap draws |
| `ssef/simulate.py` | Scenario simulation (hold / up / down SOFR) |
| `ssef/robustness.py` | Lag grid, start-date sensitivity, race window grid |
| `ssef/provenance.py` | Data hashes, versions, seeds, git commit -> `run_info.json` |
| `run_m1.py` | Driver: plain-language report + tables in `results/m1/` |
| `tests/test_m1.py`, `conftest.py` | Tests (OLS vs statsmodels, no look-ahead, recovery, size/power of the bootstrap tests, mutation-checked) |

`m1_noise.py` and `run_m1_noise.py` are superseded by this package; keep them until the package is reviewed.

## Conventions and decisions to confirm

* Time step: one observation = 1/252 year. kappa = -ln(1 + a) / dt, where a is the coefficient on r(t-1) in the ECM.
* SOFR timing: `--sofr-lag strict` (default) uses a SOFR value only on dates AFTER its publication date. It differs from the
  updater convention on most days; the Blueprint leaves the lag open.
* z = daily USD/SGD return, SGD per USD, so z > 0 means the SGD weakened. Verify against the MAS series.
* dr lags: BIC choice (8 on the current data; residuals are white from 8). The reviewer's "one or two lags" leave
  autocorrelated residuals (Ljung-Box p ~ 0).
* The break date is estimated by Gregory-Hansen from the same data: tests that condition on it are exploratory.
  Gregory-Hansen's own critical values account for the search; the conditional bootstrap p-values do not.
* The linear Gaussian model allows negative rates (Blueprint 5.5).

## Left for others (not done here)

* M0 report headline (bias-corrected c1 with bootstrap CI), conditional EVT, VaR backtests, S-REIT repricing-gap step.
* Updaters: two MAS API keys are hard-coded (rotate them; use environment variables) and dates are compared as strings.
* Merge `mzero/` into `ssef/` (the README says ssef): `ssef/loaders.py` is written to take over `mzero/data.py`.
* Anchor variable: adding forward points would make CIP hold by construction; settle with the mentor first.
* Which MAS field is the SORA series (overnight vs compounded): run `python updaters/update_sora.py --inspect`.
