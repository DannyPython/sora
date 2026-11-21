# sora

SSEF 2027: is the classical Vasicek short-rate model misspecified for SORA? (Blueprint v8c)
M0 = classical Vasicek (constant long-run mean). M1 = the long-run level moves with SOFR and USD/SGD (error-correction form).

## Quick start

```
pip install -r requirements.txt
python run_m0.py              # M0: gate, calibration, forecasts      -> results/m0/
python run_m1.py              # M1: gate, ECM, valid tests, race      -> results/m1/   (about 30 s; --fast for 8 s)
python run_backtest.py        # VaR backtest RW vs M0 vs M1            -> results/backtest/ (about 10 s; --report for the text)
python run_density.py         # predictive-density validation RW vs M0 vs M1 -> results/density/ (a few seconds)
python run_evt.py             # EVT tail validation of the M0 / M1 residuals  -> results/evt/ (about 1 min; --fast for 15 s)
python run_validation.py      # size and power of the tests on simulated data -> results/validation/ (about 15 min; --fast for 3 min)
python check_data.py          # health check of the three data files (exit code 1 on any FAIL)
python -m pytest              # all tests
```

Refreshing the data needs MAS API keys: copy `.env.example` to `.env`, fill it in (it is git-ignored), then
`python update_all.py` (add `--dry-run` to fetch and validate without writing). Keys must never be committed.

## Layout

| Path | Role |
|---|---|
| `ssef/loaders.py` | All loaders (SORA, SOFR, USD/SGD), the as-of SOFR merge on publication dates, `build_dataset`, data checks |
| `ssef/ols.py` | The one OLS routine (HAC, nested F, delta method). M0 is M1 with columns dropped |
| `ssef/m0.py` | M0: calibration (AR(1) + p lagged changes, the same M0 as the ECM), structural parameters, delta-method CIs, exact simulator (forecasts live in `race.py`) |
| `ssef/gate.py` | M0 gate: ADF/KPSS, `adf_gate` (c1 with bootstrap CI; `p=0` is the pure AR(1)) |
| `ssef/report.py` | M0 plain-language report (headline is the bootstrap view when the gate fails) |
| `ssef/restricted_ecm.py` | M1 error-correction regression, level shift, lag selection; M0 = the same regression with columns dropped |
| `ssef/cointegration.py` | M1 gate: unit roots, Engle-Granger over lags/trend, Gregory-Hansen, slope stability |
| `ssef/nested.py` | Bootstrap tests valid with an I(1) regressor (anchor test, ECM cointegration test) |
| `ssef/race.py` | Random walk vs M0 vs M1 at 1/20/60 days, bootstrap Clark-West |
| `ssef/backtest.py` | VaR backtest (Proposal 4.7): 95/99% VaR of the SORA level from RW/M0/M1 (normal, historical, EVT-GPD tails), Kupiec, Christoffersen, tick-loss head-to-head (Giacomini-White), power |
| `ssef/density.py` | Predictive-density validation (Proposal 4.7): PIT of the VaR backtest's predictive distributions, KS, Berkowitz, bias / width / serial dependence / clustering, tail shares |
| `ssef/evt.py` | EVT validation (Proposal 4.6): mean-excess plot, GPD fit by threshold with bootstrap KS, threshold rule, M0 vs M1 tail difference, episode concentration |
| `ssef/validate_tests.py` | Validation of the validation: size and power of the classical / HAC / bootstrap F tests, Engle-Granger, the ECM bootstrap gate and CUSUM on simulated data with known truth (spurious, M0, M0 + level shift, M1 with chosen beta1 and speed) |
| `ssef/noise.py`, `ssef/simulate.py`, `ssef/robustness.py` | M1 noise checks, scenarios, sensitivity |
| `ssef/provenance.py` | Run provenance: data hashes, versions, seeds, git commit -> `run_info.json` |
| `ssef/validate.py`, `check_data.py` | Data validation |
| `run_m0.py`, `run_m1.py`, `run_backtest.py`, `run_density.py`, `run_evt.py`, `run_validation.py` | Reproducible drivers (`run_validation.py` is opt-in, slow) |
| `updaters/`, `update_all.py` | Data refresh from MAS and the NY Fed (see `data/README.md`) |
| `data/` | The three daily series and their data dictionary (`data/README.md`) |
| `tests/` | `test_m0.py`, `test_m1.py`, `test_backtest.py`, `test_density.py`, `test_evt.py`, `test_validate_tests.py`, `test_data.py` |
| `README_M1.md` | M1 notes and open decisions |

## Conventions

* One observation = one step of `dt = 1/252` year (business days; weekends not stretched).
* SOFR is used only after its publication date (`strict` by default); FX return `z > 0` means the SGD weakened.
* The linear Gaussian models allow negative rates.
* Every random step is seeded; every run writes `run_info.json` with the data hashes and the git commit.

## VaR backtest (`run_backtest.py`)

* Same models, data, lags and rolling window as the race; only the predictive distribution around the point forecast is new.
* VaR = quantile of the SORA **level** h days ahead. `up` = rates rise (the S-REIT borrower's risk), `down` = rates fall.
* Three tails per model: `normal` (the Vasicek Gaussian assumption), `hs` (window residual quantile), `evt` (GPD above the top
  10% of the window's standardised residuals, PWM fit; `--gpd mle` for maximum likelihood).
* Horizons > 1 day use non-overlapping origins (overlapping hits break the Kupiec/Christoffersen distributions), so h = 20
  has only ~37 points: the power table shows how little such a test can detect.
* Head-to-head uses the quantile (tick) loss with a Giacomini-White test, valid for nested models under a rolling window.
* Not done yet: GARCH-filtered (conditional) EVT; the breach clustering in the `down` tail is the evidence that it is needed.

## Density validation (`run_density.py`)

* Uses exactly the predictive distributions of the VaR backtest (same models, window, origins); the question is whether the
  WHOLE distribution is right, with every held-out point instead of the ~8 breaches of a 99% VaR.
* PIT = predictive cdf of the outcome, uniform if the model is right. `normal` = Vasicek Gaussian, `empirical` = window residuals.
  `z = Phi^-1(PIT)` is iid N(0,1): `mean_z < 0` forecasts run above outcomes; `sd_z < 1` distribution too wide (risk overstated).
* Tests: KS (uniform), Berkowitz LR (mean 0, sd 1, no AR(1)), Ljung-Box on z and z^2, tail shares against 5% / 1%.
* Horizons > 1 use non-overlapping origins; h = 20 leaves ~37 points, too few for the tests to say much.
* The predictive scale is a constant (window residual sd), so volatility clustering is rejected by construction.

## EVT validation (`run_evt.py`)

* GPD fitted to the standardised M0 and M1 residuals (same sample; each divided by its own sd) above several thresholds, both tails.
* KS p-values by parametric bootstrap (the textbook p-value is invalid with estimated parameters); `choose_threshold` is a
  convention: lowest threshold with >= 30 exceedances and KS p >= 0.10.
* M0 vs M1: difference in the GPD shape and in the 99% quantile with a paired block-bootstrap interval.
* Episode check: exceedance clusters, share in the densest 120 days, and the fit without that window. A large change in xi
  means the tail rests on one episode (the 2022-2023 tightening).
* Not done yet: declustering and GARCH-filtered (conditional) EVT.
