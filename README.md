# sora

SSEF 2027: is the classical Vasicek short-rate model misspecified for SORA? (Blueprint v8c)

## M0 (null model: classical Vasicek)

```
pip install -r requirements.txt
python run_m0.py            # Block 1-3 + M0 side of Block 7; writes results/m0/
python -m pytest            # unit tests
```

| File | Role |
|---|---|
| `ssef/data.py` | Block 1: load `data/Domestic_Interest_Rates_clean.csv` (SORA, decimals) |
| `ssef/gate.py` | Block 2: ADF / KPSS on level and first difference |
| `ssef/m0.py` | Block 3: AR(1) OLS, structural parameters, delta-method CIs, std. residuals, exact simulator, rolling forecast |
| `run_m0.py` | Reproducible driver; saves tables, residuals, forecasts and `run_info.json` (data hash, versions, seed) |

Convention: one observation = one step of `dt = 1/252` year (business-day steps; weekends not stretched).
