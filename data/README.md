# Data dictionary

Three daily series, all in `data/`, all CSV with `dd/mm/yyyy` dates. Refresh with `python update_all.py`
(keys in `.env`, see `.env.example`) and verify with `python check_data.py`.

| File | Columns | Unit | Source | Calendar |
|---|---|---|---|---|
| `Domestic_Interest_Rates_clean.csv` | `SORA Value Date`, `SORA Publication Date`, `SORA` | % per annum | MAS API (domestic interest rates, daily) | Singapore business days |
| `SOFR_clean.csv` | `SOFR Effective Date`, `SOFR Publication Date`, `SOFR` | % per annum | NY Fed Markets Data API (public) | US business days |
| `USDSGD_clean.csv` | `Date`, `USD/SGD` | SGD per 1 USD, end of day | MAS API (exchange rates, end of period, daily) | Singapore business days |

## Conventions that matter for the analysis

* **Publication date.** SORA's comes from the API. SOFR's is not in the API: the updater sets it to the *next row's
  effective date* (the NY Fed publishes SOFR on the next US business day at about 8am New York time). The latest SOFR
  row has a blank publication date until the next rate arrives; this is the only row allowed to be blank.
* **No look-ahead.** `ssef.loaders.known_sofr` uses a SOFR value only on SORA dates *after* its publication date
  (`strict`, default) or on/after it (`same-day`). SOFR appears at about 8pm Singapore time, after the SORA day.
* **Calendars differ.** SORA and USD/SGD share the Singapore calendar; SOFR follows US holidays, so a few dozen SORA
  dates have no SOFR of their own. The as-of merge carries the last published value forward (past values only).
* **FX sign.** USD/SGD is SGD per USD, so a positive daily return `z` means the SGD weakened.
* **Study window.** All three start on 1 Oct 2021 (`START_DATE` in each updater). The series end on different days
  because they are refreshed at different times; `check_data.py` warns if they drift more than 5 days apart.

## Known caveats

* **Which SORA is it?** The proposal says "daily compounded SORA", but daily moves of up to 148 bp suggest the
  *overnight* rate. Run `python updaters/update_sora.py --inspect` to list the fields the API offers. A different
  field can be downloaded into its *own* file with `--field <name> --csv data/<file>.csv`; it never overwrites this one.
* The linear models used downstream allow negative rates; the data themselves are all positive.
* Values can be revised by the source: `update_*.py` print how many stored values an update changed.

## What `check_data.py` verifies

Header and date format, strictly increasing unique dates, numeric values inside a plausible range
(SORA/SOFR -1 to 20 %, USD/SGD 0.5 to 3), no weekend rows, calendar gaps over 7 days, publication dates never before the
value date, SOFR publication = next effective date, freshness (default 7 days), and consistency between files
(every SORA date has an FX rate, same start date, end dates within 5 days).
