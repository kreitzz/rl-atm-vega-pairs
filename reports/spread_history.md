# Historical IV spread state

Tested the spread's own history as additional context. The 63-observation z-score comes closest to the existing ladder-sized model, but none of these additions improves its point-estimate test Sharpe.

## Results

All variants retain current AMD–NVDA IV spread, known earnings countdowns, and 30-calendar-day trailing RV for each stock. No VIX or momentum is included in this focused experiment. Dates match the previous sizing run exactly: 3,922 walk-forward test entries across 2010–August 2025.

| Additional spread history | Fixed-size annualized Sharpe | Ladder annualized Sharpe |
| --- | ---: | ---: |
| None | 0.692 | **0.735** |
| Trailing mean | 0.522 | 0.578 |
| Z-score | 0.633 | 0.729 |
| Trailing mean + z-score | 0.534 | 0.508 |
| 21-observation change | 0.586 | 0.706 |
| Mean + z-score + change | 0.367 | 0.306 |
| Feature set selected on preceding validation | 0.628 | 0.452 |

The simple short-higher-IV baseline is 0.444. Sharpes use the same annualized 30-day payoff convention and proportionally scaled 0.5-vol-point full-size pair cost. The ladder is 0/25/50/75/100% of the 0.001 per-leg, per-entry maximum vega per reference-capital dollar.

The ladder z-score model is only 0.006 Sharpe below the reference. Its paired 95% block-bootstrap difference interval is **[-0.083, 0.071]**: this run does not distinguish the two reliably. On the 188-entry nonoverlapping diagnostic, the reference is 0.852, z-score is 0.824, and spread change is 0.822. These remain repeated exploratory experiments rather than fresh holdout tests.

## Features

For spread `s[t] = IV_AMD[t] - IV_NVDA[t]` in volatility points:

- **Historical mean:** the mean of the 63 available daily spread observations strictly before entry.
- **Historical z-score:** `(s[t] - historical_mean) / max(historical_sample_SD, 0.1)`. The SD uses the same strictly prior 63 observations; the 0.1-vol-point floor handles a flat historical series.
- **Recent change:** `s[t] - s[t-21]` over available daily observations.

These windows count available observations, not a fully populated exchange calendar; missing quotes are not interpolated. The raw current spread remains an input in every model. Neither the current spread nor any later observation enters the historical mean or SD. No additional rows are dropped because the common panel already starts after the feature warm-up.

Each feature set receives the same tree grid, seeds, expanding-window schedule, outcome-boundary purging, and prior-validation selection. Both fixed and ladder sizing are rerun on the identical panel; the original 0.692/0.735 controls reproduce. The ladder search retains the fixed policy as a training incumbent. No test-period result was used to change the feature definitions or retune the models.

## Reproduce and inspect

```sh
python scripts/build_spread_history.py
python scripts/run_walk_forward.py --panel artifacts/spread_history/dataset.parquet --output artifacts/spread_history/fixed --features-json config/spread_history_features.json --reference-model rv30cal --comparison-model spread_zscore --cost 0.5
python scripts/run_walk_forward.py --panel artifacts/spread_history/dataset.parquet --output artifacts/spread_history/ladder --features-json config/spread_history_features.json --reference-model rv30cal --comparison-model spread_zscore --sizing-ladder 0 0.25 0.5 0.75 1 --max-vega-per-dollar 0.001 --cost 0.5
python scripts/report_sizing.py --fixed artifacts/spread_history/fixed --ladder artifacts/spread_history/ladder --output reports/spread_history
python -m unittest discover -s tests -v
```

- [Sizing comparison and intervals](spread_history/sizing_comparison.csv)
- [Ladder allocations](spread_history/sizing_allocations.csv)
- [Annual ladder P&L, reference-capital return, and Sharpe](spread_history/sizing_annual.csv)

The builder uses the previously generated RV/VIX and short-RV/momentum panels. Local feature data and predictions remain under `artifacts/spread_history/`. Twenty tests pass, including explicit checks that the historical mean and SD exclude the current and future observations, the 21-observation change is aligned correctly, and a flat historical spread is handled deterministically. Annual P&L totals reconcile for all strategies.
