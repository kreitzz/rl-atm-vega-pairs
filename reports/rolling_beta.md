# Rolling AMD-on-NVDA return beta

Tested rolling return beta as an additional state variable. Neither 63-day nor 126-day beta improves the main walk-forward payoff-Sharpe result; the current best model remains the no-beta 30-day-RV model with the vega-sizing ladder.

## Definition

`beta = Cov(AMD daily return, NVDA daily return) / Var(NVDA daily return)`

This is the OLS slope with an intercept, treating AMD as the dependent return and NVDA as the explanatory return. It uses CRSP split-adjusted ex-distribution **simple** close-to-close returns over the most recent 63 or 126 consecutive trading-day observations through entry close. Require a complete window and a denominator above 1e-12; otherwise the feature is missing.

Beta enters only as a state input. Initial dollar vega remains balanced between the option legs, and daily delta hedging retains the existing simulation assumption. No beta-derived change to the hedge ratio or reward is introduced.

## Matched results

All models retain current IV spread, both earnings countdowns, and 30-calendar-day trailing RV for both stocks. No VIX, momentum, or historical-spread variables are added in this focused test.

| Additional beta input | Fixed-size annualized Sharpe | Ladder annualized Sharpe |
| --- | ---: | ---: |
| None | 0.692 | **0.735** |
| 63 trading days | 0.569 | 0.585 |
| 126 trading days | 0.581 | 0.533 |
| Both windows | 0.546 | 0.564 |
| Variant chosen using preceding validation | 0.633 | 0.539 |

Same 3,922 test entries in 2010–August 2025, same training and validation dates, same hyperparameter grid, same five seeds, same 0.5-vol-point full-size pair cost. No observations are removed by beta availability. The controls reproduce the previous 0.692/0.735 result exactly.

The paired block-bootstrap 95% interval for 63-day-beta ladder Sharpe minus no-beta ladder Sharpe is **[-0.351, 0.027]**. This is an exploratory comparison, not evidence that return beta is inherently unhelpful. Historical test data has been inspected repeatedly.

On the 188-entry nonoverlapping diagnostic, ladder Sharpes are 0.852 without beta, 0.667 with 63-day beta, 0.702 with 126-day beta, and 0.710 with both. The fixed-size 126-day model does improve that particular diagnostic (0.907 versus 0.788); it does not improve the primary full-entry metric. No model was selected using this diagnostic.

All Sharpes remain annualized 30-day net payoff ratios (`sqrt(365/30)`), not daily marked portfolio-return Sharpes. The ladder allows 0/25/50/75/100% of a 0.001 per-leg, per-entry maximum vega per reference-capital dollar, with costs proportional to size.

## Reproduce and inspect

```sh
python scripts/build_rolling_beta.py --source-root /path/to/convert-rv-rl/data/raw/wrds
python scripts/run_walk_forward.py --panel artifacts/rolling_beta/dataset.parquet --output artifacts/rolling_beta/fixed --features-json config/rolling_beta_features.json --reference-model rv30cal --comparison-model beta63 --cost 0.5
python scripts/run_walk_forward.py --panel artifacts/rolling_beta/dataset.parquet --output artifacts/rolling_beta/ladder --features-json config/rolling_beta_features.json --reference-model rv30cal --comparison-model beta63 --sizing-ladder 0 0.25 0.5 0.75 1 --max-vega-per-dollar 0.001 --cost 0.5
python scripts/report_sizing.py --fixed artifacts/rolling_beta/fixed --ladder artifacts/rolling_beta/ladder --output reports/rolling_beta
python -m unittest discover -s tests -v
```

- [Sizing comparison and intervals](rolling_beta/sizing_comparison.csv)
- [Allocation diagnostics](rolling_beta/sizing_allocations.csv)
- [Annual ladder P&L and Sharpe](rolling_beta/sizing_annual.csv)

Local feature data, candidate settings, and predictions are in `artifacts/rolling_beta/`. Twenty-three tests pass, including recovery of a known regression slope with an intercept, future-data exclusion, missing-window handling, and zero-denominator handling. Annual P&L totals reconcile for every model.
