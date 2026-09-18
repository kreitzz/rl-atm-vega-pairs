# Current reference model

Reference cost: **1.0 vol point per full-size pair round trip**, chosen by the user. The cost scales with absolute size; flat costs zero. This is a modeling assumption, not a calibrated bid/ask estimate.

State: IV spread, earnings countdowns for AMD and NVDA, and 30-calendar-day trailing RV for both stocks. Signed size choices: -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1. Maximum initial vega per reference-capital dollar is 0.001 per leg per new entry.

The existing separately retrained 1.0-cost run is the reference: annualized 2010–August 2025 walk-forward payoff Sharpe **0.667**, with 77.9% of entry dates traded and average size 43.8% of maximum. No additional parameter search was performed to adopt it.

Sharpes use `sqrt(365/30)` on 30-day net payoffs including flat dates. Yearly outcomes are attributed to entry year, not settlement year. These are not daily marked portfolio-return Sharpes. 2025 entries end August 29.

| Year | Annualized payoff Sharpe |
| --- | ---: |
| 2010 | 0.724 |
| 2011 | -0.331 |
| 2012 | -0.092 |
| 2013 | 1.783 |
| 2014 | 2.538 |
| 2015 | 1.576 |
| 2016 | 1.187 |
| 2017 | 1.177 |
| 2018 | 1.243 |
| 2019 | 0.186 |
| 2020 | 0.453 |
| 2021 | -0.733 |
| 2022 | -0.635 |
| 2023 | 0.080 |
| 2024 | -0.139 |
| 2025 | 1.922 |

[Annual P&L and Sharpe CSV](reference_model_annual.csv). Earlier reports preserve their original 0.5-cost assumptions and now explicitly pin that cost in their reproduction commands.

```sh
python scripts/run_walk_forward.py --panel artifacts/short_rv_momentum/dataset.parquet --output artifacts/reference_model --features-json config/reference_features.json --reference-model rv30cal --comparison-model rv30cal --sizing-ladder 0 0.25 0.5 0.75 1 --max-vega-per-dollar 0.001 --cost 1.0
```

The run already exists locally at `artifacts/execution_costs/cost_1/`.
