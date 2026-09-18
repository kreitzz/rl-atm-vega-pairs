# Shorter trailing RV and 12–1 momentum

Tested 5- and 10-trading-day trailing realized volatility as replacements for the existing 30-calendar-day mark, plus 12–1 momentum for AMD and NVDA. Neither addition improved the matched walk-forward result in this model.

## Results

All learned variants include IV spread and both earnings countdowns. Each cell below is an annualized net payoff Sharpe after the same 0.5-vol-point pair cost, across 3,922 walk-forward entries in 2010–August 2025.

| RV horizon | No momentum, no VIX | With momentum, no VIX | No momentum, with VIX | With momentum and VIX |
| --- | ---: | ---: | ---: | ---: |
| 30 calendar days | 0.692 | -0.161 | 0.050 | -0.078 |
| 5 trading days | 0.426 | -0.088 | 0.164 | -0.148 |
| 10 trading days | 0.373 | -0.094 | 0.030 | -0.023 |

The short-higher-IV rule scores 0.444. Selecting the feature variant using each fold's preceding validation data scores 0.166. The best retrospective test result should not be confused with what that validation selection procedure chose.

The 188-entry nonoverlapping check gives 0.788, 0.432, and 0.345 for 30-day, 5-day, and 10-day RV without momentum or VIX. Their momentum counterparts score -0.177, -0.106, and -0.141. This diagnostic uses one fixed starting offset.

The matched 30-day reference is 0.692, versus 0.527 in the preceding run, because this experiment removes the earlier training rows without a full year of momentum history for **every** variant. Test dates are unchanged. This change illustrates sensitivity to the training sample and is not an improvement from adding momentum.

For context, block-bootstrap 95% intervals for the no-momentum/no-VIX versions are [0.331, 1.033], [0.043, 0.816], and [-0.048, 0.738] for 30-, 5-, and 10-day RV. These are exploratory comparisons over repeatedly inspected history, not a fresh confirmatory holdout or multiple-testing-adjusted result.

## Feature definitions

- Short RV: `100 * sqrt(252 * mean(log(1 + retx)^2))` over the last 5 or 10 consecutive trading-day returns, ending at entry. All returns in the window must be present. These use the same split-adjusted ex-distribution CRSP returns as the 30-day RV mark.
- 12–1 momentum: cumulative **total return** from 252 trading days before entry through 21 trading days before entry, using CRSP `ret`. At entry index `t`, compound returns `t-251` through `t-21`, inclusive: 231 daily returns. Exclude the latest 21 daily returns. Express momentum in percentage points and calculate separately for each stock.
- The existing 30-day RV is a calendar-day window; the short marks use trading-day windows. This distinction is intentional and recorded in the feature configuration.
- Earnings, IV spread, prior-close VIX, reward calculation, and costs retain their earlier definitions. RV and momentum never use forward returns.

The common complete panel begins December 31, 2002. Momentum removes 230 earlier entries from the previous common panel; all twelve variants use the same remaining rows. Each two-year fold retains the earlier expanding training / two-year validation / two-year test procedure with unsettled-outcome purging. The same tree grid, minimum leaf sizes, five policy-search seeds, and Sharpe objective are used. There is no new tuning after inspecting these results.

## Artifacts and reproduction

- [All strategy scores](short_rv_momentum_scores.csv)
- [Annual P&L and Sharpe for all variants](short_rv_momentum_annual.csv)
- Feature registry: `config/short_rv_momentum_features.json`

Annual P&L is the sum of entry-cohort 30-day net payoffs at $1 vega per leg, not a percentage return on capital. Annualized payoff Sharpe uses `sqrt(365/30)` and includes flat entry dates.

```sh
python scripts/build_short_rv_momentum.py --source-root /path/to/convert-rv-rl/data/raw/wrds
python scripts/run_walk_forward.py --panel artifacts/short_rv_momentum/dataset.parquet --output artifacts/short_rv_momentum/walk_forward --features-json config/short_rv_momentum_features.json --reference-model rv30cal --comparison-model rv5td_momentum
python scripts/report_annual.py --predictions artifacts/short_rv_momentum/walk_forward/predictions.csv --output reports/short_rv_momentum_annual.csv --models rv30cal rv5td rv10td rv30cal_momentum rv5td_momentum rv10td_momentum
python -m unittest discover -s tests -v
```

Build the original RV/VIX panel first if needed. Detailed candidates, selected configurations, feature metadata, and predictions remain local under `artifacts/short_rv_momentum/`. Fifteen tests pass, including exact momentum-window boundaries, omission of the latest month, and short-RV missing-data behavior. Annual P&L totals reconcile to the aggregate results for every strategy.
