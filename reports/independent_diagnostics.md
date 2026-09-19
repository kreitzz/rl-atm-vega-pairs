# Why the additional pairs lose

The result is a mixture of concentrated bad years and weak net edge outside them. Low IV correlation is not supported as the main explanation: all three additional pairs have substantially higher daily ATM-IV-change correlation than AMD/NVDA. Their IV-change regression slopes are still below one, so high correlation does not establish an exact equal-vega hedge.

All diagnostics refer to the four independent sizing-ladder policies, the 2.5%-of-average-entry-IV pair cost, and common 2010–August 2025 entry dates. Correlations use daily changes, not IV levels; the rolling summary uses 252 observations. They describe the evaluation history and were not used as a lookahead selection screen.

| Pair | Daily IV-change correlation | Beta, first IV change on second | Median rolling correlation |
| --- | ---: | ---: | ---: |
| AMD/NVDA | 0.468 | 0.578 | 0.438 |
| XOM/CVX | 0.816 | 0.695 | 0.817 |
| JPM/BAC | 0.805 | 0.633 | 0.808 |
| HD/LOW | 0.742 | 0.668 | 0.640 |

## Is one year responsible?

| Pair | Winning / losing / flat years | Worst year | Full Sharpe | Sharpe excluding worst year |
| --- | --- | ---: | ---: | ---: |
| AMD/NVDA | 11 / 5 / 0 | 2021 | 0.635 | 0.691 |
| XOM/CVX | 7 / 9 / 0 | 2021 | -0.070 | 0.030 |
| JPM/BAC | 6 / 6 / 4 | 2010 | -0.012 | 0.071 |
| HD/LOW | 7 / 9 / 0 | 2018 | -0.105 | -0.032 |

XOM/CVX's 2021 loss of 450.55 proxy units exceeds its aggregate loss of 320.68, but it also loses in nine of sixteen years. JPM/BAC's 2010 loss of 279.64 similarly exceeds its aggregate loss of 43.71; four years are entirely flat. HD/LOW remains negative even without its worst year, 2018, and loses in nine years. Dropping a worst year is an ex-post sensitivity diagnostic, not a valid improved backtest or an actionable exclusion rule. P&L units are proxy dollars per $1 maximum pair-leg vega before the common 25% portfolio weight; 2025 is partial.

## Policy generalization and costs

Each selected tree and policy was refitted using only its historical training fold. All replayed test actions and validation Sharpes match the stored run. Median fold scores are descriptive summaries, not the Sharpe of concatenated returns.

| Pair | Median training Sharpe | Median validation Sharpe | Median test-fold Sharpe | Gross payoff per unit absolute size | Cost per unit absolute size |
| --- | ---: | ---: | ---: | ---: | ---: |
| AMD/NVDA | 1.255 | 0.643 | 0.643 | 5.556 | 1.238 |
| XOM/CVX | 0.913 | 0.272 | 0.017 | 0.445 | 0.641 |
| JPM/BAC | 0.876 | 0.063 | 0.008 | 0.732 | 0.788 |
| HD/LOW | 0.777 | 0.436 | 0.103 | 0.250 | 0.650 |

The three additional policies learn positive training performance that weakens substantially later. This is consistent with overfitting, changing relationships, or features that do not capture a persistent edge; it cannot distinguish these explanations by itself. Their gross selected-trade payoff fails to cover modeled costs. Strong IV co-movement can provide a hedge relationship without providing a forecastable realized-minus-implied relative-value opportunity.

Simple controls do not reveal an obvious missed edge: always long the first name, always short it, and shorting the higher-IV name all have negative after-cost Sharpe for each additional pair over these same dates. These controls trade all dates at full size, so they are not matched-date attribution controls. They do not prove that a better strategy is impossible.

A diagnostic that suppresses an entire test fold when its selected policy had nonpositive preceding validation Sharpe does not rescue these pairs: Sharpe is -0.071 for XOM/CVX, -0.089 for JPM/BAC, and -0.105 for HD/LOW. No filters or exclusions were adopted from these diagnostics.

Leg attribution shows that XOM losses exceed CVX gains; JPM losses slightly exceed BAC gains; and LOW losses exceed HD gains. Per-leg modeled cost is half of that leg's 2.5%-of-IV width, which reconciles to the pair cost.

## Annual net proxy P&L

| Entry year | AMD/NVDA | XOM/CVX | JPM/BAC | HD/LOW |
| --- | ---: | ---: | ---: | ---: |
| 2010 | -40.67 | -5.23 | -279.64 | 16.59 |
| 2011 | 154.39 | -107.67 | -44.02 | 3.00 |
| 2012 | -62.45 | 43.36 | 154.03 | -7.98 |
| 2013 | 554.74 | 49.84 | 188.54 | 117.23 |
| 2014 | 1279.73 | -4.28 | 0.00 | 49.70 |
| 2015 | 1210.50 | -73.27 | 0.00 | -17.41 |
| 2016 | 2217.30 | 57.78 | -13.87 | 106.48 |
| 2017 | 872.31 | -52.86 | 0.00 | 13.95 |
| 2018 | 939.80 | -16.62 | 7.68 | -252.40 |
| 2019 | 203.46 | 46.28 | 12.13 | -115.84 |
| 2020 | 243.85 | 448.26 | 8.35 | -8.12 |
| 2021 | -309.27 | -450.55 | -5.89 | -60.70 |
| 2022 | -242.07 | -148.71 | -17.35 | -164.82 |
| 2023 | 7.02 | -160.85 | -57.84 | -30.85 |
| 2024 | -82.21 | 3.16 | 0.00 | 55.97 |
| 2025 | 767.01 | 50.69 | 4.17 | -46.03 |

[Summary](independent_diagnostics/summary.csv) · [Annual P&L, gross/cost breakdown, and Sharpe](independent_diagnostics/annual.csv) · [Training/validation/test folds](independent_diagnostics/folds.csv) · [Leg attribution](independent_diagnostics/legs.csv) · [Simple controls](independent_diagnostics/simple_baselines.csv)

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.diagnose_independent_pairs
```

This uses the existing local proportional-cost independent run. No new market-data files are published. All Sharpes retain the annualized 30-day proxy-payoff convention. Results are exploratory on repeatedly inspected history; descriptive correlations and worst-year exclusions do not establish causality or generalizable pair-selection rules.
