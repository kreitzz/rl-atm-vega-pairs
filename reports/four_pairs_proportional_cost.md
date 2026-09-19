# Four pairs with proportional IV costs

The requested cost rule did not produce a diversification improvement in the retrained experiment. AMD/NVDA alone scores 0.635 versus 0.565 for four independently trained policies under a shared cap and -0.106 for the joint pair-and-size policy.

## Cost definition

For each entry date and pair, full-size pair cost is `0.025 * (IV_A + IV_B) / 2`, with IV expressed in annualized percentage points. Multiply by the absolute pair allocation. Two 40-vol names therefore cost 1 point as a pair; two 20-vol names cost 0.5 points; a 40/20 pair costs 0.75 points. This interpretation preserves the previous 1-point **pair** cost at 40 vol; it does not charge a separate full width on each leg at both entry and exit. If each leg's quoted full width is 2.5% of IV, this is equivalent to paying half that width per leg once. It is a modeling assumption, not a calibrated quote estimate.

Costs are computed from entry IV only and included in training, validation selection, and test payoffs. The rule raises average AMD/NVDA cost while reducing costs for the other pairs. Flat allocations cost zero.

| Pair | Mean entry IV | Mean full-size pair cost |
| --- | ---: | ---: |
| AMD/NVDA | 48.17% | 1.204 |
| XOM/CVX | 22.56% | 0.564 |
| JPM/BAC | 27.10% | 0.678 |
| HD/LOW | 24.07% | 0.602 |

## Matched comparison

All entries are the same 3,922 dates from 2010 through August 2025. All strategies are retrained with unchanged features, search settings, expanding-window folds, boundary purges, and shared entry budgets. The third column isolates the cost change by keeping the previous fixed-cost policy's decisions unchanged; the final column includes retraining and validation reselection.

| Model | Fixed 1-point cost | Old policy, new costs | Retrained, new costs |
| --- | ---: | ---: | ---: |
| AMD/NVDA only | 0.667 | 0.633 | 0.635 |
| XOM/CVX only | -0.113 | -0.032 | -0.070 |
| JPM/BAC only | -0.124 | -0.069 | -0.012 |
| HD/LOW only | -0.120 | -0.043 | -0.105 |
| Four policies, equal budgets | 0.494 | 0.518 | 0.499 |
| Four policies, shared cap | 0.561 | 0.580 | 0.565 |
| Joint pair-and-size policy | -0.059 | -0.064 | -0.106 |

The three additional standalone policies still have negative aggregate net payoffs after retraining. Lower costs help their fixed-policy scores but do not establish sufficient net edge. The joint model's deterioration after retraining shows that changing its learned allocations can outweigh the direct benefit of lower costs. More flexibility does not guarantee better test performance, and the joint tree does not contain the original AMD/NVDA policy as a guaranteed fallback.

## Annual retrained results

| Entry year | AMD/NVDA | Equal budgets | Shared cap | Joint |
| --- | ---: | ---: | ---: | ---: |
| 2010 | -0.051 | -0.388 | -0.525 | 1.393 |
| 2011 | 0.195 | 0.006 | 0.066 | -2.660 |
| 2012 | -0.107 | 0.180 | 0.136 | 0.203 |
| 2013 | 1.775 | 2.597 | 2.615 | 0.704 |
| 2014 | 2.555 | 2.736 | 2.634 | -2.073 |
| 2015 | 1.552 | 1.356 | 1.506 | -0.610 |
| 2016 | 1.163 | 1.250 | 1.296 | 0.484 |
| 2017 | 1.139 | 1.070 | 1.126 | 0.021 |
| 2018 | 1.191 | 0.772 | 0.822 | 0.408 |
| 2019 | 0.363 | 0.194 | 0.030 | 0.207 |
| 2020 | 0.392 | 0.491 | 0.483 | 0.266 |
| 2021 | -0.752 | -1.703 | -1.813 | 0.191 |
| 2022 | -0.799 | -1.198 | -1.212 | -1.154 |
| 2023 | 0.026 | -0.654 | -0.775 | 0.125 |
| 2024 | -0.196 | -0.050 | -0.151 | 0.294 |
| 2025 | 1.884 | 1.957 | 1.811 | 0.314 |

These are annualized 30-day proxy-payoff Sharpes using `sqrt(365/30)`, not daily marked option-and-hedge portfolio Sharpes. The same handpicked pairs and repeatedly inspected history limit the strength of the inference. The earlier fixed-cost experiment and reference remain unchanged.

[Comparison CSV](four_pairs_proportional_cost/comparison.csv) · [Costs and attribution](four_pairs_proportional_cost/costs_and_attribution.csv) · [Annual results](four_pairs_proportional_cost/annual.csv) · [Selected configurations](four_pairs_proportional_cost/selected_configs.csv)

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.run_four_pairs --source-root /path/to/wrds --cost-rate 0.025
python -m unittest discover -s tests -q
```

Detailed panels and predictions remain local under `artifacts/four_pairs_cost_0.025/`. Thirty-two tests pass, including scalar and variable-cost moment calculations against direct payoffs, the 40/20-vol examples, cross-pair covariance, and shared-budget constraints. Saved test rewards were independently reconciled to the entry-IV costs and actions.
