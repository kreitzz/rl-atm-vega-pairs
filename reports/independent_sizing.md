# Independent sizing ladder comparison

The ladder improves combined annualized payoff Sharpe from **0.405** for retrained full-size/flat policies to **0.499**. Holding the ladder policies' trade dates and directions fixed while forcing every active position to full size produces **0.438**, so the benefit is not entirely due to selecting different trades.

All comparisons use 3,922 common test dates in 2010–August 2025, the proportional entry-IV cost rule, and four fixed equal pair budgets. Costs scale with absolute position size in every variant. The full-size/flat models are independently retrained and validation-selected with the same five features, six tree settings, five search seeds, and purged expanding-window folds. The last column is a diagnostic rescore without retraining; it does not represent a separately optimized full-size model.

| Pair | Full-size/flat, retrained | Ladder | Ladder decisions, forced full size |
| --- | ---: | ---: | ---: |
| AMD/NVDA | 0.575 | 0.635 | 0.569 |
| XOM/CVX | -0.063 | -0.070 | -0.066 |
| JPM/BAC | -0.145 | -0.012 | -0.060 |
| HD/LOW | -0.146 | -0.105 | -0.042 |
| combined | 0.405 | 0.499 | 0.438 |

## Size usage

Percentages of all test entry dates; size is relative to each pair's fixed budget, not the total portfolio budget.

| Pair | Flat | 25% | 50% | 75% | 100% | Mean size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AMD/NVDA | 22.8% | 17.1% | 27.7% | 19.9% | 12.5% | 45.5% |
| XOM/CVX | 40.4% | 14.6% | 6.2% | 15.6% | 23.3% | 41.7% |
| JPM/BAC | 58.6% | 12.4% | 18.7% | 10.3% | 0.0% | 20.2% |
| HD/LOW | 58.3% | 19.6% | 8.8% | 3.4% | 9.9% | 21.7% |

The ladder improves aggregate Sharpe, but it does not turn the three additional standalone pairs profitable. Size is a learned allocation, not a calibrated confidence probability. These remain annualized 30-day proxy-payoff Sharpes, not daily marked option-and-hedge returns; model choices have used repeatedly inspected history.

[Comparison CSV](independent_sizing/comparison.csv) · [Size usage](independent_sizing/size_usage.csv) · [Annual portfolio comparison](independent_sizing/annual.csv)

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.run_four_pairs --source-root /path/to/wrds --cost-rate 0.025 --sizing fixed
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.run_four_pairs --source-root /path/to/wrds --cost-rate 0.025 --sizing ladder
```

The default remains the independent ladder. No additional dataset files are published. Thirty-two tests pass; the fixed-policy action set and matched evaluation dates were checked directly.
