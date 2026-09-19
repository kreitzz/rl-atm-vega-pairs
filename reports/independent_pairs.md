# Four independent pair policies

The current four-pair setup trains AMD/NVDA, XOM/CVX, JPM/BAC, and HD/LOW separately. Each policy sees only its own IV spread, two earnings countdowns, and two trailing 30-day RVs. It selects direction and a 0/25/50/75/100% size within its own fixed budget. There is no shared state or cross-pair allocation decision.

Each pair receives 25% of the total entry-vega budget. Unused budget stays idle; one pair's decision cannot resize another pair. All four may trade simultaneously. Each trade is initially vega-neutral within its pair. The cap applies to each entry cohort, not all overlapping open positions.

Cost per full-size pair is `0.025 * (IV_A + IV_B) / 2`, scaled by its absolute allocation. The existing expanding-window training, validation, purging, and sizing search remain unchanged.

| Policy | Annualized payoff Sharpe, 2010–August 2025 |
| --- | ---: |
| AMD/NVDA | 0.635 |
| XOM/CVX | -0.070 |
| JPM/BAC | -0.012 |
| HD/LOW | -0.105 |
| Combined, fixed equal budgets | **0.499** |

[Sizing comparison](independent_sizing.md): the ladder improves combined Sharpe from 0.405 for retrained full-size/flat policies to 0.499.

This is the `equal_budget` strategy already evaluated in the [proportional-cost experiment](four_pairs_proportional_cost.md), not the shared-cap or joint strategy. [Yearly results](four_pairs_proportional_cost/annual.csv) include this strategy and each individual pair. Sharpes use the existing annualized 30-day proxy-payoff convention, not daily marked option-and-hedge P&L.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.run_four_pairs --source-root /path/to/wrds --cost-rate 0.025
```

The runner now defaults to independent policies and fixed equal budgets. Historical joint/shared-cap comparisons require `--include-joint`. Local panels and predictions remain ignored. The shared CSV in `data/` still contains the original AMD/NVDA dataset; the runner builds the additional pairs from the local WRDS source files.

[Why the additional pairs lose: annual, correlation, and policy diagnostics](independent_diagnostics.md).

[Stock-return versus IV dependence diagnostics](correlation_gap.md).
