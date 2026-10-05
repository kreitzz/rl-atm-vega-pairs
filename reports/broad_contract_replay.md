# All-pair actual-contract replay

Attempted all **358 pairs × 3 policies** from 2018-01-02 through 2025-08-29: 350 broad-screen pairs and eight additional legacy pairs. **39 pairs have at least one unresolved policy run.** Full-universe portfolio statistics remain unavailable; failed runs are not replaced with cash.

## Common-complete diagnostic only

The following basket excludes every pair with a failure in any of the three models, using the same pair roster across models. This is an ex-post data-complete subset and can have selection bias; it is not the requested full-universe result. Inactive pairs and pairs without a fitted historical policy remain in cash.

| Basket | Policy | Pairs | Total return | CAGR | Annualized excess Sharpe | Max drawdown |
|---|---|---:|---:|---:|---:|---:|
| broad | RL sizing, no memory | 311 | 19.88% | 2.40% | -1.337 | -0.25% |
| broad | RL sizing + memory | 311 | 19.89% | 2.40% | -1.310 | -0.26% |
| broad | NN fixed size + memory | 311 | 16.93% | 2.06% | -3.646 | -0.73% |
| legacy_extra | rl_reference (legacy features) | 8 | 20.01% | 2.41% | -0.375 | -0.49% |
| legacy_extra | rl_sized (legacy features) | 8 | 19.29% | 2.33% | -0.494 | -0.81% |
| legacy_extra | nn_fixed (legacy features) | 8 | 15.80% | 1.94% | -1.110 | -1.16% |

Returns include interest on uninvested cash. Sharpe uses daily returns in excess of the cash rate, annualized with √252. The 2025 annual table is year-to-date.

## Unresolved runs

| Failure | Pair-policy runs |
|---|---:|
| invalid held quote | 66 |
| missing held option | 16 |
| invalid underlying | 13 |
| held underlying corporate action | 5 |

The simulator stops on these cases rather than inventing held-position prices or liquidating retroactively. These counts identify data/lifecycle blockers, not trading losses. Three missing WPX/RCL quotes on March 13, 2020 were checked against the raw downloaded partitions and are absent there as well ([check](broad_contract_replay/raw_missing_quote_check.csv)).

## Method and limits

- Replays saved out-of-sample proxy-trained actions with their original historical screen membership. No new monthly actual-contract retraining or beta eligibility filter is applied in this experiment.
- Closest eligible ATM call to 30 days, selected from prior-session information. Initial dollar-vega matching, daily stock delta hedge, integer contracts and overlapping cohorts.
- Execution at midpoint plus **0.10 volatility point per option fill**, plus option commissions, stock execution, financing and stock borrow. The existing liquidity filters and NAV-scaled exposure, reserve and stress limits remain active.
- Fixed equal $1 million accounts per unique pair. No capital recycling from inactive pairs; no cross-pair hedge netting. The original proxy portfolio used changing fold rosters, so its allocation and payoff Sharpe are not directly comparable.
- Broad policies: RL sizing without memory, RL sizing with profitability memory, and fixed-size NN with memory. Legacy extras use their original relative-value-feature policies; their feature treatment differs, so they are reported separately.
- History begins in 2018 because these are the saved broad experiment’s out-of-sample signals, despite earlier contract data being available.
- Peak replay memory: 0.69 GiB. Each completed sleeve passed option/stock cash-ledger and ending-position reconciliation.

[Every pair/policy result](broad_contract_replay/status.csv) · [Unresolved cases](broad_contract_replay/unresolved.csv) · [Diagnostic membership](broad_contract_replay/diagnostic_membership.csv) · [Annual diagnostic results](broad_contract_replay/diagnostic_annual.csv) · [Breadth](broad_contract_replay/diagnostic_breadth.csv)

Reproduce using `python -m scripts.cache_broad_contracts`, `python -m scripts.run_broad_contract_replay`, then `python -m scripts.report_broad_contract_replay`. Shared data and detailed ledgers stay in ignored local directories.
