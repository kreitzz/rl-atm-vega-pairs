# VIX and momentum: expiry-policy ablation

Comparison on 346 common complete pairs, January 2018–August 2025. Each pair retains its $1m capital sleeve, including inactive sleeves. All variants use beta 0.85–1.15 and return correlation ≥0.5. Monthly training, validation, recent refits, profitability gates, costs and actual expiry accounting are unchanged. Both RL sizing and fixed-size NN are retrained for each feature set.

VIX is the strictly previous available close, at most seven calendar days stale. Momentum is each leg’s compounded CRSP total return from 252 trading days ago to 21 days ago (231 returns), in percentage points, respecting security-link intervals. Missing momentum is zero with a separate availability flag; no trading dates are dropped. No extra momentum-difference feature was added.

| Features added | RL annualized excess Sharpe | NN annualized excess Sharpe |
|---|---:|---:|
| baseline | 0.106 | -0.125 |
| vix | 0.226 | -0.081 |
| momentum | -0.007 | -0.232 |
| both | 0.117 | -0.451 |

## Annualized excess Sharpe by year

| Year | RL baseline | RL VIX | RL momentum | RL both | NN baseline | NN VIX | NN momentum | NN both |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2018 | -1.110 | -0.998 | -0.355 | -0.302 | -0.447 | -1.435 | -0.380 | -1.640 |
| 2019 | -0.005 | 0.044 | 0.007 | -0.015 | -0.916 | -0.206 | -0.523 | -0.248 |
| 2020 | 0.065 | 0.065 | 1.135 | 0.065 | 0.479 | 0.095 | -0.524 | 0.078 |
| 2021 | 0.231 | 0.503 | 0.377 | 0.556 | -0.030 | -0.209 | -0.145 | -1.374 |
| 2022 | 0.631 | 0.579 | 0.611 | 0.818 | 0.040 | 0.330 | -0.154 | -0.236 |
| 2023 | 0.261 | 0.631 | -1.057 | -0.878 | -0.523 | -0.302 | 0.913 | -0.610 |
| 2024 | -0.096 | 0.119 | -0.079 | 0.087 | 0.191 | 0.340 | -0.412 | -0.376 |
| 2025 | 0.130 | -0.326 | 0.203 | 0.048 | -0.082 | -1.569 | -0.649 | -1.091 |

## Activity and gross edge

| Variant | Model | Entries | Active pairs | Gross P&L | Execution costs |
|---|---|---:|---:|---:|---:|
| baseline | rl_sized | 296 | 35 | $123,934 | $71,087 |
| baseline | nn_fixed | 379 | 62 | $105,253 | $174,912 |
| vix | rl_sized | 280 | 35 | $172,623 | $73,225 |
| vix | nn_fixed | 389 | 63 | $135,551 | $177,976 |
| momentum | rl_sized | 329 | 35 | $75,103 | $72,178 |
| momentum | nn_fixed | 395 | 63 | $35,312 | $178,204 |
| both | rl_sized | 299 | 35 | $136,327 | $73,915 |
| both | nn_fixed | 366 | 64 | $-173,267 | $166,377 |

Daily cash-excess Sharpe is annualized using √252. Yearly percentage returns in the CSV include interest on unused capital; excess P&L removes the cash benchmark. 2025 ends in August. Gross P&L excludes execution, stock borrow and financing.

All periods have been inspected before; these are exploratory feature comparisons, not an untouched holdout. Overlapping trades and shared names are dependent. The validation gate still validates the earlier fitted proposal, not the exact recent-refit weights. Incomplete backtests are disclosed in status and excluded consistently from all variants.

Audit: 358 pairs checked; original states, causal feature alignment, entry eligibility, action ladders and completed-outcome training cutoffs verified. Both momentum inputs available on 99.55% of state rows. NN warnings: 3582/8026 validation fits and 1356/2947 recent fits. Daily ledger reconciliation runs inside every completed simulation.

[Period comparison](expiry_vix_momentum/comparison.csv) · [Annual percentage returns and Sharpes](expiry_vix_momentum/annual.csv) · [Activity](expiry_vix_momentum/activity.csv) · [Pair attribution](expiry_vix_momentum/pair_attribution.csv) · [Coverage and errors](expiry_vix_momentum/status.csv) · [Audit](expiry_vix_momentum/audit.json)
