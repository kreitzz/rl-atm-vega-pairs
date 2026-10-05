# Trailing return skewness and kurtosis

Feature-only change relative to the recent-refit experiment: 63-session sample skewness and Fisher excess kurtosis for each leg, their pairwise differences, and two availability flags. A normal distribution has population excess kurtosis zero. These are historical return moments, not option-implied smile/skew or implied risk-neutral moments.

Use CRSP retx histories already present in the asset panels, reindexed to the trading calendar. All 63 observations must be available and variance nonzero. Missing values become neutral zero plus an explicit availability flag; pair differences are zero unless both legs are available. The source asset panels inherit the older standardized-IV availability calendar; missing sessions are not compressed together. Signal-close observations are available for next-session execution.

Two models: independent RL sizing and fixed-size NN. Both retain their existing features, costs, contract labels, monthly five-year refit window, beta gates and risk limits. The original historical validation enable/disable decisions remain fixed. Thus the gate does not independently validate the added features or refitted weights. Profitability-filtered variants use the previously selected rules, without retuning.

Attempted all 358 pairs. 2 pairs are unresolved in at least one compared policy; all tables use a matched complete roster and retain cash-only sleeves. Ex-post coverage selection can be biased. 2018–2021 and 2022–August 2025 are separated, but both periods have been inspected before. This is exploratory research, not an untouched test.

| Model | Variant | Period | Pairs | Return | Excess vs cash | Annualized excess Sharpe |
|---|---|---|---:|---:|---:|---:|
| rl_sized | previous_filtered | development_2018_2021 | 348 | 5.112% | 0.015% | 0.242 |
| rl_sized | previous_filtered | later_2022_2025 | 348 | 15.875% | -0.034% | -0.513 |
| rl_sized | previous_filtered | full | 348 | 21.799% | -0.018% | -0.131 |
| rl_sized | refit | development_2018_2021 | 348 | 5.093% | -0.005% | -0.048 |
| rl_sized | refit | later_2022_2025 | 348 | 15.799% | -0.110% | -0.973 |
| rl_sized | refit | full | 348 | 21.696% | -0.121% | -0.515 |
| rl_sized | refit_filtered | development_2018_2021 | 348 | 5.104% | 0.007% | 0.117 |
| rl_sized | refit_filtered | later_2022_2025 | 348 | 15.863% | -0.045% | -0.639 |
| rl_sized | refit_filtered | full | 348 | 21.777% | -0.039% | -0.278 |
| rl_sized | moments | development_2018_2021 | 348 | 5.126% | 0.028% | 0.295 |
| rl_sized | moments | later_2022_2025 | 348 | 15.778% | -0.130% | -1.129 |
| rl_sized | moments | full | 348 | 21.713% | -0.104% | -0.444 |
| rl_sized | moments_filtered | development_2018_2021 | 348 | 5.115% | 0.017% | 0.332 |
| rl_sized | moments_filtered | later_2022_2025 | 348 | 15.872% | -0.037% | -0.573 |
| rl_sized | moments_filtered | full | 348 | 21.799% | -0.018% | -0.142 |
| nn_fixed | previous_filtered | development_2018_2021 | 348 | 5.072% | -0.025% | -0.496 |
| nn_fixed | previous_filtered | later_2022_2025 | 348 | 15.852% | -0.057% | -0.711 |
| nn_fixed | previous_filtered | full | 348 | 21.728% | -0.089% | -0.607 |
| nn_fixed | refit | development_2018_2021 | 348 | 4.965% | -0.132% | -0.976 |
| nn_fixed | refit | later_2022_2025 | 348 | 15.742% | -0.166% | -1.477 |
| nn_fixed | refit | full | 348 | 21.489% | -0.327% | -1.186 |
| nn_fixed | refit_filtered | development_2018_2021 | 348 | 5.082% | -0.016% | -0.259 |
| nn_fixed | refit_filtered | later_2022_2025 | 348 | 15.877% | -0.032% | -0.436 |
| nn_fixed | refit_filtered | full | 348 | 21.765% | -0.051% | -0.350 |
| nn_fixed | moments | development_2018_2021 | 348 | 4.953% | -0.145% | -1.224 |
| nn_fixed | moments | later_2022_2025 | 348 | 15.773% | -0.136% | -1.119 |
| nn_fixed | moments | full | 348 | 21.507% | -0.310% | -1.174 |
| nn_fixed | moments_filtered | development_2018_2021 | 348 | 5.072% | -0.026% | -0.419 |
| nn_fixed | moments_filtered | later_2022_2025 | 348 | 15.884% | -0.024% | -0.294 |
| nn_fixed | moments_filtered | full | 348 | 21.762% | -0.055% | -0.348 |

Daily cash-excess Sharpe is annualized with √252. Returns include cash interest. 2025 is partial. Skewness and kurtosis are tested together, so this does not identify their individual contributions.

[Results](contract_tail_moments/comparison.csv) · [Annual results](contract_tail_moments/annual.csv) · [Activity](contract_tail_moments/activity.csv) · [Feature availability](contract_tail_moments/availability.csv) · [Unresolved](contract_tail_moments/unresolved.csv)

Audit: all original states preserved; strict completed-outcome timing, baseline enable gates, beta eligibility, action ladders and profitability filters verified. Both legs' moments are available on 99.93% of beta-eligible signal rows. 122 tests passed. Maximum ledger error 1.02e-09 dollars; 128 NN convergence warnings retained. Conservative two-worker peak-memory bound 1.27 GiB.

[Audit details](contract_tail_moments/audit.json)
