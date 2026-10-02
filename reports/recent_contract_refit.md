# Recent-outcome refit experiment

All 358 pairs attempted. Same contract marks, costs, sizing ladders, beta screens, historical pair membership and risk limits as the aligned learning experiment. NN remains full-size/flat.

The original model uses five years ending before a separate one-year validation window. This experiment retains exactly the original monthly enable/disable decision, then refits enabled models on five years of outcomes completed strictly before the deployment month. Thus the historical gate validates the model specification/proposal, not the newly refitted weights. No new full-book validation of those weights is claimed.

At least 504 usable completed outcomes are required in the new window. If shifting the window forward leaves fewer, that model-month stays flat and the insufficient-history status is logged.

Two predefined changes: recent refit alone, and recent refit plus the previous development-selected profitability gate (RL: positive 91/365-day directional net profit; NN: positive 365-day profit). No new gate tuning. Earlier 2018–2021 and later 2022–August 2025 results are separate; all dates have already been inspected in prior research, so this is not an untouched holdout.

Comparisons use the same complete roster across all compared policies, retaining cash-only sleeves. 2 pairs have an unresolved evaluation in at least one compared policy. Ex-post complete-roster selection can be biased.

| Model | Variant | Period | Pairs | Return | Excess vs cash | Annualized excess Sharpe |
|---|---|---|---:|---:|---:|---:|
| rl_sized | baseline | development_2018_2021 | 348 | 5.073% | -0.025% | -0.218 |
| rl_sized | baseline | later_2022_2025 | 348 | 15.784% | -0.124% | -1.124 |
| rl_sized | baseline | full | 348 | 21.658% | -0.159% | -0.642 |
| rl_sized | previous_memory | development_2018_2021 | 348 | 5.112% | 0.015% | 0.242 |
| rl_sized | previous_memory | later_2022_2025 | 348 | 15.875% | -0.034% | -0.513 |
| rl_sized | previous_memory | full | 348 | 21.799% | -0.018% | -0.131 |
| rl_sized | recent | development_2018_2021 | 348 | 5.093% | -0.005% | -0.048 |
| rl_sized | recent | later_2022_2025 | 348 | 15.799% | -0.110% | -0.973 |
| rl_sized | recent | full | 348 | 21.696% | -0.121% | -0.515 |
| rl_sized | recent_memory | development_2018_2021 | 348 | 5.104% | 0.007% | 0.117 |
| rl_sized | recent_memory | later_2022_2025 | 348 | 15.863% | -0.045% | -0.639 |
| rl_sized | recent_memory | full | 348 | 21.777% | -0.039% | -0.278 |
| nn_fixed | baseline | development_2018_2021 | 348 | 4.996% | -0.101% | -0.968 |
| nn_fixed | baseline | later_2022_2025 | 348 | 15.742% | -0.166% | -1.474 |
| nn_fixed | baseline | full | 348 | 21.525% | -0.292% | -1.217 |
| nn_fixed | previous_memory | development_2018_2021 | 348 | 5.072% | -0.025% | -0.496 |
| nn_fixed | previous_memory | later_2022_2025 | 348 | 15.852% | -0.057% | -0.711 |
| nn_fixed | previous_memory | full | 348 | 21.728% | -0.089% | -0.607 |
| nn_fixed | recent | development_2018_2021 | 348 | 4.965% | -0.132% | -0.976 |
| nn_fixed | recent | later_2022_2025 | 348 | 15.742% | -0.166% | -1.477 |
| nn_fixed | recent | full | 348 | 21.489% | -0.327% | -1.186 |
| nn_fixed | recent_memory | development_2018_2021 | 348 | 5.082% | -0.016% | -0.259 |
| nn_fixed | recent_memory | later_2022_2025 | 348 | 15.877% | -0.032% | -0.436 |
| nn_fixed | recent_memory | full | 348 | 21.765% | -0.051% | -0.350 |

Sharpe uses daily excess returns and √252. Total return includes cash interest. 2025 is partial.

[Comparison CSV](recent_contract_refit/comparison.csv) · [Annual results](recent_contract_refit/annual.csv) · [Fit audit](recent_contract_refit/fits.csv) · [Unresolved](recent_contract_refit/unresolved.csv)

Audit: 1,432 evaluations, 2,331 refit decisions; strict completed-outcome timing, historical enable gates, beta eligibility and action ladders verified. 3 enabled model-months stayed flat for insufficient recent history. Maximum ledger error 6.76e-10 dollars. 82 NN convergence warnings retained. 119 unit tests passed. Conservative two-worker peak-memory bound 1.32 GiB.

[Trading activity](recent_contract_refit/activity.csv) · [Audit](recent_contract_refit/audit.json)
