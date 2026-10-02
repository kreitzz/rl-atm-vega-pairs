# Contract entry gate experiment

Frozen monthly contract-trained proposals, unchanged sizing ladder, costs and risk limits. Changed entry paths are fully resimulated. All 358 pairs attempted; comparisons retain the same complete pairs across every policy, including cash-only sleeves. Unresolved pairs are excluded explicitly, which can introduce selection bias.

- `memory_year`: chosen direction has positive mean completed net shadow profit over 365 calendar days, with at least 63 completed entered cohorts.
- `memory_both`: also positive over 91 days, with at least 21 cohorts.
- `commission_budget`: estimated round-trip commissions at signal close are at most 0.40 vol points (the existing four option-fill penalties). Actual commissions are still charged.
- The baseline is eligible for selection. Choose separately for RL and NN by broad-basket 2018–2021 annualized daily excess Sharpe; assess 2022 onward separately. These periods have been inspected previously: this is exploratory, not an untouched test.
- Gates use only stored as-of signal states; profitability memory excludes outcomes ending on or after the signal date. Shadow cohorts overlap, so observation counts are not independent samples. Gates only suppress new entries; existing positions follow their original lifecycle.

| Model | Gate | Period | Pairs | Return | Excess vs cash | Annualized excess Sharpe |
|---|---|---|---:|---:|---:|---:|
| rl_sized | baseline | development_2018_2021 | 348 | 5.073% | -0.025% | -0.218 |
| rl_sized | baseline | later_2022_2025 | 348 | 15.784% | -0.124% | -1.124 |
| rl_sized | baseline | full | 348 | 21.658% | -0.159% | -0.642 |
| rl_sized | memory_year | development_2018_2021 | 348 | 5.107% | 0.009% | 0.128 |
| rl_sized | memory_year | later_2022_2025 | 348 | 15.854% | -0.055% | -0.656 |
| rl_sized | memory_year | full | 348 | 21.770% | -0.047% | -0.275 |
| rl_sized | memory_both | development_2018_2021 | 348 | 5.112% | 0.015% | 0.242 |
| rl_sized | memory_both | later_2022_2025 | 348 | 15.875% | -0.034% | -0.513 |
| rl_sized | memory_both | full | 348 | 21.799% | -0.018% | -0.131 |
| rl_sized | commission_budget | development_2018_2021 | 348 | 5.095% | -0.003% | -0.053 |
| rl_sized | commission_budget | later_2022_2025 | 348 | 15.878% | -0.031% | -0.545 |
| rl_sized | commission_budget | full | 348 | 21.781% | -0.035% | -0.296 |
| nn_fixed | baseline | development_2018_2021 | 348 | 4.996% | -0.101% | -0.968 |
| nn_fixed | baseline | later_2022_2025 | 348 | 15.742% | -0.166% | -1.474 |
| nn_fixed | baseline | full | 348 | 21.525% | -0.292% | -1.217 |
| nn_fixed | memory_year | development_2018_2021 | 348 | 5.072% | -0.025% | -0.496 |
| nn_fixed | memory_year | later_2022_2025 | 348 | 15.852% | -0.057% | -0.711 |
| nn_fixed | memory_year | full | 348 | 21.728% | -0.089% | -0.607 |
| nn_fixed | memory_both | development_2018_2021 | 348 | 5.070% | -0.027% | -0.690 |
| nn_fixed | memory_both | later_2022_2025 | 348 | 15.885% | -0.023% | -0.401 |
| nn_fixed | memory_both | full | 348 | 21.761% | -0.056% | -0.517 |
| nn_fixed | commission_budget | development_2018_2021 | 348 | 5.050% | -0.047% | -0.868 |
| nn_fixed | commission_budget | later_2022_2025 | 348 | 15.841% | -0.067% | -1.317 |
| nn_fixed | commission_budget | full | 348 | 21.691% | -0.126% | -1.074 |

Development-selected policies: rl_sized: memory_both, nn_fixed: memory_year.

Unresolved pairs in the comparison union: 2. Peak RSS: 0.70 GiB.

[Full results](contract_entry_gates/comparison.csv) · [Annual Sharpe](contract_entry_gates/annual.csv) · [Pair results](contract_entry_gates/status.csv) · [Unresolved](contract_entry_gates/unresolved.csv)

[Selected-policy pair contributions](contract_entry_gates/selected_pair_contributions.csv) · [Profit concentration](contract_entry_gates/selected_concentration.csv) · [Industry contributions](contract_entry_gates/selected_industry_contributions.csv)

Trading activity on the common broad roster: rl_sized: 2,722 → 514 entries (memory_both); nn_fixed: 1,989 → 389 entries (memory_year).

The development-selected RL filter improves later-period Sharpe but still loses versus cash. Much of the improvement is avoided trading; it is not evidence of positive net alpha. All tested NN variants also have negative development Sharpe, so the selected NN is a comparative benchmark, not a supported trading recommendation.
