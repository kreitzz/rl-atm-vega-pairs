# Monthly learning on actual contracts

Attempted **358 pairs** with independent RL sizing and a full-size/flat NN, 2018-01-02–2025-08-29. **2 pairs have an unresolved evaluation in at least one model.** Cash-only pairs remain in the capital denominator. Any incomplete basket below is explicitly a common-complete subset, not a full-universe result.

| Basket | Policy | Scope | Pairs | Total return | CAGR | Annualized excess Sharpe | Max drawdown |
|---|---|---|---:|---:|---:|---:|---:|
| broad | rl_sized | common_complete_subset | 348 | 21.66% | 2.59% | -0.642 | -0.01% |
| broad | nn_fixed | common_complete_subset | 348 | 21.52% | 2.58% | -1.217 | -0.02% |
| legacy_extra | rl_sized | full | 8 | 21.76% | 2.61% | -0.048 | -0.08% |
| legacy_extra | nn_fixed | full | 8 | 21.15% | 2.54% | -0.518 | -0.22% |

Sharpe is calculated from daily cash-excess returns and annualized with √252. Total returns include cash interest; 2025 is partial. Equal $1 million sleeves, with idle budgets kept in cash.

## Matched comparison with the previous replay

Both experiments below use the same 311 broad pairs that completed all compared policies. This ex-post common-data roster can be biased. The new run changes marks, rewards, refit schedule, features and eligibility together; it is not an isolated ablation.

| Experiment | Policy | Total return | Annualized excess Sharpe |
|---|---|---:|---:|
| frozen_proxy | rl_reference | 19.88% | -1.337 |
| frozen_proxy | rl_sized | 19.89% | -1.310 |
| frozen_proxy | nn_fixed | 16.93% | -3.646 |
| contract_trained | rl_sized | 21.65% | -0.639 |
| contract_trained | nn_fixed | 21.53% | -1.119 |

## What is aligned

- State IVs come from the eligible selected ATM/approximately-30-day call contracts. Standardized surface IV and future RV are not training states or reward inputs.
- Labels use observed option mid marks, daily Black–Scholes delta hedges, integer contract counts, 0.10-vol-point execution penalty per option fill, $0.65 option commissions, 1 bp stock execution, financing, stock borrow, and the simulator’s lifecycle conventions.
- Every month, proposal models fit a rolling five-year history ending before a separate one-year validation period. At least 504 fully observed label rows are required. Outcomes must finish strictly before validation starts. No refit includes the validation period.
- Each fitted proposal is replayed on that historical validation year through the same overlapping-book, NAV-scaled risk-controlled simulator used for evaluation. It is enabled only if validation daily excess Sharpe is positive. Unresolved validation books do not pass. New validation entries stop 35 calendar days before its end, so all holdings can close before the current prediction month.
- RL uses an eight-leaf direct Sharpe policy search and actions −1 through +1 in 0.25 increments. The NN has one 16-unit tanh layer, train-only scaling, and separately predicts the two full-size net directional outcomes; it chooses −1/0/+1. Both receive the same states and validation gate.
- States include contract spread, trailing RV and earnings inputs, spread reversion, completed 91/365-day directional net-profit memory and observation counts, signal-close commission-per-vega estimate, contract maturity/moneyness, and pair beta/correlation.
- Historical broad pair rosters are retained. Monthly eligibility uses only preceding-close information: 126-session beta of the first leg on the second between 0.75 and 1.25, correlation at least 0.5. This gate is applied to proposal-training rewards, validation actions and evaluation actions. The eight legacy extras are reported separately; unlike the frozen replay, they now use the same new learning specification.

## Limits and verification

**The proposal optimizer still uses sums of standalone shadow cohorts.** Its objective is not an exact optimizer of the overlapping risk-controlled book. Full-book held-out validation and final evaluation enforce the actual risk rules. Profit memory is completed directional shadow-trade history, not feedback from the currently deployed policy’s own path. Validation starts with an empty book; final evaluation carries holdings across model-refit dates.
Corporate-action and missing-held-quote cases remain unresolved rather than imputed or retrospectively liquidated. Unresolved training trajectories are unknown labels and excluded from fitting, not assigned zero profit; this missing-outcome selection remains a limitation. Historical earnings knowledge, midpoint execution and the existing reserve/stress model remain assumptions. This is repeated research on the same history, not an untouched test set or proof of live profitability.
Peak process memory: 0.79 GiB. Sum of worker peak RSS: 2.34 GiB (a conservative bound, not a simultaneous measurement). Completed evaluations pass cash/hedge/option-ledger reconciliation. Fit records are checked for strict training/validation/evaluation timing. Batched labels matched all 3,399 AMD/NVDA training dates and all nine actions to numerical precision; tests also cover split adjustments, dividends, missing marks and missing Greeks. Reusing prepared market lookups reproduced the saved full-history AMD/NVDA NAV exactly.
695 of 7921 NN fits reported a convergence warning; these remain visible in the fit audit and are not selectively removed.

[Every pair result](aligned_contract_learning/status.csv) · [Annual returns and Sharpe](aligned_contract_learning/annual.csv) · [Matched comparison](aligned_contract_learning/matched_comparison.csv) · [Matched annual comparison](aligned_contract_learning/matched_annual.csv) · [Trading breadth](aligned_contract_learning/breadth.csv) · [Unresolved cases](aligned_contract_learning/unresolved.csv) · [Training label coverage](aligned_contract_learning/labels.csv) · [Monthly fit audit](aligned_contract_learning/fits.csv)

Reproduce the bounded three-worker run with `python -m scripts.run_aligned_contract_batch`, then `python -m scripts.report_aligned_contract_learning`. Each worker processes pairs sequentially. Worker input manifests preserve resume checks; labels and detailed ledgers stay in ignored local artifacts.
