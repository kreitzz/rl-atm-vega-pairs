# Maturity pooling inside frozen P25 states

Checked all 608 saved proposals. 558 have complete full-size standalone paths; 50 have no valid standalone entry and are excluded from outcome statistics. Each complete proposal is assigned using its original monthly tree, and every sizing code reproduces the archived policy. No tree, optimizer or funded portfolio was rerun.

## Holding-duration associations after grouping by the same frozen state

| Population | Outcome | Correlation | Rows in usable cells | Distinct monthly states |
|---|---|---:|---:|---:|
| training_windows | target | 0.029 | 12812 | 582 |
| training_windows | daily_sd | 0.009 | 12812 | 582 |
| training_windows | path_max_drawdown | 0.102 | 12812 | 582 |
| later_proposals | target | -0.050 | 399 | 57 |
| later_proposals | daily_sd | -0.059 | 399 | 57 |
| later_proposals | path_max_drawdown | 0.055 | 399 | 57 |

Each variable is centered within its own fit-month/leaf population. Cells need at least four rows and two distinct holding durations. CSVs also show within-state/pair centering and pre/post-2022 results. Training windows repeatedly use overlapping examples. Later proposals are after their tree fit, but the strategy configuration was retrospectively selected and this analysis was requested after observing performance; this is not an untouched holdout.

## Complete later proposals by mean maturity at entry

| Entry DTE | Proposals | Mean full-size profit label | Median daily P&L SD | Median daily risk / original training-state median | Mean chosen size |
|---|---:|---:|---:|---:|---:|
| 21-27 | 77 | 1.47 | 2.59 | 1.40 | 71.4% |
| 28-34 | 418 | 2.21 | 1.89 | 0.94 | 74.4% |
| 35-45 | 63 | 1.84 | 2.18 | 1.01 | 71.4% |

Profit and daily risk use the original fixed-450-vega label units, not percentage investment returns. Relative daily risk compares each subsequent path against its own original training state; the median across proposals is not a portfolio risk estimate. Proposal paths include entries rejected by funded risk caps, so these are signal/representation diagnostics, not funded returns or subgroup Sharpes.

## Short versus long maturity inside training states

165 monthly states have at least three short (<28D) and three long (>=35D) examples. Their median longer/shorter daily-risk ratio is 0.98; the 10th–90th percentile range is 0.50–2.06. This variation checks whether maturity is uniformly interchangeable; it does not identify an optimal sizing change.

## How to interpret this check

Small conditional associations support maturity pooling as a working approximation; strong or persistent associations would motivate a targeted controlled experiment. A zero overall association does not establish uniformly homogeneous states. Risk/return differences can arise from market date, name, IV, overlapping trades and other inputs. Per-leaf samples are small, and many monthly states contain too few later proposals for a reliable maturity contrast.

Reproduce with `python -m scripts.diagnose_p25_maturity --label-checks-only`. Outputs remain beside the earlier maturity diagnostics: state_conditional_correlations.csv, state_later_proposals.csv, state_later_maturity_summary.csv and state_training_band_contrasts.csv. Verified source hashes and resource use are in label_protocol.json.
