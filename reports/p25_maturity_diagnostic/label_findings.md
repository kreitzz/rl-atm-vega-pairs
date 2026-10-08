# Original P25 labels and frozen-state diagnostic

845 distinct examples actually used by the 90 saved trees. Original full-size paths reproduce every target; frozen leaf counts and target means reproduce every tree. No training, optimizer search or new strategy replay.

## Does a longer holding period mean a larger label or greater risk?

| Saved-path quantity | Holding-days correlation | Within pair and quarter |
|---|---:|---:|
| target | -0.013 | 0.003 (n=669) |
| abs_target | 0.014 | -0.107 (n=669) |
| daily_sd | 0.048 | -0.087 (n=669) |
| realized_quadratic_exposure | 0.086 | 0.088 (n=669) |
| path_max_drawdown | 0.133 | 0.076 (n=669) |

The label is cumulative standalone P&L above cash divided by the fixed 450 vega budget, not a return standardized to 30 days. Daily standard deviation excludes zero padding after settlement. Quadratic exposure sums squared daily P&L: it can rise with time or unusually large moves, and is not an estimate of independent-day variance. These are associations, confounded by names, IV and market dates; overlapping trades prevent treating rows as independent trials.

## Do frozen states pool materially different durations?

360 / 582 monthly leaf populations contain both mean-entry maturities below 28 days and at least 35 days. Windows reuse examples, so this count is not independent evidence.
Latest tree (2025-06-01): 7 leaves; 6 contain both bands. See label_frozen_leaf_ranges.csv for duration spans and within-state risk dispersion.

## Do open positions coincide with rejected new entries?

| Decision status | Count | Same pair open | Long-held pair open | Single-leg tail open |
|---|---:|---:|---:|---:|
| entered | 207 | 83 | 19 | 4 |
| integer_vega_mismatch | 3 | 3 | 2 | 0 |
| no_eligible_contract | 50 | 39 | 12 | 1 |
| risk_deleveraging | 69 | 69 | 0 | 3 |
| risk_limit_entry | 279 | 258 | 54 | 6 |

Inventory is reconstructed from actual prior entries still open after that day's expiry settlements. "Long holding" means average actual leg holding at least 35 calendar days. Counts concern the same pair. A rejected trade with open inventory is evidence that the mechanism can be present; it does not prove maturity caused rejection or that a shorter holding policy would improve Sharpe. Risk caps can reject a trade even with no existing same-pair position.

## Reproduce

`python -m scripts.diagnose_p25_maturity --label-checks-only`

Outputs: label_examples.csv, label_maturity_summary.csv, label_duration_correlations.csv, label_frozen_leaf_ranges.csv, label_admission_inventory.csv, label_admission_summary.csv and label_protocol.json. Core training dates, directional labels and frozen models are verified against source manifests. Centered correlations restrict pair/quarter cells to at least four examples and at least two durations; see CSVs for sample counts and earlier/later subperiods.
