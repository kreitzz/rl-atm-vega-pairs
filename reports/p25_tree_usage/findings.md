# Which monthly P25 trees and variables were used?

The current strategy has 90 saved monthly fits, January 2018–June 2025. **73 trees scored at least one qualified proposal**, and **69 trees produced at least one funded entry**. All 608 proposals and 207 entries were mapped to their original policy. This is a read-only census of the current strategy, not the five representation-experiment variants.

## Variables per tree

The full fitted trees use **2–6 distinct variables**, with median **5** and mean **4.48**. Only 11 of 90 trees use exactly three variables; 78 use more than three.

| Distinct variables in whole fitted tree | Trees | Share of 90 |
|---|---:|---:|
| 2 | 1 | 1.1% |
| 3 | 11 | 12.2% |
| 4 | 28 | 31.1% |
| 5 | 44 | 48.9% |
| 6 | 6 | 6.7% |

## Most frequently used variables

The first count includes any split in the whole tree. The second collapses branches whose leaves all choose the same size. Counts are structural usage, not measures of incremental Sharpe or causal feature importance.

| Variable | Trees with any split | Share of 90 | Trees after constant-size pruning | First split |
|---|---:|---:|---:|---:|
| Past profitability, 365 days (`profit_365d`) | 46 | 51.1% | 39 | 12 |
| Observation count, 365 days (`count_365d`) | 42 | 46.7% | 37 | 12 |
| Pair correlation (`correlation`) | 39 | 43.3% | 36 | 22 |
| Mean implied volatility (`iv_mean`) | 35 | 38.9% | 26 | 7 |
| Past profitability, 91 days (`profit_91d`) | 34 | 37.8% | 27 | 2 |
| Oriented relative IV gap (`oriented_iv_gap`) | 32 | 35.6% | 17 | 1 |
| Oriented volatility-premium gap (`oriented_vrp_gap`) | 31 | 34.4% | 27 | 9 |
| Spread rank strength (`rank_strength`) | 30 | 33.3% | 16 | 4 |
| Pair beta (`beta`) | 25 | 27.8% | 20 | 6 |
| Commission in volatility points (`commission_vol_points`) | 23 | 25.6% | 22 | 3 |
| Mean realized volatility (`rv_mean`) | 23 | 25.6% | 17 | 9 |
| VIX (`vix`) | 21 | 23.3% | 21 | 2 |
| Observation count, 91 days (`count_91d`) | 13 | 14.4% | 12 | 1 |
| Mean days to expiry (`dte_mean`) | 9 | 10.0% | 6 | 0 |

There are 492 raw split nodes across all 90 trees; 112 belong to subtrees that always choose the same size. Pruning yields 1–6 distinct variables per sizing tree, median 4. A value of zero means that monthly policy chooses one size regardless of its tree partition.

## Which tree was used for a trade?

The model is selected by the **signal month**. The signal is evaluated at its close, and execution is on the next trading session. A month-end signal can therefore execute in the next month using the earlier month policy. The model is never chosen by later performance. Each accepted trade retains its original entry sizing while the book is marked and hedged.

14 proposals cross a calendar-month boundary between signal and execution. Monthly dates with zero proposals represent fitted models that did not score a qualified proposal. The daily book continues during months without new entries and through the August 2025 runoff.

The full 90-row **monthly_trees.csv** lists fit date, proposal and funded-entry counts, first/last signal and execution dates, training cutoff, states, first split, every split variable and every leaf sizing action. **proposal_tree_mapping.csv** lists all 608 proposals with their exact model, leaf, size, admission result and variables visited along their decision path.

## Whole tree versus one decision path

Different branches can use different variables. A trade visits one root-to-leaf path, so the number of variables it encounters can be smaller than the number in the whole tree. The following counts track actual proposal paths rather than all possible branches.

| Distinct variables on a proposal path | Proposals | Share of 608 |
|---|---:|---:|
| 1 | 70 | 11.5% |
| 2 | 112 | 18.4% |
| 3 | 261 | 42.9% |
| 4 | 158 | 26.0% |
| 5 | 7 | 1.2% |

## Checks and reproduction

All 90 original models were read from the hash-verified archive with the saved sklearn version. The earlier feature-frequency census was reproduced exactly. Tree application and independent path traversal reproduce all 608 saved leaf assignments and sizing actions. No model was fitted, no CEM search was run, and no portfolio was replayed.

`python -m scripts.diagnose_p25_tree_usage` in the pinned environment. One numerical thread, 20% duty cycle, 384 MiB RSS stop. Charts: feature_usage.png, tree_structure.png and monthly_usage.png. Raw statistics: feature_usage.csv, monthly_trees.csv, distributions.csv and proposal_tree_mapping.csv. Provenance: protocol.json and summary.json.
