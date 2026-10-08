# P25 sizing: original inputs versus PCA factors

**Keep the current full-input representation for the decision tree.** It had the highest portfolio Sharpe in both the validation and later evaluation periods. Restricting the model to the three core economic inputs reduced performance, and none of the PCA transformations improved it.

Five predeclared representations were refitted across all 90 monthly, 18-month completed-outcome windows. Each reran the same tree and 50 × 192 cross-entropy sizing search. All 608 proposals were replayed through the same $10 million funded ledger. The current full-input baseline reproduces every monthly tree, sizing choice, daily NAV, trade and decision. Eight numerical and leakage tests passed; every portfolio ledger reconciled and ended with closed inventory.

| Representation | 2018–2021 validation Sharpe | 2022–August 2025 evaluation Sharpe | Evaluation profit above cash | Evaluation entries |
|---|---:|---:|---:|---:|
| Current full inputs (`raw_full`) | **0.573** | **1.263** | **$248,673** | 110 |
| Three core inputs (`core3`) | 0.390 | 0.880 | $180,336 | 107 |
| Three PCA factors (`pca3`) | 0.335 | 1.013 | $212,425 | 102 |
| PCA retaining at least 80% of variance (`pca80`) | 0.389 | 0.945 | $195,554 | 112 |
| All PCA factors (`pca_all`) | 0.461 | 0.852 | $160,850 | 106 |

Validation selected **the current full-input model** using only 2018–2021 returns. All later results are reported, but later ranking is descriptive; no configuration was selected using later Sharpe. All scores are annualized daily cash-excess Sharpe after the inherited execution, hedging and financing costs.

## Interpretation

**The three core inputs are useful heuristics, but they are not an adequate replacement for the broader input set in this experiment.** Removing the other economic inputs reduced later Sharpe by 0.382, from 1.263 to 0.880, and reduced evaluation profit above cash by about $68,337. The full model also won in validation. The paired 95% interval for the later Sharpe change is [-0.622, -0.124] using 126-session blocks, and remains below zero with 21- and 63-session blocks. This supports a collective contribution from the additional sizing inputs; it does not identify which individual variable caused the gain.

**PCA did not provide a better representation for the current tree and sizing search.** Retaining every PCA direction preserves variation in all nonconstant economic inputs, yet its later Sharpe was 0.852 versus 1.263 for the original representation. The loss therefore cannot be attributed solely to discarded principal components. A tree splits one coordinate at a time; rotating the coordinates changes the partitions it can learn with the same eight-leaf limit. The result is consistent with the existing economic coordinates being better suited to this tree, without proving a universal disadvantage for PCA.

**Explained input variance is not a measure of predictive usefulness.** The 80% version needed six to eight economic factors per monthly fit. Retaining more PCA variance did not improve later Sharpe: three factors scored 1.013, the 80% version 0.945, and all factors 0.852. Those differences between PCA variants are uncertain: the paired intervals for the reported PCA-versus-PCA contrasts include zero. Their ranking also changed between periods: all factors was the best PCA variant in validation but the weakest in later evaluation. These results do not establish that every omitted direction is noise; they demonstrate why variance share alone cannot choose the representation.

**The current model is already offered 15 economic inputs, plus 12 availability flags.** The three core inputs are IV mean, the direction-oriented and normalized IV gap, and historical rank strength. The full model additionally has volatility-premium, realized-volatility, beta, correlation, maturity, cost, VIX, profitability-memory and observation-count inputs. Seeing only a few of these in a particular monthly tree does not mean the remaining inputs were unavailable. The PCA diagnostic's roughly 50% figure describes the first three principal components of this broader set, not the information captured by three original variables.

The full model had a smaller later cash-relative maximum drawdown, 0.495%, than every alternative (0.686%–0.745%). It did not dominate every year: PCA variants sometimes won in individual years, and all five versions had negative Sharpe in 2024. Keeping the current representation is supported by this controlled comparison, not by a claim of global optimality or performance in every regime.

## Paired uncertainty versus the current model

| Candidate | Evaluation Sharpe change | 95% interval, 126-session blocks |
|---|---:|---:|
| core3 | -0.382 | [-0.622, -0.124] |
| pca3 | -0.250 | [-0.512, +0.032] |
| pca80 | -0.318 | [-0.577, -0.054] |
| pca_all | -0.410 | [-0.689, -0.036] |

## What the comparison tests

- core3 versus raw_full tests whether three economic sizing inputs suffice. All availability flags remain; they are constant in every training window.
- pca3 versus core3 tests a compact factor representation against the three economic heuristics.
- pca_all versus raw_full changes splitting coordinates without discarding variation in nonconstant economic inputs.
- pca80 and pca3 versus pca_all test compression within the PCA representation. The 80% threshold is fixed in advance; dimension is determined from each training window.
- PCA loadings/means/scales use training inputs only. No target enters the PCA fit. Trees learn the original regression target; CEM optimizes training shadow-book Sharpe; reported performance is independently replayed funded portfolio Sharpe.

## Limits

The 2022–2025 history was already inspected. This is a causal walk-forward comparison on previously researched history, not a fresh untouched holdout. Later completed outcomes may train later monthly fits. Paired bootstrap intervals preserve common dates and serial blocks, but do not adjust for prior strategy search, multiple comparisons or representation selection. Block-length sensitivity at 21, 63 and 126 sessions is provided, with 2,000 replications for each contrast and block length.

Entry screens, universe, direction and contracts stay frozen. A three-input sizing model still inherits qualification screens using other economic inputs. This experiment cannot establish that those other variables are irrelevant to trade selection. It also does not test PCA residual signals across stock returns, supervised factors, neural networks, other feature families, or globally optimal trees.

The old $0.65-per-contract training labels, actual $0.03 execution commission, 0.10 vol-point option penalty, 1 bp stock hedge costs and all financing/risk conventions are preserved. Continued inventory across the 2022 boundary is included. The evaluation ends on August 29, 2025, so 2025 is partial.

## Reproduction and files

`python -m scripts.run_p25_representation` with the pinned requirements.txt environment plus matplotlib. Single worker, one numerical thread, 20% CPU duty, 384 MiB RSS stop; a STOP file under artifacts/contracts/p25_representation interrupts safely. Peak worker RSS was about 292 MiB. Original models and data are read-only. The findings' interpretation is an editorial addition to the runner's generated numerical report.

- protocol.json and selection.json: predeclared cases, date splits, package versions and validation-only selection.
- periods.csv, annual.csv and sharpe_intervals.csv: funded metrics and paired intervals.
- fits.csv: 90 monthly fits per representation, dimensions, training-only variance retained and exact baseline checks.
- allocation_comparison.csv: all 608 proposed sizing actions.
- comparison.png: validation/evaluation Sharpe and later cash-relative wealth.
- artifacts/contracts/p25_representation: fitted transformations, trees, leaf actions and complete audited funded ledgers.
