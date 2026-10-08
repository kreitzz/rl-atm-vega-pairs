# PCA diagnostic for the selected P25 sizing policy

This inspects the 27-input `features/control` policy associated with 1.263 funded excess Sharpe in 2022–August 2025. No policy was changed or backtest rerun.

## Method and scope

- Reconstructed all 90 monthly 18-month training windows exactly; row counts and latest completed outcomes match saved fit metadata.
- Every signal date and outcome end precedes its fit cutoff. The main correlation/loadings snapshot is January 2022; June 2025 is a separate later historical snapshot.
- PCA uses 15 economic inputs, centered and scaled to sample standard deviation within each window. The 12 availability flags are reported separately. Constant economic fields receive zero weight.
- Uses the model's existing zero-filled inputs. Observations are pair/date training examples; overlapping trades and repeated pairs are not independent samples.
- Targets are not read as PCA inputs. Only eligibility and completed-outcome metadata reproduce the training population; no profit correlation or significance test is performed.
- PC signs are arbitrary. Adjacent-window stability uses the top-three subspaces, so sign flips or rotations within a subspace do not count as instability.
- Later windows are descriptive. This is not a new holdout, a PCA sizing policy, or evidence of improved Sharpe.

## Results

- January 2022: n=70; first PC explains 33.4%; first three explain 62.7%; 6 PCs are required for 80%.
- June 2025: n=83; first three explain 52.4%; 7 PCs are required for 80%.
- Across monthly windows, first-three variance share ranges from 49.1% to 63.8%; median 54.9%.
- Adjacent top-three subspace overlap: median 0.960, minimum 0.657 (1 means identical subspaces). Overlapping training windows naturally increase apparent stability.
- Three-core-input check: January 2022 first PC explains 53.8%, first two 89.0%.
- Availability flags: 0 flag/window combinations vary; minimum available fraction 100.0%.

Strongest January 2022 correlations:

| Inputs | Pearson | Spearman |
|---|---:|---:|
| iv_mean / rv_mean | 0.897 | 0.900 |
| iv_mean / commission_vol_points | 0.771 | 0.759 |
| oriented_iv_gap / correlation | -0.769 | -0.664 |
| oriented_iv_gap / beta | 0.754 | 0.720 |
| rv_mean / correlation | 0.729 | 0.758 |

Dominant standardized inputs in the January 2022 components:

- PC1: iv_mean (+0.89), rv_mean (+0.89), correlation (+0.87), oriented_iv_gap (-0.84).
- PC2: count_365d (+0.83), count_91d (+0.60), profit_365d (-0.56), beta (+0.51).
- PC3: profit_91d (+0.75), rank_strength (-0.50), oriented_vrp_gap (+0.45), oriented_iv_gap (+0.39).

Most frequently used tree split variables:

| Variable | Months with a split | Months after pruning constant-size subtrees |
|---|---:|---:|
| profit_365d | 46 | 39 |
| count_365d | 42 | 37 |
| correlation | 39 | 36 |
| iv_mean | 35 | 26 |
| profit_91d | 34 | 27 |
| oriented_iv_gap | 32 | 17 |
| oriented_vrp_gap | 31 | 27 |
| rank_strength | 30 | 16 |

There are 112 splits across the saved models inside subtrees that always choose one size. Those subtrees can be collapsed for understanding the sizing mapping. This does not measure profitability importance, and does not prove every retained split changes allocations on future qualified proposals.

## Interpretation

There is measurable redundancy: January 2022 IV and RV have correlation 0.897, and two components capture 89.0% of variation within the three core inputs. However, the full economic input set needs six components in January 2022 and seven in June 2025 to reach 80%. These snapshots do not support treating all 15 inputs as just three linear factors.

The saved trees draw on profitability and observation history as well as volatility and correlation. Frequent splits and large PCA loadings answer different questions; neither establishes that a feature improves portfolio Sharpe. The 112 constant-size-subtree splits also demonstrate that predicting the regression target and changing the sizing action are distinct.

PCA can identify shared variation, but principal components are mixtures rather than feature selections. Use these results to nominate related feature groups for future retrained ablations; no input is dropped on the basis of this diagnostic. Adjacent training windows overlap, and the January 2022 / June 2025 samples have only 70 / 83 observations, so apparent stability is not an independent robustness test.

## Files and reproduction

`python -m scripts.diagnose_p25_pca` reads bundled data directly. It uses one numerical thread, serial processing, 20% duty cycle and a 384 MiB RSS stop. The 35% system-memory floor applies on macOS, matching the existing supervisor; no cache restoration is required.

- `correlations.png`: January 2022 Pearson/Spearman relationships.
- `variance_and_stability.png`: cumulative variance and monthly subspace stability.
- `component_loadings.png`: January 2022 and June 2025 top-three loadings.
- CSVs provide correlations, all variance shares, top-three weights/loadings, monthly summaries, availability, and tree split usage.
- `protocol.json`: source/model hashes, exact procedure, environment and export limitations.
