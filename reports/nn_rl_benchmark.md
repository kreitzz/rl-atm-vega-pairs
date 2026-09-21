# Simple NN versus RL sizing with both relative-value features

The requested benchmark compares a small supervised neural network choosing full-size long, flat, or short against the existing independent Sharpe-policy search using its 25% sizing ladder. Both receive the same seven inputs and use identical training, validation, test dates, and costs. Added features are inputs, not hard trading filters.

The combined sized policy scores 1.461 versus 1.036 for the NN and 1.110 for the matched original-state sized reference. However, the new-feature full-size Sharpe policy scores 1.497: sizing does not win on every basket. Excluding silver, the sized result is 0.188; the remaining broad edge is modest.

## Portfolio results

| Basket | Original states, RL sizing | Both features, NN full-size/flat | Both features, RL full-size/flat | Both features, RL sizing |
| --- | --- | --- | --- | --- |
| combined | 1.110 | 1.036 | 1.497 | 1.461 |
| combined_without_silver | 0.086 | -0.261 | -0.034 | 0.188 |
| etf_stock | 1.134 | 1.313 | 1.656 | 1.608 |
| stocks | 0.145 | -0.356 | 0.170 | 0.191 |

Stock baskets span 2010–August 2025; ETF/stock and combined baskets span 2018–August 2025. All models within a basket use its common calendar. Portfolios use fixed equal pair budgets with unused allocation idle. Combined includes 12 stock pairs and 10 longer-history ETF/stock pairs. ETF/ETF comparisons and the two 2024-start pairs are reported individually below. All Sharpes are annualized 30-day proxy-payoff Sharpes, not daily marked option-and-hedge returns.

## Per-pair results

| Pair | Original states, RL sizing | Both features, NN full-size/flat | Both features, RL full-size/flat | Both features, RL sizing |
| --- | --- | --- | --- | --- |
| AMAT_LRCX | -0.000 | -0.260 | 0.222 | -0.061 |
| AMD_NVDA | 0.369 | -0.000 | 0.416 | 0.467 |
| AMGN_GILD | -0.052 | 0.050 | 0.252 | 0.240 |
| ARKK_TSLA | 0.809 | 1.015 | 0.591 | 0.809 |
| C_WFC | -0.539 | -0.351 | -0.555 | -0.491 |
| DUK_SO | 0.135 | 0.301 | 0.363 | 0.304 |
| GSLC_SPYM | 2.471 | 2.534 | 2.220 | 2.471 |
| HD_LOW | -0.059 | 0.052 | -0.099 | 0.054 |
| JPM_BAC | 0.014 | -0.120 | -0.180 | -0.003 |
| MSFT_ORCL | 0.202 | -0.312 | -0.261 | -0.214 |
| OIH_HAL | -0.003 | -0.278 | 0.022 | -0.018 |
| OIH_SLB | -0.107 | -0.060 | 0.201 | 0.093 |
| OIH_XOP | -0.078 | -0.494 | 0.174 | -0.060 |
| PPH_AMGN | -0.020 | 0.176 | 0.040 | 0.186 |
| PPH_BMY | 0.098 | 0.247 | 0.471 | 0.487 |
| PPH_MRK | 0.379 | 0.603 | 0.555 | 0.322 |
| SIL_AG | 1.158 | 0.307 | 1.155 | 1.121 |
| SIL_HL | 1.151 | 1.528 | 1.693 | 1.639 |
| T_VZ | -0.100 | -0.141 | -0.132 | -0.193 |
| UNP_CSX | 0.276 | -0.098 | 0.241 | 0.360 |
| UPS_FDX | -0.130 | -0.259 | -0.173 | -0.121 |
| XOM_CVX | -0.040 | -0.195 | -0.070 | -0.036 |
| XOP_APA | -0.140 | -0.152 | -0.203 | -0.181 |
| XOP_COP | -0.228 | 0.026 | -0.274 | -0.075 |
| XOP_DVN | -0.216 | -0.344 | -0.121 | -0.110 |

The sized policy with both features beats the NN in 18/25 pairs, the original-state sized policy in 15/25, and the both-feature full-size policy in 15/25. Counts include the short-history comparisons and are descriptive, not independent trials. Each pair uses its own available test history.

## Annual combined-basket Sharpe

| Year | Original states, RL sizing | Both features, NN full-size/flat | Both features, RL full-size/flat | Both features, RL sizing |
| --- | --- | --- | --- | --- |
| 2018 | 0.657 | 0.078 | 0.613 | 1.008 |
| 2019 | -0.128 | -0.227 | -0.036 | 0.537 |
| 2020 | 1.376 | 0.521 | 1.188 | 1.718 |
| 2021 | 0.503 | 0.025 | 0.443 | 0.153 |
| 2022 | 1.196 | 2.433 | 2.395 | 2.190 |
| 2023 | 0.297 | 0.518 | 1.094 | 1.610 |
| 2024 | 1.951 | 1.976 | 2.356 | 1.875 |
| 2025 | 5.007 | 2.760 | 4.751 | 3.660 |

Excluding silver:

| Year | Original states, RL sizing | Both features, NN full-size/flat | Both features, RL full-size/flat | Both features, RL sizing |
| --- | --- | --- | --- | --- |
| 2018 | 0.424 | 0.491 | 0.643 | 0.854 |
| 2019 | -0.780 | -0.210 | -0.249 | 0.321 |
| 2020 | 1.067 | 0.427 | 0.890 | 1.519 |
| 2021 | -1.229 | -1.476 | -1.298 | -1.308 |
| 2022 | -0.013 | -0.002 | -0.237 | -0.520 |
| 2023 | -0.595 | -1.524 | -2.063 | -1.467 |
| 2024 | -1.167 | -0.726 | -0.880 | -0.940 |
| 2025 | 2.408 | -0.248 | 2.467 | 1.189 |

2025 ends in August. The sized basket excluding silver loses in each of 2021–24, despite a positive full-period score. This is not a consistently profitable broad-universe result.

## Paired differences

| Basket | Left model | Right model | Sharpe difference | 95% low | 95% high |
| --- | --- | --- | --- | --- | --- |
| stocks | rl_both_sized | nn_both_fixed | 0.547 | 0.164 | 0.912 |
| stocks | rl_both_sized | rl_base_sized | 0.046 | -0.157 | 0.243 |
| stocks | rl_both_sized | rl_both_fixed | 0.021 | -0.170 | 0.216 |
| etf_stock | rl_both_sized | nn_both_fixed | 0.295 | 0.054 | 0.625 |
| etf_stock | rl_both_sized | rl_base_sized | 0.473 | 0.182 | 0.756 |
| etf_stock | rl_both_sized | rl_both_fixed | -0.048 | -0.219 | 0.159 |
| combined | rl_both_sized | nn_both_fixed | 0.424 | 0.123 | 0.771 |
| combined | rl_both_sized | rl_base_sized | 0.351 | 0.072 | 0.663 |
| combined | rl_both_sized | rl_both_fixed | -0.037 | -0.266 | 0.218 |
| combined_without_silver | rl_both_sized | nn_both_fixed | 0.449 | 0.050 | 0.827 |
| combined_without_silver | rl_both_sized | rl_base_sized | 0.102 | -0.217 | 0.413 |
| combined_without_silver | rl_both_sized | rl_both_fixed | 0.222 | -0.078 | 0.460 |

Intervals use 2,000 paired circular block resamples with blocks of 63 entry dates. They preserve cross-pair alignment within the basket, but do not account for model selection, repeated examination of history, or retraining uncertainty.

## Benchmark definitions

- **State:** IV spread, both capped earnings countdowns, both trailing 30-calendar-day RVs, `(trailing_RV_A-IV_A)-(trailing_RV_B-IV_B)`, and `(past_mean_spread-current_spread)/past_std_spread`. Spread moments use the prior 252 available panel observations, excluding today, with minimum 126. ETF countdowns remain constant 31. The RV feature is a transformation of existing inputs; the historical-spread feature adds history.
- **NN:** one hidden layer of 16 tanh units; supervised mean-squared-error forecast of the next 30-day gross relative payoff. Features and target are standardized using training data only. L-BFGS training uses seed 2026, maximum 10,000 iterations / 200,000 function calls, tolerance 1e-4. Training fits two regularization strengths (0.01, 1.0); validation chooses among those and cost multipliers (1, 2, 4), for six candidates. Go long if predicted payoff exceeds multiplier × entry cost; short if below its negative; otherwise flat. Actions are strictly -1, 0, +1. There is no randomly held-out early-stopping sample.
- **RL sizing:** the existing training-reward regression tree plus cross-entropy policy search maximizing annualized payoff Sharpe. This is direct historical policy search, not a neural actor-critic or TD learner. Tree candidates use 4/8/16 leaves × minimum 126/252 observations, selected by validation Sharpe. Five search seeds, 80 iterations, and population 256 are unchanged. Actions are -1 through +1 in 0.25 steps. Each sizing search is warm-started from the best training full-size policy.
- **Sizing control:** the same new-feature tree/policy method with only -1/0/+1 actions. Its tree configuration is independently selected on validation. This helps distinguish sizing benefits from architecture/objective differences between the NN forecast and Sharpe policy. It is not the sized policy forcibly rounded to full size.
- **Original-state reference:** fresh sized-policy fits with the five original inputs, on exactly the same feature-complete samples as the new models. The new historical feature removes 126 initial panel observations. SIL/HL's 2016 fold then falls below 504 training rows, so all four models start its test in 2018; other retained folds preserve their original test dates, which are asserted. Reference scores can therefore differ from previously saved experiments because initial training history changes.
- **Validation:** expanding training, preceding two-year validation, following two-year test; forward 30-day outcomes are purged at boundaries. At least 504 training, 252 validation, and 126 test rows per fold. Test outcomes do not choose configurations, directions, sizes, or NN thresholds.
- **Payoff/cost:** action × `[(forward_RV_A-IV_A)-(forward_RV_B-IV_B)]` minus `abs(action) × 0.025 × (IV_A+IV_B)/2`. Sharpe is `sqrt(365/30) × mean / sample_std`, retaining flat dates. Costs are paid per new entry; budgets are per cohort, not the sum of overlapping holdings.

## Fit and data checks

151 NN selections across 25 pairs; 2 selected fits reported an optimizer convergence warning. The NN iteration budget was increased from an initial 2,000-iteration run after inspecting training convergence, before comparing test performance. All NN fits were rerun under the final common settings, with tree-policy actions preserved. This is an optimization check, not a test-score-based choice of NN budget.

Training-sample sensitivity is material: the original-state stock basket is 0.145 here versus 0.317 in the prior run; AMD/NVDA is 0.369 versus 0.635. The model, search, test dates, and costs are unchanged, but the historical-feature warmup removes early training observations. Use the freshly matched reference to assess the new features; the sensitivity itself is a limitation of the current tree/policy approach.

Saved predictions were checked against the payoff formula and action menus for every pair/model. All training/validation outcome boundaries were verified. The benchmark tests also check NN cost thresholds and historical-feature causality.

The underlying universe and period have already been explored. Silver extreme-IV concentration and GSLC liquidity limitations remain; retraining does not validate execution or create a fresh holdout. See the [ETF input audit](etf_exploration.md) and [winner/loser review](winner_loser.md).

[Pair scores and exposure](nn_rl_benchmark/scores.csv) · [Annual pair results](nn_rl_benchmark/annual.csv) · [Annual portfolio results](nn_rl_benchmark/portfolio_annual.csv) · [Selected configurations](nn_rl_benchmark/configs.csv) · [All candidates](nn_rl_benchmark/candidates.csv) · [Paired differences](nn_rl_benchmark/contrasts.csv)

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.run_nn_rl_benchmark --workers 3 --resume
python -m scripts.report_nn_rl_benchmark
python -m unittest discover -s tests -q
```

Detailed predictions/checkpoints remain in ignored `artifacts/nn_rl_benchmark/`; no additional raw market data or reward columns are published in `data/`.

Selected NN optimizer warnings:
| Pair | Fold | Iterations |
| --- | --- | --- |
| JPM_BAC | 2020 | 10000.000 |
| XOP_DVN | 2024 | 10000.000 |
