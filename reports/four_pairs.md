# Four-pair experiment

Adding XOM/CVX, JPM/BAC, and HD/LOW did not improve aggregate annualized payoff Sharpe in this first pass. All four use the same 3,922 test entry dates, January 2010–August 2025, with a 1.0-vol-point full-size pair cost. The AMD/NVDA control reproduces the existing reference actions and rewards exactly.

| Strategy | Annualized payoff Sharpe |
| --- | ---: |
| AMD/NVDA only | 0.667 |
| XOM/CVX only | -0.113 |
| JPM/BAC only | -0.124 |
| HD/LOW only | -0.120 |
| Four policies, 25% budget each | 0.494 |
| Four policies, shared cap | 0.561 |
| Joint pair-and-size policy | -0.059 |

## What was tested

Each pair uses its IV spread, both earnings countdowns, and both trailing 30-calendar-day RVs. The historical earnings-date availability assumption is unchanged. All four have 5,685 matched model rows from December 31, 2002 to August 29, 2025. Pair order determines positive direction: long the first ticker's vega, short the second's.

- **Individual policies:** the existing tree-state/Sharpe-search procedure is independently trained per pair. The equal-budget baseline allocates 25% of the maximum entry budget to each policy and retains its learned size. It does not redistribute unused allocation.
- **Shared-cap baseline:** independently chosen signed sizes are divided by `max(1, sum(abs(sizes)))` each entry date. This permits any pair to use the budget when other pairs are inactive. Resulting allocations need not be exact 25% increments.
- **Joint policy:** a multioutput tree observes all 20 state inputs, learns states from the four training payoff labels, and direct policy search maximizes the Sharpe of the total net pair payoff. There are 321 signed allocation combinations in 25% increments with total absolute allocation at most 100%, including all-flat. It can select, size, and combine any of the four pairs. No lookahead outcomes enter its observations.

The shared entry budget is a sum of per-pair one-leg dollar-vega magnitudes, with equal opposite vega inside each pair. At the existing 0.001 maximum vega per reference-capital dollar, full allocation is $1,000 long and $1,000 short initial vega per $1 million reference capital. Sizes are fixed at entry. This caps each new cohort; it does not impose a cap on the sum of overlapping outstanding cohorts.

All models use expanding training, preceding two-year validation, and subsequent two-year test blocks, with 30-day outcomes purged at the boundaries. Trees use 4/8/16 leaves and minimum leaf sizes 126/252; policy search uses five seeds, 80 iterations, and 256 candidates. The best training seed is retained, and tree configuration is selected by validation Sharpe. The joint tree uses unscaled multioutput payoff labels, so higher-variance pairs influence its partition more. The joint model changes the state partition and does not mathematically contain the existing AMD/NVDA policy as a guaranteed fallback.

## What hurt performance

The three additional standalone policies have positive gross proxy P&L but negative net P&L after the common 1-point execution cost. Their low correlations with AMD/NVDA do not compensate for those losses. The joint model also changes AMD/NVDA's decisions and adds losing allocations to the other pairs. Its training Sharpes range from 0.94 to 1.80, while aggregate test Sharpe is negative; this is consistent with poor generalization from the larger decision space, not proof that diversification cannot work.

As an additional diagnostic, going entirely flat in folds where the selected joint model has nonpositive validation Sharpe gives aggregate test Sharpe **-0.066**. That also fails to improve the reference.

## Annual comparison

| Entry year | AMD/NVDA | Equal budgets | Shared cap | Joint |
| --- | ---: | ---: | ---: | ---: |
| 2010 | 0.724 | 0.708 | 0.527 | 1.417 |
| 2011 | -0.331 | -0.513 | -0.519 | -2.656 |
| 2012 | -0.092 | -0.217 | 0.060 | -0.348 |
| 2013 | 1.783 | 2.231 | 2.182 | -0.036 |
| 2014 | 2.538 | 2.616 | 2.552 | -3.279 |
| 2015 | 1.576 | 1.407 | 1.575 | -0.489 |
| 2016 | 1.187 | 1.222 | 1.319 | 0.567 |
| 2017 | 1.177 | 1.038 | 1.092 | 0.640 |
| 2018 | 1.243 | 0.907 | 0.886 | 0.424 |
| 2019 | 0.186 | -0.037 | -0.161 | 0.286 |
| 2020 | 0.453 | 0.568 | 0.482 | 0.274 |
| 2021 | -0.733 | -1.883 | -2.069 | 0.430 |
| 2022 | -0.635 | -1.130 | -1.153 | -0.537 |
| 2023 | 0.080 | -0.767 | -0.687 | -0.032 |
| 2024 | -0.139 | -0.142 | -0.266 | 0.118 |
| 2025 | 1.922 | 1.923 | 1.779 | -1.144 |

Sharpes retain the existing `sqrt(365/30)` annualization of 30-day net entry payoffs, including flat dates. These are not daily marked portfolio Sharpes or executable option-and-hedge returns. The pairs were handpicked; this experiment does not implement systematic universe selection or establish a pristine out-of-sample result. The single-pair reference itself was developed on repeatedly inspected history. No features, costs, or pair choices were retuned after this run to improve its test score.

[Overall scores](four_pairs/scores.csv) · [Annual scores and net proxy P&L](four_pairs/annual.csv) · [Pair attribution](four_pairs/attribution.csv) · [Selected configurations](four_pairs/selected_configs.csv) · [Standalone payoff correlations](four_pairs/reward_correlations.csv)

## Reproduce

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.run_four_pairs --include-joint --source-root /path/to/wrds
python -m unittest discover -s tests -q
```

The runner needs the local WRDS source files plus `data/current_dataset.csv`. New panels and detailed predictions remain local and ignored under `artifacts/four_pairs/`; no additional market-data files are published. Use `--resume` only to resume this exact experiment's completed local folds. Thirty tests pass, including joint reward moments against direct calculations, cross-pair covariance, shared-budget and cost accounting, all-flat behavior, and fold purging. The AMD/NVDA reference and joint leg-to-total attribution were also reconciled numerically.
