# Broad universe: does prior profitability help RL allocate its pair budget?

The policy, rather than an external profit-ranking rule, decides whether each pair deserves a trade and how much of its fixed budget to use. The new state summarizes completed out-of-sample trades from a reference RL strategy for that same pair. A poorly performing history can lead to flat or smaller positions, but no such behavior is imposed.

The input-only historical screens produce 743 distinct pairs across 2008–24. The 2018–August 2025 experiment considers 350 unique pairs in 563 pair/fold combinations; 308 pairs have at least one eligible fitted evaluation. The candidate roster changes every two years.

**Finding:** the added profit-history state does not improve the broad basket: sized RL falls from 2.788 to 2.606 annualized proxy Sharpe. Its change interval includes zero. Nor does it consistently allocate more after profits: mean size is about 42.0% after positive history versus 42.4% after negative history; only 101 of 230 pairs observed in both regimes have larger average size after profits. These are descriptive associations, not proof that historical profitability cannot be useful.

## Matched annualized proxy Sharpe

| Basket | RL without profit state | NN full-size/flat + profit state | RL full-size/flat + profit state | RL sizing + profit state |
| --- | --- | --- | --- | --- |
| all | 2.788 | 1.519 | 2.647 | 2.606 |
| without_metals | 2.568 | 1.264 | 2.504 | 2.397 |

`without_metals` is a concentration sensitivity excluding SIC groups 104, 106, 109 and SIL/stock pairs; it is not an optimized pair-selection rule or an exclusion of every mining business. Pair budgets are equal within each fold. Insufficient prior model history and unfilled allocation remain cash, with no reweighting to the remaining pairs.

## Extreme-mark sensitivity

| Model | Baseline Sharpe | Skip IV >100 Sharpe | Net P&L | Net P&L from IV >100 | Extreme-date profit fraction |
| --- | --- | --- | --- | --- | --- |
| rl_no_memory | 2.788 | 1.853 | 2100.052 | 1396.417 | 0.665 |
| nn_memory | 1.519 | -0.497 | 1852.069 | 2126.204 | 1.148 |
| rl_memory_fixed | 2.647 | 1.673 | 3237.458 | 2396.397 | 0.740 |
| rl_memory | 2.606 | 1.520 | 1773.713 | 1193.964 | 0.673 |

This diagnostic suppresses an entry if either leg's standardized IV exceeds 100 vol, retains idle budget, and uses the same 1,805 complete-outcome dates. It is not a newly adopted or independently validated rule. High IV can be real; this flags reliance on extreme marks for actual-contract quote checks. Contributions use the actual per-fold equal pair budgets. [Weighted pair contributions](broad_pairs/pair_contributions.csv).

## Does RL reduce exposure after losses?

Positive/negative history means the reference strategy's trailing mean net payoff is above/below zero, with at least 126 completed entry cohorts. Exposure is absolute size, not direction. The entry-weighted table describes behavior; differences can also reflect other changing state variables.

| Model | History state | Pair entries | Trade fraction | Mean absolute size |
| --- | --- | --- | --- | --- |
| nn_memory | negative_history | 94192 | 0.858 | 0.858 |
| nn_memory | positive_history | 131942 | 0.875 | 0.875 |
| nn_memory | short_or_flat_history | 776 | 0.773 | 0.773 |
| rl_memory | negative_history | 94192 | 0.686 | 0.424 |
| rl_memory | positive_history | 131942 | 0.722 | 0.420 |
| rl_memory | short_or_flat_history | 776 | 0.701 | 0.533 |
| rl_memory_fixed | negative_history | 94192 | 0.548 | 0.548 |
| rl_memory_fixed | positive_history | 131942 | 0.577 | 0.577 |
| rl_memory_fixed | short_or_flat_history | 776 | 0.677 | 0.677 |
| rl_no_memory | negative_history | 94192 | 0.648 | 0.389 |
| rl_no_memory | positive_history | 131942 | 0.696 | 0.402 |
| rl_no_memory | short_or_flat_history | 776 | 0.621 | 0.234 |

Within-pair comparisons reduce differences caused solely by which names populate the positive and negative groups:

| Model | Pairs with both states | Pairs sizing more after profits | Median size difference | Mean size difference |
| --- | --- | --- | --- | --- |
| nn_memory | 230 | 102 | -0.008 | -0.010 |
| rl_memory | 230 | 101 | -0.015 | -0.019 |
| rl_memory_fixed | 230 | 94 | 0.000 | -0.017 |
| rl_no_memory | 230 | 107 | -0.008 | -0.022 |

## Annual all-candidate results

| Year | RL without profit state | NN full-size/flat | RL full-size/flat | RL sizing |
| --- | --- | --- | --- | --- |
| 2018 | 1.239 | -0.838 | 1.024 | 1.268 |
| 2019 | 2.007 | 1.246 | 0.886 | 1.622 |
| 2020 | 1.719 | -0.541 | 2.217 | 2.094 |
| 2021 | 1.228 | -1.985 | 1.709 | 1.493 |
| 2022 | 4.283 | 2.201 | 4.044 | 3.800 |
| 2023 | 4.061 | 3.957 | 4.114 | 3.634 |
| 2024 | 6.253 | 4.225 | 5.438 | 6.056 |
| 2025 | 7.342 | 3.966 | 7.102 | 6.861 |

2025 is partial through August. Sharpe uses sqrt(365/30) on net entry-cohort proxy payoffs, including flat entries. It is not the Sharpe of a daily marked option portfolio.

## Paired Sharpe differences

| Basket | Compared with | RL-memory Sharpe change | 95% low | 95% high |
| --- | --- | --- | --- | --- |
| all | rl_no_memory | -0.182 | -0.445 | 0.094 |
| all | nn_memory | 1.087 | 0.650 | 1.588 |
| all | rl_memory_fixed | -0.041 | -0.349 | 0.255 |
| without_metals | rl_no_memory | -0.170 | -0.457 | 0.156 |
| without_metals | nn_memory | 1.134 | 0.648 | 1.708 |
| without_metals | rl_memory_fixed | -0.107 | -0.400 | 0.237 |

Intervals use 2,000 paired circular block resamples of 63 entry dates. They do not correct for model selection, repeated research on this history, changing universe construction, or missing outcomes.

## Highest full-period fitted scores, for diagnosis only

| Pair | Entries | RL without profit state | NN full-size/flat | RL full-size/flat | RL sizing |
| --- | --- | --- | --- | --- | --- |
| DVN/SWN | 147 | 5.730 | 5.692 | 6.262 | 6.352 |
| SWN/AR | 147 | 5.522 | 5.525 | 6.005 | 6.005 |
| INFN/COMM | 262 | 2.579 | 2.412 | 3.216 | 3.144 |
| PSEC/KKR | 417 | 3.140 | 1.705 | 2.089 | 3.110 |
| GS/WETF | 489 | 2.553 | 2.498 | 2.789 | 2.920 |
| BLK/WETF | 489 | 2.798 | 1.809 | 2.554 | 2.798 |
| KGC/NEM | 417 | 2.886 | 0.419 | 2.539 | 2.759 |
| WETF/JEF | 488 | 2.641 | 0.017 | 2.666 | 2.728 |
| FCEL/RUN | 274 | 2.725 | 1.714 | 2.972 | 2.725 |
| AEM/KGC | 417 | 2.753 | 0.868 | 2.546 | 2.665 |

This retrospective table is not the universe selector and requires at least 126 scored entry dates for display. Different pairs enter different folds and have different sample lengths. The model does not receive full-period profitability or the list above. Per-pair and annual Sharpes with fewer than two scored observations are left undefined in the CSVs.

## Universe and timing

- At each two-year boundary, use the preceding 252 available return sessions. Require at least 200 complete observations of IV, return, trailing RV, earnings state, and call activity; median aggregate call volume at least 100 and median call open interest at least 500.
- Use the latest security-name SIC classification effective strictly before that boundary. Within each three-digit SIC group, choose up to four names by historical median call open interest, deduplicating CRSP identifiers. Form all pairs whose matched trailing return correlation is at least 0.4. Cap at 180 stock pairs by rotating across industries, with higher minimum-leg open interest first within each industry. Add previously specified ETF comparisons only when they pass the same input screen. No profit measure enters this screen.
- CRSP returns are restricted to their OptionMetrics link intervals. Securities with different permanent return identifiers are not spliced. Historical candidates include acquired/delisted names; ticker labels in aggregated pair tables are display labels and can differ across historical screens.
- This remains the locally downloaded universe, not a complete point-in-time U.S. options universe. Current security-type metadata and source-history/earnings coverage influence which assets are available. At preparation, an asset needs 504 IV observations and 20 recorded stock earnings dates. These availability constraints and non-point-in-time earnings-calendar assumptions remain limitations; historical screening does not eliminate all selection bias.

## Profitability state and benchmark

- Reference RL: the seven IV/RV/value inputs from the prior benchmark; a fixed eight-leaf tree, minimum 126 rows; the usual five-seed, 80-iteration, 256-population Sharpe search and 25% sizing ladder. It refits every two years from 2006 using only outcomes completed before the fit date. No in-sample fitted trade rewards enter the profitability state.
- For each new entry, collect the last 252 reference trade cohorts whose 30-day end date is strictly earlier than entry. Four inputs are their mean net payoff, annualized payoff Sharpe, active-trade fraction, and observation count divided by 252. Flat reference trades contribute zero. Unknown history has zero values plus count zero. A zero variance history has Sharpe input zero.
- These are shadow trades from a common reference RL policy, not endogenous feedback from each newly augmented policy's own changing decisions. The augmented RL and NN both observe this same historical track record. Modeling feedback from the newly learned policy itself would require a different sequential training setup.
- Each candidate is trained independently. Compare a seven-input sized RL policy, an eleven-input sized RL policy, its independently validation-selected full-size/flat control, and a full-size/flat NN receiving the same eleven inputs. No joint allocation policy or cross-pair profit-ranking gate is introduced.
- Training expands; the two preceding years are used for model validation, followed by two test years. Thirty-day outcomes are purged at both boundaries. Eligibility requires 504 prior training and 252 prior validation rows, with no minimum future test-length requirement.
- RL uses the established six tree configurations and training-seed selection. The NN has one 16-unit tanh layer, training-only feature/target scaling, L-BFGS maximum 10,000 iterations, regularization 0.01/1, and validation-chosen entry-cost thresholds 1/2/4. NN actions are strictly -1/0/+1; sized RL actions range from -1 to +1 in 0.25 increments.
- Pair payoff and costs are unchanged: action times the difference of forward RV-minus-entry-IV across legs, less absolute size times 2.5% of average entry IV. Initial vega neutrality, daily delta hedging, standardized IV marks, and fixed cohort entry budgets remain assumptions.

## Missing outcomes and checks

The full screened calendar has 1926 entry dates; 1805 have jointly complete portfolio outcomes across all four models. 121 dates are excluded from all comparisons because at least one active candidate trade has no complete forward RV outcome. These missing payoffs are not set to zero. This complete-outcome analysis can still be biased, particularly around delistings or corporate actions, and is not an executable portfolio backtest.

| Basket | Fold | Candidate pairs | Fitted pairs | Calendar dates | Joint complete dates | Unresolved pair dates |
| --- | --- | --- | --- | --- | --- | --- |
| all | 2018 | 103 | 96 | 503 | 466 | 63 |
| without_metals | 2018 | 97 | 90 | 503 | 466 | 63 |
| all | 2020 | 132 | 121 | 505 | 464 | 56 |
| without_metals | 2020 | 128 | 117 | 505 | 464 | 56 |
| all | 2022 | 175 | 149 | 501 | 501 | 0 |
| without_metals | 2022 | 167 | 141 | 501 | 501 | 0 |
| all | 2024 | 153 | 140 | 417 | 374 | 43 |
| without_metals | 2024 | 145 | 132 | 417 | 374 | 43 |

13 of 506 selected NN fits report an optimizer convergence warning. All saved action menus, net reward calculations, reference-history states, and model purge boundaries were audited. Tests cover forward-RV equivalence, strict maturity timing, future-outcome independence, and preservation of unknown active-trade payoffs.

Aggregate activity is across all call strikes/expiries; it does not verify executable ATM 30-day spreads. Extremely high IV marks can drive the proxy. [Extreme-IV contribution](broad_pairs/extreme_iv_attribution.csv) is descriptive, not a new trading rule.

[Candidate screens](broad_pairs/candidate_screens.csv) · [Pair scores](broad_pairs/scores.csv) · [Annual results](broad_pairs/portfolio_annual.csv) · [Exposure by past-profit state](broad_pairs/behavior.csv) · [Within-pair exposure differences](broad_pairs/within_pair_behavior.csv) · [Selected model configurations](broad_pairs/configs.csv)

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.build_broad_pairs --source-root /path/to/wrds
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.run_broad_memory --workers 4 --resume
python -m scripts.report_broad_memory
python -m unittest discover -s tests -q
```

Market panels, reference trades, and predictions stay under ignored `artifacts/broad_pairs/`. The shared market-data CSV is unchanged.
