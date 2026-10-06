# Q sizing on the original 30-day volatility proxy

Status: complete. Run: `artifacts/q_sizing/20261005T060028513720Z`.

3,922 entry dates, 2010-01-04–2025-08-29. AMD/NVDA only. All Sharpes use sqrt(365/30) on net entry-cohort proxy payoffs, including flat dates. These are not actual option-contract or daily NAV returns.

**Result:** original CEM 0.735; plain entry action values 0.838; validation-selected entry utility 0.859; sequential Q -0.281 versus constrained CEM 0.636.

Entry improvement is +0.123, with paired 95% interval [-0.122, 0.370]. The interval includes zero, so a reliable advantage is not established. The better entry point estimate comes from a different action-value/utility rule; it remains a one-decision bandit control. The genuinely sequential capacity-control version did not improve performance.

| Model | Payoff Sharpe | Nonzero entries | Average size |
|---|---:|---:|---:|
| cem | 0.735 | 3055 | 0.455 |
| entry_q | 0.859 | 3739 | 0.677 |
| entry_q_plain | 0.838 | 3846 | 0.981 |
| sequential_q | -0.281 | 984 | 0.251 |
| sequential_q_plain | -0.081 | 82 | 0.005 |
| same_budget_cem | 0.564 | 2118 | 0.198 |
| constrained_cem | 0.636 | 2838 | 0.248 |
| cash | 0.000 | 0 | 0.000 |
| short_higher_iv | 0.444 | 3922 | 1.000 |
| always_long | -0.329 | 3922 | 1.000 |
| always_short | 0.135 | 3922 | 1.000 |

## Settings and interpretation

The CEM reference reproduces the original five inputs, six tree configurations, five search seeds, nine signed size choices, outcome purging, two-year validation/test blocks, and fixed 0.5-point cost. Historical payoff Sharpe: 0.735323; rerun difference: +0.000000.

Entry Q is a terminal action-value control, not sequential learning. It uses the same fitted tree partitions and estimates mean/second-moment utility per action. Risk weights 0, 0.1, 0.5 are selected on preceding validation Sharpe; the training-only payoff SD supplies their scale. CEM has six validation candidates; entry Q has eighteen. Plain Q and risk-adjusted Q are reported separately.

Sequential Q uses capacities [1.0, 5.0, 10.0] full-size equivalents; values are selected on validation only. Existing positions reserve absolute initial vega until their exact 30-calendar-day maturity. Costs are paid at entry, gross rewards at maturity, and final tickets settle during runoff. Entry-cohort reporting exactly reconciles to these settlement rewards. No early exits or synthetic option marks.

Tree-based fitted Q iteration learns nine action values for each of the three risk weights together. The inputs include the five current signals, current IV levels, available capacity, and a 30-slot reservation schedule with signed allocations and observable entry-IV/RV context. Forward RV, terminal rewards, raw dates and fold IDs are excluded. The source CSV lacks the complete intratrade return path, so this is partially observed, not an exactly sufficient Markov state. Counterfactual training rows add coverage, not independent market days.

FQI: 32 sweeps, gamma 0.98, 12 single-worker extra trees, depth 8, minimum leaf 64. It optimizes discounted expected utility, not exact Sharpe. The smaller-budget constrained CEM control is refitted on the reference market partition, using 64 policies and 24 search iterations plus the original policy as incumbent. It sees capacity through execution clipping, but does not use Q's inventory features. The sequential comparison therefore mixes learning method, representation, objective and search budget; it is not a pure algorithm ablation.

The 32-sweep FQI budget is fixed rather than a convergence claim; final mean target changes are saved per fold. Episode books restart flat at each split, with separate maturity runoff. Earnings dates retain the original known-at-entry assumption rather than a new point-in-time audit.

## Validation selection and uncertainty

- 2010: entry Q {'max_leaves': 4, 'min_leaf': 252, 'risk': 0.5}; sequential Q {'budget': 1.0, 'risk': 0.5, 'risk_index': 2, 'validation': 0.2907817118799215}; constrained CEM {'budget': 10.0, 'validation': 1.683264797120259, 'training_sharpe': 1.172853958623787, 'population': 64, 'iterations': 24}.
- 2012: entry Q {'max_leaves': 16, 'min_leaf': 126, 'risk': 0.0}; sequential Q {'budget': 10.0, 'risk': 0.5, 'risk_index': 2, 'validation': 0.574396969945711}; constrained CEM {'budget': 10.0, 'validation': 0.6200960555147857, 'training_sharpe': 1.290494933734239, 'population': 64, 'iterations': 24}.
- 2014: entry Q {'max_leaves': 16, 'min_leaf': 126, 'risk': 0.0}; sequential Q {'budget': 1.0, 'risk': 0.5, 'risk_index': 2, 'validation': 0.0}; constrained CEM {'budget': 5.0, 'validation': -0.16490875681321654, 'training_sharpe': 1.125862302390777, 'population': 64, 'iterations': 24}.
- 2016: entry Q {'max_leaves': 4, 'min_leaf': 126, 'risk': 0.0}; sequential Q {'budget': 10.0, 'risk': 0.5, 'risk_index': 2, 'validation': 1.012353570500732}; constrained CEM {'budget': 10.0, 'validation': 2.157520720697095, 'training_sharpe': 1.2060201069177585, 'population': 64, 'iterations': 24}.
- 2018: entry Q {'max_leaves': 16, 'min_leaf': 126, 'risk': 0.5}; sequential Q {'budget': 10.0, 'risk': 0.5, 'risk_index': 2, 'validation': 0.14617042796578658}; constrained CEM {'budget': 5.0, 'validation': 1.2500314870274953, 'training_sharpe': 1.3628976098737728, 'population': 64, 'iterations': 24}.
- 2020: entry Q {'max_leaves': 16, 'min_leaf': 252, 'risk': 0.5}; sequential Q {'budget': 10.0, 'risk': 0.1, 'risk_index': 1, 'validation': 0.5274573788754936}; constrained CEM {'budget': 10.0, 'validation': 1.1191955705526817, 'training_sharpe': 1.334334048057628, 'population': 64, 'iterations': 24}.
- 2022: entry Q {'max_leaves': 4, 'min_leaf': 126, 'risk': 0.0}; sequential Q {'budget': 10.0, 'risk': 0.5, 'risk_index': 2, 'validation': 0.7069277263332391}; constrained CEM {'budget': 10.0, 'validation': 0.20673230058199685, 'training_sharpe': 1.3333674639515347, 'population': 64, 'iterations': 24}.
- 2024: entry Q {'max_leaves': 16, 'min_leaf': 252, 'risk': 0.5}; sequential Q {'budget': 1.0, 'risk': 0.5, 'risk_index': 2, 'validation': 0.0}; constrained CEM {'budget': 5.0, 'validation': 0.5249622993744113, 'training_sharpe': 1.2526621476723396, 'population': 64, 'iterations': 24}.

63-entry paired circular-block bootstrap, 1,000 draws:
- entry_q_minus_cem: 95% interval [-0.122, 0.370].
- sequential_q_minus_constrained_cem: 95% interval [-1.315, -0.516].
- sequential_q_minus_same_budget_cem: 95% interval [-1.223, -0.433].

All historical years have already been inspected. Intervals describe fixed realized policies; they do not account for repeated research, refitting or all selection uncertainty. Risk-shaped utility is never substituted for unshaped payoff in reported Sharpe.

## Yearly results

Payoffs are attributed to entry year; 2025 ends in August. The same frozen two-year-fold models are used.

| Entry year | Original CEM | Entry utility | Sequential Q | Constrained CEM |
|---|---:|---:|---:|---:|
| 2010 | 0.843 | 0.442 | 0.294 | 0.638 |
| 2011 | -0.215 | 0.319 | -0.211 | -0.102 |
| 2012 | -0.047 | -0.374 | -0.520 | 0.220 |
| 2013 | 1.814 | 2.909 | -0.609 | 1.799 |
| 2014 | 2.606 | 2.824 | 0.000 | 1.466 |
| 2015 | 1.650 | 2.167 | 0.000 | 1.439 |
| 2016 | 1.231 | 1.237 | -0.323 | 1.196 |
| 2017 | 1.248 | 1.359 | -0.065 | 1.189 |
| 2018 | 1.302 | 1.279 | -0.492 | 0.922 |
| 2019 | 0.286 | 0.386 | -0.741 | 0.239 |
| 2020 | 0.534 | 0.347 | 0.000 | 0.626 |
| 2021 | -0.598 | -0.644 | 0.000 | -0.621 |
| 2022 | -0.681 | 0.585 | 0.165 | -0.490 |
| 2023 | 0.544 | -0.824 | -0.911 | 0.186 |
| 2024 | 0.006 | -0.076 | 0.000 | 0.105 |
| 2025 | 2.028 | 1.949 | 0.000 | 1.886 |

## Verification

18 focused tests passed. Saved models reproduced all 3,922 test-date decisions across eight folds. Replacing unseen test outcomes left every selected sequential-Q decision unchanged. Ticket settlement, proportional costs, split purging and capacity checks passed.

Training took 39.6 minutes at the configured one-core, 50% duty limit; observed peak RSS was 0.560 GiB. The frozen training source is kept separately from the reporting source snapshot.

## Files and reproduction

`scripts/run_q_sizing.py` contains the full isolated experiment; `tests/test_q_sizing.py` checks payoff reconciliation, causal features, action masks, terminal settlement and deterministic learning.

```powershell
python -m unittest tests.test_q_sizing tests.test_sizing tests.test_walk_forward
python -m scripts.run_q_sizing --smoke
python -m scripts.run_q_sizing
```

Each timestamped ignored run directory keeps settings, source/version hashes, dated actions, validation grids, fold checkpoints and trusted-local model pickles. Only load pickles generated by this experiment. A STOP file in that run directory requests cooperative termination. One CPU core, 50% cooperative duty, one numerical thread and 3 GiB RSS stop. The memory guard runs between operations rather than enforcing a hard allocation ceiling.
