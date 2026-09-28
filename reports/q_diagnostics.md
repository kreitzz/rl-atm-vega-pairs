# Why the Q-learner isn't doing better

These are diagnostics for the two spread-feature Q-learners in [q_learning.md](q_learning.md): the plain Q-learner, and the Q-learner that also sees whether its open trade is winning (Q+P&L). Both pairs are covered, over 2010–2024.

`scripts/diagnose_q_learning.py` reruns the same walk-forward fits and records a trade log, the per-seed Q-tables, the whole validation grid and state-visit counts. The trade log reconciles exactly to the daily P&L. It adapts three existing repo diagnostics to the Q-learner's daily-position format:
- fold generalization and gross-vs-cost (`diagnose_independent_pairs.py`)
- random-direction controls (`test_direction_controls.py`)
- leg attribution (`report_positions.py`)

## Summary

| | AMD/NVDA | JPM/BAC |
| --- | --- | --- |
| Are the long/short calls better than random? | **Yes.** Beats 99% of 5,000 random-direction versions of the same trades | **No.** About the 50th percentile, the same as a coin flip |
| Edge before costs, per trade | +1.7 to +1.9 vol pts | −0.4 to 0.0 |
| Cost per trade | 1.2 (costs take about two-thirds of the gross) | 0.6 |
| Does training find a signal? | Yes (training Sharpe 0.4–1.1) | Mostly no (training Sharpe near 0 in 6 of 8 folds) |
| Main problem | Noise and instability: thin states, seed variation, a few losing trades held to expiry | No signal to learn |

**AMD/NVDA has a real edge, and the Q-learner finds it,** but its results are noisy. **JPM/BAC has nothing for the Q-learner to learn,** so no tuning of the learner will fix it. That pair needs better inputs, or should be dropped.

## Detail

### 1. Overfitting: train vs validation vs test Sharpe

| Fold | AMD Q+P&L train / val / test | JPM Q+P&L train / val / test |
| --- | --- | --- |
| 2010 | 0.48 / 0.54 / 0.03 | 1.39 / 0.53 / −1.20 |
| 2012 | 1.04 / 0.66 / −0.11 | 0.00 / 0.00 / 0.00 |
| 2014 | 1.11 / −0.30 / 0.70 | 0.03 / 0.21 / −0.27 |
| 2016 | 0.56 / 0.00 / 0.00 | 0.45 / 1.90 / 0.44 |
| 2018 | 0.42 / 0.90 / 0.78 | −0.50 / 0.70 / −0.08 |
| 2020 | 0.37 / 0.39 / −0.75 | −0.03 / 0.68 / 0.72 |
| 2022 | −0.16 / 0.54 / 0.27 | −0.01 / 0.72 / 1.44 |
| 2024 | 0.68 / 0.45 / 0.86 | 0.06 / 1.38 / −0.47 |

- **AMD/NVDA:** training finds a signal that only partly carries over to test. This is moderate overfitting.
- **JPM/BAC:** training Sharpe is about zero in most folds. Its high validation scores come from luck, not from learning. Larger trees would not help either pair.

### 2. Data per state

| | AMD Q | AMD Q+P&L | JPM Q | JPM Q+P&L |
| --- | ---: | ---: | ---: | ---: |
| States never visited in training | 21% | 27% | 50% | 42% |
| Median training days per visited state | 31 | 14 | 91 | 21 |
| Visited states with fewer than 30 days | 49% | 63% | 36% | 52% |

Adding the P&L flag roughly halves the data behind each state. Many Q-values rest on fewer than 30 days of heavily overlapping history, so they are mostly noise.

### 3 and 4. Edge, costs and exits

| | AMD Q | AMD Q+P&L | JPM Q | JPM Q+P&L |
| --- | ---: | ---: | ---: | ---: |
| Trades | 238 | 229 | 109 | 196 |
| Mean gross / cost / net per trade | 1.71 / 1.21 / 0.50 | 1.87 / 1.18 / 0.69 | −0.43 / 0.66 / −1.09 | −0.04 / 0.61 / −0.65 |
| Gross win rate | 63% | 74% | 64% | 78% |
| Median days held | 11 | 8 | 11 | 3 |
| Share exited early by the agent | 88% | 90% | 78% | 87% |
| Early exits beat holding to day 21 | 55% | 51% | 47% | 48% |
| Total gain from exiting early vs holding | +314 | +285 | +96 | +30 |

- **Early exits help on average,** mostly by cutting trades that would have turned bad. But they beat holding only about half the time, so the gain comes from a few large avoided losses.
- **The trades the agent does not exit are the problem for AMD Q+P&L.** Its 18 trades held to day 21 average −14.7 net, about −264 in total, which wipes out much of the +420 from the 207 early exits. These are losing trades it rode to expiry: the "late and losing" states are among the thinnest in the Q-table.
- **JPM/BAC wins often but loses on average** (a 78% gross win rate with a negative mean gross). Many small gains are outweighed by a few large losses, the pattern of a short-vol exposure in the tails.

### 5. Seed stability (test Sharpe per fold)

| Fold | AMD Q ensemble (seed min, max) | AMD Q+P&L ensemble (seed min, max) |
| --- | --- | --- |
| 2012 | −0.47 (−3.06, −0.47) | −0.11 (−2.41, −0.04) |
| 2024 | 0.69 (−4.25, 0.34) | 0.86 (−1.53, 1.41) |

The typical standard deviation across seeds is 0.4–1.0 Sharpe within one fold. A single seed is unusable, and even the 5-seed average is noisy. This explains why small input changes (spread → ratio) swung the results by about ±0.5.

### 6. Does validation pick good settings?

Rank correlation between validation and test Sharpe across the 8-setting grid:

| | AMD Q | AMD Q+P&L | JPM Q | JPM Q+P&L |
| --- | --- | --- | --- | --- |
| Folds with a positive correlation | 7 of 8 | 7 of 8 | 4 of 8 | 4 of 8 |

For AMD/NVDA, validation mostly points the right way. For JPM/BAC it is no better than chance, and in several folds a different setting would have done much better (for example, grid best 0.93 vs selected 0.00 in 2012). Two years of validation is too short to choose settings for a pair with no signal.

### 7. Where the P&L comes from

- **AMD/NVDA:** almost all gross P&L comes from the AMD leg (for example +447 AMD vs −20 NVDA for Q+P&L), and both directions are profitable. 2019 alone is about a third of the plain Q-learner's P&L.
- **JPM/BAC:** 2010–11 loses about 150 of the roughly 120 total loss; the other years net slightly positive, which is an after-the-fact observation, not a rule. Long-JPM/short-BAC trades lose, mainly on the short-BAC-vol leg (BAC vol spikes).

## What this implies

1. **Don't enlarge the tree or the state.** States are already data-starved, and JPM/BAC's problem isn't resolution.
2. **Reduce noise first.** Options:
   - fewer states: drop the early/late age split, or merge the P&L flag into position only
   - more seeds, averaged (for example 20)
   - fixed settings instead of per-fold validation selection
3. **Fix held losing trades.** A stop-loss (a forced exit when the open-trade loss exceeds a threshold, set on training data) would directly target AMD's −264 from trades held to expiry. Alternatively, pool the "late" states so they have enough data.
4. **Make costs matter more in learning.** Costs take about two-thirds of AMD's gross edge, so a minimum hold or a higher entry bar would trade less.
5. **JPM/BAC:** the learner isn't the bottleneck. Look for inputs with a real signal first, or treat the pair as having no exploitable edge.

## Reproduce

```sh
python -m scripts.diagnose_q_learning --pair AMD NVDA
python -m scripts.diagnose_q_learning --pair JPM BAC
```

Trade logs and grids stay in the ignored `artifacts/q_diagnostics/`. Aggregates are in [q_diagnostics/](q_diagnostics/).
