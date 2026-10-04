# Q-learning for ATM vega pairs: JPM/BAC and AMD/NVDA

**Exploratory.** This asks whether a tabular Q-learner in the style of the stat-arb paper beats our current tree + cross-entropy model. The paper is *Advanced Statistical Arbitrage with Reinforcement Learning*: a Q-table over discretised spread moves, with buy/hold/sell actions. Test years had already been inspected in earlier work. The one design change made after seeing the first test results (the open-trade P&L state) is reported alongside the original, not in place of it.

## Result

No model shows a statistically reliable edge. Every 95% bootstrap interval includes zero.

| Pair | Q hybrid | Q hybrid + open-trade P&L | Paper replication | Threshold rule | Current model |
| --- | ---: | ---: | ---: | ---: | ---: |
| AMD/NVDA | 0.17 | **0.24** | −0.04 | 0.09 | **0.25** |
| JPM/BAC | −0.31 | −0.32 | 0.03 | −0.22 | −0.37 |

Annualized daily Sharpe of marked P&L, 2010–2024, net of costs.

| AMD/NVDA | Q + P&L state | Current model |
| --- | ---: | ---: |
| Sharpe 95% CI (63-day block bootstrap) | [−0.12, 0.66] | [−0.24, 0.58] |
| Annual P&L / vol (vol pts per $1 vega) | 10.6 / 44.3 | 13.2 / 53.0 |
| Max drawdown | −120 | −179 |
| Days invested | 52% | 69% |
| Trades | 229 round trips | 2,592 daily cohort entries |

- **AMD/NVDA:** the Q-learner with open-trade P&L in its state matches the current model's Sharpe, with a smaller drawdown and about a tenth of the entries. The simpler Q hybrid and the paper replication are weaker.
- **JPM/BAC:** the Q-learner does not rescue the pair. Its loss comes mostly from the 2010–11 fold (Sharpe −1.1). From 2012 onward it is flat or positive in most folds, but that is a descriptive observation, not a usable rule.
- **Paper replication:** close to zero on both pairs. Its reward, A·(θ − X), pays for trading toward the mean of the IV spread, not for the realized-vs-implied P&L that drives these positions.

### Fold Sharpes (two-year test blocks)

| Test block | AMD Q | AMD Q+P&L | AMD current | JPM Q | JPM Q+P&L | JPM current |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2010–11 | −0.19 | 0.03 | 0.14 | −1.14 | −1.20 | −1.26 |
| 2012–13 | −0.47 | −0.11 | −0.97 | 0.00 | 0.00 | 0.45 |
| 2014–15 | 0.31 | 0.70 | 0.76 | −0.27 | −0.27 | −1.98 |
| 2016–17 | −0.76 | 0.00 | 1.02 | 1.02 | 0.44 | −0.46 |
| 2018–19 | 1.58 | 0.78 | 0.83 | 0.68 | −0.08 | 0.17 |
| 2020–21 | 0.03 | −0.75 | −0.20 | 0.72 | 0.72 | −0.14 |
| 2022–23 | −0.37 | 0.27 | −0.52 | 0.00 | 1.44 | −0.74 |
| 2024 | 0.68 | 0.86 | −1.03 | 0.00 | −0.47 | 0.00 |

A Sharpe of 0.00 means the selected policy stayed flat for the whole block.

## Method

**Economics.** Each position is long one pair leg's 30-day ATM vega and short the other's, delta-hedged daily, with equal initial vega per leg. Daily P&L per leg (vol points per $1 of initial vega) is:

- **Carry:** 100·(r² − IV₀²/252) / (2·IV₀·21/252). Summed over a full 21-trading-day hold, this equals 100·(RV² − IV₀²)/(2·IV₀) ≈ RV − IV, which is the repo's payoff.
- **IV mark:** remaining vega √((21−k)/21) times the IV change since entry. It disappears at a full hold and applies on early exit.

The full-hold payoff has correlation 0.92 with the repo's original RV − IV label. Positions close automatically after 21 trading days. The cost is the repo's current rule, 2.5% of the average pair IV per round trip, half charged on entry and half on exit.

**Models**

| Model | State | Actions | Learning |
| --- | --- | --- | --- |
| Q hybrid | Tree leaf (≤8, from the 5 inputs) × {flat, long/short × early/late in hold} | Flat: stay, enter long, enter short. In position: hold, exit | Q-learning on daily P&L; ε = 0.1, α = 0.05, 20 epochs, 5 seeds averaged |
| Q hybrid + P&L | As above, plus whether the open trade is winning or losing | Same | Same |
| Paper replication | Last 4 daily IV-spread moves coded ±1/±2 (256) × position | Buy / nothing / sell | Paper settings: α = 0.1, γ = 0.99, ε = 0.1, 10 episodes on AR(1)/OU paths fitted to training |
| Threshold rule | z-score of the IV spread (training mean and sd) | Short the rich leg if \|z\| > 1; exit at z = 0 | None |
| Current model | Tree leaf (≤16) | 9-size ladder, 30-day cohort opened daily | Cross-entropy Sharpe search, as in `run_four_pairs.py` |

- **Why every action gets updated each step.** The agent's trades do not move the market, so every action available in the visited state receives its exact one-step backup. The ε-greedy choice only decides which path is followed.
- **Walk-forward.** There are 8 folds: expanding training, the preceding 2 years for validation, and 2-year test blocks (the last covers 2024 only). The Q-learner's training path ends 45 days before validation starts. For each Q model, validation Sharpe picks the tree leaves (4/8), minimum leaf size (126/252) and γ (0.9/0.98).
- **Selected settings.** AMD/NVDA mostly chose 4–8 leaves and γ = 0.9. JPM/BAC mostly chose 4 leaves.
- **Scoring.** Every model is scored on the same daily marked P&L, annualized with √252. The current model's book opens a 1/21-sized cohort each day, so its exposure is at most 1 unit, like the Q-learner's.

## Checks

- **Panel.** For AMD/NVDA, the panel built from `data/wrds_q/` reproduces `data/current_dataset.csv` exactly (IV spread, trailing and forward RV, earnings countdowns).
- **Current model.** Trained on the original label and scored with the repo's proxy Sharpe, the reimplementation gives 0.47 for AMD/NVDA, against the published 0.64. The published figure includes January–August 2025, the pair's strongest year (1.92), which this data does not cover: CRSP returns end in December 2024.
- **Accounting.** A single simulated trade's P&L equals the full-hold payoff minus cost, and the cohort book gives the same total.

## Caveats

- **The P&L is a proxy.** It comes from constant-maturity IV and equal-weighted variance carry, not daily marked contract prices.
- **Weak evidence either way.** The samples are short, the rewards are noisy, and the test periods have been viewed before, so the differences between models are not statistically meaningful.

## Reproduce

```sh
python scripts/pull_wrds_q_data.py                    # writes data/wrds_q/*.csv (WRDS login)
python -m scripts.run_q_learning --pair JPM BAC
python -m scripts.run_q_learning --pair AMD NVDA
```

Outputs go to the local, ignored `artifacts/q_learning/<pair>/`. Aggregate results are in [q_learning/](q_learning/).

## Experiment 2: IV ratio instead of IV spread

**The single change:** the tree input `spread` (IV_A − IV_B) is replaced by `iv_ratio` (IV_A / IV_B). The other four inputs, the models, costs, folds, validation grid and seeds are unchanged. The spread run is kept as-is and selected with `--features spread` (the default). This variable was chosen after looking at hindsight correlations, including post-2010 data, so this is not a clean out-of-sample test.

| Pair | Model | Sharpe, spread | Sharpe, ratio | Max DD, spread → ratio | Trades, spread → ratio |
| --- | --- | ---: | ---: | --- | --- |
| AMD/NVDA | Q hybrid | 0.17 | −0.32 | −140 → −318 | 238 → 272 |
| AMD/NVDA | Q hybrid + P&L | 0.24 | −0.11 | −120 → −269 | 229 → 315 |
| AMD/NVDA | Current model | 0.25 | 0.24 | −179 → −174 | 2,592 → 2,424 |
| JPM/BAC | Q hybrid | −0.31 | **0.31** | −183 → −32 | 109 → 118 |
| JPM/BAC | Q hybrid + P&L | −0.32 | −0.25 | −199 → −169 | 196 → 99 |
| JPM/BAC | Current model | −0.37 | −0.54 | −85 → −109 | 1,419 → 1,591 |

Sharpe is the annualized daily Sharpe, 2010–2024, net of costs. The paper replication and threshold rule do not use the tree and are unchanged.

**The swap does not help consistently:**
- AMD/NVDA: the Q-learners get worse and the current model is unchanged.
- JPM/BAC: the plain Q-learner turns positive (0.31; CI −0.13 to 0.73), while the other two models get worse.

That JPM/BAC gain comes from trading less: it is invested 13% of days instead of 31%.

Swapping in an almost equivalent input moves the Q-learners' Sharpe by ±0.5. This instability is itself the main finding: the Q-learners are sensitive to small changes in the state definition, so single backtest numbers should not be over-read.

**How the trees use the ratio.** It appears in all 48 selected trees (8 folds × 3 models × 2 pairs). It is the first split in only 6 of the 24 AMD/NVDA trees and in none of the JPM/BAC trees. The trailing-RV and earnings inputs usually take the top split, because the tree minimizes squared error on a payoff dominated by large-move days. Dropping those inputs is the natural next test.

```sh
python -m scripts.run_q_learning --pair JPM BAC --features ratio   # writes artifacts/q_learning_ratio/
```
