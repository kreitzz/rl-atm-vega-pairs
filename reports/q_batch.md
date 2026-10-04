# Q-learning revision: state information, exhaustive updates, and data gaps

The revised learner improves AMD/NVDA's point estimate, but does **not** establish a reliable advantage over cross-entropy policy search. Better economic state inputs look more useful than simply adding seeds or holding-age buckets. JPM/BAC remains negative. All results are exploratory: these test years have already influenced multiple rounds of research.

## What I disagree with in the existing approach

1. **Its state is not sufficient for its own reward formula.** `Market.step` depends on both legs' entry IV, current IV, and exact holding age. The original tree sees spread, earnings countdowns, and trailing RV, followed by a coarse age bucket. Two observations in the same Q state can therefore represent materially different trades. A winning/losing flag does not resolve this.
2. **Exploration needlessly determines which inventory states get trained.** Updating every action at a visited state still misses positions and holding ages that the sampled trajectory never reaches. In this price-taking simulator, outcomes for other feasible entry dates and directions are available. We can pool all of them deterministically.
3. **Training omits some terminal liquidation costs.** In `train_q`, `if last: target = r` omits the final exit charge for positions that have not matured. `simulate` does charge it. The new implementation includes it and stops bootstrapping at episode end. This is a correctness fix, not evidence that it explains the performance gap.
4. **The objectives and position books differ.** Q maximizes discounted expected P&L; cross-entropy searches for payoff Sharpe. Q holds one full-sized position with early exits; the reference opens overlapping, sized cohorts. Even common daily scoring cannot isolate the learning algorithm from these differences.
5. **The shared panel silently compresses missing dates.** AMD/NVDA has 18 missing interior dates relative to available pair-return dates: 2007-09-11 and 17 trading dates from 2020-07-27 through 2020-08-18. Row-to-row IV changes can span a gap while the stored next return covers only one day. This warrants a separate data treatment, not another hyperparameter.

The earlier diagnostics' claims that AMD has a “real edge” and JPM has “nothing to learn” are stronger than this evidence supports. The results show instability and negative estimates, respectively; they do not prove either proposition. Randomizing directions while keeping exits chosen by the original policy fixed is also a conditional diagnostic, not a complete strategy-level significance test.

## What to show or hide from Q

| Information | Recommendation | Reason |
| --- | --- | --- |
| Forward RV, next-day returns, full-hold reward, outcome date | Hide from action inputs | Outcomes belong in training rewards/targets only. The tree's supervised target is allowed only after it has matured within training. |
| Position direction and remaining life | Keep | These determine legal actions and reward mechanics. Exact age is conceptually useful, although its isolated ablation did not improve performance. |
| Current IV level | Keep | Spread alone loses information about cost and variance-carry scale. |
| Trailing RV for each leg | Compress initially | Test a relative variance-carry feature instead of asking a small tree to infer the nonlinear combination of four raw levels. |
| Entry IV for each leg / current mark relative to entry | Keep compact context; improve this next | The new sign-of-entry-spread-displacement flag helps in some runs. It is still only an approximation to the full entry-IV state. |
| Cumulative trade P&L or “winning/losing” | Omit from the initial compact model | Sunk carry is not itself a reason to exit. This flag fragments states and mixes carry with the economically relevant IV mark. It can matter if the objective explicitly includes drawdown or capital constraints. |
| Earnings countdown | Separate ablation | Actual historical announcement dates are not necessarily the dates known beforehand. Only promote this input after a point-in-time calendar audit. |
| Raw date, year, fold ID, future regime labels | Hide | They invite memorization or leakage rather than a deployable signal. |

The economic inputs tested here are spread, average IV, and

`relative_carry = (RV_A_trailing² - IV_A²)/(2 IV_A) - (RV_B_trailing² - IV_B²)/(2 IV_B)`.

This is an observable trailing proxy, not a forecast of future RV. It matches the structure of the simulator's variance-carry payoff more closely than spread alone. Earnings are added only in the earnings ablation. No new external market signal was introduced.

## Implemented experiment

[`run_q_batch.py`](../scripts/run_q_batch.py) constructs an empirical transition/reward table from every feasible date, entry age, direction, and legal action. It performs synchronous Bellman updates to convergence using sparse matrices. This is fitted tabular Q iteration, not cross-entropy search, and has no exploration seed or constant learning-rate noise.

The state is a training-fitted tree leaf times inventory state. Inventory is either the original early/late split, exact ages 1–20, or exact ages plus a two-way signed entry-spread-displacement flag. Entry is age zero and maturity closes the position after 21 intervals. Trees retain minimum leaf size 252. Each variant selects 4/8 leaves and gamma 0.90/0.98 on validation, then a separately reported selector chooses among the five variants on validation alone. Ties keep the first candidate. The 45-calendar-day training purge, original folds, costs, and marked-P&L formula are retained.

There are five variants and four configurations each: 20 validation candidates per fold. This is more selection freedom than the legacy Q grid; it can overfit validation. Test statistics are never used in code to select a candidate. Cash is not an extra validation candidate in this experiment, so a negative-validation policy can still be selected.

Exhaustive counterfactual rows are **not independent new observations**. Coarse states also mix several ages and entry IVs. Convergence of this aggregated Bellman operator does not prove that the aggregated state is Markov or that its policy generalizes.

## Results on the existing panel treatment

Annualized daily marked-P&L Sharpe, 2010–2024, net of the existing cost rule:

| Variant | AMD/NVDA | JPM/BAC |
| --- | ---: | ---: |
| Legacy Q+P&L, published reference | 0.238 | -0.321 |
| Cross-entropy, published reference | 0.250 | -0.370 |
| Exhaustive Q, original inputs/coarse age | 0.114 | -0.392 |
| + exact age, original inputs | -0.015 | -0.235 |
| Economic inputs, exact age | 0.257 | -0.440 |
| Economic inputs + earnings | 0.446 | -0.440 |
| Economic inputs + entry context | 0.329 | -0.203 |
| **Variant selected on validation each fold** | **0.296** | **-0.093** |

The AMD cross-entropy reference was rerun and reproduced 0.2496. The paired 63-day circular block-bootstrap interval for the selected Q minus cross-entropy Sharpe is **[-0.347, 0.569]**, around a difference of **0.047** (2,000 draws). The legacy Q reference was reproduced on the first two folds; remaining legacy Q and JPM reference figures are taken from the saved report summaries, not claimed as fresh reruns.

The 0.446 earnings result is an after-the-fact best variant, not the result of the validation selector. Its own 95% interval is [-0.079, 0.893]. It should not replace the selected result in a headline.

## Gap-boundary sensitivity (default for the new script)

The gap-aware run splits training and evaluation into contiguous episodes, removes full-hold labels crossing a gap, charges liquidation at the final observed close, and prohibits bootstrapping across the boundary. It does not manufacture missing prices or returns. This is a research sensitivity test: identifying a future data outage and liquidating immediately before it is not a live trading rule. Complete data are still needed for a deployable backtest.

| Variant | AMD/NVDA | JPM/BAC |
| --- | ---: | ---: |
| Original inputs/coarse age | 0.091 | -0.392 |
| Original inputs/exact age | -0.127 | -0.235 |
| Economic inputs | 0.211 | -0.449 |
| Economic inputs + earnings | 0.311 | -0.449 |
| Economic inputs + entry context | 0.355 | -0.253 |
| **Validation-selected** | **0.371** | **-0.141** |

For the selected AMD policy, the 95% Sharpe interval is **[-0.084, 0.779]**, max drawdown is **-166.7** vol-point P&L units, and there are **229** round trips. For JPM, the interval is **[-0.526, 0.295]**. These are not statistically reliable positive Sharpes. Boundary treatment changes training labels and validation selection as well as test accounting, so differences between tables are not a pure estimate of gap-related P&L.

Rerunning cross-entropy with the same gap-boundary treatment gives AMD Sharpe **0.224**. The selected Q advantage is **0.147**, with paired 95% interval **[-0.265, 0.694]**. This comparison also fails to establish a reliable advantage. Bootstrap intervals describe these fixed realized policies; they do not account for all prior research choices or re-fit models inside each draw.

A final **spread-only** control, with exact age and the same gap treatment, scored **0.049** for AMD and **-0.466** for JPM. It was run after the five-variant experiment and was not added retrospectively to its selector. This supports investigating economic inputs rather than stripping the state down to spread alone, without proving those inputs have a stable predictive edge.

## What I would do next

First repair the missing-date history and verify point-in-time earnings inputs. Then freeze the compact economic/entry-context representation for a genuinely new evaluation period. Do not select the earnings variant merely because its already-viewed test Sharpe is highest.

For a cleaner algorithm comparison, use the same position book and net-P&L objective for Q and cross-entropy. A useful next design is entry-only, fixed-horizon Q with the same cohort sizing, followed by a separate early-exit experiment. This separates direction/sizing quality from the much harder inventory-control problem. If full sequential control is retained, pool continuous entry-IV and age information with regularized function approximation instead of repeatedly multiplying table states.

Neither “more seeds” nor a stop-loss is the main recommendation. The actionable change from these experiments is to improve economic state information and data integrity, while keeping the uncertainty visible. The new model is a research candidate, not a demonstrated replacement for cross-entropy.

## Reproduction and checks

```powershell
conda activate xfund
python -m scripts.run_q_batch --pair AMD NVDA
python -m scripts.run_q_batch --pair JPM BAC
python -m scripts.run_q_batch --pair AMD NVDA --legacy-gaps
python -m scripts.run_q_batch --pair JPM BAC --legacy-gaps
python -m scripts.compare_q_batch --pair AMD NVDA --references current_cem
python -m scripts.compare_q_batch --pair AMD NVDA --references current_cem --respect-gaps --out artifacts/q_batch_strict
python -m scripts.run_q_batch --pair AMD NVDA --variants spread_only --out artifacts/q_batch_spread_strict
python -m scripts.run_q_batch --pair JPM BAC --variants spread_only --out artifacts/q_batch_spread_strict
python -m unittest tests.test_q_batch tests.test_first_pass tests.test_walk_forward tests.test_rv_vix
```

Seven new tests cover terminal liquidation, expiry, reconciliation with the existing simulator, feature allowlisting, deterministic exhaustive state coverage, gap boundaries, and exclusion of labels crossing gaps. The focused suite passes 17 tests. Each experiment saves daily P&L, validation grids, selected configurations, and model snapshots under `artifacts/q_batch/`, `artifacts/q_batch_strict/`, or `artifacts/q_batch_spread_strict/`. Compact result snapshots are retained in [q_batch/](q_batch/). Existing Q scripts and reports were preserved.
