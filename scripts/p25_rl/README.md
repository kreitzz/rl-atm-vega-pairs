# Pair IV/RV trading with fitted Q learning

**Result:** three completed iterations do not establish a reliable improvement over the original threshold proposal strategy. The earlier-selected action-specific capacity Q scores 0.974 versus 0.821 in validation, then 0.802 versus 1.055 in the later period. The highest later Q point estimate is 1.072, with a paired improvement interval spanning zero. Matched immediate-reward controls also leave the benefit of continuation unestablished. Existing strategy configuration remains unchanged.

Read the [short analysis](../../reports/p25_rl_research/Summary.txt), [complete findings](../../reports/p25_rl_research/Findings.txt), or [all results](../../reports/p25_rl_research/results.csv). The selected research model is retained for reproducibility; it is not a demonstrated replacement for the benchmark.

This experiment asks whether sequential allocation can improve the repository's existing IV/RV pair proposals. It preserves their contracts, directions, historical screens and execution rules. Its research target is the opportunity cost of portfolio capacity: an entry changes inventory, risk usage and the ability to admit subsequent proposals.

The motivation comes from the frozen-policy diagnostic. An additive sizing model performed better on later unrestricted standalone trades, but the ranking reversed after actual portfolio admission. This suggests testing allocation inside the funded ledger. It does not establish that a Q learner will improve returns.

## Run and read

From the repository root, with the project's Python environment active:

The standard-library parent starts one resource-limited worker at a time and
freezes earlier selection before later evaluation:

```powershell
python -m scripts.p25_rl.research --phase all
```

To resume one stage, use `--phase validation`, `evaluation` or `report`.
Individual family commands remain available for diagnostics. Complete all
three family validations and `temporal --phase validation`, then run
`summary --phase select` before any family's later `evaluation` phase.

All runners use one numeric worker, one thread, a 20% CPU duty limit and a 384 MiB memory limit. A `STOP` file in the corresponding experiment's artifact directory requests a stop at the next budget check. Do not run numeric workers concurrently.

Read the combined human-readable findings in `reports/p25_rl_research/`. **Its `selection.json` determines the selected implementation using earlier validation only.** Later performance does not choose the strategy. Each family's frozen specification, validation scores, fitted models and full ledgers remain in its separate report/artifact directory.

## Decision and learning

At each completed close with proposals, one portfolio action chooses skip, half sizing or full sizing for that day's proposal batch. The extended portfolio arm can instead admit only pairs absent from the current book. That membership is frozen at the decision close. Original risk and quote checks still apply when orders execute next session.

State inputs summarize current proposal economics and the actual book: stress and reserve usage, pair crowding, shared-name exposure, maturity release, recent volatility, drawdown and vega exposure. This is an aggregate, partially observed representation of the portfolio, not the full contract-level Markov state.

Training uses eight simulated behavior histories on the same historical prices. Fitted Q iteration estimates action values from actual funded rewards and discounted next-state values. Unlike the repository's earlier CEM searches, the update contains a Bellman continuation term. Counterfactual histories supply different inventory states; they do not create independent market observations.

Interval rewards sum daily cash-excess returns between opportunity closes. Risk-shaped arms additionally penalize squared daily returns using a training-only scale. Discounting uses elapsed trading sessions. This utility is not an exact Sharpe objective. All reported performance uses the original unshaped funded returns and costs.

## Three bounded experiment families

The first allocation iteration uses a shared shallow critic. Its declared arms are:

| Arm | Change |
| --- | --- |
| `q_capacity` | Sequential Q with inventory and unshaped rewards |
| `q_risk` | Adds the predeclared risk penalty |
| `q_conservative` | Adds conservative action selection relative to full sizing |
| `q_portfolio` | Adds the unheld-pair portfolio action to the conservative arm |

The second allocation iteration (`action_q.py`) fits separate shallow forests for each action. It tests capacity, risk-shaped and conservative Q plus a myopic control. This addresses an earlier validation diagnostic: the shared critic could ignore its action inputs. It reuses the same eight behavior ledgers and changes the value-function representation. Results go to `reports/p25_action_q/` and `artifacts/contracts/p25_action_q/`.

The third family (`management.py`) learns daily held-position reductions, described below. Across the three families, myopic controls set the discount to zero. The initial allocation family's no-inventory ablation removes book features. Fixed full, fixed half, diversification and congestion rules provide simple controls; archived tree/CEM and threshold alternatives provide historical context. Conservative deployment subtracts ensemble disagreement from action advantage; it retains the ordinary max-Q training backup. This is a model heuristic, not a confidence interval.

`temporal.py` adds two matched, unshaped myopic controls for the unshaped
action-specific allocation and held-position Q arms. Only the discount changes
to zero; features, action choices, rewards and critic architecture stay fixed.
These controls test whether continuation adds value and are ineligible for
selection. Their separate outputs use `p25_temporal_q` directories.

## Chronology and selection

Initial Q fits use transitions ending strictly before 2020. Validation covers 2020–2021 in an unbroken portfolio replay, with fixed full sizing before 2020. Every validation arm stops new proposals after October 2021 and runs inventory off through December. Partial histories at a fit cutoff remain nonterminal; their next-state values can require approximation near that boundary.

The joint selection considers nine declared Q candidates: four shared-critic allocation arms, three action-specific allocation arms and two held-position arms. Myopic and no-inventory controls are ineligible. A candidate must have at least 20 validation entries, nonnegative cash-excess profit in both validation years, and validation Sharpe exceeding fixed full sizing by more than 0.01. Otherwise the threshold proposal strategy with full sizing remains selected.

Final models refit on transitions ending before 2022. Evaluation runs through August 2025, carrying the original fixed-full portfolio's actual December 2021 inventory and NAV into 2022. Every frozen arm is reported, including failures. No later result retunes a model or changes the validation selection.

The final entry fits have 229 distinct decision dates. Simulated behavior histories produce 1,431 supported transitions for the three sizing actions, but share those market dates. The daily management experiment has 1,007 distinct training dates. More simulated histories cannot supply missing market regimes. The approximate book state and noisy off-policy continuation values are plausible explanations for weak generalization; the results do not isolate their individual effects.

Risk-shaped management made 251 later entries versus the benchmark's 75, repeatedly closing while fixed proposals replenished the book. Saved option commissions and execution penalties rose to $52,189 versus $8,075, while reserve use fell. This documents expensive turnover alongside lost performance. Changes in holding duration and admission also affect returns, so the loss cannot be attributed entirely to fees.

## Interpretation and checks

The study has 608 inherited proposals on 495 distinct dates. Training replicas share market shocks, and the validation period is short. All evaluation years have already been researched. Block intervals describe uncertainty in the realized comparisons; they do not make this a fresh holdout or erase earlier strategy selection.

The instrumented engine preserves the original accounting function and supplies detached completed-close snapshots. Baseline reproduction, reward reconciliation and continuous handoff checks guard against an apparent gain caused by accounting or timing changes. Costs, quote fallbacks, universe construction and option-assignment assumptions remain inherited research limitations. Existing strategy source, saved models and result snapshots are preserved.

The completed study passed 31 focused tests, exact baseline reproduction, policy/model reload checks, causal training cutoffs, funded accounting reconciliation and both block-bootstrap comparisons. `completion.json` records report verification and output hashes. Full-history baseline intervals have positive lower bounds; later-period significance depends on block length. Neither result establishes that the baseline is uniquely significant or optimal.

An additional mechanical diagnostic checks whether opposing accepted entry orders in the same option can save fees by netting across pairs:

```powershell
python -m scripts.p25_rl.netting
```

It finds zero opposing contract groups and zero savings in this baseline. It changes no orders and is not a new funded strategy test. Its outputs are separate from the Q-learning findings.

## Separate daily held-position experiment

```powershell
python -m scripts.p25_rl.management --phase behavior
python -m scripts.p25_rl.management --phase validation
```

This distinct Q experiment preserves full-size entries and learns daily hold, retain half of original quantities, or close the held book. Reductions execute next session; positions never re-expand. State inputs include current versus entry IV, trailing 20/60-half-life realized variance carry, remaining maturity, liquidation cost, crowding and actual portfolio risk. Flat daily states retain continuation value because future fixed proposals can still arrive.

Four fixed behavior histories support an unshaped Q arm, a risk-shaped Q arm and a myopic control. Separate shallow forests estimate each action's value, ensuring that a shared state partition cannot discard every action input. The action support counts remain explicit. Training/validation timing and October 2021 runoff match the entry experiment. Its isolated outputs are `reports/p25_inventory_q_management/` and `artifacts/contracts/p25_inventory_q_management/`. The combined selection across all three families is frozen using earlier validation before any family's later evaluation. Use the command sequence above to preserve that order.
