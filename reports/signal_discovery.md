# Actual-contract signal discovery

First-pass result: no tested rule or fitted feature set produced positive average net payoff, either across all evaluable opportunities or the nonoverlapping sensitivity sample. There were 19,310 evaluation opportunities across 113 pairs. This does not establish that no signal exists; it rejects these specific simple mappings of the current features as sufficient evidence of net edge.

Chronological calls/calls signal screen, before the RL policy or profitability-memory gate selects trades. Existing historical pair membership, beta 0.85–1.15 and correlation ≥0.5 filters remain. Calls-only reference features match the contract type. This does not test a broader unscreened universe or independently forecast individual legs.

Quarterly pooled ridge regressions predict the two directional net standalone payoffs. Each fit uses at most five years of observations whose entire outcomes ended strictly before the quarter. Training starts in 2018; evaluation runs from 2020 through July 10, 2025. Equal total training weight per pair; fixed regularization; train-only imputation, clipping and scaling. Choose the direction with the larger prediction only when predicted net payoff is positive. No threshold or feature-set winner is selected on these results. Raw-rule baselines always choose their economic direction, without a fitted cost hurdle. Cash/no trading earns zero excess payoff.

Labels include the existing option-entry penalty, commissions, daily delta hedging, stock execution, borrow and financing through expiry. Numbers below are net dollars per initial mean leg vega, not capital returns or portfolio Sharpe. Vega is matched initially. Standalone labels omit overlapping-book capacity limits.

| Model | Trades | Net / trade | Net / opportunity | Nonoverlap net / trade | Direction rank correlation |
|---|---:|---:|---:|---:|---:|
| raw_iv_spread | 19,310 | -0.643 | -0.643 | -0.809 | 0.082 |
| raw_rv_value | 19,310 | -0.883 | -0.883 | -0.593 | 0.049 |
| rv_value_only | 11,611 | -0.860 | -0.517 | -1.128 | 0.035 |
| spread_only | 8,431 | -0.540 | -0.236 | -0.129 | 0.052 |
| training_mean | 11,005 | -0.814 | -0.464 | -0.840 | 0.016 |
| value | 13,214 | -0.694 | -0.475 | -1.092 | 0.065 |
| value_context | 15,262 | -0.861 | -0.681 | -0.349 | 0.033 |
| value_context_surface | 15,271 | -0.818 | -0.647 | -0.834 | 0.038 |
| value_context_surface_memory | 15,693 | -0.761 | -0.618 | -0.941 | 0.044 |

Source coverage: 26,334 resolved calls/calls opportunities out of 28,884 screened pair-dates. Evaluation is conditional on both directional labels resolving; this is future outcome availability, not an executable filter. Missing histories can bias this sample. All models share the same evaluable dates.

This is repeatedly explored research history, not an untouched holdout. Outcomes overlap and pairs share stocks. Nonoverlap rows thin each pair by outcome end date as a sensitivity diagnostic; they do not eliminate cross-pair dependence or constitute an executable scheduling rule. Earnings features retain the old 31-day cap, so they cannot accurately represent all expiry-specific event exposure. Surface marks remain candidate state variables only, never executable premiums. No annualized Sharpe is inferred from terminal cohort payoffs.

[Annual diagnostics](signal_discovery/annual.csv) · [Pair diagnostics](signal_discovery/pairs.csv) · [Fit timing audit](signal_discovery/fit_audit.csv) · [Original IV reconciliation](expiry_signal_gap.md)

The next distinct experiment should predict each underlying’s realized variance over its selected contract’s actual horizon, then map that forecast to delta-hedged contract P&L and compare legs. The current IV–RV baseline uses trailing RV directly; it is not a fitted forward-variance forecast. Expiry-specific earnings exposure must replace the capped event feature before testing that extension. A promising forecast must first survive a simple fixed-size strategy replay before adding RL sizing.

Validation: 154 quarterly model fits passed strict completed-outcome timing checks; three unit tests cover purging, held-out-label independence, train-only transformations, and flat-payoff accounting. Peak worker RSS was 0.49 GiB, with one worker and a 50% duty-cycle cap.
