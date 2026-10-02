# Signal-rank policy experiments

342/346 pairs; January 2018–August 2025. Calls/calls only, unchanged costs, sizing ladder, beta/correlation filters, monthly walk-forward schedule, validation gate and risk limits. All four new variants share the prior calls-only matched-label training rows and completion cutoffs.

| Variant | Annualized excess Sharpe | P&L above cash | Entries | Active pairs |
|---|---:|---:|---:|---:|
| Original calls-only reference | 0.235 | $108,573 | 227 | 31 |
| Prior calls-only, matched training rows | 0.238 | $110,052 | 227 | 30 |
| Full state, profitability gate removed | -0.240 | $-294,539 | 1513 | 58 |
| IV spread/reversion ranks, original profitability gate | -0.231 | $-84,517 | 276 | 37 |
| IV spread/reversion ranks, no profitability gate | -0.630 | $-513,401 | 1719 | 60 |
| Spread/reversion + term-slope ranks, no profitability gate | -0.297 | $-462,256 | 1840 | 62 |

Pairs with an incomplete book in any new variant are excluded from every aggregate comparison, including both baselines. Their errors remain in incomplete.csv; no failed book is treated as flat or assigned a payoff. This is a common-complete diagnostic comparison, not a claim of a complete full-universe backtest.

Predeclared follow-up to the signal/profitability study: (1) remove the hard profitability gate while retaining the full state, including memory inputs; (2) replace the state with trailing IV-spread and spread-reversion ranks while retaining the original gate; (3) combine the rank state with removal of that gate; (4) add the term-slope difference rank to (3). No hard-coded reversal of memory or direction based on full-sample outcomes. The policy learns direction, flat decisions and the existing sizing ladder from past completed net contract outcomes.

Ranks compare the current signal to at most 252 observed signal rows including itself, with 63 observations required. Missing ranks use 0.5 with an availability flag. Term slope is masked when unavailable. These transformations are prefix-invariant and use no future observations. The original tree/CEM Sharpe objective, seed and search budget are unchanged. Removing the profitability gate does not remove beta/correlation, contract availability, positive validation Sharpe, or portfolio risk checks.

All four new variants were specified before viewing their backtest results. However, feature selection was motivated by correlations on this same 2018–2025 history: these are adaptive research comparisons, not an untouched holdout or proof of improvement. Overlapping outcomes and shared names create dependence. Annualized excess Sharpe uses daily returns above the cash benchmark and includes all cash sleeves. Validation still checks the pre-refit proposal rather than the exact deployed weights. Initial vega matching and daily delta hedging do not prevent later vega drift; each leg is held to its own expiry. American early assignment remains unmodeled.

[Annual results](signal_rank_policy/annual.csv) · [Pair results](signal_rank_policy/matched_pair_results.csv) · [P&L components](signal_rank_policy/pnl_components.csv) · [Audit](signal_rank_policy/audit.csv) · [Incomplete books](signal_rank_policy/incomplete.csv)

## Findings

None of the four variants improved on the prior calls-only matched-row policy. On the 342 common complete pairs, that baseline has 0.238 annualized excess Sharpe and 227 entries. Removing only the profitability gate increases entries to 1,513 and reduces Sharpe to -0.240. Its gross option-plus-hedge P&L is approximately $51,556, versus $287,061 in option/hedge execution costs before borrow, dividends and financing. The original baseline has approximately $142,667 gross option-plus-hedge P&L and $59,590 execution costs. These are observed accounting components, not a separate frictionless-policy backtest.

Replacing the full state with spread/reversion ranks while retaining the gate also underperforms (-0.231). Removing the gate lowers that to -0.630; adding term-slope ranks improves it to -0.297, still negative. A weak negative standalone correlation for recent profitability does not establish that its hard gate should be removed: this controlled removal hurt performance. Likewise, directional rank correlation does not establish sufficient net trading edge.

Four pairs are excluded from every common-roster result: DVN/MRO (unsupported distribution), RCL/NCLH (missing held option), OIH/HAL and OIH/SLB (invalid held quotes). Six of 1,384 pair/variant books are incomplete; the other 1,378 passed the audit. The rank policy retaining the gate completed all 346 pairs and has -0.141 full-universe Sharpe with 330 entries. The other new variants do not have complete full-universe results. The earlier best full-universe result remains 0.247; no reference configuration was changed.

[Full-universe completeness](signal_rank_policy/full_universe_completeness.csv) · [Concentration sensitivity](signal_rank_policy/concentration_sensitivity.csv) · [Validation status counts](signal_rank_policy/fit_status_counts.csv)
