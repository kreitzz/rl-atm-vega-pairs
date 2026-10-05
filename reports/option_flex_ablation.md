# Matched call/put ablations

All new variants permit calls/calls or puts/puts only. 346/346 pairs are common to every completed variant. The previous unrestricted result is context, not a permitted strategy. Same dates (January 2018–August 2025), costs, beta/correlation filters, capital and portfolio risk limits.

| Variant | Annualized excess Sharpe | P&L above cash | Entries | Active pairs |
|---|---:|---:|---:|---:|
| Original calls-only reference | 0.226 | $105,789 | 280 | 35 |
| Previous unrestricted CC/CP/PC/PP (context only) | -0.024 | −$10,598 | 277 | 34 |
| Original entries → calls/calls (control) | 0.226 | $105,789 | 280 | 35 |
| Original entries → puts/puts | -0.009 | −$4,855 | 69 | 23 |
| Retrain calls only (control) | 0.226 | $105,789 | 280 | 35 |
| Calls only, matched-type training rows | 0.247 | $115,428 | 275 | 34 |
| Calls/calls or puts/puts + sizing | 0.086 | $38,644 | 205 | 30 |
| Same type + original call-memory gate | 0.183 | $88,930 | 289 | 38 |
| Same type, full size or flat | 0.028 | $12,474 | 84 | 23 |
| Same type, 4× policy-search iterations | 0.062 | $27,166 | 203 | 30 |
| Same type + put-memory inputs | 0.101 | $49,520 | 191 | 27 |

Original-entry substitutions preserve the calls-only reference’s actual entry signal dates, direction and requested sizing ladder. Puts can differ in availability, fitted vega, integer quantities and risk acceptance, so fills/counts may differ. No new put-memory gate or retraining is applied to these substitutions. The calls-only substitution and original-row retraining must reproduce every reference NAV exactly.

The matched-type base has 17 actions (CC/PP at eight sizes, plus flat). Its four ablations change one item each: use original call memory for the entry gate; remove sizing (five actions); increase search generations from 50 to 200 at 192 proposals each; or add eight completed-put-outcome memory inputs. Tree target, seed, hyperparameters, monthly training/validation/refit and all other features remain fixed. Calls-only on matched-type rows isolates the complete-label sample/runoff change. Shared finite CC/PP outcomes and the later completion date are used in all matched-type fits.

All results are research comparisons on previously inspected data, not an untouched test or evidence from picking the best ablation. Sharpe uses daily cash-excess returns annualized by √252, including cash-only sleeves. Outcomes overlap and share underlyings. Same-type options still require integer vega sizing; initial mismatch is capped at 10%, delta is hedged daily, and vega can drift. Each leg remains held to its own expiry. European fitting/cash intrinsic settlement and the absence of American early assignment modeling are unchanged. Validation still checks the pre-refit proposal rather than the exact deployed weights.

[Signal/profitability study](option_signal_relationships.md) · [Policy direction and sizing](option_policy_signal_quality.md) · [Annual results](option_flex_ablation/annual.csv) · [Pair results](option_flex_ablation/matched_pair_results.csv) · [Contract types](option_flex_ablation/entry_types.csv) · [P&L components](option_flex_ablation/pnl_components.csv) · [Audit](option_flex_ablation/audit.csv) · [Incomplete books](option_flex_ablation/incomplete.csv)
