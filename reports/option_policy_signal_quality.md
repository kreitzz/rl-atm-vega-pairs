# Policy direction and sizing diagnostics

These diagnostics condition on nonflat walk-forward proposals with resolved standalone outcomes. They are not an unbiased test of all opportunities. Net payoff is measured per actual initial vega at the requested ladder size, avoiding the mechanical correlation between position size and dollar P&L. The payoff remains a standalone fixed-capital counterfactual; portfolio fills can have different integer quantities and net hedge/financing costs. The filled subset flags proposals that became portfolio entries, not a cohort attribution of the portfolio NAV.

| Policy | Resolved proposals | Better direction | Positive net payoff | Size vs net-per-vega Spearman |
|---|---:|---:|---:|---:|
| cc_matched_rows | 1159 | 60.5% | 56.0% | 0.077 |
| cc_original | 1188 | 61.0% | 56.6% | 0.072 |
| reference_calls | 1188 | 61.0% | 56.6% | 0.072 |
| same_type | 769 | 64.2% | 58.3% | -0.163 |
| same_type_cc_memory | 1003 | 61.5% | 56.5% | 0.006 |
| same_type_memory_features | 680 | 62.2% | 58.1% | 0.002 |
| same_type_no_sizing | 582 | 61.9% | 56.4% | nan |
| same_type_search_4x | 758 | 63.9% | 57.9% | -0.184 |

“Better direction” means the chosen long/short orientation beat its reverse. It need not have positive net payoff after costs. Constant full-size variants have no size correlation. Overlapping outcomes, repeated pairs and shared underlyings invalidate IID significance tests; none are reported.

Put substitution filled 69 of 280 original entry dates. Among 61 common filled dates with complete standalone labels, mean put-minus-call net payoff per initial vega was -0.0174 (median 0.2634). This is an ex-post matched-date shadow-payoff diagnostic, not an implementable selection rule or full-portfolio P&L attribution.

[All proposal/filled-subset metrics](option_flex_ablation/policy_signal_quality.csv) · [Outcomes by ladder size](option_flex_ablation/policy_sizing_quality.csv) · [Memory gate conditional outcomes](option_flex_ablation/memory_gate_outcomes.csv) · [Put entry rejections](option_flex_ablation/frozen_put_rejections.csv) · [Matched-date shadow payoffs](option_flex_ablation/matched_substitution_shadow_payoffs.csv) · [Feature relationships](option_signal_relationships.md)
