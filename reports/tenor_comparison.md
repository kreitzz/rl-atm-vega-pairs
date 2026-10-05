# ATM option maturity comparison

346/346 pairs complete for all three tenors. Common-roster results:

| Target DTE | Annualized excess Sharpe | Entries | Active pairs | P&L above cash | Median entry DTE |
|---|---:|---:|---:|---:|---:|
| 14 | -0.232 | 71 | 10 | $-49,474 | 16 |
| 30 | 0.215 | 256 | 30 | $96,246 | 31 |
| 45 | -0.003 | 170 | 22 | $-585 | 44 |

Targets 14, 30 and 45 calendar days at next-session entry, with respective windows 7–21, 21–45 and 35–55 days. Original selection priority is preserved: nearest tenor, then nearest strike, within the existing moneyness/liquidity limits. Calls/calls only; each leg held to its own expiry, daily delta hedge, initial integer vega mismatch ≤10%. Same costs, risk limits, sizing ladder, beta .85–1.15 and correlation ≥.5.

Original full RL state including VIX, surface context and directional profitability memory; monthly five-year training, separate prior-year validation, then recent-five-year completed-outcome refit. The inherited validation gate tests pre-refit weights. No rank-only state or gate removal. Spread, RV-relative value, reversion, contract attributes, labels and profitability memory reflect the traded tenor; cached surface-context features stay fixed across tenors. Thirty-day calls reuse their original native labels and states, without the matched-put sample restriction. This deliberately avoids imposing irrelevant put availability on other maturities.

A common 60-calendar-day entry runoff applies to both validation and final evaluation, January 2018–August 2025. This differs from the older 50-day runoff; thus the fresh 30D control need not equal either older 0.226 native-call or 0.247 matched-put-row Sharpe. Contract availability and training samples differ by tenor and are reported. The three targets were declared before the run; all periods are previously inspected research history, not an untouched holdout.

Same vega is not same gamma or tail risk. No early assignment is modeled. Missing/unsupported held contracts make a book incomplete; failed pairs are excluded from every common-roster comparison and remain explicit in the full-roster completeness table. They are never assigned zero returns. No reference policy has been promoted.

[Comparison and full-roster status](tenor_comparison/comparison.csv) · [Annual results](tenor_comparison/annual.csv) · [Pair results](tenor_comparison/pair_results.csv) · [P&L components](tenor_comparison/pnl_components.csv) · [Coverage](tenor_comparison/coverage.csv) · [Actual entry maturities](tenor_comparison/entry_tenors.csv) · [Incomplete books](tenor_comparison/incomplete.csv) · [Audit](tenor_comparison/audit.csv) · [Real-data payoff reconciliation](tenor_comparison/real_data_reconciliation.json)

## Findings

All 1,038 pair/tenor books completed and passed the audit; no pairs were excluded. Sixteen targeted tests passed, and 36 real-data fixed-capital action/entry trajectories reconciled with the full expiry simulator. The two training workers each remained below roughly 1.3 GB peak resident memory.

Changing maturity alone did not improve this policy. The 30D control has 0.215 annualized excess Sharpe, versus -0.232 for 14D and -0.003 for 45D. Actual median entry maturities are 16, 31 and 44 calendar days. Both legs share an expiry in 80.3%, 65.6% and 92.9% of entries respectively; all three have median expiry gap zero. Thirty-day results therefore do not benefit from a higher share of exactly matching expiries in this experiment.

The 14D policy is sparse: 71 entries across 10 active pairs. Two M/KSS entries account for -$72,759 above-cash P&L, versus -$49,474 for the entire 14D book. Replacing that sleeve with cash after observing the result yields 0.125 Sharpe; this is an ex-post sensitivity diagnostic, not an investable selection rule, and still trails the 30D control. The 45D policy has a strong 2023 (1.468 Sharpe), but that does not persist across the other years. The 30D book also has concentration: removing its largest winner GES/TJX reduces Sharpe to 0.116.

These results do not establish that every 14D or 45D signal is unprofitable. They show that changing tenor with this state, liquidity selection, learner, gating and risk framework did not restore performance on the inspected research history. No new policy was promoted.

[Concentration sensitivity](tenor_comparison/concentration_sensitivity.csv)
