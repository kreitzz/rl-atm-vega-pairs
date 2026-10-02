# Why the broad proxy and contract replay differ

The old 2.606 proxy Sharpe and the new −1.310 daily cash-excess Sharpe measure different portfolios and different payoff series. The following diagnostics identify concrete differences. They use the 311 broad pairs that completed all three policies; the 39 unresolved pairs remain excluded and the results are not a full-universe conclusion.

## Entry filters and IV marks

Same saved RL-with-memory decisions, same 311-pair union, fixed pair budgets, and a common complete-outcome calendar of 1,835 signal dates. Sixty-six dates with unknown proxy outcomes are excluded; the final 35 calendar days are reserved for runoff.

| Proxy calculation | Annualized 30-day payoff Sharpe |
|---|---:|
| All saved entries on this matched roster/calendar | 2.567 |
| Only entries accepted by the contract simulator | 1.445 |
| Accepted entries, scaled to actual initial vega | 1.430 |
| Above, replacing standardized IV with same-day selected-contract IV | 0.244 |
| Above, using the contract IV on the following execution session | 0.172 |

These are all **proxy payoff** Sharpes, using √(365/30). The final two rows retain the original forward RV, horizon and proxy costs; only entry IV is replaced. They do not calculate actual option P&L or resolve gamma path dependence, differing maturities, early exits or changed cash costs.

The accepted, actual-vega-scaled proxy produces $18.443 million of summed cohort rewards across the 311 $1 million sleeves. Substituting same-day contract IV reduces that to $2.213 million, an **88.0% reduction**; using execution-session IV reduces it to $1.553 million. These are proxy reward dollars, not financed portfolio NAV returns. The correction uses mean initial leg vega consistently; it does not additionally change the proxy to reflect small residual initial vega imbalances.

The largest IV substitutions affect SIL/HL, KGC/AG, HL/AG, RIG/GLNG and SLB/RIG. Mean absolute differences between standardized signal IV and execution-contract IV are approximately 14–28 vol points across these pairs' accepted entries. Our same-day Black–Scholes IV and the vendor IV **for the same selected contracts** differ by only about 0.06–0.28 points on average for those five pairs. This supports a standardized-mark/contract mismatch rather than a large Black–Scholes inversion discrepancy in those cases. It does not establish that the standardized source is erroneous: interpolation, maturity, moneyness and source-quality effects still need separation.

## Most signals did not become trades

Across the full simulator calendar, 30,403 of 136,315 nonzero RL-memory signals entered: **22.3%**. The largest rejection categories were risk-limit entry (45,654), no eligible contract (36,039), and unavailable/ineligible execution quote (11,442). Stress loss was included in 45,177 of the risk-entry rejections. These counts cover all simulator dates, not only the proxy's complete-outcome subset.

The original proxy did not apply these contract-level and book-risk checks. Historical aggregate call activity does not guarantee an eligible selected ATM contract. This materially changes which opportunities are evaluated.

## Cost bridge on identical actual positions

The following retains every realized option trade and hedge, changing only cash costs and recomputing financing exactly. It does **not** retrain, resize, or reapply risk interventions under alternative costs. Sharpe here uses daily marked cash-excess returns and √252, so it is not directly comparable to the proxy table above.

| RL sizing + memory diagnostic | Annualized daily excess Sharpe |
|---|---:|
| Recorded net result | −1.310 |
| Remove only 0.10-vol execution penalty | −0.811 |
| Remove only option commissions | 0.445 |
| Remove option commissions and option penalty | 0.946 |
| Remove option costs and stock execution costs | 1.333 |
| Also remove stock borrow | 1.421 |

Option commissions totaled **$7.109 million**, versus **$2.029 million** in option execution penalties and **$1.564 million** in stock execution costs. Commissions average 1.023 vol points when divided by summed mean initial leg vega; unlike a fixed vol penalty, per-contract fees depend strongly on contract vega. The proxy used a single cost equal to 2.5% of average entry IV and the saved policy learned that older payoff/cost system.

The negative net Sharpe accompanies a **19.89% total NAV return versus 21.82% in cash** over January 2018–August 2025. It describes a small but persistent cash-relative loss, not a large absolute portfolio collapse.

## Interpretation

The old result relies heavily on opportunities removed by contract/risk checks and on standardized IV spreads that differ from selected-contract spreads. A positive gross directional component remains in the actual-position cost diagnostic, but the current execution and commission model consumes it. A clean next experiment must train and evaluate on the same contract IVs, actual net option/hedge payoffs, eligibility and risk rules. Neither result alone settles whether an implementable strategy works.

[Cost bridge](broad_contract_replay/gap_cost_bridge.csv) · [Per-pair costs](broad_contract_replay/gap_pair_costs.csv) · [Entry counts](broad_contract_replay/gap_entry_counts.csv) · [Proxy filter bridge](broad_contract_replay/gap_proxy_entries.csv) · [IV substitution by pair](broad_contract_replay/gap_entry_marks.csv)

Reproduce with `python -m scripts.diagnose_broad_replay_gap` followed by `python -m scripts.diagnose_broad_entry_marks`. The cost diagnostic verifies that unchanged cash flows reproduce each saved NAV within $0.0000001. IV substitutions assert complete finite selected-contract marks for the matched accepted entries.
