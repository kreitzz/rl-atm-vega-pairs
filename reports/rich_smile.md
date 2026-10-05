# Richer smile information: fixed 30D policy experiment

346/346 pairs common to all completed books, January 2018–August 2025.

| State | Annualized excess Sharpe | Entries | Active pairs | P&L above cash |
|---|---:|---:|---:|---:|
| control | 0.215 | 256 | 30 | $96,246 |
| smile_levels | 0.152 | 263 | 32 | $67,883 |
| smile_levels_changes | 0.160 | 253 | 32 | $69,651 |

Two predeclared additions to the full reference state: (1) richer smile levels; (2) those levels plus 5- and 21-session changes. Each preserves the original features, same 30D call selections, native call labels, full profitability gate, monthly refit, sizing ladder, beta/correlation screen, costs, risk limits and common 60-day runoff. The exact control is the preceding maturity experiment’s 30D policy. Nothing was tuned after viewing these new outcomes.

For each underlying/date, fit eligible cached calls at the selected ATM contract’s exact expiry. Interpolate IV at BS call deltas 0.10, 0.25, 0.50, 0.75 and 0.90 only between observed fitted points whose delta gap is ≤0.15. No extrapolation. Standard contracts, positive bid, open interest ≥100 and relative quoted width ≤25%; rejected or missing points remain unavailable.

Eight normalized shape descriptors: 10- and 25-delta downside/upside wing premiums relative to the selected ATM IV; 25-delta call-minus-put-equivalent risk reversal; 25-delta butterfly versus 50-delta IV; and downside/upside 10-versus-25-delta tail slopes. All are divided by current selected ATM IV. The downside measures are inferred from ITM calls under the existing European BS/dividend convention, not observed OTM put IVs. Each descriptor enters for both legs and their difference, with explicit availability flags. Changes use exact 5/21 trading-session lags on the market calendar; a missing endpoint stays unavailable. Zero-filled missing values always have a zero availability flag.

The original depth-limited tree/CEM learner and search budget stay fixed. The extra inputs do not guarantee better partitions. Training and memory use only past completed outcomes. The inherited validation gate evaluates pre-refit weights, not the exact deployed refit. All dates have already been inspected in research; this is an adaptive comparison, not an untouched holdout. Initial vega matching can drift, each leg settles at its own expiry, and American early assignment is still omitted.

Incomplete books, if any, are disclosed and their pairs excluded from every common comparison, including the control. No incomplete book is treated as cash. No reference policy is promoted automatically.

[Annual results](rich_smile/annual.csv) · [Pair results](rich_smile/pair_results.csv) · [Feature coverage](rich_smile/coverage.csv) · [P&L components](rich_smile/pnl_components.csv) · [Audit](rich_smile/audit.csv) · [Incomplete books](rich_smile/incomplete.csv)

## Findings

All 692 new pair/variant books completed and passed the audit. The control exactly reproduces the prior 30D portfolio NAV and 0.215 annualized excess Sharpe. Richer smile levels yield 0.152 (263 entries, 32 active pairs); levels plus recent changes yield 0.160 (253 entries, 32 active pairs). Neither improves on the control. The reference remains unchanged.

Coverage substantially limits interpretation: on eligible research signal dates, both legs have bracketed 25-delta upside IV in 18.2% of observations, 25-delta downside in 10.0%, 10-delta upside in 9.2%, and 10-delta downside in 8.9%. Both-leg 25-delta butterfly coverage is only 3.4%. These figures reflect same-expiry call-only inputs plus the declared liquidity and interpolation-gap rules; they do not mean that every missing target lacks a nearby tradable contract. No extrapolation was introduced to improve coverage after seeing results.

This experiment changes information while retaining ATM trades. The follow-up now permits independently selected delta buckets with calls/calls or puts/puts, and 14-, 30-, or 45-day target expiries. Its separate [contract-choice protocol](curve_trading_protocol.md) preserves the reference state and tests the expanded set of tradable contracts; the results above remain those of the ATM-only smile-state experiment.
