# Hold-to-expiry diagnostic

Frozen historical signals, sizing actions and profitability-memory gates. No model or reward labels have been retrained for expiry holding. Changed lifecycles affect available capital and accepted entries; this is not a matched-inventory comparison.

Each leg settles separately at intrinsic value on the last supplied trading session on or before its expiry. No option exit spread, commission or modeled exit penalty. Daily stock hedging and its transaction costs continue, including final hedge liquidation. Assumes cash settlement and no early exercise/assignment for these American calls. Different expiries can leave one leg outstanding.

Interim NAV still uses observed option midpoints. Daily BS delta uses current fitted IV; failed fits use the last valid IV with current spot, remaining time, rate and declared-dividend PV. Missing or invalid quotes and unsupported adjustments remain explicit incomplete histories. Risk limits apply to entry; existing risk breaches are reported rather than triggering sales. Insolvency still fails the simulation.

348 pairs attempted; 347 common complete pairs compared, including cash-only sleeves. 1 pairs excluded from both sides because at least one replay is unresolved; this ex-post common-roster selection can bias results.

| Policy | Lifecycle | Annualized excess Sharpe | Cumulative excess return |
|---|---|---:|---:|
| earlier_filtered_rl | previous_exits | -0.131 | -0.018% |
| earlier_filtered_rl | hold_to_expiry | -0.163 | -0.031% |
| surface_filtered_rl | previous_exits | -0.265 | -0.035% |
| surface_filtered_rl | hold_to_expiry | -0.040 | -0.007% |
| surface_filtered_nn | previous_exits | -0.255 | -0.037% |
| surface_filtered_nn | hold_to_expiry | -0.215 | -0.050% |

2018–August 2025. Sharpe uses daily excess returns and √252. Returns use $1m starting capital per pair. All periods have previously been inspected.

[Annual results](expiry_holding/annual.csv) · [Status and fallback counts](expiry_holding/status.csv) · [Comparison](expiry_holding/comparison.csv)

The current surface RL policy has 410 entries, with an average 31.34 calendar days held per leg. Gross hedged P&L including stock dividends is $85,625; option entry costs are $47,786 and hedge execution costs $39,288. Net excess P&L after borrow and financing versus cash is −$25,888 across the fixed $347m initial basket. The earlier RL policy has 404 entries and −$108,590 net excess P&L; NN has 334 entries and −$172,953. NN's less-negative Sharpe accompanies worse net excess P&L, so it should not be described as an unqualified improvement.

The unresolved replay is CCL/RCL, surface NN: held option 129807663 is missing on March 13, 2020. The pair is excluded from every displayed variant. Daily risk breaches persist by design: 740 pair-days for earlier RL, 1,038 for surface RL and 1,267 for surface NN. Failed-IV hedge fallbacks occur on 394, 448 and 438 distinct date/contract observations, respectively. Thus eliminating forced sales does not establish that the book always satisfies the prior risk limits.

All 135 tests passed, including independent expiries, weekend expiry, settlement without an expiry quote, dividend handling, failed-IV fallback, insufficient runoff, and entry blocking without forced liquidation. Maximum ledger reconciliation error is below $0.000000001. The sequential replay took 67 seconds and peaked at 0.734 GiB resident memory.

[Holding periods and cost totals](expiry_holding/holding_and_costs.csv). Before interpreting this as a maturity-trained strategy, regenerate completed-outcome labels and profitability memory through each leg's actual expiry, then retrain and validate monthly with this lifecycle. Existing labels, memory and model weights still describe the previous exit rules.
