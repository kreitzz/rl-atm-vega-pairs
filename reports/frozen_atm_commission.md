# Frozen ATM strategy: commission replay

Frozen saved monthly ATM30 call/call decisions. Full history starts flat in 2018; the matched recent test starts flat in 2022. Both use all 346 original pair sleeves, $1m each, including cash sleeves. No refitting or retrospective pair removal. Only commission changes; NAV-based sizing, integer rounding, daily delta hedges and the original risk limits are reapplied.

| Period | Commission | Annualized excess Sharpe | Entries | P&L above cash |
|---|---:|---:|---:|---:|
| 2018-01-02 to 2025-08-29 | $0.65 | 0.276 | 238 | $110,465 |
| 2018-01-02 to 2025-08-29 | $0.03 | 0.350 | 238 | $140,627 |
| 2022-01-03 to 2025-08-29 | $0.65 | 0.292 | 192 | $71,735 |
| 2022-01-03 to 2025-08-29 | $0.03 | 0.396 | 192 | $97,548 |

Sharpe = daily cash-excess mean / sample standard deviation × √252, excluding the initial session. Midpoint plus a 0.10-vol-point entry penalty per leg and 1 bp stock execution are unchanged. Initial integer vega mismatch ≤10%; option quantities then remain fixed apart from splits and each leg settles at its own expiry. No daily option-vega rebalancing. Settlement carries no modeled commission.

The full-history $0.65 column uses the saved original ledger. Its active-pair replay is independently checked against that ledger before the lower-fee run. Recent-period columns are fresh simulator replays, so they need not equal a slice of the older continuously funded portfolio. The policy still contains its original $0.65-trained cost inputs and validation/memory gates: this isolates execution fees rather than retraining for cheaper execution.

This is previously explored research history, not an untouched holdout. Replacing the largest winner with cash is an ex-post concentration diagnostic only.

[Annual results](frozen_atm_commission/annual.csv) · [Pair results](frozen_atm_commission/pairs.csv) · [Comparison](frozen_atm_commission/comparison.csv)
