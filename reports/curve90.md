# 90-day-only contract test

Completed both policies for all 346 roster pairs, with no failed pairs. Evaluation: 2018–August 29, 2025.

| Policy | Annualized cash-excess Sharpe | Pair entries | Active pairs | Net excess P&L |
|---|---:|---:|---:|---:|
| 90D ATM calls | 0.210 | 13 | 3 | $12,521 |
| 90D flexible strikes, calls or puts | -0.262 | 47 | 5 | $-29,873 |

Trading was sparse: 3 and 5 active pairs, respectively. The roster includes 199 predeclared cash-only pairs. AMD/NVDA contributed $15,605 in both policies; the flexible policy lost most on HAL/SLB (−$19,907), BAC/JPM (−$12,700), and BAC/WFC (−$12,124). Aggregate dollar P&L sums the $1 million pair sleeves. Cash interest is excluded from the reported Sharpe and excess P&L.

Target 90 days, entry window 75–105 days; one maturity target, independent delta buckets, calls/calls or puts/puts. Initial integer vega matching, daily delta hedge, hold each leg to its own expiry. Same costs/risk limits and original 30D reference state. The new ATM control also trades 90D calls. Seventy-two curve combinations are screened annually using past training data; monthly sizing policy and chosen-contract profitability gates remain unchanged.

A 110-calendar-day runoff applies to both validation and final entry eligibility. Older 30D/14–45D results used 60 days, so those are contextual comparisons, not a matched maturity ablation. Evaluation history is already explored research data, not an untouched holdout. Early assignment and daily option-vega rebalancing are not modeled.

[Annual results](curve90/annual.csv) · [Pair results](curve90/pair_results.csv) · [Contract choices](curve90/contract_mix.csv) · [Incomplete sleeves](curve90/incomplete.csv)
