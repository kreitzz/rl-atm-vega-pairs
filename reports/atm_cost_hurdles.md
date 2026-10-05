# Training cost as a profitability hurdle

Predeclared grid: $0.03, $0.15, $0.30, $0.65 per option contract. All evaluation books pay $0.03. The two intermediate policies rebuild rewards, commission features, completed-outcome profitability memory, directional gates and prior-year validation using the higher training cost. The extra cost is a conservative profitability hurdle, not a calibrated probability of profit.

| Training hurdle | Full-history Sharpe | Recent Sharpe | Full entries | Active pairs | Full P&L above cash |
|---|---:|---:|---:|---:|---:|
| $0.03 | 0.129 | 0.196 | 323 | 36 | $61,016 |
| $0.15 | 0.172 | 0.226 | 288 | 33 | $71,383 |
| $0.30 | 0.302 | 0.333 | 277 | 33 | $127,123 |
| $0.65 | 0.350 | 0.396 | 238 | 30 | $140,627 |

Sharpe improves with the hurdle in both displayed periods, while full-history entries decline. The original $0.65-trained policy remains the strongest of these four tested settings. This supports a conservative profitability buffer in this sample, without establishing an optimal hurdle or calibrated confidence probability. Removing the largest winner after the fact gives full-history Sharpes of 0.029, 0.060, 0.198 and 0.243 respectively; the more conservative settings also fare better in that concentration diagnostic.

Final audit: all 1,384 new pair/hurdle/period windows completed. Original memory error was below 3e-12 and daily ledger error below $1e-9. Memory permissions are nested as the hurdle rises: 33,558, 32,284, 31,191 and 28,740 eligible direction-days respectively. Actual entry commissions are $0.03 in every new trading book. Five regression checks passed. Peak worker memory was 0.78 GiB, with the existing 50% CPU cap.

Full history: January 2018–August 2025. Recent test: January 2022–August 2025, starting flat with the same $1m per pair. All 346 original pairs remain, including cash sleeves. Same ATM30 call/call contracts, signed 25/50/75/100% ladder, monthly beta 0.85–1.15 and correlation ≥0.5 screen, actual-contract rewards, monthly eight-leaf direct Sharpe search, completed-outcome purging, prior-year validation and recent-five-year refit.

Training validation pays the selected hurdle; evaluation resets commission to $0.03. The 0.10-vol-point option entry penalty, 1 bp daily stock hedge execution, financing, borrowing and risk limits are identical everywhere. Option quantities are fixed after initially matching vega within 10%, apart from supported splits, and each leg settles at its own expiry. No daily option-vega rebalancing. Training-cost changes can alter directions, dates and sizes, so this is more than filtering a frozen prediction by a confidence threshold.

The $0.03 and $0.65 endpoints reuse their saved monthly policies; $0.15 and $0.30 are fresh fits of the same algorithm. Original $0.65 memory and curve gates are checked before fitting. Accounting, expiry and commission checks run on every simulated book. This is exploratory sensitivity on previously examined history; selecting the highest reported Sharpe would not constitute untouched validation. Largest-winner replacement is an ex-post concentration diagnostic.

[Annual results](atm_cost_hurdles/annual.csv) · [Comparison](atm_cost_hurdles/comparison.csv) · [New pair results](atm_cost_hurdles/pairs.csv)

[Memory permissions](atm_cost_hurdles/memory_permission_totals.csv) · [Audit](atm_cost_hurdles/audit.json)
