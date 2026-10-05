# Training cost as a profitability hurdle

Follow-up grid: $1.00 and $1.50 per option contract, compared with the saved $0.03/$0.15/$0.30/$0.65 policies. All evaluation books pay $0.03. The two new policies rebuild rewards, commission features, completed-outcome profitability memory, directional gates and prior-year validation using the higher training cost. The extra cost is a conservative profitability hurdle, not a calibrated probability of profit.

| Training hurdle | Full-history Sharpe | Recent Sharpe | Full entries | Active pairs | Full P&L above cash |
|---|---:|---:|---:|---:|---:|
| $0.03 | 0.129 | 0.196 | 323 | 36 | $61,016 |
| $0.15 | 0.172 | 0.226 | 288 | 33 | $71,383 |
| $0.30 | 0.302 | 0.333 | 277 | 33 | $127,123 |
| $0.65 | 0.350 | 0.396 | 238 | 30 | $140,627 |
| $1.00 | 0.286 | 0.397 | 207 | 31 | $109,729 |
| $1.50 | 0.186 | 0.288 | 168 | 27 | $65,434 |

Full-history Sharpe peaks at the saved $0.65 hurdle and then falls at $1.00/$1.50. Recent Sharpe is effectively tied at $0.65/$1.00 (0.396/0.397) before falling to 0.288 at $1.50. The new higher costs reduce trade counts but do not reliably improve performance. Replacing the largest winner with cash gives full-history Sharpe 0.171 at $1.00 and 0.054 at $1.50, versus 0.243 at $0.65.

Compared with the $0.65-trained policy at identical $0.03 execution, $1.50 loses $31,956 more on CENX/AA and removes AU/NEM and SPWR/CSIQ, missing $20,870 and $19,536 of their prior positive P&L. These are observed policy-change contributions, not a causal isolation of individual model inputs.

Final audit: all 1,384 new pair/hurdle/period windows completed. Actual entry commission remains $0.03. Original memory and curve gates were reproduced; higher-cost memory permissions remain nested. Largest ledger error was below $1e-9. Six regression checks passed. Peak worker memory was 0.73 GiB under the 50% CPU cap.

Full history: January 2018–August 2025. Recent test: January 2022–August 2025, starting flat with the same $1m per pair. All 346 original pairs remain, including cash sleeves. Same ATM30 call/call contracts, signed 25/50/75/100% ladder, monthly beta 0.85–1.15 and correlation ≥0.5 screen, actual-contract rewards, monthly eight-leaf direct Sharpe search, completed-outcome purging, prior-year validation and recent-five-year refit.

Training validation pays the selected hurdle; evaluation resets commission to $0.03. The 0.10-vol-point option entry penalty, 1 bp daily stock hedge execution, financing, borrowing and risk limits are identical everywhere. Option quantities are fixed after initially matching vega within 10%, apart from supported splits, and each leg settles at its own expiry. No daily option-vega rebalancing. Training-cost changes can alter directions, dates and sizes, so this is more than filtering a frozen prediction by a confidence threshold.

The $0.03/$0.15/$0.30/$0.65 policies reuse saved results; $1.00 and $1.50 are fresh fits of the same algorithm. This follow-up was motivated by the preceding cost sweep. Original $0.65 memory and curve gates are checked before fitting. Accounting, expiry and commission checks run on every simulated book. This is exploratory sensitivity on previously examined history; selecting the highest reported Sharpe would not constitute untouched validation. Largest-winner replacement is an ex-post concentration diagnostic.

[Annual results](atm_higher_cost_hurdles/annual.csv) · [Comparison](atm_higher_cost_hurdles/comparison.csv) · [New pair results](atm_higher_cost_hurdles/pairs.csv)

[Memory permissions](atm_higher_cost_hurdles/memory_permission_totals.csv) · [Audit](atm_higher_cost_hurdles/audit.json)
