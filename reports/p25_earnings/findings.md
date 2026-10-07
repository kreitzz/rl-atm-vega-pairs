# Earnings-aware sizing on the retained strategy

Adding earnings inputs did not improve the retained policy in this first comparison. Both variants changed sizing while keeping all 608 proposals, option contracts, direction rules, costs and account limits. The saved 1.003 Sharpe checkpoint remains unchanged.

| Policy | Annualized excess Sharpe | Trades | Profit above cash | Annualized growth relative to cash |
| --- | ---: | ---: | ---: | ---: |
| Retained policy | 1.003 | 212 | $322,637 | 0.342% |
| Historical dates assumed known | 0.977 | 212 | $289,868 | 0.308% |
| Countdown from prior reports | 0.956 | 212 | $283,477 | 0.301% |

Each leg gets days until earnings and an indicator for earnings before its selected expiry. Four availability flags bring the input count from 33 to 41. Missing information never removes a proposal. The action menu remains 50%, 75% and 100%, with monthly fits on 18 months of completed outcomes. There is no earnings exit or entry ban.

The historical-date version assumes the eventual reported earnings date was known at the signal close, with a countdown capped at 90 days. We lack an archive of schedules as announced at the time. The other version estimates the countdown from strictly earlier reports; changing future report dates does not change that input. Before-open versus after-close timing is unresolved. Neither is a new untouched test period.

## Yearly annualized excess Sharpe

| Year | Retained | Historical dates | Prior reports |
| --- | ---: | ---: | ---: |
| 2018 | 0.396 | -0.268 | -0.268 |
| 2019 | -0.577 | -0.531 | -0.535 |
| 2020 | 1.581 | 1.583 | 1.583 |
| 2021 | -0.580 | -0.580 | -0.580 |
| 2022 | 2.462 | 2.599 | 2.572 |
| 2023 | 1.604 | 1.619 | 1.493 |
| 2024 | -0.785 | -0.784 | -0.784 |
| 2025 (partial) | 1.231 | 1.232 | 1.233 |

2025 ends on 29 August. Full percentage returns are in [annual.csv](annual.csv).

The largest reductions in return above cash occur in 2018 and 2022. In 2022 the earnings-aware policies have a higher yearly Sharpe but less profit. The weak 2024 result barely changes. Earnings input therefore does not resolve the strategy’s recent losing regime in this comparison.

## What changed

Historical dates assumed known changes the requested size on 30 of 608 proposals. It shares 205 accepted entries with the original book, admits 7 different entries and no longer admits 7 original entries.
Countdown from prior reports changes the requested size on 59 of 608 proposals. It shares 204 accepted entries with the original book, admits 8 different entries and no longer admits 8 original entries.

The earnings features are evaluated through the existing allocation model; this result does not establish that earnings are economically irrelevant. Added information can change tree partitions and estimated sizing rules without improving performance on subsequent trades. The sample supplies roughly 60–300 completed examples per monthly fit.

The original first fit and the complete funded baseline reproduce exactly. Six tests check event boundaries, direction, missing data, ETF treatment and independence from future reports. Both new ledgers reconcile and finish with all inventory closed. The first replay attempt hit the memory stop; model scoring was separated from replay and stock reads were restricted to proposed names. The optimized replay preserved the original ledger exactly.

Additional files: [feature usage](feature_usage.csv), [event-conditioned sizes](event_sizing.csv), [allocation changes](allocations.csv), [block-bootstrap intervals](sharpe_change_intervals.csv), [protocol](protocol.json). Bootstrap intervals do not correct for strategy selection. No model, window or threshold was selected using this experiment’s outcomes. The wider strategy was previously selected on these dates.
