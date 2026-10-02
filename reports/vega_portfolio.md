# Portfolio-level vega-neutral test

Completed January 2, 2020–August 29, 2025, with $1 million initial capital for each strategy. These are actual marked-option and stock-hedge cash ledgers. Sharpe is annualized from daily cash-excess returns using √252. Total returns include interest on cash.

| Allocator | Annualized excess Sharpe | Total return | CAGR | Max drawdown | Excess P&L |
|---|---:|---:|---:|---:|---:|
| forecast | 0.019 | 16.78% | 2.78% | -3.27% | $1,171 |
| equal_rank | -0.775 | 15.11% | 2.52% | -0.39% | $-15,527 |
| Cash | Undefined (zero excess volatility) | 16.66% | — | 0.00% | $0 |

## Annual results

| Year | Forecast Sharpe | Forecast return | Rank Sharpe | Rank return | Cash return |
|---|---:|---:|---:|---:|---:|
| 2020 | 0.928 | 2.61% | -1.222 | 0.23% | 0.52% |
| 2021 | 2.054 | 2.79% | -0.351 | -0.04% | 0.13% |
| 2022 | -1.057 | -0.29% | -1.946 | 1.40% | 1.80% |
| 2023 | -0.723 | 3.58% | -0.759 | 4.71% | 4.86% |
| 2024 | 0.258 | 6.14% | -1.317 | 5.10% | 5.39% |
| 2025 | -1.649 | 1.01% | -0.300 | 2.95% | 3.04% |

2025 is partial through August 29.

## Specification

- Universe: 308 cached underlying IDs in the original 346-pair roster. Monthly eligibility takes the union of names in historically eligible pairs. Positions are not coupled into pairs and there is no additional pairwise-beta gate. Upstream universe/screen construction and data availability remain inherited limitations.
- Signal: the prior HAR+quoted-IV variance forecast (5/21/63-session variance, DTE, quoted IV), pooled across securities. Each monthly decision uses the preceding close, with models fitted on outcomes completed before that signal quarter. No RL/NN policy was trained in this first construction test. Forecast training still comes from the prior screened/resolved history; it is not a new broad-universe label set.
- At most the 15 highest and 15 lowest demeaned forecast-value scores enter the candidate set. Forecast allocation maximizes approximate relative variance value less two option-fill penalties/commissions. Rank allocation targets equal long/short vega per candidate, then scales to the same risk limits; it trades only if aggregate estimated edge exceeds those option costs. Both are long/short relative-value books. Stock hedging, financing and borrow costs are charged in the ledger, but not fully forecast by this entry objective.
- Contracts: approximately 30-day ATM calls, 21–45 DTE and 0.25–0.75 delta at entry, positive uncrossed quotes, at least 100 OI, relative bid/ask width ≤25%, stock ≥$5 at signal and execution. Contract IDs stay fixed between monthly rebuilds. All holdings close one trading session before the earliest expiry, or at the next monthly rebuild. This changes holding schedules versus the previous expiry experiments.
- Daily balancing preserves initial relative vega targets where feasible and rescales for current risks. Integer contracts only; net vega must be within the larger of $10 per volatility point or 1% of gross vega after each close. Delta is hedged per name at each close. Neutrality is an end-of-day constraint, not an intraday guarantee.
- Limits per $1 million NAV: $20,000 gross vega, $2,000 per-name vega, 2× gross stock-hedge notional, 80% capital reserve, 5% independent-name stress loss for ±20% spot and ±10 volatility-point shocks, 1% absolute gamma P&L for a 1% spot move. A 1% pretrade capital cushion accommodates fees. These stress scenarios are not guarantees against larger moves.
- Industry buckets use cached SIC prefixes, not GICS sectors. Each has a gross-vega cap of 35% of the overall gross-vega budget and a signed cap of 10% of that budget. These are budget-based limits, not caps on the fraction of actual utilized exposure. Missing/conflicting industries share an unknown bucket.
- Names with earnings within seven days (or unknown countdowns) have an aggregate gross-vega cap of 25% of the total budget, updated daily. Countdown data uses final historical event dates assumed known, not an archived announcement schedule. The forecast itself excludes earnings.
- No more than 100 contracts per name or 10% of signal-date option open interest, whichever is smaller; verified forward splits adjust this inventory cap. No position increases when delta is outside 0.15–0.85 or the current entry-quality filter fails. Whole-contract rounding may reduce the book further.
- Execution costs on every option adjustment: midpoint plus a separate 0.10-vol-point penalty and $0.03 per contract per fill (user-specified institutional assumption). Stock hedge execution is 1 bp, stock borrow is 0.3% annualized, and negative cash costs the financing rate plus 1%. Ordinary stock dividends flow through the hedge. This midpoint assumption does not demonstrate executable fills.
- Actual held midpoints are retained, including wide quotes. Crossed held bid/ask quotes are swapped, consistent with the inherited engine. Missing fitted Greeks use the last observed IV with current spot/time/rate/dividend PV. Missing held marks, unsupported corporate adjustments, insolvency or ledger failures stop the test rather than deleting the affected dates. American early assignment is not modeled.

## Audit and interpretation

This is an initial allocation experiment on repeatedly explored history, not an untouched test or a clean fixed-pair ablation: universe use, rebalancing and holding schedules also changed. It does not establish that RL improves portfolio construction. Both complete ledgers reconcile and end flat; all daily net-vega checks pass. No pair is removed because of subsequent profitability.

The first development draft omitted contract-count/liquidity caps. It accumulated tens of thousands of low-vega penny-stock contracts and was invalid as a tradable sizing proposal; a held quote widening also caused a large marked loss. That draft is preserved locally under `unbounded_contract_draft`. The general price, OI, contract-count and off-ATM increase limits above were added after that failure and apply identically to both final allocators. Therefore these controls are a disclosed research revision, not a pre-registered untouched specification.

Peak worker RSS: 0.55 GiB, with one worker and a 50% duty-cycle cap.

[Annual data](vega_portfolio/annual.csv) · [P&L attribution](vega_portfolio/attribution.csv) · [Exposure audit](vega_portfolio/risk_audit.csv) · [Fit timing](vega_portfolio/fit_audit.csv)
