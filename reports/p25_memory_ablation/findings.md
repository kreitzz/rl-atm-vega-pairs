# Does the profitability-memory gate help?

The original gates have the highest historical Sharpe, while removing both gates produces 6.3 times as many trades and 2.4 times the profit above cash. Keeping only the count minimums performs worse than removing the entire gate. The ungated learner also beats its fixed-full-size control, 0.835 versus 0.713. Block-bootstrap intervals for changes versus the retained strategy span zero, so the ranking does not establish a reliable Sharpe advantage for the gates.

Two declared ablations change entry qualification and retrain the same sizing policy monthly. Both keep the 33 inputs, including profitability memory and observation counts. Spread ranks, pair selection, contracts, costs, financing, sizing actions and account limits remain unchanged. Fixed-full-size controls isolate the screen change without a learned sizing change.

| Entry screen | Proposals | Trades | Annualized excess Sharpe | Profit above cash |
| --- | ---: | ---: | ---: | ---: |
| Original gates, learned size | 608 | 212 | 1.003 | $322,637 |
| Counts only, learned size | 3484 | 1014 | 0.626 | $527,463 |
| No memory gate, learned size | 4744 | 1334 | 0.835 | $786,690 |
| Counts only, fixed full size | 3484 | 661 | 0.708 | $598,544 |
| No memory gate, fixed full size | 4744 | 873 | 0.713 | $702,500 |
| Original gates, fixed full size | 608 | 133 | 0.769 | $272,338 |

The full memory gate excludes 87.2% of otherwise qualified proposals. The positive-profit tests alone exclude 82.5% of proposals that meet the count minimums. Restrictions therefore materially reduce trading opportunities; the result table separates profit from risk-adjusted performance.

## Yearly annualized excess Sharpe

| Year | Original gates | Counts only | No memory gate |
| --- | ---: | ---: | ---: |
| 2018 | 0.396 | -0.195 | -0.755 |
| 2019 | -0.577 | -0.100 | 0.214 |
| 2020 | 1.581 | 1.471 | 2.774 |
| 2021 | -0.580 | -0.762 | 0.072 |
| 2022 | 2.462 | 1.584 | 1.351 |
| 2023 | 1.604 | 0.259 | 0.295 |
| 2024 | -0.785 | 0.546 | 0.589 |
| 2025 (partial) | 1.231 | -1.086 | -1.383 |

The count-only variant keeps at least 21 completed observations over 91 days and 63 over 365 days. The ungated variant also removes those two minimums. Evaluation proposals never use future trade outcomes or future quote coverage as a filter. Monthly training uses only outcomes completed before the fit date.

The broader native quote cache is streamed in small batches into an indexed disk store. Its complete baseline NAV, decisions and trades reproduce the retained checkpoint. All new ledgers reconcile and finish with closed inventory. One worker remains capped at 20% of one core and 384 MiB RSS.

This is a comparison on previously researched 2018–2025 history, not a fresh holdout. More proposals can compete for the same inventory and risk budget; accepted portfolios therefore change. The learner can reduce a proposal to 50%, but its unchanged action menu cannot reject it. No threshold is tuned and the retained checkpoint is preserved.

This tests whether the gates help within the current architecture, not whether these exact cutoffs are optimal. Removing a hard gate while also introducing a learned zero-exposure action would be a separate experiment. Memory counts measure completed simulated daily candidate cohorts, often with overlapping holdings; they are not independent trades. Memory retains the original $0.65-per-contract label costs, while funded execution uses $0.03.

See [yearly results](annual.csv), [proposal funnel](proposal_funnel.csv), [screen audit](screen_audit.csv), [admission reasons](admission.csv), [fit dates](fits.csv), and [block-bootstrap Sharpe changes](sharpe_change_intervals.csv). The intervals do not adjust for strategy selection. 2025 ends on 29 August.
