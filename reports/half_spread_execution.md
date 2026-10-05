# Halfway-between-mid-and-side execution

[Capital-exhaustion audit](drawdown_audit.md): fixed dollar-vega sizing continues as NAV falls, without margin limits. The 100% figures are cumulative account losses, not single-trade losses; the AMD/NVDA closing hedge was independently verified across 1,142 sessions.

Buys fill at `(mid + ask) / 2`; sells fill at `(mid + bid) / 2`, at both entry and exit. This pays one quarter of the full quoted width per fill, half the original spread-crossing charge. Commissions remain $0.65 per contract per transaction. Saved signals, contract selections, sizing, hedging, financing and quote-quality rules are unchanged. Every simulated order receives the assumed price; fill probability is not modeled.

| Four-pair portfolio, Jan 2–Aug 29, 2025 | Bid/ask return | Halfway return | Bid/ask excess Sharpe | Halfway excess Sharpe |
| --- | ---: | ---: | ---: | ---: |
| Sized RL | 3.97% | 5.58% | 0.49 | 1.28 |
| Full-size/flat NN | 0.71% | 3.70% | -0.47 | 0.18 |
| Short higher IV, RL sizes | 1.73% | 3.34% | -0.69 | 0.20 |

Returns include cash and are period returns, not annualized. Sharpe is annualized with sqrt(252) using daily returns less cash. Halfway-execution RL returns **2.57% above cash**, on initial capital. Four independent $1 million accounts retain their original fixed dollar-vega budgets.

| Sized RL pair, 2025 | Halfway return | Annualized excess Sharpe |
| --- | ---: | ---: |
| AMD/NVDA | 11.98% | 1.65 |
| XOM/CVX | 4.13% | 0.39 |
| JPM/BAC | 4.54% | 0.69 |
| HD/LOW | 1.67% | -0.42 |

The full 2010–August 2025 experiment still fails to establish a profitable strategy. AMD/NVDA RL exhausts capital July 18, 2014 (previously January 15); XOM/CVX does so February 8, 2021 (previously July 3, 2019). JPM/BAC's cumulative return improves from -88.29% to **-47.25%** and HD/LOW from -68.38% to **-27.66%**. All four full-size/flat NN accounts still exhaust capital. No full-history portfolio Sharpe excludes the failed accounts.

These runs retain the previously disclosed crossed-quote envelope sensitivity: five repaired held quotes in the 2025 RL book. They do not resolve the strict-quote data failures. The signals remain frozen from the earlier proxy-based training; no model was retrained or selected using these new results.

All 12 pair/model combinations in 2025 have identical trades, quantities and commissions to the original run. Their option spread charges halve exactly; hedge costs and positions match; spread savings plus changed interest reconcile to every daily NAV difference within $0.000001. All completed runs pass the cash-flow audit. Unit tests: 74 passing.

[Portfolio comparison](selected_pair_backtest/half_spread/portfolio_comparison.csv) · [All outcomes](selected_pair_backtest/half_spread/status.csv) · [Annual results for completed runs](selected_pair_backtest/half_spread/annual.csv) · [Comparison checks](selected_pair_backtest/half_spread/comparison_checks.csv).

```sh
python -m scripts.run_selected_pair_backtest --no-risk-controls --reuse-cache --cache-root artifacts/contracts/four_pairs/cache --output artifacts/contracts/four_pairs_half_spread --report reports/selected_pair_backtest/half_spread --option-spread-fraction 0.5 --repair-crossed-held-quotes
python -m scripts.report_half_spread
```
