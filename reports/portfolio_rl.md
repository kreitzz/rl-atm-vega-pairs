# RL portfolio sizing versus matched benchmarks

Evaluation: January 3, 2022–August 29, 2025. All four policies start with $1 million, use the same monthly candidate portfolios and signals, $0.03 per option contract per fill, 0.10-vol-point execution penalty, daily delta/vega balancing, and identical liquidity/risk limits. Results use continuous actual-option cash-ledger replays. Total returns include cash interest; Sharpe uses daily cash-excess returns annualized with √252.

| Policy | Annualized excess Sharpe | Total return | Max drawdown | Excess P&L | Funded months | Mean gross vega |
|---|---:|---:|---:|---:|---:|---:|
| forecast | -0.669 | 9.66% | -3.42% | $-62,402 | 44 | 398 |
| equal_rank | -0.936 | 14.88% | -0.26% | $-10,232 | 44 | 38 |
| nn_fixed | -0.317 | 13.96% | -3.42% | $-19,396 | 20 | 148 |
| rl_sized | -0.434 | 13.44% | -2.16% | $-24,664 | 29 | 208 |
| Cash | Undefined | 15.90% | 0.00% | $0 | 0 | 0 |

## Annual results

| Year | Forecast Sharpe | NN Sharpe | RL Sharpe | NN return | RL return |
|---|---:|---:|---:|---:|---:|
| 2022 | -1.131 | -1.400 | -1.675 | -0.54% | -0.26% |
| 2023 | -0.778 | -0.658 | -0.845 | 4.11% | 3.90% |
| 2024 | 0.213 | 1.187 | 0.709 | 7.36% | 6.72% |
| 2025 | -1.880 | -0.601 | -0.662 | 2.52% | 2.58% |

2025 is partial through August 29. These results must not be compared directly with the earlier 2020–2025 full-period Sharpe: the matched evaluation starts in 2022 because 2020–2021 supplies initial policy training.

## What learns

- RL is the existing direct Sharpe policy-search approach: a shallow regression-tree state partition (at most three leaves, minimum six training months per leaf) and cross-entropy search over state-dependent sizing actions. The ladder is cash, 25%, 50%, 75%, or 100% of the forecast allocator’s initial portfolio. It learns portfolio timing/sizing; individual-name selection and relative weights still come from the constrained forecast allocator.
- The NN benchmark has one 16-unit tanh hidden layer. It predicts full-size monthly net excess payoff and chooses full-size or cash. It receives exactly the same monthly state features and completed training months as RL. Architecture, regularization, seed, thresholds and search budget were fixed before inspecting evaluation results; there is no selection of a winning test seed.
- State features aggregate the existing candidate signals: median IV, mean absolute forecast value, RV/IV variance ratio, value dispersion, earnings share, relative quote width and DTE, plus trailing six-/twelve-completed-month full-size portfolio profitability. Memory comes from the fixed forecast portfolio, not the learned policy’s future or unobserved outcome.
- Both policies refit monthly on at most five years of completed episodes. The last outcome date must precede the preceding-close signal date. At least 18 training months are required. States, scalers, imputation and tree partitioning use training data only. Neither policy sees current-month reward during fitting.
- Training rewards come from independent monthly portfolio episodes at each ladder size, with exact contract quantities, daily rebalancing and full costs. Each episode starts with $1 million and ends flat by the next monthly rebuild. The RL objective sums those daily excess-dollar paths. Final evaluation replays the chosen actions continuously, with actual evolving NAV and integer constraints; it is not a splice of standalone episode returns.
- The sizing hook leaves the full-size baseline unchanged. Lower sizing is applied before integer rounding and daily risk projection. The same forecast-model training inputs and constrained allocator are shared across every comparison. No new turnover-reduction rule was introduced in this experiment.

## Limits and checks

This is a small monthly training sample on already explored research history, not a fresh holdout or proof of live profitability. Monthly shadow rewards and the continuous NAV-dependent book need not match exactly. The NN optimizes a payoff forecast while RL optimizes Sharpe; this comparison does not isolate reinforcement learning from differences in objective or policy family. A gross-sizing layer is narrower than unrestricted RL portfolio construction. Different policies can use different amounts of risk under identical limits.

Inherited data assumptions include historical earnings dates treated as known, EOD midpoint execution with a penalty, daily rather than intraday neutrality, existing quote/Greek fallbacks, and no American early assignment. The prior general contract-count/liquidity controls were developed after an earlier failed sizing draft. No evaluation period or name is excluded for poor performance.

All four ledgers reconcile, finish flat, and pass daily net-vega checks. All 44 policy fits pass strict completed-outcome timing checks. NN convergence warnings: 0. Peak worker RSS: 0.51 GiB, with one worker and the 50% duty-cycle cap.

[Annual data](portfolio_rl/annual.csv) · [Monthly decisions and fit audit](portfolio_rl/decisions.csv) · [Sizing frequencies](portfolio_rl/sizing.csv) · [Costs](portfolio_rl/costs.csv) · [Base portfolio specification](vega_portfolio.md)
