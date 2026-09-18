# RL for Short-Term ATM Vega Pair Trading

Research into reinforcement learning (RL) for short-term relative-value trading in at-the-money (ATM) options. The goal is to learn when to enter, size, rebalance, and exit paired volatility positions after accounting for execution costs and risk.

**Status:** Project specification only. Data ingestion, simulation, training, and evaluation are not yet implemented.

## Research question

Can an RL policy improve the net, risk-adjusted performance of a vega-balanced ATM options pair relative to simple spread-based trading rules under the same execution assumptions and risk limits?

The initial scope is a pair of related underlyings with liquid options, similar option maturities, and short holding periods. The asset universe, bar frequency, holding horizon, and expiry range remain research choices. Initial experiments will use a fixed pair so pair selection does not obscure the policy's contribution.

## Trading setup

- Construct an ATM straddle on each underlying using a call and put at the same strike and expiry. Select the strike nearest the forward where reliable forward estimates are available; otherwise use a documented spot-based convention.
- Match expiries across the two legs and define explicit strike reselection, roll, and exit rules. Track the actual held contracts between decisions rather than treating a rolling ATM quote series as a tradable instrument.
- Buy volatility on one underlying and sell volatility on the other. Size the positions to approximately offset dollar vega, subject to contract rounding and exposure limits.
- Delta hedge each underlying according to a fixed, costed rule for the initial experiments. Compare joint hedge optimization only as a later extension.
- Study intraday and multi-day holding horizons separately, with explicit overnight and expiry policies.

For signed straddle quantities `q_A` and `q_B` and per-contract dollar vegas `V_A` and `V_B`, target:

```text
q_A * V_A + q_B * V_B ≈ 0
q_B ≈ -q_A * V_A / V_B
```

Vega inputs must share the same volatility units and include contract multipliers. Use integer contract quantities in the simulator and monitor residual exposure. Vega balance does not eliminate gamma, theta, skew, jump, correlation, or liquidity risk; option and hedge P&L must be valued directly.

## RL formulation

| Component | Initial design |
| --- | --- |
| Observations | Lagged ATM implied-volatility spread and rolling z-score; realized volatility; term structure and skew; Greeks; time to expiry; underlying returns; bid/ask spreads and quote age; current inventory, hedges, holding time, and available capital. |
| Actions | Start with discrete target positions: flat, long A volatility / short B volatility, or the reverse, at a small set of bounded sizes. |
| Transitions | Execute after the observation timestamp using an explicit latency convention; mark actual option contracts and hedge positions; apply costs, fills, and risk limits. |
| Reward | Change in marked portfolio equity after trading costs, normalized by fixed starting capital, less documented exposure and turnover penalties. |
| Episodes | Contiguous market windows with explicit treatment of overnight holdings, rolls, and end-of-episode liquidation costs. |
| Constraints | Gross and net vega, delta, gamma, position size, margin/capital use, quote freshness, liquidity, maximum holding time, and drawdown limits. |

Portfolio equity should already include option P&L, hedge P&L, commissions, slippage, financing, and applicable borrow costs; do not subtract those costs twice in the reward. Record unpenalized net P&L separately from the shaped training reward. Hard risk limits belong in the environment and must apply to every policy and baseline.

Start with PPO as a candidate policy learner for the discrete action space. Algorithm choice is an experiment, not an assumed source of edge.

## Data and simulator requirements

1. Timestamped option bid/ask quotes, sizes where available, underlying prices, contract identifiers, strikes, expiries, multipliers, and corporate-action information.
2. Consistent implied-volatility and Greek estimates, with documented rates, dividends, pricing conventions, and missing-data handling.
3. Synchronized, causal features: reject stale or crossed quotes and fit transforms using training data only.
4. Conservative fills using bid/ask prices and explicit slippage, commissions, hedge costs, and execution delay. Model partial or failed paired execution rather than assuming both legs always fill simultaneously.
5. Capital and margin accounting for short options, with exercise/assignment and expiry handling or rules that close positions before those events.
6. An auditable ledger of orders, fills, contracts, hedges, exposures, rewards, and equity.

Historical replay assumes actions do not change subsequent market prices. Bound trade size by liquidity and test sensitivity to fills and market impact before drawing conclusions.

Do not commit licensed market data, credentials, or large training artifacts. Document data provenance and access requirements when a source is selected.

## Evaluation plan

- Use chronological training, validation, and held-out test periods with walk-forward evaluation. Purge overlapping outcome windows and apply a documented embargo appropriate to the holding horizon.
- Choose pairs, feature transforms, policy settings, and thresholds using training/validation data only. Freeze choices before evaluating the final test period.
- Compare against no trading, fixed-size spread mean reversion, a rolling z-score entry/exit rule, and a supervised signal with deterministic sizing. Give every strategy the same costs, capital, and risk constraints.
- Report net P&L, risk-adjusted returns with a stated annualization convention, maximum drawdown, turnover, holding times, capital usage, exposure distributions, and transaction-cost decomposition.
- Repeat training across random seeds and report uncertainty using time-aware resampling. Evaluate performance by volatility regime and pair, including adverse periods.
- Stress spreads, slippage, execution latency, hedge frequency, stale quotes, and failed fills. Compare policies with and without inventory and cost features.

Success means repeatable out-of-sample improvement over the strongest baseline after costs, with acceptable drawdown and capital usage. A negative result is a valid research outcome.

## Implementation roadmap

- [ ] Select the pair universe, data source, sampling interval, expiry range, and holding horizon.
- [ ] Build and validate contract selection, aligned quotes, and causal features.
- [ ] Implement a deterministic accounting engine and test hedge P&L, multipliers, costs, rolls, and terminal liquidation.
- [ ] Establish spread-rule baselines and a reproducible evaluation harness.
- [ ] Expose the simulator through a Gymnasium-compatible environment.
- [ ] Train an initial RL policy and run walk-forward comparisons and ablations.
- [ ] Produce a research report with assumptions, failure cases, and reproducible results.

Proposed stack: Python, NumPy/pandas, Parquet, Gymnasium, PyTorch, and Stable-Baselines3. Dependencies will be pinned when implementation begins.

## Repository workflow

The default branch is `main`. Create feature branches for data, simulator, policy, and evaluation work. Future experiments should record the commit, configuration, data version, random seed, and evaluation window.

There are no training commands or executable strategy components yet. The first implementation milestone is a validated simulator and baseline report.
