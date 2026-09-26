# RL for Short-Term ATM Vega Pair Trading

Four independent policies for AMD/NVDA, XOM/CVX, JPM/BAC, and HD/LOW, assuming initial vega neutrality and daily delta hedging. Each uses its own IV spread, earnings countdowns, and trailing 30-day RV, with a 0/25/50/75/100% sizing ladder and a fixed equal budget. Pair cost is 2.5% of average entry IV, scaled by size.

[Current setup and results](reports/independent_pairs.md): combined annualized payoff Sharpe **0.499**. [Original AMD/NVDA reference](reports/reference_model.md).

[NN versus RL benchmark](reports/nn_rl_benchmark.md): full-size/flat neural network versus independently sized RL policies, with both relative-value features.

[Broader universe and prior-profitability state](reports/broad_pairs.md).

[Actual-contract AMD/NVDA pilot](reports/contract_pilot.md): daily option-and-stock-hedge P&L, costs, and cash accounting.

[AMD/NVDA dataset (CSV)](data/current_dataset.csv): 5,685 daily observations, December 2002–August 2025. Includes the five state inputs, entry IVs, forward 30-day RVs, and outcome date. Rewards are calculated during training; forward RV is not a state input. Earnings dates within 30 days are assumed known at entry.

**Academic use only.** Third-party data remain subject to OptionMetrics, CRSP, Compustat, and institutional license terms; this notice grants no additional data rights.
