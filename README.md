# RL for Short-Term ATM Vega Pair Trading

Four independent policies for AMD/NVDA, XOM/CVX, JPM/BAC, and HD/LOW, assuming initial vega neutrality and daily delta hedging. Each uses its own IV spread, earnings countdowns, and trailing 30-day RV, with a 0/25/50/75/100% sizing ladder and a fixed equal budget. Pair cost is 2.5% of average entry IV, scaled by size.

[Current setup and results](reports/independent_pairs.md): combined annualized payoff Sharpe **0.499**. [Original AMD/NVDA reference](reports/reference_model.md).

[NN versus RL benchmark](reports/nn_rl_benchmark.md): full-size/flat neural network versus independently sized RL policies, with both relative-value features.

[Q-learning revision](reports/q_batch.md): exhaustive batch Q updates, economic state ablations, and missing-date sensitivity; AMD/NVDA improves in point estimate, without a statistically reliable advantage over cross-entropy.

[Broader universe and prior-profitability state](reports/broad_pairs.md).

[Actual-contract AMD/NVDA pilot](reports/contract_pilot.md): daily option-and-stock-hedge P&L, costs, and cash accounting.

[Contract-based Q learner](reports/contract_q.md): train directly on actual call-contract and hedge P&L, with next-session bid/ask execution and fixed contract IDs. The initial 2024 run selected cash; a profitable contract policy has not yet been established.

[Finite contract episodes](reports/contract_episodes.md): separate entry forecasts and Q exits. The selected fixed-horizon policy earned −0.021% above cash in 2024 and +1.156% through August 2025 on only two trades per period; Q still stayed flat.

[Contract data coverage for all previously explored pairs](reports/contract_data.md).

[AMD/NVDA dataset (CSV)](data/current_dataset.csv): 5,685 daily observations, December 2002–August 2025. Includes the five state inputs, entry IVs, forward 30-day RVs, and outcome date. Rewards are calculated during training; forward RV is not a state input. Earnings dates within 30 days are assumed known at entry.

**Academic use only.** Third-party data remain subject to OptionMetrics, CRSP, Compustat, and institutional license terms; this notice grants no additional data rights.
