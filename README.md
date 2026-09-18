# RL for Short-Term ATM Vega Pair Trading

Exploring reinforcement learning for short-term AMD/NVDA options pair trading, assuming vega-neutral execution in at-the-money options across the pair.

State inputs: AMD–NVDA 30-day IV spread, days until earnings for each stock (assumed known within 30 days), trailing 30-day realized volatility for each stock, and prior-close VIX. Actions are long AMD/short NVDA, the reverse, or flat. Assume equal initial dollar vega and daily delta hedging; score trades using subsequent 30-day realized versus entry implied volatility.

[Latest results](reports/rv_vix_pass.md): maximize annualized net payoff Sharpe after an assumed 0.5-vol-point round-trip pair cost. Full-state test Sharpe: 0.137; matched spread-only model: 0.743; short-higher-IV baseline: 0.956.
