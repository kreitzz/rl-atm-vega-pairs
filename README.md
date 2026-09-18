# RL for Short-Term ATM Vega Pair Trading

Exploring reinforcement learning for short-term AMD/NVDA options pair trading, assuming vega-neutral execution in at-the-money options across the pair.

First pass: the sole state input is AMD 30-day IV minus NVDA 30-day IV. Actions are long AMD/short NVDA, the reverse, or flat. Assume equal initial dollar vega and daily delta hedging; score trades using each leg's subsequent 30-calendar-day close-to-close realized volatility versus entry IV as a P&L proxy.

[First-pass results](reports/first_pass.md): maximize net payoff Sharpe with an assumed 0.5-vol-point round-trip pair cost. The policy stays flat 68.5% of the time; test payoff Sharpe is 0.193 versus 0.274 for shorting the higher-IV stock (unannualized).
