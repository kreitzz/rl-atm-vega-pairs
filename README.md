# RL for Short-Term ATM Vega Pair Trading

Exploring reinforcement learning for short-term AMD/NVDA options pair trading, assuming vega-neutral execution in at-the-money options across the pair.

State inputs: AMD 30-day IV minus NVDA 30-day IV, plus days until earnings for each stock, assuming earnings dates within 30 days are known. Actions are long AMD/short NVDA, the reverse, or flat. Assume equal initial dollar vega and daily delta hedging; score trades using each leg's subsequent 30-calendar-day close-to-close realized volatility versus entry IV as a P&L proxy.

[Results](reports/earnings_pass.md): maximize annualized net payoff Sharpe with an assumed 0.5-vol-point round-trip pair cost. Test Sharpe: earnings-aware 0.392, spread-only 0.672, short-higher-IV baseline 0.956 (30-day horizon scaling).
