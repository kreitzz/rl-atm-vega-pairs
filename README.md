# RL for Short-Term ATM Vega Pair Trading

Exploring reinforcement learning for short-term AMD/NVDA options pair trading, assuming vega-neutral execution in at-the-money options across the pair.

First pass: the sole state input is AMD 30-day IV minus NVDA 30-day IV. Actions are long AMD/short NVDA, the reverse, or flat. Assume equal initial dollar vega and daily delta hedging; score trades using each leg's subsequent 30-calendar-day close-to-close realized volatility versus entry IV as a P&L proxy.
