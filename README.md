# RL for Short-Term ATM Vega Pair Trading

Exploring reinforcement learning for short-term AMD/NVDA options pair trading, assuming vega-neutral execution in at-the-money options across the pair.

State inputs: AMD–NVDA 30-day IV spread, days until earnings for each stock (assumed known within 30 days), trailing 30-day realized volatility for each stock, and prior-close VIX. Actions are long AMD/short NVDA, the reverse, or flat. Assume equal initial dollar vega and daily delta hedging; score trades using subsequent 30-day realized versus entry implied volatility.

[Walk-forward results, 2010–2025](reports/walk_forward.md): annualized net payoff Sharpe after an assumed 0.5-vol-point round-trip pair cost: spread 0.215, + earnings 0.366, + RV 0.527, + VIX -0.020; simple short-higher-IV baseline 0.444.

[Annual P&L and Sharpe](reports/walk_forward_annual.md).

[Short RV and 12–1 momentum experiment](reports/short_rv_momentum.md): on matched training dates, 30-day RV remains strongest; 5/10-day RV and momentum did not improve results.

[Dynamic entry-sizing ladder](reports/sizing_ladder.md): 0/25/50/75/100% of maximum vega per dollar. The 30-day RV model's annualized payoff Sharpe rises from 0.692 to 0.735; the improvement remains uncertain.

[Historical spread experiment](reports/spread_history.md): a trailing 63-observation z-score scores 0.729 with the ladder versus 0.735 without it; no clear improvement.

[Rolling beta experiment](reports/rolling_beta.md): 63/126-day AMD-on-NVDA return beta did not improve the current best model.
