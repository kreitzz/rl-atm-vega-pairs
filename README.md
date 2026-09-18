# RL for Short-Term ATM Vega Pair Trading

Reinforcement learning for AMD/NVDA ATM options, assuming equal initial dollar vega and daily delta hedging. Score trades using subsequent 30-day realized versus entry implied volatility.

**Reference execution cost: 1.0 vol point per full-size pair round trip**, scaling with trade size. Current model: IV spread, earnings countdowns, 30-day trailing RV, and a 0/25/50/75/100% vega-sizing ladder. Annualized walk-forward payoff Sharpe: **0.667**.

[Reference model and yearly results](reports/reference_model.md) · [Cost sensitivity](reports/execution_costs.md) · [Prior feature experiments](reports/rolling_beta.md)

[Position and leg-level P&L breakdown](reports/reference_positions.md).

[Matched direction controls](reports/direction_controls.md): the policy beats constant directions, but the persistence-preserving randomization result is less decisive.
