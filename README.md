# RL for Short-Term ATM Vega Pair Trading

AMD/NVDA ATM vega-neutral pairs, assuming daily delta hedging. Current model uses IV spread, earnings countdowns, and trailing 30-day RV, with a 0/25/50/75/100% sizing ladder and 1.0-vol-point pair execution cost.

[Current finalized dataset (CSV)](data/current_dataset.csv): 5,685 daily observations, December 2002–August 2025. Includes all five state inputs, each leg’s entry IV and forward 30-day RV, the outcome date, and gross pair payoff. Forward RV and payoff are labels, not state inputs. Earnings countdowns assume actual dates within 30 days were known at entry.

[Current model, training command, and yearly results](reports/reference_model.md). Annualized walk-forward payoff Sharpe: **0.667**.

**Academic use only.** Third-party data remain subject to OptionMetrics, CRSP, Compustat, and institutional license terms; this notice grants no additional data rights.
