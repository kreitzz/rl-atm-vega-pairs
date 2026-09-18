# RL for Short-Term ATM Vega Pair Trading

Reinforcement learning for AMD/NVDA ATM options, assuming equal initial dollar vega and daily delta hedging. Score trades using subsequent 30-day realized versus entry implied volatility.

**Reference execution cost: 1.0 vol point per full-size pair round trip**, scaling with trade size. Current model: IV spread, earnings countdowns, 30-day trailing RV, and a 0/25/50/75/100% vega-sizing ladder. Annualized walk-forward payoff Sharpe: **0.667**.

[Reference model and yearly results](reports/reference_model.md) · [Cost sensitivity](reports/execution_costs.md) · [Prior feature experiments](reports/rolling_beta.md)

[Position and leg-level P&L breakdown](reports/reference_positions.md).

[Matched direction controls](reports/direction_controls.md): the policy beats constant directions, but the persistence-preserving randomization result is less decisive.

**Current dataset:** [Download/view the AMD/NVDA CSV](artifacts/short_rv_momentum/dataset.csv) · [Parquet for training](artifacts/short_rv_momentum/dataset.parquet). It contains **5,685 daily observations from December 31, 2002 through August 29, 2025**, with 36 columns plus the entry date and no missing values. The start reflects feature warm-up; walk-forward evaluation begins in 2010.

| Columns | Meaning |
| --- | --- |
| `iv_amd`, `iv_nvda`, `spread` | 30-day 50-delta call IV as the ATM proxy; spread = AMD minus NVDA. Volatility is in annualized percentage points. |
| `amd_earnings_days_capped`, `nvda_earnings_days_capped` | Days to actual earnings, capped at 31; assumes earnings dates within the 30-day horizon were known at entry. Historical announcement-time availability is not verified. |
| `trailing_rv_amd`, `trailing_rv_nvda` | Trailing 30-calendar-day close-to-close RMS log-return volatility, annualized using 252 trading days. |
| `vix_previous_close`, `rv_5td_*`, `rv_10td_*`, `momentum_12_1_*` | Extra experimental inputs: prior available VIX close, shorter trailing RV, and 12-minus-1-month total-return momentum. Excluded from the current reference state. |
| `end`, `rv_amd`, `rv_nvda`, `long_reward` | Forward 30-calendar-day evaluation endpoint, subsequent realized volatilities, and gross long-AMD/short-NVDA payoff: `(RV_AMD − IV_AMD) − (RV_NVDA − IV_NVDA)`. These are outcome labels, not policy inputs. |

Other columns retain earnings dates and diagnostic flags. Use only the five inputs in [the reference feature configuration](config/reference_features.json) to reproduce the current state. Payoffs approximate daily delta-hedged, initially vega-neutral trading; they are not actual option-and-hedge P&L.

`git pull` includes these files, earlier experiment panels, [reference predictions](artifacts/execution_costs/cost_1/predictions.csv), and [Cboe VIX history](data/raw/VIX_History.csv). Install `requirements.txt`, then use [the reference training command](reports/reference_model.md); rebuilding panels from source requires the original WRDS data.

**Academic-use license notice:** Shared for noncommercial academic research by authorized collaborators. OptionMetrics, CRSP, and Compustat data sourced through WRDS, and Cboe data, remain subject to their respective provider and institutional license terms. This repository grants no additional rights to third-party data; access and redistribution require permission under those licenses. An academic purpose alone does not grant that permission.
