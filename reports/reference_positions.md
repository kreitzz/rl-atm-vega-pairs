# Reference-policy position and P&L breakdown

[Matched always-long, always-short, and randomized-direction controls](direction_controls.md).

Reference: 1.0 vol point per full-size pair round trip; current IV spread, earnings countdowns, 30-day trailing RV, and the dynamic vega-sizing ladder. Attribution covers 3,922 walk-forward entry dates, January 2010–August 2025.

The policy tilts toward **short AMD vega / long NVDA vega**, but it is not a constant-direction strategy. That direction accounts for 60.3% of actual trades, 60.8% of summed absolute entry allocations, and 67.2% of net proxy P&L. The reverse direction contributes the remaining 32.8% of net P&L. There are 867 flat dates (22.1% of entry opportunities).

## Direction and leg attribution

P&L is in simulated dollars per **$1 of maximum initial dollar vega per leg**. Every trade uses its actual ladder fraction; multiply the figures by the chosen maximum dollar vega to scale. These are sums of 30-day proxy payoffs, not percentage returns or daily marked option P&L.

| Direction | Trades | Share of trades | Size-weighted share | AMD gross P&L | NVDA gross P&L | Pair costs | Net P&L |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Short AMD / long NVDA | 1,841 | 60.3% | 60.8% | 5,383.74 | 1,009.20 | 1,044.00 | **5,348.93** |
| Long AMD / short NVDA | 1,214 | 39.7% | 39.2% | 2,109.10 | 1,178.29 | 673.25 | **2,614.14** |
| Flat | 867 | — | — | 0.00 | 0.00 | 0.00 | 0.00 |
| Total | 3,055 traded | 100% | 100% | **7,492.84** | **2,187.49** | **1,717.25** | **7,963.08** |

AMD accounts for 77.4% of gross P&L, NVDA 22.6%. Costs are retained at pair level because the model does not identify actual per-leg execution charges. Short-AMD trades have a 59.5% net win rate; long-AMD trades have a 51.6% net win rate. Average sizes conditional on trading are similar: 56.7% versus 55.5% of maximum.

The tilt is not stable across all subperiods: since 2021, short AMD represents only 40.8% of trades and 42.3% of summed absolute entry size. “Size-weighted share” measures allocations at entry, not time-weighted live inventory or marked Greeks.

## Attribution formula

For signed entry size `a` (positive = long AMD):

- AMD gross = `a * (RV_AMD - IV_AMD)`.
- NVDA gross = `-a * (RV_NVDA - IV_NVDA)`.
- Total pair cost = `1.0 * abs(a)`.
- Net = AMD gross + NVDA gross - pair cost.

Dollar vega is held constant in this proxy. Actual changing Greeks, daily hedge cash flows, and intraperiod marks are not decomposed. Each full 30-day outcome is assigned to entry year in the annual table; 2025 entries end August 29 and outcomes extend through September 28.

## Detailed outputs

- [Direction totals](reference_positions/directions.csv)
- [Year-by-year direction and leg P&L](reference_positions/annual_directions.csv)
- [Sizing-level breakdown](reference_positions/sizes.csv)
- [Total leg attribution](reference_positions/legs.csv)

Reproduce with `python scripts/report_positions.py`. Checks reconcile both gross legs to the original trade reward, infer and verify the 1.0-point proportional cost, and reconcile all direction totals to the reference policy's net P&L.
