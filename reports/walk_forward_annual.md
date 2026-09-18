# Annual walk-forward P&L and Sharpe

Net P&L assumes $1 of initial dollar vega per leg on every nonflat entry and includes the 0.5-vol-point pair cost. No capital base is specified, so these are dollar P&L figures, not percentage returns. Multiply P&L by the dollar-vega sizing to scale the experiment.

Each trade’s full 30-day outcome is assigned to its entry year; this is not cash settlement-year or daily marked portfolio accounting. Sharpe uses the annualized 30-day payoff convention, including zero payoffs on flat entry dates. 2025 includes entries through August 29 and outcomes through September 28.

| Year | RV model net P&L | RV model Sharpe | RV + VIX net P&L | RV + VIX Sharpe |
| --- | ---: | ---: | ---: | ---: |
| 2010 | 623.19 | 0.839 | 1,074.56 | 1.421 |
| 2011 | -54.71 | -0.063 | -389.60 | -0.623 |
| 2012 | -2,286.25 | -2.153 | -2,072.51 | -1.905 |
| 2013 | 622.13 | 0.455 | -1,034.21 | -0.761 |
| 2014 | 2,132.95 | 2.756 | 415.63 | 0.361 |
| 2015 | 2,144.70 | 1.671 | -282.87 | -0.178 |
| 2016 | 1,754.97 | 0.671 | -379.53 | -0.276 |
| 2017 | 1,999.99 | 1.963 | 91.77 | 0.434 |
| 2018 | 925.84 | 0.738 | 1,154.67 | 0.893 |
| 2019 | 716.03 | 0.875 | 611.20 | 0.784 |
| 2020 | 498.88 | 0.517 | 436.11 | 0.447 |
| 2021 | -1.27 | -0.002 | 689.83 | 1.377 |
| 2022 | 360.67 | 0.585 | 360.67 | 0.585 |
| 2023 | -712.48 | -0.831 | -712.48 | -0.831 |
| 2024 | 230.18 | 0.352 | -317.86 | -0.633 |
| 2025 | 707.72 | 1.366 | 46.07 | 0.429 |

Both models also include IV spread and both earnings countdowns. See the [walk-forward methodology](walk_forward.md) and [all-strategy annual CSV](walk_forward_annual.csv).

Reproduce with `python scripts/report_annual.py` after the walk-forward run. Annual P&L sums reconcile to the overall result for every strategy.
