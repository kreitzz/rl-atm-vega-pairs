# Contract-horizon realized-variance forecasts

First forecast each underlying’s close-to-close realized variance from next-session entry close through the selected call’s last trading day at expiry. Split-adjusted stock prices match the simulator convention; targets use 252 × mean squared log return, with no entry-day return. Unique security/date training rows avoid duplicating a stock merely because it occurs in several pairs.

Quarterly pooled log-variance ridge models use completed outcomes only, a five-year window, equal total weight per security, train-only clipping/imputation/scaling and a training-residual smearing correction. HAR uses trailing 5/21/63-session variance plus actual DTE. HAR+IV adds quoted-contract implied variance. The earnings variant additionally uses the uncapped countdown, event count inside the holding period and event count per holding session. No model or threshold was selected using these results.

**Earnings caveat:** event features use final historical earnings dates, assumed known on the signal date; no point-in-time announcement archive is available. The earnings-free variants are the primary comparison. Dates are daily, so before-open versus after-close timing is unresolved. Missing event coverage is imputed using training data, not treated as proof of no earnings.

Translate each forecast to approximate long-option profit per initial vega using 100 × (forecast variance − IV²)/(2 × IV), then subtract the two legs. Trade only when its magnitude exceeds the trailing completed-outcome estimate of symmetric directional net drag, equal-weighted across pairs. This is a historical empirical hurdle, not an exact decomposition of fees; both signs and all models use the same hurdle. Actual results use cached net daily-delta-hedged hold-to-expiry pair labels, including the existing execution, borrow and financing assumptions.

| Forecast | Trades | Net / initial vega per trade | Net / opportunity | Nonoverlap net / trade |
|---|---:|---:|---:|---:|
| har | 15,876 | -0.861 | -0.708 | -0.718 |
| har_iv | 9,684 | -1.210 | -0.607 | -0.011 |
| har_iv_earnings | 10,363 | -1.113 | -0.598 | -0.066 |
| trailing_rv21 | 16,121 | -0.901 | -0.752 | -0.479 |

Quoted-contract implied variance beat every tested forecast on both mean squared error and QLIKE loss (lower is better). All translated trading rules had negative average net payoff in the full and nonoverlapping samples. Each fitted rule was negative in five of six evaluation calendar years; 2025 is partial.

| Variance forecast | Mean squared error | QLIKE |
|---|---:|---:|
| implied_variance | 0.1189 | 0.3085 |
| trailing_rv21 | 0.1871 | 0.4580 |
| har | 0.1408 | 0.4166 |
| har_iv | 0.1349 | 0.3557 |
| har_iv_earnings | 0.1348 | 0.3605 |

Matched 19,310 of 19,310 previously evaluable pair opportunities; missing histories or feature warmup are excluded identically across models.

This is a signal/standalone-payoff test, not a financed portfolio backtest: it does not enforce overlapping-book capacity or establish annualized Sharpe. Terminal cohort P&L cannot be used as daily returns. Missing/resolved outcomes condition the sample; the prior beta/membership screens remain. Nonoverlap thinning is diagnostic and does not eliminate dependence between pairs. Historical research data has already been repeatedly explored. Vega is matched at entry, not rebalanced daily. The variance-to-vega approximation omits path-dependent gamma weighting and jumps; improved variance accuracy need not yield improved trading returns.

[Forecast accuracy](variance_forecast/forecast_accuracy.csv) · [Annual pair outcomes](variance_forecast/annual.csv) · [Fit timing](variance_forecast/fit_audit.csv) · [Historical cost hurdles](variance_forecast/cost_hurdles.csv)


Checks: all matched pair outcome-end dates equal the later leg expiry session; completed training ends precede prediction quarters. Three unit tests cover holding-window boundaries, uncapped earnings horizons, and held-out-target independence. Peak worker memory: 0.43 GiB; one worker at 50% duty cycle.
