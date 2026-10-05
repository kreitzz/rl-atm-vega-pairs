# NAV-scaled pair-trading reference

Risk controls are now on by default in `scripts.run_selected_pair_backtest`, with halfway-to-mid option fills. Legacy fixed-dollar sizing requires `--no-risk-controls`; old results remain in separate folders. The policies remain frozen, without retraining or tuning limits against the new returns.

| Rule, per independent pair account | Setting |
| --- | --- |
| Full-size new-entry vega per leg | $450 × current pre-entry NAV / $1 million, then multiplied by the policy's size |
| Gross outstanding vega cap | $20,000 × current NAV / $1 million |
| Gross stock hedge notional cap | 2× NAV |
| Capital reserve | 50% of gross stock hedge notional + gross option midpoint value, capped at 80% NAV |
| Stress-loss budget | At most 5% NAV |
| Stress scenarios | Independently shock each stock by −20%, 0%, +20% and its IV by −10, 0, +10 vol points |

The stress model reprices calls with the existing Black–Scholes/dividend approximation at unchanged maturity, keeps the current delta hedge fixed through the instantaneous shock, and sums the worst loss for each name. It credits no correlation between the two underlyings. IV has a positive floor. The reserve is a conservative research approximation, **not broker-approved margin**. The 5% stress budget is not a guaranteed maximum realized loss.

Sizing uses today's marked account equity. Entry gates use projected NAV after option execution fees/spread and the anticipated stock-hedge transaction cost. Existing books are also checked every day: if a limit is exceeded, close the oldest whole pair cohorts at the configured execution prices until compliant. No new cohort is entered on a day of forced risk reduction. Overnight losses occur before this close-based intervention; no losses are erased or backfilled.

## Results

All four RL accounts complete the full history without capital exhaustion. Returns include cash and are cumulative for the indicated period. Sharpe is annualized excess-over-cash Sharpe based on daily NAV returns.

| Four-pair sized RL portfolio | Return | Above cash, initial-capital basis | Annualized excess Sharpe | Maximum drawdown |
| --- | ---: | ---: | ---: | ---: |
| January 2010–August 2025 | -5.12% | -30.72% | -1.66 | -16.59% |
| January–August 2025 | +4.12% | +1.11% | 1.15 | -0.17% |

The 2025 NN portfolio returns 3.51%, with excess Sharpe 0.52 and maximum drawdown -0.41%. Its full-history AMD/NVDA run stops on an unsupported underlying split adjustment on July 20, 2021. No full-history NN portfolio result omits that failure.

| RL pair, full history | Cumulative return | Maximum drawdown | Worst daily return |
| --- | ---: | ---: | ---: |
| AMD/NVDA | -35.81% | -50.21% | -3.88% |
| XOM/CVX | +0.98% | -10.32% | -0.66% |
| JPM/BAC | +7.03% | -7.59% | -1.16% |
| HD/LOW | +7.33% | -6.16% | -1.32% |

This addresses the inappropriate fixed-dollar exposure as capital falls, but does not establish a profitable signal after costs. All four full-history RL pair accounts underperform cash. Cumulative drawdown remains possible even with daily risk limits.

The July 2014 AMD/NVDA example changes substantially: on July 17 the new run has $811,363 NAV and one active cohort, versus $56,911 and 18 cohorts in the previous halfway-fill run. Its gross stock hedge notional is $508,332. Positions differ because the risk gates reject/close trades; the subsequent day's gain is not a like-for-like reduction of the old losing position.

## Validation and remaining limits

All completed runs reconcile option and stock cash flows to NAV and close their original contract positions. Every end-of-day book satisfies the configured hedge, reserve, stress and vega caps, and every new entry respects its NAV-scaled vega budget. The historical RL book's maximum gross stock hedge leverage is 1.25× at the pair-account level. Unit tests: 80 passing. Peak experiment memory: about 402 MiB.

The execution scenario still assumes every order fills halfway between mid and the relevant quoted side, plus $0.65/contract and the existing hedge/financing costs. It retains explicit crossed-quote envelope repairs: two held quotes in the full-history AMD/NVDA RL run and one in its separate 2025 run. This is not a fresh holdout, broker-margin backtest, or simulation of limit-order fill probability. Corporate-action handling remains incomplete as noted above.

[All pair outcomes](selected_pair_backtest/risk_controlled/status.csv) · [Portfolio results](selected_pair_backtest/risk_controlled/portfolio.csv) · [Annual portfolio returns](selected_pair_backtest/risk_controlled/portfolio_annual.csv) · [Risk-limit audit](selected_pair_backtest/risk_controlled/risk_audit.csv) · [Comparison with fixed-dollar sizing](selected_pair_backtest/risk_controlled/portfolio_comparison.csv).

```sh
python -m scripts.run_selected_pair_backtest --reuse-cache --cache-root artifacts/contracts/four_pairs/cache --risk-controls --repair-crossed-held-quotes
python -m scripts.report_risk_controlled
python -m unittest discover -s tests -q
```
