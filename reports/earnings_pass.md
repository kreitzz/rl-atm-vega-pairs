# Earnings-aware policy pass

Actual historical earnings dates are assumed known when they fall within the 30-calendar-day holding window. The policy uses AMD–NVDA 30-day IV spread plus days until earnings for each stock. In this first tabular pass, each countdown is bucketed into **0–30 days** or **more than 30 days**. No exact date beyond the holding window affects the action.

The policy maximizes annualized net payoff Sharpe with the same 0.5-vol-point round-trip pair cost. Validation selects 10 IV-spread buckets × 4 earnings combinations = 40 states. Training, validation, and test dates and the 30-day reward match the [spread-only run](first_pass.md).

| Strategy | Annualized test payoff Sharpe | Dates traded |
| --- | ---: | ---: |
| IV spread + earnings calendar | 0.392 | 30.5% |
| IV spread only | 0.672 | 31.5% |
| Short higher IV / long lower IV | 0.956 | 100% |

All figures use `sqrt(365/30) × mean(net 30-day payoff) / sample SD`, including flat entry dates. This is horizon-based annualization, not a daily marked portfolio Sharpe or serial-correlation adjustment. The earnings model's 95% block-bootstrap Sharpe interval is **[-0.126, 0.858]**. Its mean net payoff is 0.744 vol points per entry opportunity. All five selected-configuration seeds chose identical test actions.

The added earnings state did not improve this exploratory test. The test period has already been inspected during preceding iterations; it is not an untouched final holdout.

## Calendar and assumptions

The local Compustat RDQ calendar supplies historical reporting dates. AMD has 102 dates from April 18, 2001 through August 4, 2026. NVDA has 102 dates from February 14, 2001 through May 20, 2026 after the supplement below. Both cover every entry and outcome window in this backtest.

The source had a 182-day NVDA gap in 2006. NVIDIA published a quarterly revenue release on August 10, 2006; the public [SEC-filed release](https://www.sec.gov/Archives/edgar/data/1045810/000104581006000027/q207pressreleaseexhibit.htm) supplies that event in `config/earnings_supplement.csv`. It is included as an earnings-related reporting event; the rest of the calendar is not independently audited against issuer releases.

The data lacks historical schedule-publication timestamps. Knowing the actual date within 30 days is an explicit simulation assumption, not a verified property of every historical event. Release times are also absent: same-day earnings count as inside the holding window, potentially including a pre-close event already reflected in entry IV.

No realized earnings returns or financial results enter the state. The optional `--earnings-mode estimated` instead forecasts the next date using the median of up to eight preceding reporting intervals, with a 91-day fallback. The default actual-date experiment is the requested known-calendar assumption.

## Reproduce

```sh
python scripts/attach_earnings.py --source-root /path/to/convert-rv-rl/data/raw/wrds
python scripts/run_first_pass.py --source-root /path/to/convert-rv-rl/data/raw/wrds --earnings --earnings-mode actual --cost 0.5 --output artifacts/actual_earnings_sharpe_pass
python -m unittest discover -s tests -v
```

The local calendar and aligned daily panel are in `artifacts/earnings/`. Full model actions, policy table, bootstrap intervals, and settings are in `artifacts/actual_earnings_sharpe_pass/`. Licensed rows stay outside Git. Seven tests pass, covering date alignment, countdown encoding, annualization, cost treatment, training optimization, and outcome separation.
