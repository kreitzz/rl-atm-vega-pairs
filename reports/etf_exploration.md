# ETF/stock and ETF/ETF exploration

The available local cache supports thirteen prespecified candidates: ten main ETF/stock pairs, one longer-history ETF/ETF comparison, and two short-history comparisons. Each pair has a separate policy, its own sizing ladder, and a fixed budget. No shared state or joint allocation policy is used.

The local download does **not** include the suggested SPY, QQQ, SMH, SOXX, XLF, XLE, or IWM surface series. This test uses the ETF histories actually available rather than assuming those missing data exist. Candidates were specified from related economic exposures and coverage before their policy results were examined; high correlation is measured, not presumed. Historical ETF constituent membership was not reconstructed.

## Findings and input-quality checks

The main basket's baseline Sharpe is 1.036, versus 1.051 with the correlation gap and 0.947 with the beta gap. The gap additions do not show a robust basket-level improvement, and the simple short-higher-IV rule scores 1.131 on the identical calendar. The headline basket result is concentrated in the silver pairs: removing SIL/HL lowers baseline Sharpe to 0.551, and removing both silver pairs lowers it to 0.027. These exclusions diagnose concentration; they are not newly selected portfolios.

- **SIL/AG** is the strongest candidate for follow-up that is less dependent on extreme marks. Full available-period Sharpe is 1.115 baseline, 1.208 with correlation gap, and 1.330 with beta gap. About 50.5% of net baseline P&L comes from the 4.7% of dates when AG IV exceeds 100 vol. Skipping those dates, with zero payoff on the full calendar, still yields 0.896. This threshold is a descriptive sensitivity check, not an adopted or independently validated trading rule.
- **SIL/HL** has 1.118 baseline Sharpe over its longer history, but 96.2% of its profit comes from the 16.0% of dates when HL IV exceeds 100 vol. Skipping them leaves Sharpe 0.082. HL's maximum surface IV is 282.7 vol and its 99th percentile is 206.6. Those marks are not proved erroneous, but their contribution requires validation against actual contract quotes. Much of this result is persistent long-SIL/short-HL exposure; a constant-direction control outperforms the learned baseline.
- **GSLC/SPYM** has 2.261 Sharpe for all three learned variants, with identical decisions, versus 2.539 for simply shorting the higher-IV fund. There are only 417 test entries. GSLC has zero aggregate call-option volume on 71.5% of those dates, median call volume zero, and median call open interest 60. Surface IV coverage therefore does not establish executable liquidity, and the assumed cost is not verified. This is an especially weak basis for claiming a tradable edge.
- The oil-services pairs show the intended high-return-correlation/lower-IV-correlation relationship, but adding the gap does not improve their results consistently. Pharmaceutical pairs have weaker return correlation and mixed feature gains. A large gap alone is not a demonstrated selection rule.

The activity figures cover 2024–August 2025 and aggregate all expirations/strikes by call/put side; they are not specifically ATM 30-day contract liquidity. SIL, HL, and AG each have positive call volume on all observed dates in that window, but this does not validate their particular surface marks or execution costs. [Activity audit](etf_exploration/activity_audit.csv) · [Extreme-IV attribution](etf_exploration/extreme_iv_attribution.csv) · [Concentration checks](etf_exploration/concentration_checks.csv).

## Main ETF/stock portfolio

The ten main pairs share the evaluation dates below, with equal fixed entry budgets and unused allocation left idle. Several pairs reuse the same ETF, so this is concentrated exposure, not ten independent bets.

| Model | Annualized payoff Sharpe | Change versus baseline | 95% paired block interval for change |
| --- | ---: | ---: | --- |
| baseline | 1.036 | +0.000 | [+0.000, +0.000] |
| correlation gap | 1.051 | +0.014 | [-0.325, +0.260] |
| beta gap | 0.947 | -0.089 | [-0.358, +0.182] |

Common test period: **2018-01-02–2025-08-29**, **1,906 entries**. The following controls use precisely that same calendar and budget, always trading each pair at full size. They distinguish a persistent directional volatility-premium difference from learned timing and sizing; unlike the learned policies, they never choose to sit out.

- always long etf: **0.587**.
- always short etf: **-1.676**.
- short higher iv: **1.131**.

## Matched main-pair results

All entries in this table use the common main-portfolio dates above. These may differ from a pair's full available test history in the downloadable scores CSV.

| Pair | Baseline | + Correlation gap | + Beta gap | Always long ETF / short stock |
| --- | ---: | ---: | ---: | ---: |
| XOP/COP | -0.156 | 0.196 | -0.282 | -0.875 |
| XOP/DVN | -0.069 | -0.070 | -0.384 | -0.582 |
| XOP/APA | -0.109 | -0.151 | -0.154 | -0.399 |
| OIH/SLB | 0.181 | -0.288 | 0.041 | -0.503 |
| OIH/HAL | 0.028 | -0.063 | -0.073 | -0.359 |
| PPH/MRK | 0.276 | 0.444 | 0.555 | -0.764 |
| PPH/BMY | 0.169 | 0.113 | 0.237 | -0.234 |
| PPH/AMGN | 0.326 | -0.112 | 0.537 | -0.558 |
| SIL/HL | 1.234 | 1.190 | 1.113 | 1.502 |
| SIL/AG | 1.112 | 1.209 | 1.333 | 0.579 |

## ETF/ETF and short-history comparisons

These are reported separately rather than shortening the main test to their common history. OIH/XOP compare oil services with exploration/production, not identical portfolios. GSLC/SPYM compare broad U.S. large-cap exposures with different weighting/index rules. ARKK/TSLA is a thematic ETF/stock comparison. A short period cannot establish robustness.

| Pair | Test start | Entries | Baseline | + Correlation gap | + Beta gap |
| --- | --- | ---: | ---: | ---: | ---: |
| OIH/XOP | 2018-01-02 | 1907 | -0.208 | -0.424 | -0.334 |
| ARKK/TSLA | 2024-01-02 | 417 | 0.180 | 0.867 | -1.319 |
| GSLC/SPYM | 2024-01-02 | 417 | 2.261 | 2.261 | 2.261 |

## Are returns tightly linked while IV behaves differently?

Median trailing 252-session estimates over each pair's full eligible test history. Beta regresses the first ticker (ETF) on the second ticker; a beta below one can reflect lower ETF volatility and is not by itself a weak relationship. IV changes use percentage-point differences. Correlation gap is calculated date by date, so its median need not equal the difference between the medians of the components.

| Pair | Return correlation | IV-change correlation | Correlation gap | Return beta | IV-change beta |
| --- | ---: | ---: | ---: | ---: | ---: |
| XOP/COP | 0.828 | 0.672 | 0.127 | 0.928 | 0.780 |
| XOP/DVN | 0.873 | 0.740 | 0.132 | 0.726 | 0.700 |
| XOP/APA | 0.836 | 0.700 | 0.117 | 0.674 | 0.700 |
| OIH/SLB | 0.912 | 0.679 | 0.214 | 0.959 | 0.619 |
| OIH/HAL | 0.923 | 0.649 | 0.267 | 0.861 | 0.542 |
| PPH/MRK | 0.615 | 0.438 | 0.170 | 0.448 | 0.550 |
| PPH/BMY | 0.571 | 0.358 | 0.242 | 0.380 | 0.424 |
| PPH/AMGN | 0.547 | 0.470 | 0.156 | 0.400 | 0.548 |
| SIL/HL | 0.856 | 0.027 | 0.748 | 0.530 | 0.001 |
| SIL/AG | 0.866 | 0.243 | 0.546 | 0.515 | 0.216 |
| OIH/XOP | 0.885 | 0.779 | 0.101 | 0.947 | 0.782 |
| ARKK/TSLA | 0.634 | 0.432 | 0.184 | 0.403 | 0.305 |
| GSLC/SPYM | 0.990 | 0.374 | 0.617 | 0.989 | 0.859 |

High return correlation and an IV-correlation gap are not sufficient evidence of mispricing. ETF diversification, changing holdings, and constituent earnings can create structural differences. The feature ablations test whether the current policy family can use these estimates; they do not establish the optimal model or rule out an edge in other formulations.

## Method and data limits

- Baseline states are IV spread, both trailing 30-day RVs, and the stock's earnings countdown when applicable. ETF countdowns are constant 31 (no corporate earnings event), so ETF/stock policies have four varying baseline inputs and ETF/ETF policies have three. This does not model the ETF's weighted constituent earnings calendar.
- Added features are trailing return correlation minus IV-change correlation, or trailing return beta minus IV-change beta. Windows span 252 sessions with at least 200 paired observations, using data through entry close. No missing quotes are filled. Same-close execution remains the existing proxy assumption.
- Independent signed sizing ladders run from zero to full size in 25% increments. Pair cost is `0.025 * (IV_A + IV_B) / 2` in vol points, scaled by absolute size. Budget caps are per new cohort, not all overlapping open positions. Shared ETF positions are not netted across pair books for costs.
- Two-year validation and test blocks follow expanding training, with 30-day outcomes purged at boundaries. A fold needs at least 504 training, 252 validation, and 126 test rows. This yields different initial test years for newer ETFs. Tree configurations, search seeds, iterations, and population match the earlier experiments; no configuration grid was tuned on the ETF test outcomes.
- OIH and PPH use only their post-December-2011 ETF-era return identifiers and corresponding IV history; pre-conversion HOLDRS histories are excluded. [OIH fund description](https://www.vaneck.com/us/en/investments/oil-services-etf-oih?redirectVE=generic), [PPH fund description](https://www.vaneck.com/us/en/investments/pharmaceutical-etf-pph/).
- [XOP](https://www.ssga.com/us/en/individual/etfs/state-street-spdr-sp-oil-gas-exploration-production-etf-xop) covers oil/gas exploration and production; [SIL](https://www.globalxetfs.com/funds/sil/) covers silver miners. Their related-stock candidates are not assumed to have constant historical ETF weights.
- SPYM is the current identifier label for the historical fund series. Its benchmark changed to the S&P 500 in January 2020; earlier continuous local history from 2018 uses the preceding large-cap benchmark. [Issuer benchmark history](https://www.ssga.com/us/en/individual/etfs/state-street-spdr-portfolio-sp-500-etf-spym). GSLC uses an [ActiveBeta large-cap index](https://am.gs.com/public-assets/documents/575230c8-24d6-11ef-870d-5172ed5a38e2). Their test begins in 2024 and is too short for strong conclusions.

All results are annualized 30-day realized-minus-implied proxy-payoff Sharpes, not actual option-and-hedge daily returns. Constant initial vega and a common cost formula remain approximations. The pairs are handpicked from the available cache, the historical data have been repeatedly inspected, and multiple models are being compared. Paired 63-entry block intervals use 2,000 resamples and do not adjust for model selection or include retraining uncertainty.

[All pair scores and allocations](etf_exploration/scores.csv) · [Annual pair results](etf_exploration/annual.csv) · [Dependence profiles](etf_exploration/profiles.csv) · [Constant controls](etf_exploration/controls.csv) · [Leg attribution](etf_exploration/legs.csv) · [Portfolio results](etf_exploration/portfolio.csv) · [Annual portfolio results](etf_exploration/portfolio_annual.csv) · [Coverage](etf_exploration/coverage.csv) · [Conditional groups](etf_exploration/conditional_groups.csv)

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.explore_etf_pairs --source-root /path/to/wrds
python -m scripts.audit_etf_inputs --source-root /path/to/wrds
python -m scripts.report_etf_pairs
python -m unittest discover -s tests -q
```

Data panels and detailed predictions remain local under ignored `artifacts/etf_exploration/`. The repository's shared market-data CSV is unchanged. Forty tests pass, including ETF/stock earnings handling, sample-size gates, dependence calculations, payoff accounting, and purging. The four-pair default is unchanged.
