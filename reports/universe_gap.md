# Wider pair universe: stock-versus-IV dependence gaps

Eight additional pairs were specified from business similarity and local data availability before their payoff results were examined: AMAT/LRCX, MSFT/ORCL, UPS/FDX, UNP/CSX, T/VZ, DUK/SO, AMGN/GILD, and C/WFC. They join the four existing pairs. This remains a handpicked universe of available historical survivors, not systematic market-wide selection or an untouched holdout.

Each pair has its own independently trained policy and sizing ladder. Compare the existing five state inputs with either one additional correlation-gap input or one additional beta-gap input. There is no joint state, cross-pair decision, or transfer of unused budgets.

## Equal-budget portfolio results

Each portfolio allocates a fixed equal share of its entry budget to each constituent pair. A variant uses the same feature specification for every constituent; there is no test-selected mixture of pair-specific winners.

| Universe | Baseline | + Correlation gap | + Beta gap |
| --- | ---: | ---: | ---: |
| Eight new pairs | -0.071 | 0.034 | -0.232 |
| Four original pairs | 0.517 | 0.400 | 0.410 |
| All twelve pairs | 0.317 | 0.326 | 0.154 |

The correlation gap helps most new pairs individually, but the eight-pair portfolio remains close to flat and the twelve-pair improvement is small. Every paired-bootstrap interval below includes zero. The beta-gap variant is weaker at portfolio level. This is useful candidate discovery, not confirmation of a generalizable diversification benefit.

## Every pair

Annualized 30-day proxy-payoff Sharpe on the same 3,918 entry dates, 2010–August 2025. No pair is omitted for poor performance.

| Pair | Baseline | + Correlation gap | + Beta gap |
| --- | ---: | ---: | ---: |
| AMD/NVDA | 0.635 | 0.449 | 0.481 |
| XOM/CVX | -0.070 | 0.090 | -0.011 |
| JPM/BAC | -0.011 | -0.014 | -0.084 |
| HD/LOW | -0.042 | -0.126 | -0.068 |
| AMAT/LRCX | 0.284 | 0.453 | 0.416 |
| MSFT/ORCL | 0.192 | 0.206 | -0.300 |
| UPS/FDX | -0.187 | -0.191 | -0.221 |
| UNP/CSX | 0.260 | 0.359 | 0.028 |
| T/VZ | -0.184 | -0.068 | -0.174 |
| DUK/SO | 0.124 | 0.160 | 0.016 |
| AMGN/GILD | -0.048 | -0.003 | 0.091 |
| C/WFC | -0.492 | -0.463 | -0.524 |

Among the eight additional pairs, adding the correlation gap improves Sharpe for **7/8**, and adding the beta gap improves it for **3/8**. These counts compare full feature ablations, not the conditional subgroup Sharpes from the earlier diagnostic. A favorable subgroup relationship does not guarantee that a refitted tree will use the extra feature effectively.

## Uncertainty in the portfolio difference

Paired 63-entry circular moving-block bootstrap, 2,000 replicates. Each draw resamples the same calendar blocks for all pairs and variants, preserving their contemporaneous dependence. Intervals describe sampling variation conditional on the fitted policies and this sample; they do not include retraining uncertainty or correct for trying multiple features and pairs.

| Universe | Feature added | Sharpe change | 95% paired block interval |
| --- | --- | ---: | --- |
| all | correlation gap | +0.009 | [-0.284, +0.297] |
| all | beta gap | -0.163 | [-0.443, +0.137] |
| new | correlation gap | +0.105 | [-0.210, +0.400] |
| new | beta gap | -0.161 | [-0.388, +0.077] |
| original | correlation gap | -0.117 | [-0.453, +0.234] |
| original | beta gap | -0.107 | [-0.433, +0.218] |

## Dependence profiles

Median trailing estimates during the test period. Betas regress the first ticker on the second. These are stock-return and IV-change regression betas, not market-index beta or option-implied equity beta. Each gap is calculated date by date before its median is taken, so its median need not equal the difference between the component medians.

| Pair | Stock correlation | IV-change correlation | Correlation gap | Stock beta | IV beta | Beta gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AMD/NVDA | 0.566 | 0.445 | 0.114 | 0.688 | 0.421 | 0.158 |
| XOM/CVX | 0.817 | 0.812 | -0.000 | 0.729 | 0.760 | 0.006 |
| JPM/BAC | 0.857 | 0.802 | 0.039 | 0.704 | 0.667 | 0.026 |
| HD/LOW | 0.782 | 0.651 | 0.103 | 0.677 | 0.627 | 0.037 |
| AMAT/LRCX | 0.858 | 0.683 | 0.162 | 0.842 | 0.684 | 0.083 |
| MSFT/ORCL | 0.532 | 0.518 | 0.019 | 0.482 | 0.498 | 0.030 |
| UPS/FDX | 0.665 | 0.558 | 0.106 | 0.526 | 0.494 | 0.041 |
| UNP/CSX | 0.735 | 0.625 | 0.112 | 0.703 | 0.536 | 0.153 |
| T/VZ | 0.709 | 0.614 | 0.068 | 0.728 | 0.715 | 0.031 |
| DUK/SO | 0.791 | 0.619 | 0.166 | 0.786 | 0.585 | 0.170 |
| AMGN/GILD | 0.517 | 0.484 | 0.012 | 0.456 | 0.382 | 0.002 |
| C/WFC | 0.730 | 0.817 | -0.057 | 0.826 | 0.848 | -0.009 |

The conditional-group CSV separately checks whether the baseline policy does better when each gap exceeds the preceding training sample's median. These are conditional Sharpes of existing decisions, not the performance of a deployable filter. No group thresholds were optimized for test outcomes.

## Data and method

- 5,678 common complete model rows, December 31, 2002–August 29, 2025; all variants and pairs share the calendar. The four omitted dates relative to the original 3,922-date test are July 2, 2010; May 9, 2011; July 3, 2012; and April 11, 2022. Training dates also differ slightly, so baseline numbers can change from earlier reports through tree/configuration selection. Compare variants within this report.
- Dependence estimates use a trailing 252-session window, with at least 200 paired valid observations and a full window of elapsed history. IV changes are differences in vol points; stock changes use CRSP ex-distribution daily returns. Neither quotes nor returns are filled across gaps. Both covariance and variance use the same matched sample. The initial perfectly-complete-window approach was discarded after its data-coverage audit excluded entire years; a small number of original-pair results had been printed before that correction. No new-pair policy result was used to choose the missing-data rule.
- State is pair IV spread, two earnings countdowns, and two trailing 30-calendar-day RVs. Added features are `stock_return_correlation - IV_change_correlation` or `stock_return_beta - IV_change_beta`, using data through entry close. This retains the existing entry-close timing assumption and actual-earnings-dates-known-within-30-days assumption.
- Pair cost is `0.025 * (IV_A + IV_B) / 2`, multiplied by absolute allocation. Each pair chooses signed sizes in 25% increments up to its fixed budget; unused allocation stays idle. The budget applies per entry cohort, not total overlapping positions.
- Expanding training, preceding two-year validation, subsequent two-year test blocks, with future payoffs purged at boundaries. For each pair/variant/fold, select 4/8/16 tree leaves and 126/252 minimum leaf counts by validation Sharpe. Each candidate uses five seeds, 80 policy-search iterations and 256 candidates. No hyperparameters were retuned to the additional pairs' test results.
- The quarterly earnings audit found three LRCX 2007 fiscal periods assigned a later common date in Compustat. Overrides use the original releases: [July 24, 2007 SEC filing](https://www.sec.gov/Archives/edgar/data/707549/000095013407015737/f32108e8vk.htm), [October 10, 2007 company release](https://newsroom.lamresearch.com/2007-10-10-Lam-Corporation-Announces-Selected-Financial-Results-for-the-Quarter-Ended-September-23%2C-2007), and [January 24, 2008 SEC filing](https://www.sec.gov/Archives/edgar/data/707549/000089161808000051/f37301e8vk.htm). Corrections were applied before any AMAT/LRCX model was evaluated.

Sharpes retain `sqrt(365/30)` scaling of overlapping 30-day realized-minus-implied proxy payoffs, including flat dates. They are not daily marked option-and-hedge portfolio Sharpes. Results are exploratory; finding profitable candidates or a feature winner in this scan is not independent confirmation of an edge. The current four-pair default is unchanged.

[All pair scores](universe_gap/scores.csv) · [Annual pair results](universe_gap/annual.csv) · [Portfolio comparisons](universe_gap/portfolio.csv) · [Annual portfolio results](universe_gap/portfolio_annual.csv) · [Dependence profiles](universe_gap/dependence.csv) · [Conditional groups](universe_gap/bins.csv) · [Coverage](universe_gap/coverage.csv) · [Selected configurations](universe_gap/configs.csv)

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.explore_universe_gap --source-root /path/to/wrds
python -m scripts.report_universe_gap
python -m unittest discover -s tests -q
```

Detailed panels and predictions stay local under ignored `artifacts/universe_gap/`. Only configuration, code, tests, and aggregate reports are published. Thirty-seven tests pass, including matched-observation beta calculations, future-data exclusion, earnings overrides, and paired-bootstrap identity.
