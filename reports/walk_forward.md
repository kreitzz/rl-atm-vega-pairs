# Expanded walk-forward evaluation: 2010–2025

The original test covered January 2021–August 2025: 1,170 daily entries but only about 56 nonoverlapping 30-day horizons. The expanded evaluation covers January 4, 2010–August 29, 2025, with **3,922 entries across eight chronological test blocks**, plus a **188-entry nonoverlapping diagnostic**.

The broader evaluation gives a more favorable result for earnings and trailing RV, while the full model including VIX still performs poorly. This also introduces regular model refreshes, so changes from the old fixed-model result cannot be attributed to test length alone.

## Results

| Feature set / strategy | Annualized payoff Sharpe | 95% block-bootstrap interval | Nonoverlapping entry Sharpe | Dates traded |
| --- | ---: | --- | ---: | ---: |
| IV spread | 0.215 | [-0.219, 0.717] | 0.093 | 55.7% |
| + earnings | 0.366 | [-0.031, 0.734] | 0.401 | 57.4% |
| + earnings + trailing RV | 0.527 | [0.156, 0.902] | 0.692 | 81.9% |
| + earnings + trailing RV + VIX | -0.020 | [-0.347, 0.332] | -0.008 | 69.5% |
| Feature set selected on preceding validation | 0.438 | [0.057, 0.798] | 0.378 | 65.5% |
| Short higher IV / long lower IV | 0.444 | [-0.009, 0.942] | 0.348 | 100% |

Every strategy pays the same 0.5-vol-point all-in round-trip pair cost. Flat dates earn zero. Sharpes use `sqrt(365/30) × mean(net 30-day payoff) / sample SD`; these remain annualized payoff proxies, not marked portfolio-return Sharpes.

The nonoverlapping sequence enters on the first available date, then the first date on or after the previous 30-calendar-day expiry. It includes flat opportunities and is only one start-offset diagnostic, not a separately optimized trading strategy. Removing overlap does not guarantee independence across volatility regimes.

Pairwise uncertainty remains substantial. The Sharpe difference for earnings-plus-RV versus spread-only is +0.312 with a paired block-bootstrap interval **[-0.107, 0.675]**; versus the simple short-higher-IV rule it is +0.084 with interval **[-0.349, 0.520]**. The broader result does not establish superiority over those alternatives.

## Test blocks

| Test block | Spread | + earnings | + RV | + VIX | Simple rule |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2010–2011 | 0.421 | 0.644 | 0.351 | 0.473 | 0.624 |
| 2012–2013 | -0.517 | -0.695 | -0.644 | -1.255 | -0.484 |
| 2014–2015 | 1.018 | 1.059 | 2.021 | 0.048 | 1.086 |
| 2016–2017 | -0.306 | 0.898 | 0.948 | -0.146 | -0.238 |
| 2018–2019 | 0.732 | 0.188 | 0.776 | 0.825 | 0.792 |
| 2020–2021 | 0.000 | 0.413 | 0.300 | 0.719 | 1.352 |
| 2022–2023 | 0.280 | 0.523 | -0.231 | -0.231 | 0.418 |
| 2024–August 2025 | 0.874 | 0.152 | 0.787 | -0.403 | 1.140 |

## Protocol

For each test start year 2010, 2012, …, 2024:

1. Train on all available earlier history beginning February 2002, stopping before the two-year validation window.
2. Validate on the two calendar years immediately preceding the test block.
3. Test the frozen policy on the next two calendar years, truncated to available data for the last block.
4. Remove training outcomes that cross into validation, and validation outcomes that cross into test. Training histories expand as time advances; each tested date appears exactly once.

Example: the 2010–2011 block trains on 2002–2007, validates on 2008–2009, and tests 2010–2011. The 2024–2025 block trains on 2002–2021, validates on 2022–2023, and tests 2024 onward. Boundary entries with unsettled outcomes are excluded from fitting and selection.

Use the same candidate grid, five seeds, feature definitions, tree partitions, policy-search procedure, costs, and calendar assumptions as the [RV/VIX experiment](rv_vix_pass.md). No additional configuration search was added. Each feature set selects its own configuration on validation. The separate validation-selected strategy also selects the feature set on validation, never on that block's test results.

This is a retrospective walk-forward study over previously available and partly inspected history. It broadens regime coverage without pretending to create a new untouched holdout. Bootstrap intervals use 63-entry circular blocks and 2,000 resamples; they do not capture all model-selection uncertainty.

## Reproduce

First build the RV/VIX panel using `scripts/run_rv_vix_pass.py`, then:

```sh
python scripts/run_walk_forward.py
python -m unittest discover -s tests -v
```

Local outputs in `artifacts/walk_forward/` include the stitched predictions, nonoverlapping entries, per-block scores, candidate and selected settings, panel hash, and summary. Twelve tests pass, including walk-forward outcome purging and calendar-based nonoverlapping selection. Market data remains excluded from Git.
