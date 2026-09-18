# Confidence-proxy vega sizing ladder

Allowing state-dependent entry sizes improved the point-estimate Sharpe for every tested feature variant, although the best absolute result's incremental improvement is uncertain.

## Matched results

Same 2010–August 2025 walk-forward dates, 3,922 entries, models, feature sets, validation procedure, and 0.5-vol-point full-size round-trip pair cost as the short-RV/momentum experiment.

| Features beyond IV spread and earnings | Fixed-size Sharpe | Ladder Sharpe | Change |
| --- | ---: | ---: | ---: |
| 30-calendar-day RV | 0.692 | **0.735** | +0.043 |
| 5-trading-day RV | 0.426 | 0.573 | +0.147 |
| 10-trading-day RV | 0.373 | 0.520 | +0.147 |
| 30-day RV + momentum | -0.161 | -0.048 | +0.113 |
| 5-day RV + momentum | -0.088 | 0.120 | +0.209 |
| 10-day RV + momentum | -0.094 | 0.232 | +0.326 |
| 30-day RV + VIX | 0.050 | 0.116 | +0.065 |
| 5-day RV + VIX | 0.164 | 0.189 | +0.024 |
| 10-day RV + VIX | 0.030 | 0.182 | +0.151 |
| 30-day RV + momentum + VIX | -0.078 | -0.024 | +0.055 |
| 5-day RV + momentum + VIX | -0.148 | 0.157 | +0.305 |
| 10-day RV + momentum + VIX | -0.023 | 0.089 | +0.112 |
| Feature set chosen by preceding validation | 0.166 | 0.368 | +0.201 |

The simple short-higher-IV baseline remains 0.444. The 30-day RV model is best retrospectively in this table; the validation-selected strategy is the separate implementable feature-selection rule.

All Sharpe figures remain annualized **30-day payoff** ratios, with `sqrt(365/30)` scaling and zero payoffs on flat dates. For the 30-day RV model, the paired 95% block-bootstrap interval for improvement is **[-0.152, 0.258]**; this does not establish a reliable improvement. Its nonoverlapping diagnostic rises from 0.788 to 0.852. These are repeated historical experiments with multiple variants, not a fresh holdout or multiplicity-adjusted test.

## Allocation and cost

The policy selects a signed fraction from `[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]`. Positive means long AMD / short NVDA; negative reverses the legs. Both legs always have equal initial absolute dollar vega.

Default maximum per leg per new trade: **0.001 dollar vega per dollar of reference capital**. With $1m reference capital, the absolute ladder is $0, $250, $500, $750, or $1,000 of vega per leg per volatility point. This scale is a configurable reporting assumption; changing it alone cannot change Sharpe under proportional costs.

At size `a`, reward is `a * long_AMD_gross_reward - 0.5 * abs(a)`. Thus a quarter-size trade pays one quarter of the full-size pair cost, and flat pays zero. The old fixed actions -1, 0, +1 retain exactly their previous rewards.

Sizing is chosen at each new entry and held for the 30-day outcome. This does not resize existing trades daily. Multiple cohorts can overlap: for the 30-day RV policy, peak summed absolute active vega per leg is 0.022 per reference-capital dollar, despite the 0.001 per-entry cap. There is no aggregate margin or capital-use constraint. Reported reference-capital returns are additive entry-cohort P&L divided by a fixed denominator, not compounded or daily marked portfolio returns.

## Confidence proxy

The absolute learned allocation is the confidence proxy: a larger size reflects a higher allocation chosen by the training Sharpe objective. It is not a calibrated win probability, posterior certainty, or guaranteed ranking of future returns. Sharpe can favor smaller sizes because of conditional volatility as well as a weaker expected payoff.

For the 30-day RV model:

| Size fraction | Test entries | Mean net payoff per unit vega on those trades |
| --- | ---: | ---: |
| Flat | 867 | — |
| 25% | 596 | -3.373 |
| 50% | 1,291 | 4.733 |
| 75% | 706 | 7.469 |
| 100% | 462 | 5.089 |

Average allocation is 45.5% of maximum across all entry opportunities. The realized ordering is not perfectly monotonic; it would be misleading to describe the ladder as calibrated confidence. These diagnostics were not used to retune the policy.

## Training and reproduction

The action search is generalized to fractional sizes. For every candidate tree, first learn the original fixed-size policy, then include it as the initial incumbent in each ladder search. This guarantees at least its training Sharpe within that tree while keeping future outcomes out of selection. Test performance has no such guarantee. The same five seeds and validation tree-selection grid are used; each size mode selects its own configuration from the common grid.

```sh
python scripts/run_walk_forward.py --panel artifacts/short_rv_momentum/dataset.parquet --output artifacts/sizing_ladder/walk_forward --features-json config/short_rv_momentum_features.json --reference-model rv30cal --comparison-model rv5td_momentum --sizing-ladder 0 0.25 0.5 0.75 1 --max-vega-per-dollar 0.001
python scripts/report_sizing.py
python -m unittest discover -s tests -v
```

- [Fixed versus ladder comparisons and uncertainty](sizing_comparison.csv)
- [Allocation diagnostics](sizing_allocations.csv)
- [Annual P&L, reference-capital return, and Sharpe](sizing_annual.csv)

Local predictions include signed per-leg vega-per-dollar targets, absolute allocation proxies, and rewards. Eighteen tests pass. Additional output checks verify every action is on the ladder, both legs offset in initial vega, costs scale correctly, and yearly P&L reconciles to the aggregate.
