# WRDS information signals versus option-pair profitability

Completed October 1, 2026. First-pass SEC disclosure and I/B/E/S analyst-revision screen on 26,334 cached, resolved ATM call/call opportunities across 128 pairs, January 2018–July 2025. No policy was retrained and the saved option checkpoint was not changed.

The broad relationships are weak and generally unstable in the later period. There are pair-specific leads with consistent signs across the two periods, but these examples were selected after inspecting many combinations. They are candidates for a future chronological test, not validated alpha.

## Pair-specific leads

Rank correlations below relate differences between the two names to subsequent directional option P&L. These examples have at least 50 observations in each period. They were selected retrospectively and are not an exhaustive list of tests.

| Pair | Signal difference | 2018–2021 correlation | 2022–2025 correlation | Early / later dates | Nonoverlapping outcomes | Nonoverlap correlation |
|---|---|---:|---:|---:|---:|---:|
| AMD/NVDA | EPS revision magnitude | +0.600 | +0.256 | 83 / 127 | 10 | +0.685 |
| CVX/XOM | Upward versus downward EPS revisions | -0.410 | -0.483 | 201 / 153 | 18 | -0.346 |
| TRV/CB | Filing uncertainty language | +0.390 | +0.361 | 264 / 195 | 34 | +0.222 |

AMD/NVDA has only 210 eligible dates: 83 in June–September 2019 and 127 in June–December 2022. Those daily observations reduce to ten nonoverlapping outcomes. Its full-sample revision-magnitude correlation is approximately +0.455, but the early/later comparison is effectively a comparison of two short market episodes. The filing-language variables also change only a few times within those episodes. Do not interpret the daily row count as hundreds of independent trades.

A positive gap correlation means that when the signal is higher in the first named leg, long-first/short-second vega tends to outperform the reverse direction. A negative correlation favors the opposite relationship. This does not mean either direction earns positive net profit, and it does not establish a causal mechanism.

## Universe-level results

Within-pair rank correlations remove persistent differences between pairs and give each eligible pair equal weight. At least 20 observed rows are required per pair in each sample. This statistic differs from an ordinary pooled rank correlation and from the mean of per-pair Spearman correlations.

| Feature | Target | Dates / pairs | Full within-pair rank correlation | Early | Later | Nonoverlap | Partial correlation controlling IV value | 90% quarter-block interval |
|---|---|---:|---:|---:|---:|---:|---:|---|
| lm_negative_gap | Direction | 20,741 / 91 | -0.019 | +0.027 | -0.020 | +0.076 | -0.013 | [-0.057, +0.019] |
| lm_negative_mean | Best-direction payoff | 20,741 / 91 | +0.025 | +0.036 | -0.002 | +0.066 | +0.024 | [-0.018, +0.066] |
| lm_uncertainty_gap | Direction | 20,741 / 91 | +0.031 | +0.054 | -0.001 | +0.062 | +0.030 | [-0.006, +0.078] |
| lm_uncertainty_mean | Best-direction payoff | 20,741 / 91 | +0.039 | +0.070 | +0.013 | +0.105 | +0.033 | [+0.005, +0.079] |
| lm_negative_change_gap | Direction | 20,213 / 88 | +0.001 | +0.010 | -0.000 | -0.051 | +0.004 | [-0.041, +0.039] |
| lm_negative_change_mean | Best-direction payoff | 20,213 / 88 | +0.011 | +0.053 | -0.010 | +0.075 | +0.011 | [-0.025, +0.046] |
| lm_uncertainty_change_gap | Direction | 20,213 / 88 | -0.020 | -0.008 | -0.014 | +0.037 | -0.020 | [-0.055, +0.019] |
| lm_uncertainty_change_mean | Best-direction payoff | 20,213 / 88 | +0.033 | +0.032 | +0.005 | +0.108 | +0.034 | [-0.000, +0.070] |
| text_change_gap | Direction | 18,917 / 84 | +0.020 | -0.063 | +0.060 | -0.024 | +0.018 | [-0.025, +0.054] |
| text_change_mean | Best-direction payoff | 18,917 / 84 | +0.014 | -0.016 | +0.057 | -0.020 | +0.014 | [-0.029, +0.057] |
| revision_breadth_gap | Direction | 20,390 / 98 | -0.016 | +0.003 | -0.041 | -0.028 | -0.022 | [-0.061, +0.026] |
| revision_breadth_mean | Best-direction payoff | 20,390 / 98 | -0.063 | -0.107 | -0.019 | -0.027 | -0.062 | [-0.096, -0.027] |
| revision_magnitude_gap | Direction | 20,390 / 98 | +0.033 | +0.044 | +0.003 | +0.009 | +0.017 | [-0.007, +0.076] |
| revision_magnitude_mean | Best-direction payoff | 20,390 / 98 | -0.042 | -0.088 | +0.020 | -0.009 | -0.042 | [-0.080, -0.005] |
| revision_activity_gap | Direction | 20,390 / 98 | +0.007 | +0.049 | -0.053 | +0.031 | +0.001 | [-0.045, +0.060] |
| revision_activity_mean | Best-direction payoff | 20,390 / 98 | -0.020 | -0.007 | -0.034 | +0.030 | -0.024 | [-0.058, +0.019] |
| forecast_dispersion_gap | Direction | 20,390 / 98 | +0.028 | +0.052 | -0.008 | +0.049 | +0.036 | [-0.023, +0.075] |
| forecast_dispersion_mean | Best-direction payoff | 20,390 / 98 | +0.025 | +0.020 | +0.002 | -0.013 | +0.020 | [-0.020, +0.076] |

All nine directional gap variables have full-sample quarter-block intervals containing zero. The strongest pooled opportunity relationship is lower analyst revision breadth versus higher best-direction payoff (about −0.063), but it weakens to about −0.019 in the later period and −0.027 in the nonoverlap sample. Average filing uncertainty has a modest positive relationship with best-direction payoff; this also weakens later. Neither supplies a demonstrated rule for selecting the profitable direction.

The best-direction target is the maximum of the two realized directional payoffs. It uses hindsight to measure the size of a possible opportunity; it is not an executable payoff, a strategy return, or evidence that a policy could choose that direction.

## Signals and availability

- SEC: Loughran–McDonald negative and uncertainty proportions; their change from the comparable filing approximately one year earlier (same form and calendar reporting quarter, 250–500 days); and one minus WRDS cosine similarity to the immediately preceding comparable-form filing (`lookback=1`). Similarity and year-over-year tone changes use different reference comparisons.
- I/B/E/S: annual EPS forecasts (`fpi=1`) for US firms with USD or unspecified currency. Use 30-day mean revision sign, mean proportional revision magnitude clipped to ±1, revision activity and forecast dispersion. Magnitude requires an absolute preceding estimate of at least $0.10. The consensus requires at least three distinct estimator/analyst entries, updated within 180 days, for the nearest future fiscal year end. These choices were fixed before computing correlations; no parameter grid was fitted.
- Pair states: first-leg minus second-leg signal for direction; average of the two legs for opportunity size. Analyst features are sampled weekly on Fridays and held for at most 14 calendar days; SEC features expire after 180 calendar days.
- SEC availability is one business day after the latest filing, acceptance or publication date. Analyst availability is one business day after the later estimate announcement and vendor activation date. Same-day estimate updates retain their time ordering. Future actual earnings fields are never extracted.
- Identifiers: historical OptionMetrics-to-CRSP links, historical CCM intervals and issuer CIK; historical CRSP-to-I/B/E/S links. Option and analyst link scores must be ≤1. Overlapping links are accepted only when all active candidate IDs agree. Unmapped or ambiguous dates remain missing.

SEC feature coverage varies by feature: roughly 19,000–21,000 evaluable dates over 84–91 pairs after the minimum-row filter. Analyst coverage is 20,390 dates over 98 pairs. Companies/ETFs lacking relevant corporate disclosures or analyst coverage are not assigned fabricated states. This is a screen on the 128 pairs with cached resolved outcomes, not a new test of all 345 checkpoint pairs.

## Outcome and inference limits

The source labels are net standalone, initially vega-matched, daily delta-hedged option/stock outcomes held through expiry. Directional edge is `(long_first_net − short_first_net)/2`, measured in dollars per initial mean-leg vega. The labels retain the older $0.65 option commission, 0.10-vol-point entry penalty, 1 bp stock execution and original financing/borrowing assumptions. They are not repriced to the current $0.03 evaluation fee, nor are they a replay of the frozen $0.65-training-hurdle checkpoint. No Sharpe or capital return is inferred from these terminal outcomes.

The source sample already conditions on historical pair membership, beta 0.85–1.15, correlation ≥0.5, entry availability and both directional outcomes resolving. Future resolution is a diagnostic sample condition, not a trading-time eligibility rule. Missing outcomes can bias correlations. Direction and initial vega matching follow each saved pair definition; inventory is not continuously rebalanced to net-zero vega.

Early observations must finish before January 1, 2022; later entries begin on or after that date. This purges outcomes spanning the boundary. These years have already been explored in previous research, so the later period is a stability check rather than an untouched holdout.

The partial correlations control contemporaneous IV spread and trailing-RV-adjusted relative value using within-pair ranks. They are descriptive regressions fitted on the same sample, not out-of-sample forecasts. Nonoverlap thinning is independent within each pair and does not remove shared-name dependence across pairs. The 90% intervals use 500 resamples of entire calendar quarters across all pairs jointly, with fixed ranks and weights. They are approximate and unadjusted for the many feature and pair comparisons. No ordinary daily-observation significance claims are made.

Historical joins and delayed timestamps address several timing risks but do not establish archived vendor vintages. I/B/E/S forecast histories can be adjusted or corrected; normalization reduces some scale sensitivity without proving point-in-time purity. Validate adjustment conventions before fitting a deployable policy.

## Next focused experiment

Predeclare revision magnitude, revision breadth and disclosure uncertainty as a small candidate set. Test pair-specific or industry interactions using only completed training outcomes and chronological validation; the opposite correlation signs across names make a universal directional mapping questionable. Require improvements over the existing value-only baseline on matched dates and costs before adding features to NN/RL training. The pair examples above may guide hypotheses but cannot be reported as independent confirmation after being selected here.

Transcripts and Revelio have not been tested in this pass. Dated employee reviews or publicly available layoff notices are a separate next feature family; resume-derived workforce histories still require a vintage audit.

## Validation and artifacts

One guarded worker at 50% of one core, a 3 GiB memory stop, and bounded read-only WRDS queries (20 seconds and at most 150,000 rows each). Peak observed worker memory across completed correlation runs was 425 MiB. Five targeted tests passed: delayed availability, historical-link ambiguity, future/stale as-of joins, removal of static pair effects, and independence from future estimate values.

Licensed rows and aligned states remain in ignored `artifacts/contracts/wrds_pair_signals/`; the report files contain aggregate diagnostics. The manifest records the outcome, aligned-data and feature-snapshot hashes. Existing checkpoint inputs and actions were not rewritten.

- [All feature correlations](wrds_pair_signals/correlations.csv)
- [Year-by-year correlations](wrds_pair_signals/annual.csv)
- [All pair correlations](wrds_pair_signals/pairs.csv)
- [Early/later and nonoverlap pair diagnostics](wrds_pair_signals/pair_periods.csv)
- [Pair coverage](wrds_pair_signals/pair_coverage.csv)
- [Identifier coverage](wrds_pair_signals/identifier_coverage.csv)
- [Research script](../scripts/test_wrds_pair_signals.py)
- [WRDS access scan](wrds_discovery.md)
