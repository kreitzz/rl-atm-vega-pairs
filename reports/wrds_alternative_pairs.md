# Other WRDS alternatives versus option-pair profitability

October 1, 2026. Extended the first correlation screen using the same 26,334 cached resolved ATM call/call opportunities over 128 pairs, January 2018–July 2025. Signal coverage varies by source. The option outcomes and existing SEC/I/B/E/S features were held fixed; no NN or RL policy was retrained.

These are exploratory fixed-feature diagnostics. Workforce and job-posting histories are reconstructed current-vintage data, not verified historical trading information. Dated reviews, insider filings and transcript versions also require publication/revision and identifier audits before a strategy backtest. No Sharpe or return improvement is inferred.

## Compact comparison

Representative directional features below are differences between the two legs. Negative correlation favors the reverse association; it does not imply either directional trade makes money. The employee-leadership and transcript-negative-change rows are promising-looking examples chosen after examining the full feature set, not independently validated selections.

| Source | Feature difference | Dates / pairs | Full correlation | Early | Later | Nonoverlap |
|---|---|---:|---:|---:|---:|---:|
| Reviews | review_leadership_gap | 5,631 / 27 | -0.075 | -0.021 | -0.097 | -0.148 |
| Layoffs | layoff_intensity_gap | 12,174 / 66 | +0.048 | +0.028 | +0.051 | +0.034 |
| Insiders | insider_imbalance_gap | 18,977 / 91 | +0.001 | -0.019 | +0.029 | +0.018 |
| Transcripts | call_negative_change_gap | 21,263 / 100 | -0.033 | +0.009 | -0.038 | -0.023 |
| Workforce, provisional | workforce_net_hiring_gap | 1,461 / 10 | +0.008 | +0.214 | -0.037 | — |
| Postings, provisional | posting_growth_gap | 8,774 / 44 | +0.017 | +0.148 | -0.042 | +0.051 |
| Institutional liquidity | ilmt_1m_gap | 4,791 / 30 | +0.018 | +0.018 | — | — |
| Credit | credit_spread_change_gap | 16,435 / 62 | -0.006 | -0.004 | -0.017 | +0.007 |

The table uses ranks within each pair and equal total weight per eligible pair. At least 20 observed rows are required per pair in each sample; consequently the nonoverlap results can use a much smaller and different pair set. Early outcomes must finish before January 2022, while later entries begin from January 2022. Later-year data are repeatedly explored research history, not an untouched holdout.

## Every tested feature

| Feature | Target | Dates / pairs | Full within-pair rank correlation | Partial IV-value control | 90% quarter-block interval |
|---|---|---:|---:|---:|---|
| review_rating_change_gap | Direction | 5,870 / 29 | -0.000 | +0.005 | [-0.083, +0.069] |
| review_rating_change_mean | Best-direction payoff | 5,870 / 29 | -0.058 | -0.059 | [-0.116, +0.002] |
| review_outlook_change_gap | Direction | 5,278 / 26 | -0.054 | -0.066 | [-0.143, +0.030] |
| review_outlook_change_mean | Best-direction payoff | 5,278 / 26 | -0.001 | -0.003 | [-0.080, +0.059] |
| review_leadership_gap | Direction | 5,631 / 27 | -0.075 | -0.077 | [-0.132, -0.012] |
| review_leadership_mean | Best-direction payoff | 5,631 / 27 | -0.064 | -0.059 | [-0.134, +0.014] |
| layoff_intensity_gap | Direction | 12,174 / 66 | +0.048 | +0.050 | [-0.022, +0.108] |
| layoff_intensity_mean | Best-direction payoff | 12,174 / 66 | -0.023 | -0.031 | [-0.065, +0.017] |
| insider_imbalance_gap | Direction | 18,977 / 91 | +0.001 | +0.007 | [-0.038, +0.044] |
| insider_imbalance_mean | Best-direction payoff | 18,977 / 91 | +0.040 | +0.036 | [+0.008, +0.072] |
| insider_buy_activity_gap | Direction | 18,977 / 91 | -0.024 | -0.017 | [-0.053, +0.005] |
| insider_buy_activity_mean | Best-direction payoff | 18,977 / 91 | -0.049 | -0.050 | [-0.086, -0.010] |
| call_uncertainty_gap | Direction | 21,491 / 101 | +0.006 | +0.008 | [-0.033, +0.049] |
| call_uncertainty_mean | Best-direction payoff | 21,491 / 101 | +0.016 | +0.015 | [-0.023, +0.056] |
| call_negative_change_gap | Direction | 21,263 / 100 | -0.033 | -0.033 | [-0.072, -0.000] |
| call_negative_change_mean | Best-direction payoff | 21,263 / 100 | +0.024 | +0.022 | [-0.006, +0.056] |
| call_qa_uncertainty_gap | Direction | 21,173 / 101 | +0.009 | +0.013 | [-0.034, +0.052] |
| call_qa_uncertainty_mean | Best-direction payoff | 21,173 / 101 | -0.007 | -0.007 | [-0.042, +0.028] |
| call_analyst_negative_gap | Direction | 21,309 / 100 | +0.002 | +0.004 | [-0.032, +0.040] |
| call_analyst_negative_mean | Best-direction payoff | 21,309 / 100 | +0.028 | +0.032 | [-0.000, +0.059] |
| workforce_net_hiring_gap | Direction | 1,461 / 10 | +0.008 | -0.043 | [-0.122, +0.152] |
| workforce_net_hiring_mean | Best-direction payoff | 1,461 / 10 | -0.030 | +0.018 | [-0.136, +0.050] |
| workforce_attrition_gap | Direction | 1,461 / 10 | +0.023 | +0.048 | [-0.106, +0.133] |
| workforce_attrition_mean | Best-direction payoff | 1,461 / 10 | +0.048 | +0.041 | [-0.053, +0.144] |
| posting_growth_gap | Direction | 8,774 / 44 | +0.017 | +0.015 | [-0.044, +0.087] |
| posting_growth_mean | Best-direction payoff | 8,774 / 44 | +0.002 | -0.000 | [-0.076, +0.082] |
| ilmt_1m_gap | Direction | 4,791 / 30 | +0.018 | +0.009 | [-0.017, +0.064] |
| ilmt_1m_mean | Best-direction payoff | 4,791 / 30 | -0.030 | -0.035 | [-0.053, -0.006] |
| bbd_gap | Direction | 0 / 0 | — | — | — |
| bbd_mean | Best-direction payoff | 0 / 0 | — | — | — |
| credit_spread_change_gap | Direction | 16,435 / 62 | -0.006 | -0.003 | [-0.053, +0.041] |
| credit_spread_change_mean | Best-direction payoff | 16,435 / 62 | -0.009 | -0.008 | [-0.048, +0.029] |
| credit_return_gap | Direction | 16,462 / 62 | -0.006 | -0.010 | [-0.048, +0.037] |
| credit_return_mean | Best-direction payoff | 16,462 / 62 | -0.000 | -0.002 | [-0.044, +0.047] |

A direction target is `(long-first/short-second net payoff − reverse-direction net payoff)/2`. A best-direction target is the maximum of both realized payoffs and assumes hindsight about direction. It measures the size of a possible opportunity, not executable strategy profit. All dollar outcomes are normalized by initial mean-leg vega.

## Features and timing

- Reviews: three-month employee overall, business-outlook and senior-leadership ratings. Overall/outlook changes compare the latest three months with the preceding three. Require at least 20 applicable reviews in each three-month mean. Monthly scores become eligible one business day after month-end plus 15 days. Reviews include all countries. No personal reviewer data or review text was downloaded. Historical edits and current company mappings remain unresolved vintage risks.
- Layoffs: log of one plus employees affected by notices in the trailing 90 days. Use notice date plus one business day, never the future effective layoff date. Weekly snapshots, with at most 14 days of carry. Zero means no observed notice in this feed, not proof of no layoffs anywhere. WARN notices have state/geographic coverage differences; notice date is a publication proxy, not a verified vendor receipt timestamp.
- Insiders: trailing 30-day open-market purchase/sale dollar imbalance and log count of purchase filings. Non-derivative common/ordinary stock only; original Form 4, excluding amendments and filings explicitly flagged as 10b5-1. Missing historical flags do not exclude all preplanned trades. Anchor to the latest filing/acceptance/publication date plus one business day. Weekly snapshots. Stop constructing zero-event observations after the last observed disclosure date; later dates remain missing.
- Transcripts: fixed small lexicons count negative and uncertainty words per 1,000 executive words, negative-rate change from the previous call, Q&A versus prepared-remarks uncertainty, and analyst-question negative words. These are crude lexical proxies, not a trained sentiment model: negation and context are not modeled. Select the earliest creation-dated version for each company/event; wait one business day after the later call or creation date. Require at least 1,000 executive words; Q&A and analyst measures have additional word-count requirements. Use at most 180 days of carry. No raw transcript text was retained.
- Workforce: three-month net external hiring and external attrition divided by average reconstructed U.S. headcount. Sum joint role/seniority/geography cells. Month-end plus 15 days, then one business day. Scaled employee estimates and reconstructed resumes can use information learned later; this fixed lag does not remove that hindsight.
- Postings: log growth of total U.S. job postings over the latest three months versus the preceding three, requiring at least 20 postings in each block. Same monthly lag as workforce. Never use future removal dates. Historical backfills, coverage/source changes and current issuer boundaries are unresolved; source mix can create apparent growth.
- Liquidity: monthly institutional trading-cost estimates, lagged 45 calendar days plus one business day. The BBD price-impact series available here stops in 2016 and has no usable 2018–2025 overlap. It is not treated as a zero signal.
- Credit: monthly market-value-weighted issuer credit-spread change and Treasury-adjusted bond returns, lagged 45 days plus one business day. Bond selection and weight changes can affect the aggregate. Historical publication timing and issuer links are unverified.

Revelio links use direct, unambiguous current CIK-to-RCID mappings, with historical OptionMetrics→CRSP→CCM issuer intervals on the equity side. Subsidiaries are not automatically rolled into current parents. Transcripts use historical GVKEY→Capital IQ company links. Numeric WRDS identifiers are normalized before joins. Ambiguous or missing links produce missing features. These checks do not establish historically correct vendor company boundaries.

## Coverage and bounded extraction

| Feature | Any observed pairs | First eligible date | Last eligible date | Nonoverlap outcomes before minimum-row filter |
|---|---:|---|---|---:|
| review_rating_change_gap | 33 | 2018-01-02 | 2025-07-10 | 314 |
| review_outlook_change_gap | 29 | 2018-01-02 | 2025-07-10 | 279 |
| review_leadership_gap | 32 | 2018-01-02 | 2025-07-10 | 303 |
| layoff_intensity_gap | 76 | 2018-01-02 | 2025-07-10 | 731 |
| insider_imbalance_gap | 106 | 2018-01-02 | 2023-11-02 | 1065 |
| insider_buy_activity_gap | 106 | 2018-01-02 | 2023-11-02 | 1065 |
| call_uncertainty_gap | 116 | 2018-01-02 | 2025-07-10 | 1255 |
| call_negative_change_gap | 115 | 2018-01-02 | 2025-07-10 | 1245 |
| call_qa_uncertainty_gap | 116 | 2018-01-02 | 2025-07-10 | 1238 |
| call_analyst_negative_gap | 115 | 2018-01-02 | 2025-07-10 | 1244 |
| workforce_net_hiring_gap | 12 | 2018-01-02 | 2025-07-10 | 90 |
| workforce_attrition_gap | 12 | 2018-01-02 | 2025-07-10 | 90 |
| posting_growth_gap | 53 | 2018-01-02 | 2025-07-10 | 518 |
| ilmt_1m_gap | 40 | 2018-01-02 | 2020-06-12 | 256 |
| bbd_gap | 0 | — | — | 0 |
| credit_spread_change_gap | 72 | 2018-01-02 | 2024-04-15 | 896 |
| credit_return_gap | 72 | 2018-01-02 | 2024-04-15 | 897 |

Broad workforce/posting queries initially hit the 20-second limit. Smaller serial batches were then attempted for unambiguous mapped companies. Failed batches stay unavailable, with the initial AMD/NVDA pilot retained where available.
- workforce: 13 attempted batches, 1 failed, 43 distinct RCIDs with returned rows; 76 mapped companies not attempted in this product.
- postings: 16 attempted batches, 0 failed, 120 distinct RCIDs with returned rows; 0 mapped companies not attempted in this product.
The workforce extraction stalled and the supervisor ended that run at its overall deadline. Saved batches were assembled locally, without restarting the remaining downloads. Its 45-company extract supports ten pairs after the observation filter. Posting extracts contain 120 companies and support 44 pairs; lack of U.S. observations and mapping ambiguity reduce coverage further. Missing companies are not assigned zero hiring or zero postings.

Underlying input history is shorter for some sources: the fetched insider transactions end in October 2023; issuer bond data end in December 2023; institutional liquidity ends in February 2020. Derived features can remain eligible for the stated carry period after the final available observation. Later empty periods are not forward-filled indefinitely.

## Limits and next experiment

The outcomes retain the older $0.65 per-contract commission, 0.10-vol-point option entry penalty, daily stock hedging costs and financing/borrow assumptions from the first screen. They are not repriced to the current $0.03 reference commission. They are standalone initially vega-matched trajectories, with no daily option-vega rebalance or overlapping-book capacity simulation. Source membership, beta/correlation filters and future outcome resolution restrict the diagnostic sample.

The partial rank correlations control IV spread and RV-adjusted relative value, not every existing state. Approximate 90% intervals resample entire calendar quarters across all pairs jointly 500 times, with fixed ranks. They are not adjusted for the 34 candidate features or subsequent pair inspection. Nonoverlap thinning reduces expiry-window repetition but does not remove shared-name dependence across pairs. Neither same-sign early/later correlations nor an unadjusted interval excluding zero proves a tradable edge.

Use these results to predeclare a small hypothesis set for a separate chronological NN/RL comparison on matched costs and opportunities. Give employee sentiment and selected transcript changes a focused test if their later and annual diagnostics support it. Verify historical vintages before treating reconstructed labor data as a backtest input. A signal screen alone cannot say whether the current policy or portfolio Sharpe improves.

## Validation

One guarded worker at 50% of one core with a 3 GiB memory stop; one read-only WRDS connection, 20-second statement timeouts and bounded company aggregates/transcript batches. Peak correlation-worker RSS observed: 476 MiB. Existing baseline outcome and feature columns were checked exactly unchanged after alignment. Licensed extracts and derived row data remain in ignored artifacts; CSV reports are aggregates.

Eleven targeted tests passed, covering numeric identifier normalization, delayed monthly availability, exclusion of future events and post-coverage zeros, future-review independence, sparse review counts and fixed lexical counts, plus the original timing/link/correlation checks.

- [All correlations](wrds_alternative_pairs/correlations.csv)
- [Annual diagnostics](wrds_alternative_pairs/annual.csv)
- [Pair correlations](wrds_alternative_pairs/pairs.csv)
- [Early/later and nonoverlap pair diagnostics](wrds_alternative_pairs/pair_periods.csv)
- [Feature coverage](wrds_alternative_pairs/coverage.csv)
- [Identifier coverage](wrds_alternative_pairs/identifier_coverage.csv)
- [Original SEC/I/B/E/S screen](wrds_pair_signals.md)
- [Research script](../scripts/test_wrds_alternative_pairs.py)
