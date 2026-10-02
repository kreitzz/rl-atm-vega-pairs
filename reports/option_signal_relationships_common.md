# Signal versus subsequent net pair payoff

Common-date sample: both CC and PP must have observed, resolved entries on the same pair/date.

All screened opportunities with observed, resolved standalone entries are included, before the policy chooses trades. CC and PP are evaluated separately. Positive directional payoff means long vega in the first name and short vega in the second was better than the reverse. Outcomes include entry costs, daily stock hedging, dividends, borrowing and financing through expiry, normalized by actual initial leg vega.

These are descriptive associations on already inspected 2018–2025 research dates. The standalone training trajectories do not apply portfolio capacity/risk limits. Missing/unresolved outcomes are disclosed in coverage. Overlapping ~30-day outcomes and common market exposures make observations dependent; no IID p-values or causal claims are made. The nonoverlapping check thins each pair/type chronologically but does not remove cross-pair dependence. Quintiles use trailing 252-observation signal ranks known at the signal date, not future outcome ranks.

| Type | Signal | Pooled Spearman | Within-pair Pearson | Equal-pair mean Spearman | Nonoverlap Spearman |
|---|---|---:|---:|---:|---:|
| CC | spread | -0.074 | -0.110 | -0.094 | -0.112 |
| CC | rv_adjusted_value | 0.018 | -0.047 | -0.039 | 0.024 |
| CC | spread_reversion | 0.076 | 0.085 | 0.093 | 0.087 |
| CC | beta | 0.012 | 0.002 | 0.012 | -0.009 |
| CC | correlation | 0.039 | 0.037 | 0.035 | 0.065 |
| CC | vix_previous_close | 0.023 | 0.032 | 0.012 | -0.001 |
| CC | memory_edge_91d | -0.069 | -0.156 | -0.096 | -0.098 |
| CC | memory_edge_365d | -0.006 | -0.054 | -0.087 | 0.009 |
| CC | surface_downside_skew_difference | -0.009 | -0.015 | 0.009 | -0.011 |
| CC | surface_smile_curvature_difference | 0.007 | 0.025 | 0.007 | 0.038 |
| CC | surface_term_slope_difference | -0.060 | -0.083 | -0.081 | -0.056 |
| CC | surface_term_bend_difference | 0.005 | 0.030 | -0.003 | 0.003 |
| PP | spread | -0.057 | -0.108 | -0.065 | -0.070 |
| PP | rv_adjusted_value | 0.009 | -0.055 | -0.048 | 0.014 |
| PP | spread_reversion | 0.054 | 0.083 | 0.062 | 0.052 |
| PP | beta | 0.016 | -0.005 | 0.004 | 0.005 |
| PP | correlation | 0.049 | 0.044 | 0.044 | 0.057 |
| PP | vix_previous_close | 0.006 | 0.018 | 0.004 | -0.016 |
| PP | memory_edge_91d | -0.075 | -0.111 | -0.101 | -0.065 |
| PP | memory_edge_365d | -0.002 | -0.043 | -0.105 | 0.013 |
| PP | surface_downside_skew_difference | -0.003 | -0.011 | 0.001 | 0.013 |
| PP | surface_smile_curvature_difference | 0.018 | 0.027 | 0.019 | 0.085 |
| PP | surface_term_slope_difference | -0.080 | -0.091 | -0.111 | -0.058 |
| PP | surface_term_bend_difference | 0.027 | 0.050 | 0.021 | 0.068 |

[Full correlations](option_flex_ablation/common_signal_correlations.csv) · [By year](option_flex_ablation/common_signal_correlations_by_year.csv) · [By pair](option_flex_ablation/common_signal_correlations_by_pair.csv) · [Signal quintiles](option_flex_ablation/common_signal_quintiles.csv) · [Coverage](option_flex_ablation/common_signal_coverage.csv)

[All available dates](option_signal_relationships.md) · [Common dates](option_signal_relationships_common.md)
