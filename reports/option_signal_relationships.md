# Signal versus subsequent net pair payoff

Each contract type uses its own available opportunities; CC/PP sample composition differs. See the common-date study for a like-for-like comparison.

All screened opportunities with observed, resolved standalone entries are included, before the policy chooses trades. CC and PP are evaluated separately. Positive directional payoff means long vega in the first name and short vega in the second was better than the reverse. Outcomes include entry costs, daily stock hedging, dividends, borrowing and financing through expiry, normalized by actual initial leg vega.

These are descriptive associations on already inspected 2018–2025 research dates. The standalone training trajectories do not apply portfolio capacity/risk limits. Missing/unresolved outcomes are disclosed in coverage. Overlapping ~30-day outcomes and common market exposures make observations dependent; no IID p-values or causal claims are made. The nonoverlapping check thins each pair/type chronologically but does not remove cross-pair dependence. Quintiles use trailing 252-observation signal ranks known at the signal date, not future outcome ranks.

| Type | Signal | Pooled Spearman | Within-pair Pearson | Equal-pair mean Spearman | Nonoverlap Spearman |
|---|---|---:|---:|---:|---:|
| CC | spread | -0.083 | -0.097 | -0.106 | -0.099 |
| CC | rv_adjusted_value | 0.026 | -0.045 | -0.029 | 0.027 |
| CC | spread_reversion | 0.080 | 0.073 | 0.099 | 0.070 |
| CC | beta | -0.004 | -0.022 | -0.058 | -0.042 |
| CC | correlation | 0.036 | 0.028 | 0.013 | 0.063 |
| CC | vix_previous_close | 0.012 | 0.020 | 0.009 | 0.006 |
| CC | memory_edge_91d | -0.040 | -0.114 | -0.118 | -0.037 |
| CC | memory_edge_365d | 0.006 | -0.054 | -0.136 | 0.045 |
| CC | surface_downside_skew_difference | -0.013 | 0.003 | 0.021 | -0.006 |
| CC | surface_smile_curvature_difference | 0.008 | 0.022 | 0.041 | 0.030 |
| CC | surface_term_slope_difference | 0.010 | -0.012 | 0.056 | 0.028 |
| CC | surface_term_bend_difference | -0.008 | 0.017 | -0.012 | -0.023 |
| PP | spread | -0.059 | -0.118 | -0.079 | -0.096 |
| PP | rv_adjusted_value | 0.009 | -0.052 | -0.044 | 0.010 |
| PP | spread_reversion | 0.055 | 0.090 | 0.076 | 0.058 |
| PP | beta | 0.015 | -0.007 | -0.018 | -0.013 |
| PP | correlation | 0.048 | 0.043 | 0.046 | 0.067 |
| PP | vix_previous_close | 0.007 | 0.016 | -0.002 | -0.015 |
| PP | memory_edge_91d | -0.072 | -0.108 | -0.100 | -0.058 |
| PP | memory_edge_365d | -0.001 | -0.042 | -0.101 | 0.011 |
| PP | surface_downside_skew_difference | -0.001 | 0.002 | 0.003 | 0.016 |
| PP | surface_smile_curvature_difference | 0.019 | 0.030 | 0.024 | 0.071 |
| PP | surface_term_slope_difference | -0.077 | -0.080 | -0.108 | -0.048 |
| PP | surface_term_bend_difference | 0.024 | 0.042 | 0.020 | 0.051 |

[Full correlations](option_flex_ablation/signal_correlations.csv) · [By year](option_flex_ablation/signal_correlations_by_year.csv) · [By pair](option_flex_ablation/signal_correlations_by_pair.csv) · [Signal quintiles](option_flex_ablation/signal_quintiles.csv) · [Coverage](option_flex_ablation/signal_coverage.csv)

[All available dates](option_signal_relationships.md) · [Common dates](option_signal_relationships_common.md)
