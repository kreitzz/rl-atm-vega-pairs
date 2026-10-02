# Causal signal mining and exit diagnostics

Reference training hurdle $0.65; executed commission $0.03. Same 345 requested pairs, option selection, sizing ladder, risk limits and monthly walk-forward procedure. All comparisons use the common complete roster; unsupported paths are recorded in coverage rather than treated as cash. Original checkpoint unchanged.

| Model | Full Sharpe | Recent Sharpe | Full entries |
|---|---:|---:|---:|
| rl_sized | 0.318 | 0.347 | 224 |
| rl_dynamics | 0.081 | 0.085 | 242 |
| rl_dynamics_gamma | 0.305 | 0.406 | 254 |
| exit_reverse | 0.318 | 0.347 | 224 |
| exit_flat_or_reverse | -0.031 | 0.071 | 270 |

Full: 2018–August 2025. Recent: 2022–August 2025, fresh start. Daily cash-excess Sharpe × √252. Actual costs include 0.10 vol point per option leg per execution, $0.03 commission, 1 bp stock execution and original financing/borrow.

Two feature families were specified before these runs: dynamics adds 5/21/63-session variance value, short RV acceleration, IV changes, spread z-scores, IV/return correlation gap and beta-scaled IV ratio. Dynamics+gamma also adds gamma per vega, gamma-weighted variance value, quoted width, open interest and expiry mismatch. Each value has a missingness flag. IV correlation requires 32 paired observations within 63 calendar trading sessions; spread z-scores require 11/32 prior observations in 21/63-session windows. Missing IV is not forward-filled. All 47 original inputs remain unchanged. Each candidate receives its own prior-year validation gate at the $0.65 training hurdle and five-year refit, with completed outcomes strictly purged at both boundaries.

Signal ranks use only 2013–2017 completed, eligible, non-overlapping hypothetical training cohorts. The target is half the difference between full-size long-pair and short-pair net payoff per target entry vega. Average within-pair Spearman correlations are descriptive; the two predeclared families are not selected using 2018–2025 performance. This remains an historically screened universe and previously inspected evaluation period.

Exit overlays reuse saved baseline entry recommendations. Reversal closes both live legs together on the next session; flat-or-reverse additionally closes when the recommendation becomes flat. Missing signal dates cause no exit. A flat recommendation can reflect eligibility or memory gates. These are diagnostic rules, not a learned exit policy, and do not forecast the held contracts. Option quantities remain fixed between entry and closure (except splits); neutrality is checked at entry, with daily delta hedging but no daily option-vega rebalancing. Early exits pay the same modeled commission and vol penalty; natural expiries do not. Freed risk capacity can change subsequent realized entries.

[Comparison](atm65_signal_mining/comparison.csv) · [Annual](atm65_signal_mining/annual.csv) · [Pairs](atm65_signal_mining/pairs.csv) · [Early ranks](atm65_signal_mining/early_rank.csv) · [Coverage](atm65_signal_mining/coverage.json)

## Signal and concentration diagnostics

Ranks below use pre-2018 completed training cohorts only. Pair observations are non-overlapping within each pair but pairs can share underlyings. These averages are descriptive, not independent significance tests.

| Candidate | Mean within-pair rank correlation | Pairs | Cohorts |
|---|---:|---:|---:|
| spread_z_63 | -0.080 | 41 | 1042 |
| dte_gap | +0.066 | 37 | 937 |
| iv_correlation_63 | -0.063 | 41 | 1043 |
| iv_change_21 | -0.062 | 33 | 870 |
| spread_z_21 | -0.058 | 40 | 1004 |
| beta_scaled_iv_ratio | -0.056 | 42 | 1069 |

| Model (full period) | Profitable / losing pairs | Top 3 share of positive P&L | Sharpe excluding largest winner |
|---|---:|---:|---:|
| exit_flat_or_reverse | 14 / 15 | 42.5% | -0.062 |
| exit_reverse | 18 / 11 | 44.5% | 0.209 |
| rl_dynamics | 20 / 13 | 42.8% | 0.017 |
| rl_dynamics_gamma | 21 / 13 | 38.6% | 0.247 |
| rl_sized | 18 / 11 | 44.5% | 0.209 |

Excluding winners is an ex-post sensitivity diagnostic: those sleeves are replaced with cash, and no policy is retrained. It is not an executable selection rule.

| Year | Baseline | Dynamics | Dynamics + gamma | Reversal exit | Flat/reversal exit |
|---|---:|---:|---:|---:|---:|
| 2018 | -0.727 | 0.440 | 0.049 | -0.727 | -1.284 |
| 2019 | 0.043 | -0.422 | -0.206 | 0.043 | -0.306 |
| 2020 | 1.146 | 1.132 | 0.992 | 1.146 | 1.132 |
| 2021 | 0.580 | 0.405 | 0.510 | 0.580 | 0.178 |
| 2022 | 0.692 | 0.498 | 0.849 | 0.692 | -0.015 |
| 2023 | 0.809 | -0.221 | 1.126 | 0.809 | 0.157 |
| 2024 | 0.144 | 0.208 | 0.442 | 0.144 | 0.126 |
| 2025 (partial) | -0.976 | -0.299 | -0.361 | -0.976 | -0.726 |

[Concentration CSV](atm65_signal_mining/concentration.csv)
