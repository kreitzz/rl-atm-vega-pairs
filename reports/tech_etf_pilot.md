# Semiconductor constituents versus SMH

Eight fixed pairs: SMH against AMD, AMAT, INTC, LRCX, MU, NVDA, TER and QRVO. These are the cached stock histories available from the [September 2017 SMH holdings](https://www.sec.gov/Archives/edgar/data/1137360/000093041317004003/c89776_ncsr.htm). This is a frozen historical constituent roster, not reconstructed daily membership. Selection precedes inspecting this experiment’s P&L, but uses retrospective data availability and previously explored stocks.

SMH option and underlying history was newly downloaded from WRDS/OptionMetrics for 2012–2025, with bounded-memory streaming. The ETF starts in December 2011; the earlier Semiconductor HOLDRS security is not spliced into it. SMH states use direct CRSP ex-distribution returns (legacy dsf through 2024 and dsf_v2 in 2025), 30-calendar-day realized volatility and an ETF earnings sentinel of 31. Existing stock states retain their prior preprocessing, including standardized-IV availability gaps. Actual contract IV and curve features are used for both legs.

Evaluation January 2019–August 29, 2025. RL has the existing nine-level sizing ladder; NN uses fixed size. Both retain VIX, profitability memory, existing curve/contract states, monthly retraining, the same training windows and validation gates. Separate retraining compares beta 0.85–1.15 with no beta cutoff. Both require correlation ≥0.5 and retain finite beta as a feature; beta is SMH-on-stock and is lagged then frozen monthly.

Execution and risk remain unchanged: near-ATM calls targeting 30 days, daily close delta hedging, hold to expiry, mid plus 0.1 vol point per leg on entry, $0.65 per contract, 1 bp stock execution, stock borrow and cash financing. Capital is $1m per pair, including flat sleeves. Reserve, hedge and stress constraints remain per pair; the combined portfolio has no additional shared-SMH risk limit.

7/8 pairs completed under all four configurations and enter the matched aggregate. Incomplete runs are disclosed in pair_results.csv.

| Beta filter | Model | Active pairs | Entries | Annualized excess Sharpe | Net excess P&L |
|---|---|---:|---:|---:|---:|
| tight | rl_sized | 1 | 5 | 0.373 | $10,792 |
| tight | nn_fixed | 1 | 5 | 0.221 | $12,057 |
| no_beta_cutoff | rl_sized | 7 | 92 | -0.822 | $-125,926 |
| no_beta_cutoff | nn_fixed | 7 | 119 | -0.460 | $-132,178 |

| Pair | Tight RL entries | Tight RL P&L above cash | No-cutoff RL entries | No-cutoff RL P&L above cash |
|---|---:|---:|---:|---:|
| SMH/AMD | 0 | $-0 | 3 | $-1,681 |
| SMH/AMAT | 0 | $-0 | 7 | $-40,687 |
| SMH/INTC | 5 | $10,792 | 23 | $-14,525 |
| SMH/LRCX | 0 | $-0 | 4 | $-5,276 |
| SMH/MU | 0 | $-0 | — | incomplete |
| SMH/NVDA | 0 | $-0 | 8 | $-3,018 |
| SMH/TER | 0 | $-0 | 27 | $-30,309 |
| SMH/QRVO | 0 | $-0 | 20 | $-30,429 |

Sharpe is daily cash-excess return annualized by √252, including idle periods. Annual percentage returns include interest on unused cash. 2025 is partial. The monthly validation gate evaluates the proposal before the recent-data refit rather than the exact deployed weights. NN convergence warnings and unresolved training labels are recorded below. This is exploratory research, not an untouched holdout.

Cross-variant checks require identical standalone payoff labels, features, costs, risk limits and evaluation dates. Only beta eligibility changes. Calls versus puts have not been tested: the current cache and replay remain calls-only. American early assignment is not modeled.

[Pair results](tech_etf_pilot/pair_results.csv) · [Yearly returns and Sharpe](tech_etf_pilot/annual.csv) · [Beta eligibility](tech_etf_pilot/eligibility.csv) · [State coverage](tech_etf_pilot/coverage.csv) · [Comparison](tech_etf_pilot/comparison.csv)

SMH has 3,052 selected ATM contract dates across 2012–August 2025; early coverage is thinner. Raw calls total 1,019,744 rows (42.4 MiB compressed). SMH split-adjusted OptionMetrics close returns agree with CRSP ex-distribution returns within 0.00000059 across 3,434 overlapping observations. The 151-test suite passes. [SMH contract coverage](tech_etf_pilot/smh_contract_coverage.csv).

SMH/MU is excluded consistently from all four aggregates: both no-cutoff books stop on May 5, 2023 because the held SMH inventory does not pass the engine’s existing verified-integer-split checks. This is a simulation limitation, not a trading loss or a zero-return result. Other complete books and all individual statuses remain visible in the CSV.

The tight-filter aggregate trades only SMH/INTC, with five entries. Its positive Sharpe is not a broad semiconductor result. Without the cutoff, all seven matched RL pairs lose money above cash; AMAT, TER and QRVO account for roughly $101k of the $126k loss. No-cutoff NN has one profitable pair, QRVO, which does not offset its other losses.

tight: audit passed for 8 pairs and 20,364 label rows; 46 unresolved shadow labels excluded from training. NN convergence warnings: 24/28 validation fits, 8/11 recent fits. Runtime 37s, two workers, peak worker RSS 0.69 GiB. [Audit](tech_etf_pilot/tight/audit.json).

no_beta_cutoff: audit passed for 8 pairs and 20,364 label rows; 46 unresolved shadow labels excluded from training. NN convergence warnings: 169/568 validation fits, 46/189 recent fits. Runtime 166s, two workers, peak worker RSS 0.65 GiB. [Audit](tech_etf_pilot/no_beta_cutoff/audit.json).
