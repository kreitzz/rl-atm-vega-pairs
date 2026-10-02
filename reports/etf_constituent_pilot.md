# ETF constituent pilot: beta gate ablation

Seven frozen pairs: OIH/SLB, OIH/HAL, OIH/RIG, OIH/BKR, SIL/HL, SIL/AG and SIL/CDE. Membership is evidenced by the [OIH March 2018 SEC holdings](https://www.sec.gov/Archives/edgar/data/1137360/000093041318001946/c91215_nq.htm) and [SIL July 2018 SEC holdings](https://www.sec.gov/Archives/edgar/data/1432353/000114420418051506/tv502272_nq.htm). BKR is the current label for the linked Baker Hughes history. These are historical snapshot constituents, not a reconstructed daily membership universe. Selection is restricted to available cached histories and is exploratory; several pairs were previously tested.

Evaluation: January 2019–August 29, 2025, after both 2018 filings. Training can use earlier completed price and option outcomes; it does not assert earlier ETF membership. Roster is frozen after selection, with no subsequent membership removals.

Two separately retrained variants: beta 0.85–1.15 versus no beta cutoff. Both retain correlation ≥0.5, finite beta as an input, VIX, the existing features, monthly refits, profitability-memory and validation gates. Beta is ETF-on-stock (the existing leg orientation), lagged and frozen monthly. Models remain independent per pair: RL with the nine-level sizing ladder versus fixed-size NN.

Costs and risk settings are identical: actual near-ATM calls targeting 30 days, daily delta hedging, hold to expiry; mid plus 0.1 vol point per leg on entry, $0.65 per contract, 1 bp stock execution, borrow and financing. Initial capital is $1m per pair, including idle sleeves. Existing position, reserve, hedge and stress limits apply per pair. Shared ETF exposure is not netted or constrained across sleeves.

7/7 pairs completed in all four model/variant combinations; only these enter every aggregate comparison. Incomplete results remain in the pair CSV.

| Beta filter | Model | Active pairs | Entries | Annualized excess Sharpe | Net excess P&L |
|---|---|---:|---:|---:|---:|
| tight | rl_sized | 3 | 37 | 0.095 | $6,335 |
| tight | nn_fixed | 3 | 39 | 0.096 | $12,175 |
| no_beta_cutoff | rl_sized | 5 | 65 | -0.277 | $-45,587 |
| no_beta_cutoff | nn_fixed | 6 | 95 | 0.012 | $3,547 |

Sharpe uses daily returns above the cash benchmark and √252, including flat periods. Annual percentage returns include interest on unused cash. The 2025 observation is partial. This is a two-sector pilot on previously inspected dates, not an untouched test. Validation evaluates the proposal before the recent-data refit, not the exact deployed weights. NN convergence warnings and unresolved shadow labels remain disclosed by each variant’s audit.

[Pair results](etf_constituent_pilot/pair_results.csv) · [Yearly returns and Sharpe](etf_constituent_pilot/annual.csv) · [Beta eligibility](etf_constituent_pilot/eligibility.csv) · [Comparison](etf_constituent_pilot/comparison.csv)

The tight RL basket gains $6.3k above cash, with $5.0k from four OIH/BKR entries. Removing the beta cutoff increases RL entries from 37 to 65 but loses $45.6k: OIH/RIG loses $30.5k and SIL/HL loses $28.9k, partly offset by other pairs. Tight-beta NN and RL have nearly identical aggregate Sharpe; neither result establishes a strong edge.

The tight beta gate excludes OIH/RIG and all SIL pairs despite median stock-return correlations of roughly 0.81–0.88, because ETF-on-stock betas are around 0.48–0.56. Removing it changes opportunity eligibility rather than vega neutrality. SIL/CDE has only 442 usable state rows across the full history and cannot satisfy the 504-row training minimum; SIL/HL and SIL/AG also have substantial training gaps. This is not a broad test of liquid index ETF constituents.

Cross-variant checks confirm identical state values apart from eligibility, identical standalone payoff labels, costs, features, risk settings and evaluation dates. The 150-test suite passes.

tight: audit passed for 7 pairs and 12,884 label rows; 183 unresolved shadow labels excluded from training. NN convergence warnings: 60/157 validation fits, 21/55 recent fits. Runtime 64s, two workers, peak worker RSS 0.63 GiB. [Audit](etf_constituent_pilot/tight/audit.json).

no_beta_cutoff: audit passed for 7 pairs and 12,884 label rows; 183 unresolved shadow labels excluded from training. NN convergence warnings: 116/265 validation fits, 34/106 recent fits. Runtime 83s, two workers, peak worker RSS 0.63 GiB. [Audit](etf_constituent_pilot/no_beta_cutoff/audit.json).
