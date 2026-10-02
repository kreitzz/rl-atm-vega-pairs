# Contract-curve choice experiment

346/346 pairs complete across both new variants. Common-roster comparison:

| Policy | Annualized excess Sharpe | Entries | Active pairs | P&L above cash |
|---|---:|---:|---:|---:|
| previous_atm_reference | 0.215 | 256 | 30 | $96,246 |
| atm_control | 0.276 | 238 | 30 | $110,465 |
| curve | 0.157 | 215 | 27 | $67,151 |

Actions: 14/30/45-day target expiries; calls/calls or puts/puts; each leg independently chooses exact original ATM or 10/25/50/75/90 absolute BS delta. Nearest eligible actual contract within 0.075 delta; no interpolation or extrapolation of traded contracts. Both legs share a target tenor, but their actual expiries may differ. Expiry is anchored to the prior tenor study’s selection, so this is a bounded grid, not every strike or every maturity. Exact ATM remains nested. Same standard-contract/OI/quote-width filters; the former 10% ATM moneyness limit is replaced by explicit delta selection.

The expanded policy considers all 216 contract combinations. Each January, it retains exact ATM30 calls plus up to seven combinations ranked by their best full-size directional shadow daily Sharpe over five years strictly before the prior validation year. Ranking requires 21 memory-permitted eligible cohorts; it does not inspect validation or evaluation outcomes. Monthly eight-leaf direct Sharpe policy search chooses the contract combination and one of eight signed sizes or flat. This is direct policy search, not a Q-learning or actor-critic implementation. The ATM control uses the same generalized optimizer with only the original ATM30 call choice; its random-search action ordering differs from the older reference.

Same native 30D reference state and observation calendar, including VIX, momentum, surface context and ATM profitability memory. No new smile-state variables are added here. The execution gate uses the selected combination’s own strictly completed 91/365-day directional profitability memory. Initial integer vega mismatch ≤10%, daily delta hedging, existing costs and portfolio risk limits, monthly beta .85–1.15 and correlation ≥.5. Option quantities are fixed after entry apart from supported splits; there is no daily option-vega rebalancing. Each leg settles at its own expiry. Early assignment is not modeled.

Execution uses observed midpoint plus/minus a 0.1-vol-point entry penalty per leg, $0.65 per option contract, and 1 bp on stock hedge trades. It does not assume crossing the full quoted option spread. Annual return columns distinguish total return, the cash benchmark, and return above cash; Sharpe uses daily cash-excess returns times √252.

Monthly five-year proposal fit, separate prior-year full-book validation, then recent-five-year completed-outcome refit if validation cash-excess Sharpe is positive. As in the inherited model, validation gates pre-refit weights. Training uses rows with complete labels across the current shortlist; unavailable contracts give known no-entry labels, while unresolved held paths are excluded only from past training, never used as deployment filters. Float32 temporary labels bound storage; all cash-flow calculations and portfolio replay use float64. Common 60-calendar-day entry runoff; January 2018–August 2025 evaluation, including previously inspected research history. This is not an untouched holdout.

Failed sleeves remain explicitly incomplete; common-roster comparisons remove them from every policy, including the old reference. Full-roster results are withheld for incomplete variants. No reference model has been promoted.

Concentration diagnostics include replacing the largest winning sleeve with cash after the fact. This is a sensitivity check, not an implementable pair-selection rule.

[Annual results](curve_trading/annual.csv) · [Pair results](curve_trading/pair_results.csv) · [Concentration](curve_trading/concentration.csv) · [Contract choices](curve_trading/contract_mix.csv) · [Choice summary](curve_trading/choice_summary.csv) · [P&L components](curve_trading/pnl_components.csv) · [Incomplete sleeves](curve_trading/incomplete.csv) · [Audit](curve_trading/audit.csv) · [Real-data reconciliation](curve_trading/real_data_reconciliation.json)
