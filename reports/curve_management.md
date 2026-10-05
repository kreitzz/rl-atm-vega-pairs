# Expiry, smile and learned-exit comparison

Same 345 requested pairs, initial vega matching, integer quantities, daily aggregate delta hedge, entry risk limits and beta/correlation eligibility. Models refit quarterly with daily entry and exit decisions, using prior-year validation and completed-outcome refits. $0.65 training commission hurdle; $0.03 executed commission; midpoint plus 0.10 vol point per option leg on entry and early exit; 1 bp stock execution and original borrow/financing. All policies use 110 calendar days of validation/final-entry runoff. The saved 30D checkpoint used monthly refits and 60 days and remains unchanged, so it is contextual rather than the matched control.

| Universe | Hold full Sharpe | Learned-exit full Sharpe | Hold recent | Learned-exit recent |
|---|---:|---:|---:|---:|
| atm30 | 0.077 | 0.015 | -0.043 | -0.058 |
| tenors | 0.028 | 0.105 | -0.111 | 0.091 |
| smile | -0.150 | 0.216 | -0.224 | 0.256 |
| both | -0.024 | 0.193 | -0.095 | 0.228 |

Full 2018–August 2025; recent 2022–August 2025 starts flat. Cash-excess daily Sharpe annualized by √252.

ATM30 is a newly retrained control under the new experiment settings, not a replay of the frozen checkpoint. Tenors chooses 14/30/90D ATM calls. Smile chooses independent lower-strike (~75-delta call), ATM or higher-strike (~25-delta call) buckets on each leg at 30D. Both combines those expiry and strike choices. All legs are calls; puts and cross-expiry pairs are outside this pass. A training-only annual screen selects at most four combinations per universe, retaining ATM30; it uses five years ending before the prior validation year. Dates and gates never depend on future execution or label resolution.

Entry states are the original 47 inputs plus nine quote-context values with availability flags: ATM IV spread, lower-minus-upper-strike IV skew difference and ATM gamma-per-vega gap at each expiry. All eight models see identical states. Entry partitions retain separate directional targets for each shortlisted combination, with the same eight-leaf Sharpe search and 25/50/75/100% sizing ladder.

The learned exit policy uses the actual fixed contracts: remaining DTE, live-leg flags, IV spread and its change from entry, current RV/IV variance value, gamma per vega, moneyness, relative vega drift, quote width, VIX, beta and correlation. It learns an eight-leaf hold/close policy by evaluating every binary leaf rule against daily marked and hedged shadow P&L, including early-exit costs and subsequent excess cash interest. Deterministic, reward-independent sampling uses at most 32 completed training cohorts per shortlisted combination per direction. Fully completed expiry paths are required before every training cutoff, even if a candidate exits earlier.

Proposal entry and exit policies train before the prior validation year. Enabled entries refit using completed outcomes from the last five years; exits also refit to manage older live positions when new entry is disabled. Training entry rewards reflect the learned exits. Memory gates continue to use strictly prior completed hold-to-expiry outcomes at the training hurdle for matched entry eligibility. The learned exit is independent of whether the new-entry policy is flat.

Inventory records original option IDs, strike, expiry, quantity and cohort. Different expiry/strike cohorts can overlap across days, with one new pair cohort permitted per signal day. Quantities stay fixed apart from verified splits; each leg naturally settles at its own expiry. Learned exits close every remaining leg of a cohort together on the next session. Hedge turnover is netted across all open cohorts within each pair sleeve. Funding and risk limits are applied per pair, consistent with the saved checkpoint. Entry decisions retain the original native-30D signal-date calendar. Initial vega matching ≤10% is enforced; daily option-vega rebalancing and early assignment remain unmodeled.

The strongest new case is the 30D smile policy with learned exits: 200 entries across 20 pairs, with 76 cohorts closed early. Its full/recent Sharpes are 0.216/0.256, below the saved monthly-refit checkpoint at 0.318/0.347. The new matched ATM30 control also uses additional curve inputs and a longer runoff, so differences from the saved checkpoint cannot be attributed solely to expiry choice or exits. Exit-enabled entry policies are retrained on exit-adjusted rewards; their comparison with hold-only policies combines entry and exit changes.

In the combined expiry/smile/exit case, 231 of 245 entries use 30D contracts, nine use 14D and five use 90D. Neither ATM-only expiry policy selects 90D. That is too little executed 90D exposure to assess a dedicated 90D strategy. Across the processed native pair/date rows, the hypothetical ATM paths have 147,874 complete 14D outcomes, 255,443 complete 30D outcomes and 76,416 complete 90D outcomes; these are data/entry-validity counts before pair eligibility and profitability gates, not independent samples.

The audit verified all 5,520 model/pair/period windows, 645 early-exit cohorts and 240 books with overlapping inventory. Largest ledger error was $6.14e-10, with no unresolved windows. Six focused tests passed, including overlapping 14/30/90D contracts, whole-cohort exits, financing and training cutoffs.

These are exploratory results on previously examined history. Unsupported paths are listed and excluded from every model on the common roster, never assigned cash P&L.

[Comparison](curve_management/comparison.csv) · [Annual](curve_management/annual.csv) · [Pairs](curve_management/pairs.csv) · [Inventory choices](curve_management/contract_mix.csv) · [Coverage](curve_management/coverage.json) · [Audit](curve_management/audit.json)
