# Employee leadership sentiment on the ATM65 checkpoint

One added signal: the first company’s trailing three-month employee leadership rating minus the second’s, plus an availability flag. Missing values are filled with zero and flagged. At least 20 leadership reviews per company; monthly publication proxy is month-end plus 15 calendar days plus one business day, with 60-day maximum carry. No direction or threshold was chosen from the test returns.

345 original matched sleeves retained; 28 have joint review observations on permitted test dates and receive new fits. Remaining sleeves reuse their own original RL or NN policy. Covered sleeves are refitted for the whole evaluation period, including missing-review dates, so this tests both the score and its availability flag. Coverage is determined by inputs and existing eligibility, never realized returns.

All 47 original inputs remain unchanged. Monthly prior-year validation and recent-five-year refits, strict completed-outcome purging, original eight-leaf Sharpe policy search / fixed-size NN, 25/50/75/100% RL ladder, beta/correlation and profitability gates, actual ATM30 call contracts, daily delta hedging and hold to expiry are unchanged. Validation/training commission is $0.65; evaluation commission is $0.03, with the original spread, hedge, financing and risk limits. Original RL monthly actions and enabled NN refits are reconstructed and checked before candidate evaluation.

| Universe | Model | Full Sharpe | Recent Sharpe | Full entries |
|---|---|---:|---:|---:|
| all | rl_baseline | 0.318 | 0.347 | 224 |
| all | rl_sentiment | 0.368 | 0.428 | 224 |
| all | nn_baseline | -0.097 | -0.099 | 221 |
| all | nn_sentiment | -0.084 | -0.103 | 217 |
| review_covered | rl_baseline | 0.194 | 0.276 | 45 |
| review_covered | rl_sentiment | 0.328 | 0.479 | 45 |
| review_covered | nn_baseline | 0.013 | 0.359 | 44 |
| review_covered | nn_sentiment | 0.057 | 0.329 | 40 |

Full: 2018–August 2025. Recent: 2022–August 2025, starting flat. Sharpe uses daily returns above cash and √252. Any unresolved contract paths are explicitly excluded from every row on a common roster.

This is an exploratory ablation on previously inspected history, following a sentiment correlation screen on the same period. Current Revelio company mappings and historical review edits have not been validated against archived vendor vintages. The date lag prevents use of later review months but cannot remove those vintage risks. No correlation-study payoff labels were used for strategy training: labels were rebuilt from the saved contract selections at the checkpoint cost settings. The checkpoint is preserved.

[Comparison](atm65_employee_sentiment/comparison.csv) · [Annual](atm65_employee_sentiment/annual.csv) · [Pairs](atm65_employee_sentiment/pairs.csv) · [Review coverage](atm65_employee_sentiment/coverage.csv) · [Accounting coverage](atm65_employee_sentiment/accounting_coverage.json)

## Verification and trade coverage

285 baseline action files and all snapshotted inputs retain their SHA-256 hashes. 1511 monthly control rows reproduced the saved RL actions; enabled original NN refits also reproduced their saved decisions. Original state values and completed-outcome cutoffs passed the audit.

| Candidate on refitted sleeves | Entries with reviews | Entries without reviews |
|---|---:|---:|
| nn_sentiment | 36 | 4 |
| rl_sentiment | 37 | 8 |

These counts refer to executed entries in the full-period book of the 28 refitted sleeves; untouched sleeves retain their original policies. Feature addition also exposes missing-data availability and can change tree partitions and validation decisions, so a performance change cannot automatically be attributed to leadership ratings alone.

[Policy changes](atm65_employee_sentiment/policy_changes.csv) · [Pair P&L changes](atm65_employee_sentiment/pair_changes.csv) · [Audit](atm65_employee_sentiment/audit.json)
