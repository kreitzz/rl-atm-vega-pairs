# Transcript signals on the employee-sentiment reference

Baseline is the current best employee-sentiment strategy: annualized excess Sharpe 0.368 full history and 0.428 since 2022. Both periods start flat; all 345 original matched sleeves remain in the requested roster. Original employee-sentiment features remain active on the same 28 sleeves.

Three transcript gaps are added together: executive negative-language change versus the previous call, uncertainty in executive answers versus prepared remarks, and negativity in analyst questions. Each first-minus-second gap has its own availability flag. Missing values are zero-filled and flagged. Latest available transcript scores carry for at most 180 days. Earliest creation-dated version is selected for each company/event; availability is one business day after the later event/creation date. Historical company links and word-count requirements are unchanged.

100 sleeves have an observed transcript feature on a permitted evaluation date and are refitted across the evaluation period. Other sleeves reuse their own current reference policy. Features and coverage were fixed before these runs, with no sign or threshold selected from evaluation returns.

Same monthly prior-year validation, recent-five-year refits, completed-outcome purging, profitability memory and beta/correlation gates, 25/50/75/100% RL sizing ladder, fixed-size NN, ATM30 calls, daily delta hedging and hold to expiry. Training/validation commission is $0.65; evaluation commission is $0.03. Spread, hedge, financing and risk settings are unchanged. Enabled current-reference refits are reconstructed and checked against their saved actions before candidate evaluation.

| Universe | Model | Full Sharpe | Recent Sharpe | Full entries |
|---|---|---:|---:|---:|
| all | rl_baseline | 0.368 | 0.428 | 224 |
| all | rl_transcripts | 0.319 | 0.481 | 237 |
| all | nn_baseline | -0.084 | -0.103 | 217 |
| all | nn_transcripts | -0.157 | -0.139 | 217 |
| transcript_covered | rl_baseline | 0.296 | 0.338 | 177 |
| transcript_covered | rl_transcripts | 0.248 | 0.391 | 190 |
| transcript_covered | nn_baseline | -0.164 | -0.257 | 174 |
| transcript_covered | nn_transcripts | -0.246 | -0.304 | 174 |

Full: 2018–August 2025. Recent: 2022–August 2025, starting flat. Sharpe uses daily returns above cash and √252. Any unresolved contract paths are explicitly excluded from every model on a common roster.

Current best checkpoint remains frozen. This comparison uses previously examined history. Transcript lexicons are simple fixed word counts; vendor historical edits, publication timing and company mappings are not fully verified against archived data versions. The training labels are rebuilt from actual saved option-contract paths at the unchanged checkpoint costs.

[Comparison](atm65_transcripts/comparison.csv) · [Annual](atm65_transcripts/annual.csv) · [Pairs](atm65_transcripts/pairs.csv) · [Coverage](atm65_transcripts/coverage.csv) · [Fit diagnostics](atm65_transcripts/fits.csv)

## Verification and trade coverage

4394 monthly control rows reproduced current-reference actions. All snapshotted sources and the best-checkpoint archive passed SHA-256 verification. Original state values and completed-outcome cutoffs are unchanged.

| Candidate on refitted sleeves | Entries with transcripts | Entries without transcripts |
|---|---:|---:|
| nn_transcripts | 167 | 7 |
| rl_transcripts | 174 | 16 |

Of 471 enabled RL recent refits, 224 split on a transcript value and 9 split on transcript availability. These counts can overlap. Adding features can also change original-feature partitions and validation decisions.

[Policy changes](atm65_transcripts/policy_changes.csv) · [Pair P&L changes](atm65_transcripts/pair_changes.csv) · [Audit](atm65_transcripts/audit.json)
