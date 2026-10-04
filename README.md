# RL for ATM Vega Pair Trading

Actual ATM options closest to 30 days, initially vega neutral and delta hedged daily, held to expiry. RL chooses direction and a 0/25/50/75/100% sizing ladder independently for each pair.

**Current best: 0.368 annualized excess Sharpe (2018–August 2025); 0.428 since 2022.** [Checkpoint](config/current_contract_reference.json) · [Results](reports/atm65_employee_sentiment.md) · [Yearly returns and Sharpe](reports/atm65_employee_sentiment/annual.csv) · [Pair results](reports/atm65_employee_sentiment/pairs.csv).

[Finalized RL inputs (compressed CSV)](data/current_rl_inputs.csv.gz): 290,151 daily rows across 139 pairs with states; [full roster](data/current_rl_pairs.csv) contains 345 sleeves, including structural cash sleeves. Includes the 47 original state inputs, two employee-leadership states, entry IVs and entry gates. Employee states are used only where `employee_features_enabled` is true. Legacy AMD/NVDA column suffixes mean the first/second leg for every pair. No forward rewards or targets; profitability-memory columns use completed past trades. [Data details and hashes](data/current_rl_data_manifest.json). Contract replay and reward reconstruction also require the licensed source option, stock, dividend and rate histories.

Training/validation commission: **$0.65 per contract**; execution: **$0.03**, plus 0.10 vol-point option penalty and 1 bp stock hedging cost. Monthly retraining, completed-outcome purging, beta 0.85–1.15 and correlation ≥0.5. Existing position, financing and risk limits are retained.

[Q-learning revision](reports/q_batch.md): exhaustive batch Q updates, economic state ablations, and missing-date sensitivity; AMD/NVDA improves in point estimate, without a statistically reliable advantage over cross-entropy.

[Broader universe and prior-profitability state](reports/broad_pairs.md).

**Future research is RL-only.** Historical NN comparisons stay saved. [Training settings](config/research_training.json) · [RL-only runner](scripts/run_atm65_rl_transcripts.py). New runs use one worker, 50% of one CPU core and a 3 GiB memory stop.

[Research findings and comparisons](reports/research_findings.md) · [New alternative-data priorities](reports/vol_alternative_data.md) · [Original ATM-IV proxy dataset](data/current_dataset.csv).

**Academic use only.** Third-party data remain subject to OptionMetrics, CRSP, Compustat, Revelio and other vendor/institutional license terms; this notice grants no additional data rights. Backward publication lags are enforced, but archived historical vendor versions are not fully verified.
