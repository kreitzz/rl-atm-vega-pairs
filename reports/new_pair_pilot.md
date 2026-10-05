# New industry-peer pilot with the current VIX policy

Twelve illustrative new pairs selected before observing their backtest P&L: APA/DVN, HES/COP, CDE/AG, BZH/LEN, MGM/HLT, AIG/PRU, AMAT/INTC, A/DHR, BA/TXT, NEE/SO, LUMN/T and EW/MDT. The sample spans multiple businesses and uses already-downloaded names. This is a manually chosen exploratory sample from historical discovery, not a random sample or an untouched universe-selection test. Ticker labels can be historical.

Independent models retain the current VIX-only features, monthly training and recent refits, hard profitability-memory and validation gates, beta 0.85–1.15, correlation ≥0.5, risk limits and costs. New entries are allowed only in historically qualifying discovery quarters; existing positions still run to expiry. Training and validation use completed historical outcomes, including history preceding discovery, as in the existing specification.

Discovery beta uses direct underlying CRSP returns; monthly policy beta continues to use the existing prepared return series, which has some IV-availability-related gaps. This preserves the current policy input pipeline but can remove otherwise eligible opportunities. No additional gate was relaxed for this pilot.

| Pair | RL entries | RL annualized excess Sharpe | RL net excess P&L | NN entries | NN net excess P&L |
|---|---:|---:|---:|---:|---:|
| APA/DVN | 5 | -0.514 | $-18,598 | 16 | $-20,826 |
| HES/COP | 14 | 0.185 | $7,325 | — | incomplete |
| CDE/AG | 0 | — | $0 | 3 | $13,718 |
| BZH/LEN | 0 | — | $0 | 0 | $0 |
| MGM/HLT | 8 | -0.349 | $-14,224 | 4 | $-23,220 |
| AIG/PRU | 0 | — | $0 | 1 | $1,947 |
| AMAT/INTC | 1 | -0.045 | $-895 | 7 | $7,209 |
| A/DHR | 0 | — | $0 | 0 | $0 |
| BA/TXT | 0 | — | $0 | 2 | $9,964 |
| NEE/SO | 2 | 0.127 | $3,588 | 3 | $11,555 |
| LUMN/T | 0 | — | $0 | 1 | $25,327 |
| EW/MDT | 0 | — | $0 | 0 | $0 |

## Portfolio comparison

11 of 12 new pairs complete under both models. Incomplete pairs are excluded consistently from aggregate comparisons and retained in the status file. Existing reference uses its fixed 346-pair roster.

| Model | Portfolio | Annualized excess Sharpe | Net excess P&L |
|---|---|---:|---:|
| rl_sized | existing_reference | 0.226 | $105,789 |
| rl_sized | new_pairs_only | -0.469 | $-30,128 |
| rl_sized | existing_plus_new | 0.162 | $75,661 |
| rl_sized | existing_plus_cash | 0.226 | $105,789 |
| nn_fixed | existing_reference | -0.081 | $-66,621 |
| nn_fixed | new_pairs_only | 0.129 | $25,674 |
| nn_fixed | existing_plus_new | -0.047 | $-40,946 |
| nn_fixed | existing_plus_cash | -0.081 | $-66,621 |

HES/COP is excluded from both models’ aggregate new-pair baskets because the NN encountered an unsupported distribution on February 8, 2024 (option 159326264). Its complete positive RL result remains visible above.

RL trades five of the twelve candidates, or four of the eleven in the matched aggregate. The matched RL basket has 16 entries; NN has 37. LUMN/T’s one NN trade contributes about $25.3k of the matched new NN basket’s $25.7k net excess profit, so the positive NN result is highly concentrated. Sparse entries do not establish persistent pair-level profitability.

January 2018–August 2025. Each pair starts with $1m; cash-only periods remain included. Sharpe uses daily cash-excess returns and √252. Percentage returns in the CSV include interest on unused capital. `existing_plus_cash` funds the added sleeves but leaves them idle, giving the same starting capital as `existing_plus_new`. Aggregate results combine independent pair sleeves: they do not enforce a new portfolio-wide concentration or shared-underlying risk budget.

All dates have already been inspected during research. The pilot is not evidence of an independently validated strategy. The monthly validation gate still evaluates the proposal before its recent-data refit, rather than the exact deployed weights. NN convergence warnings are retained in the fit logs.

[Pair results](new_pair_pilot/pair_results.csv) · [Period comparison](new_pair_pilot/comparison.csv) · [Yearly returns and Sharpe](new_pair_pilot/annual.csv) · [Historical membership](new_pair_pilot/universe.json) · [Run status](new_pair_pilot/status.csv) · [Fits](new_pair_pilot/fits.csv) · [Audit](new_pair_pilot/audit.json)


Audit passed across 12 pairs: completed-label memory timing, action ladders, beta eligibility and settlement/cost checks. Quarterly historical membership and fit cutoffs also pass. The full 148-test suite passes. NN convergence warnings: 127/247 validation fits and 37/96 recent refits. Runtime 1.6 minutes; peak worker RSS 0.72 GiB with two workers.
