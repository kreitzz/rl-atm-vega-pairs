# Pair selection and beta versus expiry performance

The attempted universe contains 350 historically screened broad pairs plus eight legacy additions. The current matched comparison retains 347 complete broad pairs. Original broad membership was formed every two years from preceding-year coverage, option activity, historical industry and stock-return correlation, without payoff ranking. Stock pairs share a three-digit SIC industry; up to four names per industry are ranked by option open interest, then pairs are distributed across industries, capped at 180 stock pairs per screen. Minimum prior return correlation is 0.4, median call volume 100 and open interest 500. Predefined ETF pairs are added if eligible.

The current contract strategy additionally screens monthly: first-leg-on-second-leg stock-return beta must lie between 0.75 and 1.25, and correlation must be at least 0.5. Estimates use the preceding 126 sessions, excluding the formation day's return, and freeze for the month. Both beta and correlation are model inputs. This is directional pairwise stock beta, not market-index beta or IV beta.

## Existing retrained trades, grouped by signal-time beta

| Absolute distance of beta from one | RL trades | RL gross P&L | RL gross P&L / summed entry vega | NN gross P&L / summed entry vega |
|---|---:|---:|---:|---:|
| ≤0.05 | 104 | $53,781 | 2.18 | −0.98 |
| >0.05–0.10 | 120 | −$10,479 | −0.48 | 0.93 |
| >0.10–0.15 | 146 | $34,263 | 1.17 | −0.33 |
| >0.15–0.25 | 177 | −$62,442 | −1.66 | −0.35 |

Entry vega is the mean absolute initial dollar vega of the two legs, per one volatility point. Gross P&L includes actual expiry payoff, daily delta hedging and hedge dividends, but excludes execution, borrow and financing. Ratios are exposure-normalized payoff summaries, not percentage returns or Sharpes. Bins are descriptive, not optimized or adopted filters.

Across 52 active RL pairs, Spearman correlation between mean entry-beta distance from one and each pair's annualized net excess Sharpe is −0.205. Across 90 active NN pairs it is −0.031. At individual-trade level, distance versus gross payoff per entry vega has rank correlation −0.079 for RL and −0.001 for NN. Mean entry return correlation versus pair net Sharpe is −0.135 for RL and −0.200 for NN.

Closer-to-one beta looks favorable for the RL near-one bucket, but the pattern is nonmonotonic and does not replicate in NN. These are selected, overlapping trades, often sharing names; pair means and outcomes also reflect different trading dates and policy choices. There is no independent-sample significance claim. Since all entries already pass the narrow beta gate, this does not estimate performance outside that range or establish that tightening the gate improves a retrained portfolio. Complete-history selection and prior research exposure remain limitations.

Sources: `config/broad_pair_research.json`, `scripts/build_broad_pairs.py`, `scripts/contract_learning_labels.py`, and saved expiry trades and monthly screens. Reproduce with `python -m scripts.diagnose_beta_performance`; signal-time eligibility is asserted for every joined entry.

[Trade buckets](expiry_signal_gap/beta_buckets.csv) · [Pair profiles and net Sharpes](expiry_signal_gap/beta_pair_profiles.csv) · [Rank correlations](expiry_signal_gap/beta_associations.csv) · [Trade profiles](expiry_signal_gap/beta_trade_profiles.csv)
