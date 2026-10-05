# Initial proxy/contract payoff reconciliation

Matched entry opportunities from 2013 through the available 2025 signal cutoff, limited to the same four pairs and dates with valid full-size shadow trades in both directions. [Results](matched_payoffs.csv).

The proxy is the original standardized-30D-IV reward `(RV_A-IV_A)-(RV_B-IV_B)`. The actual-contract directional component is `(net long-A/short-B payoff - net short-A/long-B payoff)/2`, divided by the common $450 target entry vega. Symmetric execution, commission and hedge costs cancel in this difference; directional financing/borrow differences remain. Actual contract maturities and early exits are retained, so the horizons are not yet matched exactly. These are overlapping shadow cohorts, not independent observations or realized policy performance. Correlation measures payoff agreement and does not establish predictive trading skill.

The original broad proxy result (2.606 Sharpe with memory) covered 350 candidate pairs during 2018–August 2025; the original four-pair proxy reference was 0.499 during 2010–August 2025. Current contract experiments cover only the four original pairs during 2013–August 2025, with different training, risk controls and Sharpe accounting. A broad-universe actual-contract replication has not been completed.
