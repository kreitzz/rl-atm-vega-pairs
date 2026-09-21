# Relative-value capture and profit concentration

The trades capture a relative volatility premium within the simulation, but the evidence is stronger for selling the richer leg and paying for a hedge than for sophisticated convergence timing. The larger universe is not dependent on one winning pair; dependence on a handful of underlying names, recent years, and extreme standardized IV marks is more substantial.

This audit uses the existing 1,805 jointly complete entry dates in 2018–August 2025, identical to the broad-universe report. It retains the original per-fold pair budgets and idle allocations. The primary policy is the stronger sized RL model without the explicit profit-history state; the memory version is included for comparison. No policy was retrained.

## What the trades buy and sell

For the primary policy, 72.0% of deployed entry vega shorts the higher-IV leg, 61.1% agrees with the trailing-RV-adjusted value signal, and 63.1% agrees with reversion toward the historical spread mean. The trade-level net win rate is 53.6%. These fractions are not probabilities of future success.

| Vega-weighted leg | Entry IV | Subsequent 30-day RV | Net contribution |
| --- | ---: | ---: | ---: |
| Long | 37.43 | 36.32 | -955.00 |
| Short | 46.56 | 40.93 | 3055.06 |

The IV gap averages 9.14 vol points, while the subsequent RV gap averages 4.62. Their difference leaves 3.47 net proxy vol points per unit deployed entry vega after costs. The long side is relatively cheaper, but not absolutely cheap against subsequent RV on average. The short side earns 145.5% of net profit, with the long side subtracting the balance.

The decomposition is algebraic, not causal attribution: `net = size × [(short IV − long IV) + (long forward RV − short forward RV)] − costs`. Costs are split by each leg's IV so the two legs reconcile exactly to the pair. Net contribution units are weighted proxy P&L per unit total maximum cohort budget, not a capital return percentage.

Trades agreeing with the RV-adjusted value signal earn 2,225.95 net proxy units; trades opposing it lose 125.90. The former account for 61.1% of exposure. This is a descriptive split on an existing input, not an independently tested exclusion rule.

## How much does the learned direction add?

| Model | Sample | higher_iv_rule | net | reverse | rv_value_rule | spread_rule | train_constant |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rl_memory | all | 2.652 | 2.606 | -4.196 | 2.422 | 2.050 | 0.303 |
| rl_memory | iv_at_most_100 | 1.362 | 1.520 | -4.375 | 1.030 | 0.260 | -0.356 |
| rl_no_memory | all | 2.885 | 2.788 | -4.269 | 2.566 | 2.359 | 0.522 |
| rl_no_memory | iv_at_most_100 | 1.704 | 1.853 | -4.674 | 1.191 | 0.525 | -0.282 |

These controls reuse each policy's selected entry dates and absolute sizes, replacing only direction. `higher_iv_rule` shorts the higher-IV leg; `rv_value_rule` follows the trailing-RV-adjusted value; `spread_rule` follows historical spread reversion. A zero control signal stays flat. `train_constant` chooses one fixed pair direction using only the corresponding historical training sample; `reverse` flips every actual trade. These are not standalone full-size strategies on every possible date.

For the primary policy, the higher-IV rule slightly exceeds learned-direction Sharpe (2.885 versus 2.788) on the same exposure schedule. With entries above 100 vol suppressed, learned direction does better (1.853 versus 1.704). A training-selected fixed pair direction is much weaker (0.522 on the full sample). Thus changing which leg is rich matters, but learned direction is not clearly superior to a simple IV comparison across the full book. The controls do not isolate the incremental benefit of entry timing or sizing.

## Concentration

| Model | Aggregation | Top N | Net contribution | Share of total net | Share of positive-contribution pool |
| --- | --- | --- | --- | --- | --- |
| rl_memory | pair | 1.000 | 96.722 | 0.055 | 0.044 |
| rl_memory | pair | 5.000 | 408.984 | 0.231 | 0.188 |
| rl_memory | pair | 10.000 | 674.615 | 0.380 | 0.309 |
| rl_memory | pair | 20.000 | 1043.369 | 0.588 | 0.479 |
| rl_memory | sector | 1.000 | 317.047 | 0.179 | 0.167 |
| rl_memory | sector | 5.000 | 1152.094 | 0.650 | 0.605 |
| rl_memory | sector | 10.000 | 1721.187 | 0.970 | 0.904 |
| rl_memory | sector | 20.000 | 1882.995 | 1.062 | 0.989 |
| rl_memory | underlying_leg | 1.000 | 191.550 | 0.108 | 0.077 |
| rl_memory | underlying_leg | 5.000 | 754.032 | 0.425 | 0.301 |
| rl_memory | underlying_leg | 10.000 | 1244.392 | 0.702 | 0.498 |
| rl_memory | underlying_leg | 20.000 | 1625.803 | 0.917 | 0.650 |
| rl_no_memory | pair | 1.000 | 140.366 | 0.067 | 0.057 |
| rl_no_memory | pair | 5.000 | 463.559 | 0.221 | 0.189 |
| rl_no_memory | pair | 10.000 | 747.654 | 0.356 | 0.305 |
| rl_no_memory | pair | 20.000 | 1149.403 | 0.547 | 0.469 |
| rl_no_memory | sector | 1.000 | 391.096 | 0.186 | 0.178 |
| rl_no_memory | sector | 5.000 | 1380.880 | 0.658 | 0.628 |
| rl_no_memory | sector | 10.000 | 1992.663 | 0.949 | 0.906 |
| rl_no_memory | sector | 20.000 | 2176.256 | 1.036 | 0.989 |
| rl_no_memory | underlying_leg | 1.000 | 217.134 | 0.103 | 0.078 |
| rl_no_memory | underlying_leg | 5.000 | 870.750 | 0.415 | 0.312 |
| rl_no_memory | underlying_leg | 10.000 | 1430.681 | 0.681 | 0.513 |
| rl_no_memory | underlying_leg | 20.000 | 1849.179 | 0.881 | 0.663 |

On the shared portfolio calendar, 203 pairs contribute positively, 103 negatively, and 1 are flat. One of the 308 fitted pairs has no rows on this calendar, leaving 307 represented here. Pair histories vary in length. Shares of total net can exceed 100% because losses elsewhere offset positive contributions; the separate positive-pool denominator avoids that interpretation problem.

Top pair contributions, primary policy:

| Pair | Share of total net | Dominant-direction exposure share | Net contribution | Contribution when IV >100 |
| --- | --- | --- | --- | --- |
| INFN/COMM | 0.067 | 0.870 | 140.366 | 147.534 |
| SFL/DHT | 0.047 | 0.564 | 99.173 | 83.017 |
| SIL/HL | 0.041 | 0.672 | 85.273 | 65.652 |
| HL/KGC | 0.033 | 0.637 | 69.562 | 63.636 |
| AEM/KGC | 0.033 | 1.000 | 69.185 | 45.673 |
| FCEL/RUN | 0.031 | 0.920 | 65.390 | 62.776 |
| SLB/RIG | 0.030 | 0.945 | 63.917 | 48.713 |
| PRU/GNW | 0.025 | 0.740 | 52.518 | 23.534 |
| KGC/MUX | 0.024 | 0.774 | 51.345 | 58.289 |
| MET/GNW | 0.024 | 0.935 | 50.924 | 25.676 |

Top underlying-leg contributions, primary policy:

| Underlying label | Share of total net | Net contribution |
| --- | --- | --- |
| KGC | 0.103 | 217.134 |
| INFN | 0.103 | 215.399 |
| HL | 0.078 | 163.568 |
| RIG | 0.068 | 142.694 |
| SFL | 0.063 | 131.955 |
| WETF | 0.057 | 118.893 |
| AG | 0.055 | 115.619 |
| SWN | 0.053 | 111.834 |
| TK | 0.052 | 109.868 |
| GNW | 0.049 | 103.717 |

Names are grouped by permanent CRSP identifier; displayed ticker aliases can be historical. An underlying's contribution aggregates its long and short legs across every pair and allocates its own execution cost. It is not the profit of an unhedged standalone strategy. For removal tests, all pairs involving the named underlying are removed together so the retained trades remain paired.

The top five name contributions (KGC, INFN, HL, RIG, SFL) explain 41.5% of net profit, versus 22.1% for the top five pairs. The top three historical pair-industry groups are metal mining (18.6%), oil/gas extraction (16.0%), and water transportation (11.6%). ETF comparisons remain in a separate group. Industry labels use [OSHA's SIC manual](https://www.osha.gov/ords/imis/sic_manual.display?id=30&tab=group); the underlying historical codes come from the local security-name records.

## Does the remainder still earn money?

| Model | Ex-post exclusion | Remaining Sharpe | Remaining net P&L |
| --- | --- | --- | --- |
| rl_memory | top_1_pairs | 2.531 | 1676.990 |
| rl_memory | top_5_pairs | 2.248 | 1364.729 |
| rl_memory | top_10_pairs | 2.025 | 1099.098 |
| rl_memory | top_20_pairs | 1.541 | 730.344 |
| rl_memory | pairs_involving_top_1_profitable_names | 2.409 | 1582.203 |
| rl_memory | pairs_involving_top_3_profitable_names | 2.164 | 1250.100 |
| rl_memory | pairs_involving_top_5_profitable_names | 1.894 | 1000.501 |
| rl_memory | top_1_sectors | 2.498 | 1456.665 |
| rl_memory | top_3_sectors | 2.043 | 925.036 |
| rl_no_memory | top_1_pairs | 2.767 | 1959.687 |
| rl_no_memory | top_5_pairs | 2.590 | 1636.493 |
| rl_no_memory | top_10_pairs | 2.429 | 1352.398 |
| rl_no_memory | top_20_pairs | 2.057 | 950.650 |
| rl_no_memory | pairs_involving_top_1_profitable_names | 2.665 | 1842.806 |
| rl_no_memory | pairs_involving_top_3_profitable_names | 2.453 | 1519.808 |
| rl_no_memory | pairs_involving_top_5_profitable_names | 2.278 | 1195.945 |
| rl_no_memory | top_1_sectors | 2.643 | 1708.957 |
| rl_no_memory | top_3_sectors | 2.236 | 1128.350 |

The primary policy retains Sharpe 2.429 without its ten best pairs, 2.057 without its best twenty, and 2.278 after removing every pair involving its five most profitable underlying legs. Rankings use the completed test history, so these are concentration diagnostics, not implementable selection rules. Budgets freed by exclusions remain idle. These separate removal tests still contain extreme-IV trades; they are not jointly cleaned results.

## Time and extreme-mark concentration

| Model | Entry year | Net contribution | Share of total net | Sharpe |
| --- | --- | --- | --- | --- |
| rl_memory | 2018 | 53.730 | 0.030 | 1.268 |
| rl_memory | 2019 | 58.430 | 0.033 | 1.622 |
| rl_memory | 2020 | 264.732 | 0.149 | 2.094 |
| rl_memory | 2021 | 89.226 | 0.050 | 1.493 |
| rl_memory | 2022 | 251.791 | 0.142 | 3.800 |
| rl_memory | 2023 | 204.922 | 0.116 | 3.634 |
| rl_memory | 2024 | 541.340 | 0.305 | 6.056 |
| rl_memory | 2025 | 309.542 | 0.175 | 6.861 |
| rl_no_memory | 2018 | 77.329 | 0.037 | 1.239 |
| rl_no_memory | 2019 | 137.326 | 0.065 | 2.007 |
| rl_no_memory | 2020 | 175.900 | 0.084 | 1.719 |
| rl_no_memory | 2021 | 60.474 | 0.029 | 1.228 |
| rl_no_memory | 2022 | 321.888 | 0.153 | 4.283 |
| rl_no_memory | 2023 | 307.173 | 0.146 | 4.061 |
| rl_no_memory | 2024 | 627.904 | 0.299 | 6.253 |
| rl_no_memory | 2025 | 392.058 | 0.187 | 7.342 |

For the primary policy, 78.5% of net profit comes from 2022–August 2025 and 48.6% from 2024–August 2025. The twenty best portfolio entry dates explain only 5.6% of total net profit. However, the best 1% of individual active pair-entry cohorts explain 43.5%, and the best 5% explain 104.9% after losses elsewhere. Thirty-day positions overlap: these cohorts are not independent trades or market episodes.

Entries with either IV above 100 vol use just 5.0% of deployed entry vega but contribute 66.5% of net profit. Excluding them leaves 1.853 Sharpe for the primary policy and 1.520 for the memory policy. This is a sensitivity check, not a newly validated filter.

A concrete concern is INFN/COMM, the largest pair contributor. On January 27, 2025 the cache shows INFN IV of 254.69 vol and subsequent RV of 4.17; the policy is short INFN and long COMM. Infinera was already subject to Nokia's announced acquisition, which closed on February 28, 2025. [Nokia transaction announcement](https://www.nokia.com/newsroom/nokia-to-acquire-infinera-to-increase-scale-in-optical-networks-and-accelerate-product-roadmap/) · [Infinera closing filing](https://www.sec.gov/Archives/edgar/data/1138639/000119312525041368/d847449d8k.htm). This is a reason to inspect actual contracts and merger-event risk, not proof that the surface mark is erroneous or the trade executable.

The interpretation is conditional on the standardized-IV/forward-RV payoff proxy, assumed execution costs, and already inspected history. The 121 entry dates with unresolved active-trade outcomes remain excluded exactly as in the preceding report; no missing active payoff is counted as zero.

[Summary and leg decomposition](trade_quality/summary.csv) · [Direction controls](trade_quality/direction_controls.csv) · [Pair contributions](trade_quality/pairs.csv) · [Underlying contributions](trade_quality/names.csv) · [Industry contributions](trade_quality/sectors.csv) · [Concentration](trade_quality/concentration.csv) · [Removal sensitivities](trade_quality/removals.csv) · [Value buckets](trade_quality/value_buckets.csv)

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.diagnose_trade_quality
```

The script reconciles leg contributions and the IV/RV decomposition to each stored pair payoff. It publishes aggregate diagnostics only; the shared market-data CSV is unchanged.
