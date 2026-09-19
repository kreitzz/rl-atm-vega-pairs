# Stock-return versus IV-change dependence

AMD/NVDA has a larger gap between stock-return and IV-change correlation than the other pairs in full-period statistics. A causal trailing version of that gap is associated with stronger existing-policy payoffs for AMD/NVDA, but the pattern is not universal across pairs. No policy or selection rule was changed for this diagnostic.

Pairwise return beta means regressing the first stock's daily return on the second stock's daily return. It is not beta to an external market index. Correlation measures co-movement; beta additionally incorporates the ratio of movement magnitudes. IV-change beta uses changes in IV percentage points, not percentage returns in IV. It is not the same quantity as option-implied equity beta or option-implied correlation.

## Full-period comparison

Common 2010–August 2025 dates. These descriptive full-period figures are not state inputs.

| Pair | Stock-return correlation | IV-change correlation | Stock-return beta | IV-change beta |
| --- | ---: | ---: | ---: | ---: |
| AMD/NVDA | 0.564 | 0.468 | 0.684 | 0.578 |
| XOM/CVX | 0.827 | 0.816 | 0.773 | 0.695 |
| JPM/BAC | 0.852 | 0.805 | 0.705 | 0.633 |
| HD/LOW | 0.798 | 0.742 | 0.677 | 0.668 |

## Causal trailing-gap diagnostic

Two possible features use 252 matched observations through the entry close:

- `correlation_gap = stock_return_correlation - IV_change_correlation`.
- `beta_gap = stock_return_beta - IV_change_beta`.

The gaps compare like statistics. A beta gap is asymmetric in ticker order and sensitive to relative stock/IV movement scales; neither gap by itself specifies which stock's volatility is cheap. Features assume execution after observing the entry close, as in the existing setup; execution at that same closing quote would require lagging them.

For each walk-forward block, classify test dates above or below the **preceding training sample's median** gap. Thresholds use no test outcomes. The existing independent ladder policies and proportional costs are unchanged. Rows below give conditional Sharpe on dates in each group, including the policy's flat dates within that group. They are **not** full-calendar Sharpes of a newly trained or gated strategy. Test group sizes differ because historical distributions shift.

| Pair | Correlation gap: low / high Sharpe | Low / high observations | Beta gap: low / high Sharpe |
| --- | --- | --- | --- |
| AMD/NVDA | 0.361 / 0.971 | 2454 / 1468 | 0.200 / 0.949 |
| XOM/CVX | -0.046 / -0.283 | 3518 / 404 | 0.116 / -0.707 |
| JPM/BAC | -0.001 / -0.804 | 3541 / 381 | -0.011 / -0.895 |
| HD/LOW | -0.130 / 0.021 | 3005 / 917 | -0.169 / 0.196 |

AMD/NVDA is the clearest promising example: correlation-gap high/low conditional Sharpe is 0.971/0.361, and beta-gap high/low is 0.949/0.200. Its average absolute gross pair payoff is also larger in the high-gap groups. However, its Spearman correlations between either gap and signed future gross payoff are small (-0.023 and -0.073). The evidence is more suggestive of a trading-environment or sizing feature than a standalone directional forecast.

Energy and banks perform worse in the high-correlation-gap group, whereas HD/LOW improves only to approximately flat. These results do not support a universal rule that a larger gap is better. Full-period association with policy payoffs can reflect market regimes, existing signals, and selection choices. Feature construction is causal, but this exploratory analysis and the underlying policy were evaluated on repeatedly inspected history; overlapping 30-day outcomes preclude treating daily rows as independent evidence. No significance or causal claim is made.

A controlled next experiment would add one gap at a time to each pair's separate state, retain the independent ladder and fixed budgets, and compare matched walk-forward outcomes. That experiment has not been run here.

[Dependence comparison](correlation_gap/summary.csv) · [Conditional groups](correlation_gap/bins.csv) · [Rank associations](correlation_gap/associations.csv) · [Annual conditional groups](correlation_gap/annual_bins.csv)

```sh
OPENBLAS_NUM_THREADS=1 python -m scripts.diagnose_correlation_gap --source-root /path/to/wrds
```

Detailed features remain local in the ignored run directory. Thirty-four tests pass, including beta-versus-correlation identities and future-data exclusion for the rolling estimates.
