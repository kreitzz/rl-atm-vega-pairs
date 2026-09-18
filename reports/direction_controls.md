# Matched direction controls

The policy outperforms always-long and always-short volatility on both legs when dates and sizes are held fixed. However, much of its P&L survives a control that preserves its broad allocation by walk-forward block and size. The evidence for precise entry-by-entry direction timing is therefore weaker than a simple coin-flip comparison suggests.

## Individual legs

All rows use the reference policy's 3,055 nonflat entries, actual size ladder, and 867 flat opportunities over 2010–August 2025. Values are net proxy dollars per $1 of maximum initial vega per leg. Half the 1.0-point pair cost is assigned to each leg solely for attribution; this is not an observed per-leg execution cost.

| Leg | Policy net P&L | Always long volatility | Always short volatility |
| --- | ---: | ---: | ---: |
| AMD | **6,634.21** | -4,133.26 | 2,416.01 |
| NVDA | **1,328.87** | -1,027.72 | -689.53 |

For AMD, a persistent short-volatility position makes money on these selected dates, but less than the policy. For NVDA, both constant directions lose after assigned costs while the policy is positive. These are comparisons conditional on the policy's entry dates and sizes, not independently trained standalone strategies.

## Vega-neutral pair controls

| Pair direction rule | Net proxy P&L | Annualized payoff Sharpe |
| --- | ---: | ---: |
| Policy | **7,963.08** | **0.667** |
| Always short AMD / long NVDA | 1,388.29 | 0.113 |
| Always long AMD / short NVDA | -4,822.79 | -0.391 |

All pair controls use opposite directions and equal initial dollar vega for the two legs. Same dates, same absolute sizes, same costs. The policy's positive results are not reproduced simply by always shorting AMD against NVDA.

## Randomized direction controls

Run 5,000 trials per method, seed 2026. Direction is randomized, and all selected dates, absolute sizes, flat dates, and cost amounts remain fixed. AMD and NVDA always take opposite randomized directions.

| Control | Mean pair net P&L | Mean pair Sharpe | 95% empirical Sharpe range | Fraction with Sharpe at least policy |
| --- | ---: | ---: | --- | ---: |
| Independent fair coin flips | -1,719.80 | -0.139 | [-0.250, -0.031] | 0 of 5,000 |
| Shuffle within fold × exact size | 6,268.49 | 0.519 | [0.472, 0.567] | 0 of 5,000 |
| Circular shift within fold × exact size | 6,294.40 | 0.522 | [0.353, 0.697] | **7.3%** |

The first control destroys the long/short bias and persistence. The second preserves the exact sign count and signed-vega mix within each walk-forward block and size level. The third preserves those quantities and each stratum's circular runs of directions by rotating its chronologically ordered signs. This retains more local dependence, though it does not preserve every dependence across different strata or the exact market calendar spacing.

For circular shifts, AMD's randomized mean net P&L is 6,025.43 versus actual 6,634.21; about 30.1% of trials reach or exceed its actual P&L. NVDA's randomized mean is 268.97 versus actual 1,328.87; about 11.1% reach or exceed actual. Pair net P&L is exceeded in about 7.2% of trials.

There are 27 fold/size strata. Only 1,730 of 3,055 trades lie in strata containing both directions; signs in one-direction strata cannot change under these mix-preserving controls. Conditioning on size and fold intentionally retains part of the policy's signal. It does not separately identify the value of entry selection, sizing, and broad regime direction.

These empirical ranges are conditional randomization diagnostics, not block-bootstrap confidence intervals for true Sharpe. Tail frequencies use a +1 finite-simulation correction in the CSV. They are not calibrated, model-selection-adjusted significance tests. The original policy and features were developed using repeatedly inspected history, and profits remain the constant-vega realized-minus-implied proxy.

## Interpretation

The simple “it is just short AMD” explanation is insufficient on matched dates and sizes. But the stronger control retains roughly 79% of the policy's net P&L on average, indicating that much of the observed outcome is associated with the retained allocation structure. This experiment does not establish that the model can reliably time direction at individual entries out of sample, nor does it validate actual option-and-hedge P&L.

## Reproduce and inspect

```sh
python scripts/test_direction_controls.py
python -m unittest discover -s tests -v
```

- [Fixed controls and leg attribution](direction_controls/fixed_controls.csv)
- [Randomized controls for both legs and pair](direction_controls/random_controls.csv)
- [Annual fixed-direction controls](direction_controls/annual_fixed_controls.csv)
- [Methodology and trial settings](direction_controls/methodology.json)

Raw randomization draws remain local in `artifacts/direction_controls/`. Twenty-six tests pass, including exact leg-to-pair P&L reconciliation, unchanged flat dates, preservation of directional mix and signed vega by stratum, and randomization reproducibility. No policy was retrained or changed for these diagnostics.
