# Wider execution-cost sensitivity

The 1–2 vol-point assumptions are stress cases, not calibrated estimates of AMD/NVDA execution. Costs are **all-in per full-size pair round trip**, not per leg or per fill. Retain 0.5 as the experiment's reference assumption until actual quotes and hedge costs are measured.

## Results

Same current best feature set (IV spread, earnings, 30-calendar-day RV), sizing ladder, dates, and walk-forward selection procedure. Each cost level retrains the policy and selects configurations on preceding validation data. The frozen-policy column instead keeps every 0.5-cost policy action and size fixed and only changes the execution charge.

| Total pair round-trip cost (vol points) | Frozen-policy annualized Sharpe | Retrained annualized Sharpe | Retrained dates traded | Average size / maximum |
| --- | ---: | ---: | ---: | ---: |
| 0.5 | 0.735 | 0.735 | 77.9% | 45.5% |
| 1.0 | 0.663 | 0.667 | 77.9% | 43.8% |
| 1.5 | 0.590 | 0.596 | 77.0% | 45.3% |
| 2.0 | 0.517 | 0.440 | 67.2% | 39.3% |

Retraining is not guaranteed to improve later performance at the new cost. At 2 points, the frozen original policy outperforms the newly fitted policy; the paired interval for retrained-minus-frozen Sharpe is [-0.238, 0.072]. Costs therefore should not be selected based on which retrained test score looks best.

These remain annualized 30-day payoff Sharpes, using `sqrt(365/30)`, after proportional costs. They are not daily marked portfolio-return Sharpes. Historical outcomes have already been inspected during multiple experiments.

## Is 2 vol points overly punitive?

The model alone cannot establish that. Under the constant-vega proxy, four equal-vega transactions occur at pair-leg level: enter AMD, enter NVDA, exit AMD, exit NVDA. An equal cost allocation gives:

| Pair round-trip cost | Equivalent concession per leg per entry/exit |
| --- | ---: |
| 0.5 | 0.125 vol points |
| 1.0 | 0.250 vol points |
| 1.5 | 0.375 vol points |
| 2.0 | 0.500 vol points |

This breakdown is illustrative and allocates the entire budget to options execution. The experiment's total cost also stands in for daily hedging and fees; those would consume part of the budget. The individual call/put fills of a straddle would be weighted by their vegas rather than each being charged the full leg amount.

At $1,000 of vega per leg, a 2-point total charge is $2,000 for the entire full-size pair round trip. A quarter-size entry pays one quarter of that. It is not $2,000 for each leg or each transaction.

To calibrate instead of assume: estimate option bid/ask concessions, commissions, and daily hedge costs in dollars for the actual quantities, then divide their total by the initial dollar vega of one pair leg. Our ATM implied-volatility surface data does not provide this execution-cost estimate. No claim is made that 0.5, 1, or 2 points is the correct market cost.

## Reproduce

```sh
python scripts/run_execution_costs.py
python -m unittest discover -s tests -v
```

Build the existing short-RV/momentum panel first. The runner uses only the current best feature set and retrains separately at each cost; momentum and VIX remain excluded. Dollar-vega scale is the existing 0.001 per leg per reference-capital dollar per new trade. No portfolio margin or aggregate capital cap is modeled.

- [Complete cost comparison and uncertainty](execution_costs/comparison.csv)
- [Yearly P&L and Sharpe at each cost](execution_costs/annual.csv)
- [Size allocations by cost](execution_costs/allocations.csv)

Local training logs, predictions, candidates, and settings are in `artifacts/execution_costs/`. Runtime checks verify identical dates and gross rewards across runs, exact reproduction of the original 0.5-cost actions, the cost formula, and monotonically decreasing frozen-policy P&L as costs widen. Twenty-three unit tests pass.
