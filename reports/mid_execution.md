# Mid-theoretical execution with explicit penalties

2013–August 2025, the same four pairs. Mid is the valuation and execution reference. A separate cash penalty replaces the quoted-spread charge:

`execution penalty = abs(contracts) × 100 × vega per share per vol point × penalty in vol points`

The central assumption is **0.10 vol points per leg per fill**. Two $450-vega legs cost $90 to open, before commissions; closing uses the remaining vega. This is a research fill assumption, not a calibrated estimate of achievable execution. The 0/0.05/0.10/0.25 grid measures sensitivity.

Commissions remain $0.65 per contract per execution; stock hedging costs 1 bp, with the existing financing and borrow assumptions. “Mid” below still includes those costs. Theoretical option values and Greeks continue to use the selected contracts and the existing BS fits. A forced exit without a current valid vega uses the last observed, split-adjusted vega; there is no future Greek fill.

| Model | Old quoted cost | Fixed trades at mid | Fixed trades, mid + 0.10 | Retrained, mid + 0.10 |
|---|---:|---:|---:|---:|
| Annual RL, base states | -0.799 | 0.252 | 0.093 | -0.083 |
| Monthly RL, base states | -0.852 | 0.126 | -0.019 | -0.246 |
| Monthly RL + memory | -2.026 | -1.105 | -1.268 | -0.702 |
| Monthly RL + beta | -0.654 | -0.014 | -0.132 | -0.113 |
| Monthly RL + memory + beta | -1.133 | -0.593 | -0.693 | -0.525 |
| Fixed-size NN + memory + beta | -1.957 | -0.404 | -0.716 | -1.011 |

All entries are annualized **daily excess-return Sharpe**. Old execution paid 25% of the full quoted spread per fill. Fixed-trade repricing preserves every contract, quantity, hedge and exit, and recomputes cash financing. It isolates the effect of costs; it does not choose new trades or resize.

The retrained run rebuilds net actual-contract labels and profitability memory under the new penalty, retrains the policies, and reruns the original NAV sizing and portfolio risk limits. Historical quote-quality filters and contract selection are unchanged. No penalty was chosen by optimizing the test-period Sharpe.

[Retrained returns, Sharpe and annual tables](contract_walkforward_mid_0p1.md) · [Comparison CSV](mid_execution_sensitivity/comparison.csv) · [All fixed-trade sensitivities](mid_execution_sensitivity/portfolio.csv) · [Cost breakdown](mid_execution_sensitivity/pairs.csv)

## Fixed-trade cost curve

| Penalty per leg/fill | Annual RL baseline Sharpe | Monthly RL + both Sharpe | NN + both Sharpe |
|---:|---:|---:|---:|
| 0.00 vol points | 0.252 | -0.593 | -0.404 |
| 0.05 vol points | 0.173 | -0.643 | -0.560 |
| 0.10 vol points | 0.093 | -0.693 | -0.716 |
| 0.25 vol points | -0.145 | -0.841 | -1.181 |

For the annual RL baseline's original trades across all four accounts, quoted-spread slippage totaled $499,493. The 0.10-vol-point penalty totals $76,506, with commissions of $146,513 retained separately.

The fixed-inventory 0.25-vol-point sensitivity has 7 pair/model/session risk-limit breaches in total; that sensitivity is not a fresh risk-managed simulation. The 0/0.05/0.10 cases have 0. Retrained completed simulations independently enforce every limit.

Peak memory for retraining/evaluation: 0.79 GiB. Accounting, penalty units, zero-penalty commissions, missing-Greek fallback, borrowing and pre-trade risk-cost checks are tested. Sixteen real-quote shadow-label comparisons, including split dates, matched the cash-account simulator.

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.reprice_contract_execution
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.run_contract_walkforward --option-penalty-vol-points 0.1
python -m scripts.report_contract_walkforward --root artifacts/contracts/walkforward_mid_0p1 --report reports/contract_walkforward_mid_0p1
python -m scripts.report_mid_execution
```
