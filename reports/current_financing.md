# Current best: financing audit

Frozen 0.368 employee-leadership RL checkpoint. Saved contracts, quantities, hedge inventories and dates are unchanged. No policy fits, resizing or renewed risk intervention. These are funding sensitivities, not retrained strategy results.

The saved model earns the closest-to-30-day OptionMetrics zero rate on net positive cash, pays that rate plus 1% on negative cash and charges 0.30% annual stock borrow on each short hedge. Both strategy and cash benchmark use the same lagged rate with actual calendar days / 365. Cash is not separated into short collateral and free cash in the original engine.

| Assumption | Full annualized excess Sharpe | Since 2022 |
|---|---:|---:|
| saved | 0.368 | 0.428 |
| segregated_collateral | 0.368 | 0.428 |
| borrow_1pct | 0.323 | 0.372 |
| borrow_3pct | 0.195 | 0.212 |
| rebate_minus_50bp | 0.336 | 0.388 |
| borrow_1pct_rebate_minus_50bp | 0.291 | 0.332 |
| borrow_3pct_rebate_minus_50bp | 0.163 | 0.172 |
| loan_plus_2pct | 0.368 | 0.428 |

All scenarios retain the original idle-cash benchmark and unencumbered-cash rate. Revised scenarios segregate 100% of preceding short-stock notional and accrue any loan spread on negative free cash. The 50 bp rebate discount applies only to short collateral. Actual collateral rounding/haircuts, settlement timing, broker credit/debit schedules and historically available stock-specific loan fees are not modeled. Each pair retains its separate cash sleeve; cross-pair hedge/stock-loan netting is not introduced.

Base replay reproduced the saved Sharpe and NAV paths. The independent interest identity reconciled position funding, interest on earlier excess P&L and any negative-cash surcharge. Sources were checked unchanged. Margin interest and short-sale borrow/rebate terms are distinct in [broker documentation](https://www.interactivebrokers.com/en/pricing/short-sale-cost.php). The cited current broker schedule is illustrative, not assumed historical institutional pricing.

The flat 0.30% fee and full benchmark-rate short collateral rebate are simplifying assumptions, not validated prime-broker terms. The most useful next refinement is historical, stock-specific stock-loan pricing and an explicit collateral rebate. A 30-day zero rate is also a proxy for actual overnight cash/debit funding.

[Sensitivity table](current_financing/comparison.csv) · [Pair funding and exposures](current_financing/pairs.csv) · [Annual Sharpe](current_financing/annual.csv) · [Source manifest](current_financing/manifest.json)
