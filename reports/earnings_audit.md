# Earnings-state audit

The earlier result does not establish that earnings information reduces the achievable Sharpe. It shows that the fitted, more flexible policy generalized poorly in this sample.

The audit reproduced identical dates, rewards, and costs for the two models. It explicitly embedded the spread-only policy in the earnings state space and confirmed identical actions on every split. The earnings model can ignore earnings; it chose a higher-training-Sharpe policy instead.

| Representation | Spread buckets | Total states | Minimum training observations per state | Training Sharpe | Validation Sharpe | Test Sharpe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Spread only | 5 | 5 | 751 | 0.415 | 0.532 | 0.672 |
| Spread + earnings | 5 | 20 | 49 | 0.938 | 0.501 | 0.088 |
| Spread only | 10 | 10 | 375 | 0.442 | 0.410 | 0.620 |
| Spread + earnings | 10 | 40 | 13 | 1.020 | 0.562 | 0.392 |
| Spread only | 20 | 20 | 187 | 0.581 | 0.405 | 0.672 |
| Spread + earnings | 20 | 80 | 6 | 1.168 | 0.375 | 0.153 |

Sharpes are annualized 30-day payoff ratios after costs. Daily rows have overlapping outcomes; 13 rows are not 13 independent earnings events.

Validation selected 5 spread-only buckets and 10 earnings-aware spread buckets. Their validation advantage was only 0.031 Sharpe in favor of the earnings model. The selected policies disagree on 591 of 1,170 test dates. Earnings-aware performance improves when neither stock has earnings inside the horizon, but loses relative to the baseline when AMD has earnings, especially in 2021.

The paired 63-entry block-bootstrap 95% interval for the test Sharpe difference (earnings minus spread-only) is **[-0.945, 0.433]**, so this sample does not establish a reliably negative effect. Excluding the 38 same-day earnings entries leaves the conclusion essentially unchanged: 0.390 versus 0.664.

For the subsequent RV/VIX experiment, use shared tree partitions with at least 126 training rows per leaf and rerun matched feature ablations. This addresses the sparse independent-state-table issue; it does not guarantee better test performance. The test period has already been inspected, so subsequent comparisons remain exploratory.

Reproduce with `python scripts/audit_earnings.py` after the original spread-only and actual-earnings runs. Detailed local results are in `artifacts/earnings_audit/`.
