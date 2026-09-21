# What distinguishes the winners and losers?

The strongest next state candidates are **IV spread relative to trailing RV spread** and **IV spread relative to its own historical distribution**. They help distinguish an unusually rich/cheap leg from a name that normally has higher volatility. Correlation alone has not made that distinction reliably.

This review uses the saved baseline policies for all 12 stock pairs and 13 ETF comparisons, without choosing each pair's best model variant. Stock results come from the twelve-pair common-calendar experiment, so some differ from the earlier four-pair run. ETF results use each pair's eligible history. Comparisons within each diagnostic use matched dates; cross-pair full-history scores can span different years. The two 2024-start comparisons are excluded from the 23-pair feature-improvement counts and portfolio experiments.

## Different explanations for different pairs

| Pattern | Evidence | Implication for state design |
| --- | --- | --- |
| Learned timing matters in several winners | AMD/NVDA Sharpe 0.635 versus -0.003 for always short AMD and 0.306 for shorting higher IV; MSFT/ORCL 0.192 versus negative constant-direction controls | Relative value and regime matter beyond choosing a permanent short leg |
| Some losers have gross edge, but too little to pay costs | XOM/CVX gross payoff per unit absolute size 0.446 vs cost 0.641; JPM/BAC 0.736 vs 0.787; XOP/DVN 0.197 vs 0.942 | Include an expected-edge/cost measure so flat can depend on economics, not just direction |
| Other losers have negative gross performance | C/WFC gross payoff per size -0.706, UPS/FDX -0.073 | Reducing costs cannot repair the existing policy's wrong-side selections |
| Some positive aggregate results lack consistency | OIH/SLB wins in only 2 of 8 calendar years despite 0.181 Sharpe; SIL/AG wins in 8 of 8; AMD/NVDA in 11 of 16 | Model regime dependence; do not identify robustness from full-period Sharpe alone |
| High correlation is insufficient | XOP/DVN median return correlation 0.873, IV-change correlation 0.740, but Sharpe -0.241 | A good hedge relationship does not establish a profitable volatility-premium difference |
| Some winners reflect persistent direction or extreme marks | SIL/HL constant long ETF / short stock beats its policy; 96% of baseline profit arises when HL IV exceeds 100 vol | Separate learned timing from premium harvesting, and inspect mark quality |
| Some apparent edges may be difficult to execute | GSLC has zero call volume on 71% of the short evaluation window | Liquidity/mark quality is an execution requirement, not evidence of alpha |

These are payoff attributions and associations, not identified economic causes. ETF diversification, earnings, changing businesses, and option supply/demand may contribute, but this dataset does not distinguish those mechanisms.

## Entry-observable feature diagnostics

We kept each saved baseline action and size, then applied fixed diagnostic gates. A rejected trade becomes zero payoff, including zero entry cost; remaining allocations are not increased. These are **post-hoc filters, not retrained RL policies or new independent holdouts**. No threshold search was performed. All performance remains annualized 30-day IV-versus-RV proxy Sharpe.

1. **RV-adjusted relative value**: `value = (trailing_RV_A - IV_A) - (trailing_RV_B - IV_B)`. Retain a trade when its direction agrees with this value. This is a transformation of the existing IV spread and two RV inputs, not new market information. An axis-aligned tree can benefit from directly receiving their difference.
2. **Historical spread deviation**: `reversion = (past_252_mean_spread - current_spread) / past_252_std_spread`. Retain a trade when its direction agrees with reversion toward the past mean. The history excludes today's observation and requires at least 126 earlier panel observations. Missing market dates are not filled; these are observation windows, not precisely 252 exchange sessions.
3. **Matured relative premium**: rolling mean of the last 252 pair outcomes, requiring 126, whose 30-day end dates are strictly before entry. Retain a trade if its direction agrees with that history, optionally requiring the mean directional edge to exceed current modeled cost. This adds realized feedback unavailable from current IV/RV levels alone. Overlapping outcomes are not independent samples.
4. **Earnings alignment**: retain trades only if both or neither names have earnings within the contract horizon. This tests a blunt alignment rule; it does not replace the existing individual countdowns. ETF earnings countdowns remain constant, without constituent-weighted earnings exposure.

| Gate | Improved / 23 longer-history pairs | Median pair Sharpe change | Improved in 2018–21 | Improved in 2022–25 |
| --- | ---: | ---: | ---: | ---: |
| RV-adjusted value agrees | 19 | +0.137 | 17 | 13 |
| Spread reversion agrees | 19 | +0.079 | 18 | 15 |
| Matured edge agrees | 12 | +0.000 | 13 | 8 |
| Matured edge covers cost | 11 | -0.001 | 13 | 8 |
| Both/neither earnings | 6 | -0.045 | 12 | 9 |

The two leading gates need not improve the same pairs. Pair counts are descriptive and correlated, not independent statistical trials. The weakening of the matured-edge gates after 2021 argues against assuming a year's realized premium stays constant.

## Portfolio sensitivity and examples

Equal fixed budgets, unallocated cash retained. Each row uses its own common calendar and the same dates for all gates. The combined portfolio has 22 pairs; ETF/ETF and short-history comparisons are excluded.

| Basket | Test period | Baseline | RV-value gate | Spread-reversion gate | Matured-edge/cost gate |
| --- | --- | ---: | ---: | ---: | ---: |
| 12 stock pairs | 2010–August 2025 | 0.317 | 0.574 | 0.394 | 0.446 |
| 10 ETF/stock pairs | 2018–August 2025 | 1.036 | 1.408 | 1.346 | 1.417 |
| Combined 22 pairs | 2018–August 2025 | 0.895 | 1.305 | 1.248 | 1.340 |
| Combined excluding both silver pairs | 2018–August 2025 | -0.054 | 0.279 | 0.279 | 0.285 |

The combined calendar has 1,905 entries; the stock calendar 3,918 and ETF/stock calendar 1,906. Removing silver is a concentration sensitivity, not a selected replacement portfolio. The improvements are not solely driven by silver, but none of these filtered results establishes a production strategy.

Examples: the RV-value gate changes AMD/NVDA 0.635 → 0.700, UNP/CSX 0.260 → 0.446, PPH/MRK 0.279 → 0.711, and PPH/AMGN 0.331 → 0.565. Historical spread reversion changes AMAT/LRCX 0.284 → 0.565. Neither gate rescues C/WFC: its RV-value score remains -0.355.

Regime check: stock-basket RV gating improves 2018–21 from -0.343 to 0.177, but worsens 2022–25 from 0.229 to 0.107. Combined excluding silver, it improves both periods (-0.222 → 0.285 and 0.267 → 0.318). This favors a flexible state feature over a universal hard exclusion rule.

Paired circular block bootstrap intervals use 63-entry blocks and 1,000 resamples. Combined RV-gate Sharpe improvement is +0.410 with interval [0.127, 0.830]; excluding silver it is +0.333 with [-0.008, 0.802]. Historical-spread improvement excluding silver is +0.333 with [0.079, 0.591]. These intervals do not correct for feature screening, reused history, or retraining uncertainty and are not confirmatory significance claims.

## Next state ablation

First add **RV-adjusted relative value** alone, then **historical spread z-score** alone, then both. Keep independent policies, budgets, ladder, proportional costs, and the same purged walk-forward setup. Assess broad pair improvement, period stability, and the basket excluding silver alongside aggregate Sharpe. Do not choose the best variant separately for each test pair.

Second priority is **matured edge divided by current cost**, with **matured outcome dispersion** as a sizing/risk input. The diagnostic evidence is mixed; this should not displace the two stronger candidates. This review computes edge/risk and cost/risk ratios but does not establish their incremental predictive value.

For mark quality, retain **each leg's IV level and IV/trailing-RV ratio** as candidate descriptors. Extremely high IV may be real event risk or a problematic mark; it must not be treated as automatic mispricing. Actual-contract spreads and quote freshness are preferable execution inputs when available. The existing aggregate activity audit cannot validate ATM 30-day execution.

No default policy or shared market-data CSV was changed. Only diagnostic code and aggregate outputs were added.

[Pair attribution](winner_loser/summary.csv) · [Annual results](winner_loser/annual.csv) · [Feature correlations and sign controls](winner_loser/signals.csv) · [Matched per-pair gates](winner_loser/filters.csv) · [Portfolio gates and block intervals](winner_loser/portfolios.csv)

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.diagnose_winners
python -m unittest discover -s tests -q
```

Requires the existing ignored panels and predictions under `artifacts/universe_gap/` and `artifacts/etf_exploration/run/`. The tests verify that unmatured future outcomes cannot enter the historical-premium state; payoff accounting is also reconciled against saved predictions during the diagnostic run.
