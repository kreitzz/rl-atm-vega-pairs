# Pair trading on actual options

[Capital-exhaustion audit](drawdown_audit.md): the fixed dollar-vega budget does not shrink with NAV and there is no margin gate. Capital-exhaustion dates combine signal outcomes, execution costs and this capital-model limitation.

[Halfway-between-mid-and-side execution sensitivity](half_spread_execution.md): the 2025 RL portfolio's annualized excess Sharpe improves from 0.49 to 1.28; the long-history result remains negative.

The saved policies show a positive 2025 result concentrated in AMD/NVDA, but **do not pass the longer actual-cost test**. These are frozen signals from the existing NN/RL experiment, not policies retrained on contract P&L. The RL variant is `rl_both_sized_action`; the benchmark is `nn_both_fixed_action`. This differs from the earlier AMD/NVDA pilot's `rl_no_memory_action`.

| January 2–August 29, 2025 | Return including cash | Return above cash¹ | Annualized excess Sharpe |
| --- | ---: | ---: | ---: |
| Four-pair sized RL | 3.97% | 0.96% | 0.49 |
| Four-pair full-size/flat NN | 0.71% | -2.30% | -0.47 |
| Short higher IV, RL sizes | 1.73% | -1.28% | -0.69 |

| Sized RL pair | Return including cash | Return above cash¹ | Annualized excess Sharpe |
| --- | ---: | ---: | ---: |
| AMD/NVDA | 11.02% | 8.00% | 1.48 |
| XOM/CVX | 2.80% | -0.21% | -0.05 |
| JPM/BAC | 3.60% | 0.59% | 0.28 |
| HD/LOW | -1.52% | -4.53% | -1.42 |

¹ Difference from an unlevered cash account, divided by initial capital. Returns are period percentages, not annualized. Sharpe uses daily NAV returns less daily cash returns, annualized with sqrt(252). Cash returned 3.01% during this period. Each pair receives an independent $1 million; the portfolio sums all four accounts with no redistribution of capital. The higher-IV control uses each pair's RL signal dates and absolute sizes; differing execution eligibility can affect actual entries.

These completed numbers use the existing crossed-bid/ask envelope sensitivity. The 2025 RL run repairs four held AMD/NVDA quotes and one held JPM/BAC quote; new entries still reject crossed quotes. **Strict quotes do not produce a complete four-pair portfolio**: AMD/NVDA stops April 4 and JPM/BAC RL stops April 3. XOM/CVX and HD/LOW complete without quote repairs. [Strict results](selected_pair_backtest/strict/status.csv).

## Full-history result

Continuous account simulations start January 4, 2010, with no annual resets or replenishment. Dollar-vega budgets stay fixed relative to initial capital, so their leverage increases as capital falls.

| Sized RL pair | Through August 2025, or failure |
| --- | --- |
| AMD/NVDA | Capital exhausted January 15, 2014 |
| XOM/CVX | Capital exhausted July 3, 2019 |
| JPM/BAC | -88.29% cumulative return; excess Sharpe -1.06 |
| HD/LOW | -68.38% cumulative return; excess Sharpe -0.98 |

All four full-size/flat NN accounts exhaust their capital. We do not publish a full-history portfolio Sharpe that silently excludes failed accounts. Results from surviving accounts and the short 2025 reset are not substitutes for the failed continuous experiment.

Execution costs matter substantially. By AMD/NVDA's RL capital exhaustion, option execution costs total **$1.020 million**, including **$435,329 in commissions**, against $1 million initial capital; stock hedging costs add $34,843. Historical low-priced shares have small dollar vega per contract, requiring many contracts for a fixed dollar-vega budget. JPM/BAC accumulates $1.072 million of option execution costs and HD/LOW $752,090 across the full history. These totals include observed bid/ask crossing plus commissions; they are cost attribution, not a separately simulated cost-free result. [AMD failure diagnostics](selected_pair_backtest/amd_failure_diagnostics.json) · [Completed-history costs](selected_pair_backtest/costs.csv).

## Execution and validation

Use the frozen daily ATM/~30-day selections, initially match dollar vega within 10%, and delta hedge daily using fitted Black–Scholes Greeks. Keep the original option IDs through exit; include prior-year selections when loading January marks. A selected contract with a failed fit is not replaced with another strike. Positions exit within 30 calendar days, before expiry or a known ex-dividend date, or when usable Greeks disappear. The quote dataset contains calls only.

The ladder is 0/25/50/75/100%, signed by direction, with a $450 target vega per leg at full size and a $20,000 gross-vega entry gate per account. Trade at observed ask/bid plus $0.65/contract each transaction; stock hedges cost 1 bp, short stock 0.3% annually, and negative cash the cash rate plus 1%. Vega neutrality applies at entry. Margin, quote depth, market impact, and arbitrary early assignment are not modeled; the model allows fractional hedge shares. Financing uses the existing pilot conventions. The European Black–Scholes approximation affects Greeks, not observed trade prices.

New signals stop 35 calendar days before the end of each requested window to allow runoff. Missing historical saved predictions mean no new trade, while every market session remains in the NAV ledger (21 missing prediction sessions per pair over the full history, zero in 2025). No pair is removed based on its outcome. Full history and 2025 share inspected data and are not fresh holdouts.

All completed runs reconcile daily option/stock cash flows to NAV, and all their contract positions close. Raw quote extraction uses 5,000-row batches; peak memory in the full experiment was about 516 MiB. Next research step: train and validate using actual net contract outcomes and execution cost per dollar vega, with an explicit capital/participation rule, before treating the earlier proxy signal as tradable.

[All outcomes and failures](selected_pair_backtest/status.csv) · [Year-by-year completed runs](selected_pair_backtest/annual.csv) · [Portfolio results](selected_pair_backtest/portfolio.csv).

```sh
python -m scripts.run_selected_pair_backtest --no-risk-controls --repair-crossed-held-quotes
python -m scripts.run_selected_pair_backtest --no-risk-controls --years 2025 --reuse-cache --cache-root artifacts/contracts/four_pairs/cache --output artifacts/contracts/four_pairs_strict --report reports/selected_pair_backtest/strict
python -m unittest discover -s tests -q
```

## Observed spread widths

The engine buys at ask and sells at bid at entry and exit. Each fill pays half the full quote width relative to midpoint, plus commissions. A round trip pays approximately one full spread per leg if width and vega stay unchanged.

Median **full** bid/ask IV widths on the actual 2025 RL entries, calculated by separately inverting bid and ask:

| Pair | First leg (vol points) | Second leg (vol points) |
| --- | ---: | ---: |
| AMD/NVDA | 0.86 | 0.72 |
| XOM/CVX | 1.40 | 1.37 |
| JPM/BAC | 0.78 | 0.78 |
| HD/LOW | 1.03 | 1.17 |

For AMD/NVDA's historical RL trades through capital exhaustion in January 2014, median entry widths were 2.47 and 1.06 vol points. At entry, the $0.65 commission alone was equivalent to median 0.93 AMD vol points per fill (0.41 NVDA), versus 0.055 and 0.049 respectively in 2025. These are per-option-leg measures; both legs incur costs.

Exit quotes can have much wider IV differences as held options become deep ITM/OTM or approach expiry. The local width/vega approximation becomes unstable as vega approaches zero, and some bid/ask prices cannot be inverted under the European model. Exit statistics retain fit-coverage counts; entry widths should not be mistaken for the whole lifetime cost. [Executed-quote spread statistics](selected_pair_backtest/execution_spreads.csv), generated with `python -m scripts.report_contract_spreads`.
