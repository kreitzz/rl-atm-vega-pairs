# Q learning on actual option contracts

The first contract-based implementation is complete and has been exercised on actual WRDS data. **It has not established a profitable trading policy.** With training through 2022, validation in 2023, and a frozen 2024 test, both fitted-Q variants stayed flat. Cash also won the validation selector. That result must not be described as successful option trading or as the earlier IV-proxy Sharpe carrying over.

The [design plan](contract_q_plan.md) explains the architecture and research references. The earlier IV-based Q scripts remain separate.

## What is actually traded

- One standard **call** on AMD and one on NVDA, with the **same expiration**, selected near ATM and approximately 30 days to maturity (eligible range 21–45 calendar days). These are delta-hedged call pairs, not straddles.
- Select actual option IDs from the signal session's chain, using lagged open interest. Execute at the next session's observed bid/ask; recheck the exact IDs and reject both legs if the entry is unavailable. No silent substitution.
- Match initial dollar vega using integer quantities, with a $450 target per leg, at most 10% entry imbalance, and one pair open at a time. Reject entry if gross stock-hedge notional exceeds the $1 million research capital budget. The gross vega cap is $20,000.
- Keep the IDs and quantities until exit. Rebalance stock delta hedges daily. Q decides whether to wait, enter in either direction, hold, or close the pair. It does not yet choose a strike, expiration, or a continuous size.
- Buy at ask and sell at bid, charge $0.65 per option contract per transaction, and charge 1 bp of stock-hedge turnover. Mark held options at midpoint.
- Exit at the holding limit, before expiration or a known ex-dividend date, on missing usable Greeks, at a vega-limit breach, or at the evaluation boundary. Unsupported held adjustments or missing/invalid held prices cause an explicit incomplete run.

## Accounting and learning

Each pair has a segregated financing account, including the option premiums and stock hedge. Negative ticket cash pays the observed cash rate plus 1%; positive ticket cash earns the modeled cash rate. Short stock costs 0.3% annually, and stock dividends enter the ledger. Reserve capital earns the cash benchmark. This is a specified financing convention, not a broker margin/rebate model.

The unshaped training reward is daily ticket P&L **above cash**, expressed in basis points of fixed initial capital. Account NAV evolves as prior NAV times the cash growth factor plus that dollar excess P&L. Daily option marks, stock-hedge price changes, interest, borrow, dividend cash, option execution costs, and hedge costs reconcile to ticket value changes.

The downside ablation uses:

`utility = reward_bps - lambda * min(reward_bps, 0)^2 / training_reward_sd`.

The scale is fitted on training only, with a 0.01 bp floor. The two tested penalties are 0 and 0.1. Evaluation always reports the unshaped actual cash P&L, never the utility score. This squared-downside utility is not equivalent to directly maximizing Sharpe and is not the previously proposed forecast-variance penalty.

The 31 predictive inputs describe observable contract IVs/Greeks, trailing RV, relative variance carry, trailing stock-return correlation, moneyness, maturity, quote widths, lagged liquidity, and actual inventory/financing state. Contract IDs are execution keys, not predictive features. The current model uses trailing RV proxies; it does not yet contain a separately fitted joint future-RV forecasting model.

Historical replay constructs both possible entry directions and each feasible hold/exit transition. This provides coverage of actions without pretending repeated counterfactuals are independent days. Models use inverse per-date observation weights within each action.

Fitted Q iteration uses separate ExtraTrees regressors for the four action codes (wait/hold shares one code), 16 trees, depth at most 6, minimum leaf size 128, discount 0.97, and 12 synchronous fitted sweeps. These are preset engineering settings, not a claim of Bellman convergence. The maximum changes at the final sweep remained about 13.1 and 22.8 bps for the two utility variants; value-function stability is an important follow-up.

Training outcomes cannot cross the training boundary. Every evaluation starts flat and charges final liquidation. Cash is an explicit validation candidate, and ties keep cash. The chosen policy is recorded before test evaluation. The pair, history, and 2024 period have previously been researched, so this is an exploratory walk-forward engineering run, not a fresh untouched holdout.

## WRDS data obtained

Authentication succeeded using the existing local credentials and the `mfe_wrds` environment; no password was requested or written by these scripts.

- **2,556,508 actual call quote rows**, plus underlying histories, zero curves, and distributions.
- Requested January 2010–September 2025; available underlying coverage ends **August 29, 2025**.
- 34 downloaded partitions, about 112 MiB, with request and checksum manifests.
- Two-pass preparation retains all observed quote histories for IDs selected from contemporaneous chains: **121,894 quote rows, 3,430 unique IDs, and 3,939 sessions**.
- 3,683 dates have an eligible contract pair before strategy, sizing, and fill checks.
- The 2010–2022 training graph has **90,774 states and 184,203 action transitions**. All those action outcomes were resolved; no training prices were repaired or censored in this run.

The preparation pass does not filter contracts by future profitability or future quote survival. Raw crossed quotes and missing Greeks remain in retained histories. Future failed entries become observed no-fill outcomes, while a missing held quote is never converted into a zero return.

## First test: calendar 2024

All methods use the same actual-contract engine, budget, costs, and accounting.

| Frozen method | Entries | Return above cash, initial-capital basis | Annualized daily excess Sharpe |
| --- | ---: | ---: | ---: |
| Validation-selected policy: cash | 0 | 0.00% | 0.00* |
| Q, unshaped net P&L | 0 | 0.00% | 0.00* |
| Q, downside penalty 0.1 | 0 | 0.00% | 0.00* |
| Fixed-hold tree policy search | 0 | 0.00% | 0.00* |
| Short higher IV / long lower IV | 14 | -0.90% | -0.37 |
| Relative trailing-carry direction | 14 | -3.82% | -1.62 |

*Zero is the reporting convention for zero excess volatility. Cash has no demonstrated trading alpha. Its modeled 2024 total return was 5.33%, entirely the cash benchmark.*

The fixed-hold learner fits a small tree and searches at most 81 flat/long/short policies using daily contract P&L on the training transition graph. This is a tractable policy-search control in the spirit of the existing CEM approach, **not a rerun of the original cross-entropy model**. There is no voluntary early exit in that control; the shared safety exits still apply.

For each mechanical rule, option execution costs totaled $12,837.90 and hedge costs $2,287.00. The higher-IV rule's maximum NAV drawdown was 1.31%; the relative-carry rule's was 2.04%. All test runs completed with closed positions. A separate replay of the higher-IV rule reproduced every daily NAV and supplied 28 pair fills (14 entries and 14 exits) with actual IDs, quantities, prices, and hedge transactions.

The no-trade result could reflect costs, weak signals, function approximation, or training instability. It does not prove no profitable contract strategy exists. No parameters were retuned using the 2024 results.

## Files and reproduction

- [Downloader](../scripts/download_q_contract_data.py): bounded, resumable, pair-specific WRDS requests.
- [Preparation](../scripts/prepare_q_contract_data.py): contemporaneous contract selection and portable compact data.
- [Execution engine](../scripts/contract_q_engine.py): contract/inventory state, next-session transitions, cash ledger.
- [Trainer and evaluator](../scripts/run_contract_q.py): fitted Q, cash/mechanical/search controls, validation selection, model replay.
- [Tests](../tests/test_contract_q.py): execution timing, accounting, data leakage, safeguards, and incomplete outcomes.
- [Aggregate test results](contract_q/summary.json) and [validation selection](contract_q/validation_selection.json).

```powershell
conda activate mfe_wrds
python -m scripts.download_q_contract_data --username YOUR_WRDS_USERNAME
python -m scripts.prepare_q_contract_data

conda activate xfund
python -m scripts.run_contract_q --test-years 2024 --iterations 12
python -m scripts.run_contract_q --test-years 2024 --replay-only
python -m unittest tests.test_contract_q tests.test_contract_engine tests.test_q_batch tests.test_first_pass tests.test_walk_forward
```

To request another pair, pass its two SECIDs and a separate output folder to the downloader and preparation script, then pass the resulting local bundle through `--data` and choose a separate model `--out`. The runner also accepts multiple test years, each with expanding training and the preceding year for validation.

Licensed raw data stay in ignored `data/q_contracts/AMD_NVDA/`; local models, partial-failure logs, validation decisions, and daily ledgers stay in ignored `artifacts/contract_q/AMD_NVDA/`. Compact bundles and model snapshots are trusted local pickle files, not an interchange format for untrusted inputs. No packages were installed into `xfund`; Parquet preparation runs in `mfe_wrds`.

The focused suite passes **36 tests**, including ten new contract-Q tests. Neither those tests nor successful EOD replay establishes executable live returns. Quote/stock synchronization, broker margin, participation/depth, borrow availability, and arbitrary early assignment remain outside this implementation.

## Next research step

Diagnose the flat-policy result on training and validation: Q-value stability, entry-cost hurdle, and shared-state approximation. Then compare a joint per-leg variance forecast and an entry-only Q model against the fixed-hold policy search under identical contract execution. Add learned sizing or contract choice only after those simpler controls demonstrate useful validation behavior. Freeze the next experiment before evaluating additional test periods.
