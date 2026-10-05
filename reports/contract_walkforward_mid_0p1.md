# Actual-contract learning with memory and monthly beta selection

Four independent pairs; 2013-01-01–2025-08-29. Each starts with $1 million; rejected pair allocations remain in cash.

| Model | Total return | CAGR | Annualized excess Sharpe | Max drawdown |
|---|---:|---:|---:|---:|
| Annual RL, base states | 22.79% | 1.63% | -0.083 | -3.22% |
| Monthly RL, base states | 19.16% | 1.39% | -0.246 | -4.89% |
| Monthly RL + memory | 13.00% | 0.97% | -0.702 | -6.58% |
| Monthly RL + beta screen | 23.50% | 1.68% | -0.113 | -0.92% |
| Monthly RL + memory + beta | 20.15% | 1.46% | -0.525 | -0.85% |
| Monthly fixed-size NN + memory + beta | 14.77% | 1.09% | -1.011 | -2.19% |
| Cash benchmark | 24.60% | 1.75% | — | 0.00% |

Sharpe uses daily portfolio returns less the same daily cash return, annualized by √252. Total return and CAGR include cash interest. The final year is partial. No failed pair is dropped from portfolio results.

[Annual returns and Sharpe](contract_walkforward_mid_0p1/annual.csv) · [Pair results](contract_walkforward_mid_0p1/status.csv) · [Portfolio results](contract_walkforward_mid_0p1/portfolio.csv) · [Selection summary](contract_walkforward_mid_0p1/selection_summary.csv) · [Risk audit](contract_walkforward_mid_0p1/audit.csv)

## Implementation

- **Memory:** trailing 91/365-calendar-day mean net P&L per actual initial mean leg vega, separately for long-A/short-B and short-A/long-B, plus observation counts. These are daily-entry shadow/reference trades, so inactive pairs retain a track record. Only trades closed before the signal date contribute. Overlapping observations are not independent samples.
- **Selection:** monthly, using the preceding 126 trading sessions of adjusted stock returns. A-on-B beta must be 0.75–1.25 and correlation at least 0.5. The screen freezes for the month and gates new entries; existing trades follow their normal exits. This filters the four established pairs, rather than discovering pairs across the full downloaded universe.
- **Retraining:** monthly, trailing five years, at least 504 fully observed training rows. Outcomes must close strictly before the fit date. The annual ablation refits in January. Hyperparameters are fixed; reported test years do not choose settings.
- **Inputs:** selected-contract BS IV spread, both earnings countdowns, both trailing 30-day RVs, RV-adjusted relative value, and trailing spread reversion; memory adds eight states. The legacy standardized-IV/forward-RV reward is not used.
- **RL:** eight-leaf state tree, minimum 63 training rows per leaf, direct cross-entropy policy search over the signed 0/25/50/75/100% ladder. Search maximizes daily excess-P&L Sharpe of overlapping standalone shadow cohorts. It is not actor-critic or Q-learning. Final evaluation applies the full NAV-scaled portfolio and risk constraints, which are not inside the training objective.
- **NN:** one 16-unit tanh hidden layer, training-only feature/target scaling, predicts the two directional net contract outcomes. Trade the higher positive prediction at full size; otherwise stay flat. Same monthly schedule, memory, beta filter and execution assumptions.
- **Execution:** next-session entry in the prior-close selected ATM/~30D call; daily BS delta hedging; integer contracts; initial vega matching; mid fills plus a separate 0.1-vol-point cash penalty per leg per fill (absolute dollar vega × 0.1); no quoted-spread charge; $0.65 option commission per contract, 1 bp stock execution, financing and borrow. Keep the existing NAV-scaled vega, hedge, reserve and stress limits.

## Data and checks

Ordinary integer stock splits require corroborating stock adjustment factors, option adjustment factors, unchanged expiry/100-share multiplier, and the divided strike within half-cent rounding. Contract and stock quantities adjust without cash execution. This matches the [OCC NVIDIA adjustment notice](https://infomemo.theocc.com/infomemos?number=48828). Unsupported actions still fail explicitly.

Shadow-label disposition: {'complete': 13378, 'no_entry': 868}. Unresolved labels never become zero-profit trades. Known entry rejections are zero; no future label availability determines live predictions.

All completed account ledgers and end-of-day risk limits reconcile. Training cutoff, action ladder and beta gating audits pass. 5/601 NN fits reported nonconvergence; those deterministic fits are retained and flagged in [fit logs](contract_walkforward_mid_0p1/fits.csv). Peak process RSS: 0.79 GiB; one pair is processed at a time.

Unit tests cover timing leakage, overlapping daily Sharpe, cash reconciliation, flat policies, split inventory/rounding, missing marks and portfolio risk limits. The accelerated labels were additionally checked against the original cash-account engine on real quotes, including NVIDIA split dates.

Limitations: fixed retrospectively selected four-pair universe; historical earnings dates assumed known; EOD mid-based BS inversion for American calls; explicit crossed-quote envelope repair on held positions; no intraday execution, assignment or broker margin model. These are research results, not a point-in-time deployable strategy.

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.run_contract_walkforward --option-penalty-vol-points 0.1
python -m scripts.report_contract_walkforward --root artifacts/contracts/walkforward_mid_0p1 --report reports/contract_walkforward_mid_0p1
```

Requires local licensed market-data caches. Raw quotes, labels, states, actions and account paths remain gitignored; source/settings hashes are recorded. `--reuse-labels --reuse-actions` requires matching input and code fingerprints. Prior frozen-policy results are preserved.
