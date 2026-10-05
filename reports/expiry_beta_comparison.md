# Tighter pairwise beta: retraining and backtest

Both ranges use the same 347 complete broad pairs, including cash-only sleeves. Each pair starts with $1m. Evaluation: January 2018–August 2025. Both models were retrained monthly under their respective eligibility screens. Costs, expiry settlement, risk sizing, profitability gates and feature set are unchanged.

| Model | Beta range | Annualized excess Sharpe | Net excess P&L | Entries | Active pairs |
|---|---|---:|---:|---:|---:|
| rl_sized | 0.75–1.25 | -0.207 | $-129,486 | 547 | 52 |
| nn_fixed | 0.75–1.25 | -0.492 | $-502,476 | 746 | 90 |
| rl_sized | 0.85–1.15 | 0.086 | $44,270 | 299 | 36 |
| nn_fixed | 0.85–1.15 | -0.097 | $-84,212 | 385 | 63 |

## Gross profit and costs

| Model | Beta range | Gross P&L | Option execution | Hedge execution | Stock borrow |
|---|---|---:|---:|---:|---:|
| rl_sized | 0.75–1.25 | $15,123 | $66,644 | $47,215 | $14,477 |
| nn_fixed | 0.75–1.25 | $-66,496 | $189,566 | $143,405 | $43,088 |
| rl_sized | 0.85–1.15 | $116,311 | $44,500 | $27,607 | $8,427 |
| nn_fixed | 0.85–1.15 | $129,825 | $102,367 | $76,259 | $23,868 |

Gross includes option payoffs, stock hedging and hedge dividends. Net excess P&L additionally reflects execution, stock borrow and financing relative to cash.

## Annualized excess Sharpe by year

| Year | RL old range | RL tighter range | NN old range | NN tighter range |
|---|---:|---:|---:|---:|
| 2018 | -0.152 | -1.110 | -1.157 | -0.447 |
| 2019 | -0.538 | -0.005 | -0.915 | -0.916 |
| 2020 | -0.192 | 0.065 | -0.158 | 0.479 |
| 2021 | -0.364 | 0.231 | 0.015 | 0.062 |
| 2022 | 0.344 | 0.631 | -0.059 | 0.096 |
| 2023 | -0.272 | 0.093 | -1.013 | -0.496 |
| 2024 | -0.459 | -0.096 | -0.365 | 0.224 |
| 2025 | 0.067 | 0.130 | -1.525 | -0.082 |

Audit passed for all 358 attempted pairs: label/memory timing, action ladders, beta eligibility, entry commissions and expiry settlement. Fit timing and daily ledger checks also pass in the training runner. NN convergence warnings remain: 1059/2676 validation fits and 388/995 recent refits. Fits are retained and flagged, not silently removed.
Runtime: 25.0 minutes with two workers; largest reported worker peak RSS 0.79 GiB.

Sharpe uses daily returns in excess of the cash benchmark, annualized with √252. Total percentage returns in the CSV include interest on unused capital; net excess P&L removes that benchmark. 2025 is partial. Different trade sets and sizing mean this measures the full retrained policy change, not only rejected entries.

The tighter threshold was chosen after inspecting these years, so this is an exploratory comparison, not an untouched holdout. Complete-history selection remains a limitation. The monthly validation gate still validates the proposal before its recent-data refit, rather than the exact deployed weights.

[Period comparison](expiry_trained_beta_085_115/beta_comparison.csv) · [Annual returns and Sharpes](expiry_trained_beta_085_115/beta_annual.csv) · [Activity](expiry_trained_beta_085_115/beta_activity.csv) · [Audit](expiry_trained_beta_085_115/audit.json)
