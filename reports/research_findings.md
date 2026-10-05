# Current research findings

The saved best strategy is the employee-leadership RL checkpoint, with annualized cash-excess Sharpe **0.368** over 2018–August 2025 and **0.428** since 2022. Both evaluations start flat. All 345 matched sleeves remain in the portfolio; 29 have trades over the full period, totaling 224 entries. The full-period net excess P&L is $145,778 under the saved conservative risk allocation. This is the working reference for improvements.

| Strategy | Full Sharpe | Since 2022 Sharpe | Full entries |
|---|---:|---:|---:|
| Original matched RL checkpoint | 0.318 | 0.347 | 224 |
| RL + employee leadership, current best | 0.368 | 0.428 | 224 |
| RL + three transcript gaps | 0.319 | 0.481 | 237 |
| RL choosing reference/transcript policy using prior-year validation | 0.331 | 0.407 | 254 |
| Saved fixed-size NN + employee leadership | -0.084 | -0.103 | 217 |

These rows use the same requested roster, contract accounting, execution costs and risk limits. Sharpe is daily return above cash times √252. Recent evaluation starts independently flat in 2022. Training/validation charges $0.65 per contract; final evaluation charges $0.03. The higher training hurdle is a policy-selection input, not the assumed live commission.

Employee leadership adds a three-month rating difference and availability flag on 28 covered sleeves. The main full-period improvements were AEO/TJX and CSX/UNP. It does not fix the 2018 and 2025 losses. The transcript family improves recent performance and the 2025 loss, but its full-period deterioration means it does not replace the best checkpoint. Pair-specific correlations vary; a promising correlation by itself has not established portfolio improvement.

Future experiments fit RL only and preserve the saved NN comparisons. The RL-only transcript runner supports single-feature ablations with independent output directories. Use the resource supervisor, for example:

```bash
python -m scripts.run_signal_discovery_guarded --module scripts.run_atm65_rl_transcripts --timeout 3600 -- --transcript-feature call_qa_uncertainty
```

The public-facing CSV contains decision inputs and gates, not forward rewards. Full replay requires separately licensed market histories. Local checkpoint archives preserve actions, states, gates, ledgers and source hashes; fitted model objects were not serialized. The archive itself remains local, and the repository contains the checkpoint manifests and aggregate evidence.

## Evidence

- [Best checkpoint results, timing and audit](atm65_employee_sentiment.md)
- [Prior checkpoint and costs](atm30_fee065_checkpoint.md)
- [NN/RL cost-hurdle comparison](atm_nn_cost_hurdles.md)
- [Higher-hurdle comparison](atm_higher_cost_hurdles.md)
- [RL sizing and objective ablations](atm65_ablation.md)
- [Loss years](atm65_loss_years.md)
- [Signal mining](atm65_signal_mining.md)
- [Transcript feature experiment](atm65_transcripts.md)
- [Prior-validation transcript selector](atm65_transcript_selector.md)
- [Tenor comparison](tenor_comparison.md)
- [Smile/curve experiment](curve_trading.md)
- [Inventory and policy exits](curve_management.md)
- [Actual-contract versus original IV proxy](broad_contract_gap.md)
- [SEC and I/B/E/S signal diagnostics](wrds_pair_signals.md)
- [Alternative data signal diagnostics](wrds_alternative_pairs.md)
- [New volatility-specific data shortlist](vol_alternative_data.md)

These are repeatedly examined historical research results. Publication-lag logic is checked, while historical vendor-vintage and issuer-link audits remain incomplete. The current best is preserved rather than overwritten by subsequent weaker trials.
