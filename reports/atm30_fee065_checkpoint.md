# ATM30 $0.65 training-hurdle checkpoint

Saved on October 1, 2026 as the current research reference. Training hurdle: $0.65 per contract. Actual execution commission: $0.03 per contract, with the existing 0.10-vol-point option entry penalty and other costs and risk limits unchanged. ATM calls on both legs, approximately 30-day expiry, entry vega matching, daily delta hedging and hold to expiry.

| Universe | Model | 2018–August 2025 Sharpe | 2022–August 2025 Sharpe |
|---|---|---:|---:|
| Matched 345 pairs | RL sizing ladder | 0.318 | 0.347 |
| Matched 345 pairs | NN fixed size | −0.097 | −0.099 |
| Original 346 pairs | RL sizing ladder | 0.350 | 0.396 |
| Original 346 pairs | NN fixed size | −0.127 | −0.146 |

Sharpes use daily returns above cash and are annualized with √252. Both periods start flat. The matched six-hurdle comparison excludes DVN/MRO from every row because its $0.30 NN evaluation crosses an unsupported dividend event. The original-universe $0.65 evaluations completed for both models.

The $0.65 hurdle is the best observed RL setting in the matched six-hurdle sweep. This is a selected result on previously examined history. RL wins against this NN with identical inputs, but architecture, loss function and sizing differ together. About 44% of NN proposal fits reached the iteration limit. The comparison does not isolate which change causes the improvement or establish statistical significance.

[Checkpoint manifest](../config/atm30_fee065_checkpoint.json) · [Current reference](../config/current_contract_reference.json) · [Full comparison and audit](atm_nn_cost_hurdles.md) · [Local snapshot](../artifacts/contracts/checkpoints/atm30_fee065_2026-10-01.tar)

The snapshot preserves historical actions, states, gates, fit logs, ledgers, reports and code; every member is checked with SHA-256. The former current-reference configuration is inside the archive. Raw market data remain in their existing locations, referenced by archived input manifests. Monthly fitted model objects were not serialized; the preserved actions support replay of the historical strategy.
