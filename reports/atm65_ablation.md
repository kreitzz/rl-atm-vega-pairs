# $0.65 policy ablation

Training hurdle $0.65; actual commission $0.03. Same checkpoint features, costs, eligibility, risk limits and monthly timing. The starting roster is the previously matched 345 pairs, including cash sleeves. Further unresolved paths are explicitly reported and excluded from every model, not assigned zero P&L.

| Model | Full Sharpe | Recent Sharpe | Full entries |
|---|---:|---:|---:|
| nn_fixed | -0.097 | -0.099 | 221 |
| nn_sized | -0.156 | -0.247 | 442 |
| tree_fixed | 0.062 | -0.014 | 118 |
| rl_fixed_frozen | 0.295 | 0.240 | 111 |
| rl_fixed | 0.001 | -0.047 | 88 |
| rl_sized | 0.318 | 0.347 | 224 |

The original RL policy with sizing remains strongest. Replaying its saved directions and signal dates at full size retains most full-period Sharpe (0.295 versus 0.318), although recent-period Sharpe falls from 0.347 to 0.240. This shows useful information in the decisions learned with sizing; the result is not merely a mechanical benefit of smaller positions. Actual executed trades differ because the simulator reapplies portfolio risk limits: 111 full-size frozen entries versus 224 original sized entries.

Retraining the Sharpe policy with only fixed-size actions performs much worse (0.0015 full, −0.047 recent). The learning problem changes when sizes are removed; a policy learned with sizing is not equivalent to learning under a fixed-size constraint. The supervised tree on the same partitions achieves 0.062 full and −0.014 recent, slightly exceeding the newly trained fixed-size Sharpe policy. These findings support the specific learned policy with sizing rather than an unconditional advantage from the Sharpe objective or RL alone.

Confidence sizing worsens NN performance. It preserves exactly the same signal directions and dates, but smaller positions reduce risk-limit entry rejections from 1,150 to 730 and increase realized entries from 221 to 442. Net P&L above cash worsens from −$51,579 to −$81,401. The training-only confidence quartiles are a predeclared heuristic; this result does not exclude better supervised sizing methods.

Full: 2018–August 2025. Recent: 2022–August 2025 starting flat. Cash-excess daily Sharpes annualized with √252.

NN-sized recovers the original enabled monthly refits, verifies identical fixed-size directions, and maps positive predicted net P&L into 25/50/75/100% size using training-only confidence quartiles among eligible, memory-permitted rows. Original NN validation gates are frozen to isolate sizing; this is not a new validation test of the sized policy. No NN fitting is repeated for disabled months. Architecture, optimizer and full-size payoff targets are unchanged.

RL-fixed-frozen maps the saved RL signals to 100% size without changing signal direction or dates. The simulator reapplies risk limits, so realized entries can change. RL-fixed is separately retrained with only short full size, long full size or cash, using the original Sharpe search. Tree-fixed uses exactly the same training tree partitions, choosing the leaf direction with highest positive mean net payoff rather than direct Sharpe search. The tree and RL-fixed variants use their own prior-year validation and recent-five-year refits.

This remains exploratory attribution on inspected history, with the inherited validation-then-refit procedure. NN convergence warnings are retained. Raw data, the $0.65 checkpoint and saved baseline results are unchanged.

The final audit passed across 2,760 new sleeve windows, with zero additional unresolved paths, 345 common pairs, exact reconstruction of original NN directions, and exact frozen RL signal mapping. All 713 enabled NN refits were recovered; disabled fits were reused as gates. Both saved baseline aggregate Sharpes and entry counts match the checkpoint. Maximum ledger reconciliation error was below $6e-10. Four targeted regression tests passed. Computation took approximately 18 minutes across the pilot and resumed batches, at 50% of one CPU core; peak worker memory was 0.66 GiB. Reader and audit fixes for sparse pairs are recorded under `artifacts/contracts/atm65_ablation/revisions`; no completed simulation or learning rule changed.

[Comparison](atm65_ablation/comparison.csv) · [Annual](atm65_ablation/annual.csv) · [Pairs](atm65_ablation/pairs.csv) · [Fits](atm65_ablation/fits.csv) · [Coverage](atm65_ablation/coverage.json) · [Audit](atm65_ablation/audit.json) · [Decision and fit counts](atm65_ablation/attribution_counts.json)
