# Contract episodes: entry forecasts and Q-managed exits

This experiment implements the redesign in [the diagnosis](contract_q_diagnosis.md), preserving the original contract Q and its saved results. The execution engine, $450 vega target per leg, integer options, next-session bid/ask fills, daily delta hedge and financing conventions are unchanged.

## Results

**The fixed-horizon entry learner now trades and earned a positive return above cash in partial-year 2025, but the Q-exit learner still does not establish an edge.** Every Q candidate stayed flat in validation and both test periods. Fixing the cash-value problem did not by itself produce useful Q-managed trades.

The fixed-horizon selector chose 10 sessions and a $250 predicted net-profit buffer. Its 2023 validation had two trades, just **$15.40 above cash**, and excess Sharpe 0.0058. This is a particularly weak selection result, not convincing evidence of a signal. The Q family selected cash.

| Period | Method | Entries | Dollars above cash | Return above cash | Excess Sharpe | Maximum NAV drawdown |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2024 | Selected fixed-horizon entry | 2 | −$208 | −0.021% | −0.044 | −0.232% |
| 2024 | Selected Q family / cash | 0 | $0 | 0.000% | — | 0.000% |
| 2024 | Short higher IV / long lower IV | 14 | −$9,048 | −0.905% | −0.373 | −1.313% |
| 2024 | Relative trailing carry | 14 | −$38,216 | −3.822% | −1.624 | −2.042% |
| 2025 through Aug. 29 | Selected fixed-horizon entry | 2 | +$11,555 | +1.156% | 0.863 | −0.278% |
| 2025 through Aug. 29 | Selected Q family / cash | 0 | $0 | 0.000% | — | 0.000% |
| 2025 through Aug. 29 | Short higher IV / long lower IV | 10 | +$41,934 | +4.193% | 1.454 | −1.120% |
| 2025 through Aug. 29 | Relative trailing carry | 10 | +$5,685 | +0.568% | 0.217 | −2.935% |

Returns above cash use $1 million initial capital and are not annualized. Excess Sharpe is annualized from daily returns; it is undefined for cash (the JSON stores zero as a convention). NAV drawdown includes cash interest. All runs completed and closed their positions. The selected fixed policy paid $1,928 in option/hedge execution costs in 2024 and $1,429 in partial-year 2025. Its total account returns, including cash interest, were 5.305% and 4.166%, respectively.

Only four selected-policy trades occurred across the two evaluation periods. The positive 2025 result is therefore fragile. The higher-IV mechanical rule earned more in 2025 but had a larger drawdown; the new model is not better across all metrics. The original cross-entropy policy was not retrained on this contract game, so these results do not establish a Q-versus-CEM advantage.

For transparency, two unselected candidates also made money in 2025: the 5-session zero-buffer fixed policy earned 1.359% above cash on four trades, and the 10-session zero-buffer policy earned 2.327% on five. Neither is substituted for the validation-selected policy. Full candidate results are retained in the aggregate summary.

### Why Q still stays flat

The old negative-cash-value defect is removed. In 2023 validation, the 10-session fixed model's best entry forecast reached +$321, with 13 positive forecasts and five exceeding the $250 buffer. The corresponding Q-managed entry model's **largest** forecast was −$723 (its mean best-direction value was −$1,019). It therefore rejects trades because its predicted net episode values are negative, not because it assigns a negative value to cash. That can reflect weak generalization of the early-trained exit policy and regression averaging as well as genuine costs; these predictions are not proof that every rejected trade loses.

A separate **post-hoc diagnostic**, never used for selection, replays the fixed policy's identical actual entry dates with the already-fitted Q exit rule:

| Period | Fixed exits: above cash | Q exits on identical entries: above cash |
| --- | ---: | ---: |
| 2024, two entries | −$208 | −$2,235 |
| 2025 through Aug. 29, two entries | +$11,555 | +$12,378 |

Q exits worsen 2024 by about $2,028 and improve partial-year 2025 by about $822. This mixed four-trade diagnostic does not establish exit-timing skill. It does show why it is important to distinguish entry rejection from exit quality. No alternative is promoted on the strength of this post-hoc comparison.

## What changed

- **An exact zero skip value.** Entry chooses skip, long AMD calls/short NVDA calls, or the reverse. A separate regression predicts each direction's complete realized net round-trip value. The learner cannot assign a loss to simply skipping the episode.
- **Finite episodes.** Compare 5- and 10-session holding horizons. Q can close early; existing safety exits still apply. The entry fill is followed by at most H sessions of exposure. Samples whose full maximum horizon crosses a block boundary are purged.
- **Separate exit Q.** Two values, close and hold, are fitted backward by remaining time. Closing terminates the episode. There is no circular infinite-horizon iteration and no sharing of flat cash and invested hold regressions. Each age uses a 64-tree ExtraTrees regressor, maximum depth 5 and minimum leaf size 40. Counterfactual states on the same date share a total weight of one at each age.
- **Cash-discounted profit.** Bellman backups are `(next-session excess dollars + next-state value) / cash_growth`. There is no additional preference discount (gamma=1). Summing cash-discounted rewards reconciles to the actual account's net excess wealth. Entry costs and final liquidation remain charged; no bonus rewards taking a position.
- **Observable forecasts.** Ridge models use both stocks' trailing 5-, 21- and 63-session squared returns to forecast 10-session common log variance and the difference between the two log variances. Their reconstruction supplies per-leg volatility forecasts. A second regularized model predicts 10-session IV changes for the same option IDs from contemporaneous contract features and trailing stock variance. These are point forecasts, not a full joint stochastic volatility model. The fixed 10-session forecast horizon is also used as a feature for the 5-session policies.
- **An explicit cost hurdle.** Features include today's estimated option round-trip spread and commission per target dollar vega, predicted relative variance carry, and expected per-leg IV changes. Training labels additionally include realized stock hedge costs and financing. Actual future execution costs never appear in inputs.

## Chronology and comparisons

| Purpose | Period |
| --- | --- |
| Fit volatility/IV forecasts and exit Q | 2010–2018 |
| Fit entry models on outcomes of the frozen exit policy | 2019–2022 |
| Select horizon and entry buffer | 2023 |
| Frozen evaluation | 2024 and available 2025 |

Entry models for both fixed holding and Q exits use the same 2019–2022 block. In particular, Q entry labels follow a policy fitted on the earlier block; they are not each ticket's hindsight-best exit. The exit models' forecast features are fitted within the earlier training block, so that block's fitted Q values are not out-of-sample estimates.

The preset grid is two horizons times two minimum predicted net-profit buffers ($0 and $250), separately for fixed holding and Q exits. Each family selects the highest positive validation excess Sharpe; cash wins otherwise. All candidates and the selections are saved before test evaluation. Candidate test results are diagnostics, not another selection round. Mechanical higher-IV and relative-carry controls keep their original maximum holding convention; they are reference baselines, not an isolated exit ablation.

Even when fixed and Q policies share a horizon, their entry targets differ. Their comparison measures the complete policy, not a causal estimate of exit timing alone. The objective maximizes expected episode dollars; it does not yet optimize the opportunity cost of tying up the single available position across a continuous stream of trades.

## Reproduction and validation

Run in the requested `xfund` environment:

```powershell
conda run --no-capture-output -n xfund python -m scripts.run_contract_episodes
conda run --no-capture-output -n xfund python -m scripts.replay_contract_episodes
conda run --no-capture-output -n xfund python -m unittest tests.test_contract_episodes tests.test_contract_q tests.test_contract_engine
```

Outputs are in `artifacts/contract_episodes/`: manifests, forecasts/episode audit, saved models, validation selection, complete daily ledgers and test summary. The data are the existing locally downloaded WRDS/OptionMetrics contract bundle; no further download is needed.

The saved policies reproduce all selected-policy daily NAVs to numerical tolerance. Aggregate results, selection, manifests and replay diagnostics are copied into `reports/contract_episodes/`; contract-level data and model files remain in the ignored artifacts directory. Only load the trusted locally generated pickle.

Tests cover exact cash-discounted reward reconciliation against the contract account, horizon/boundary handling, policy-following labels instead of hindsight exit selection, zero skip value and recovery of a deliberately predictable profitable synthetic trade after costs, alongside the existing contract execution/accounting tests. All 26 pass.

## Limitations

This remains an exploratory historical experiment. The 2024 period was already examined in earlier work; 2025 is a partial year. Daily quotes and stock marks are not guaranteed synchronous executable fills. The engine is not a broker margin, depth or assignment simulator. Positions remain delta-hedged calls rather than straddles. No earnings-calendar feature was added in this experiment, and the forecast models do not guarantee convergence of the relative IV spread. Training spans materially different liquidity regimes. Short holding periods also increase the importance of crossing option spreads.
