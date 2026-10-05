# Why the first contract Q stayed flat, and the next game design

This audit uses the saved first-run models, 2010–2022 training data and 2023 validation data. It does not refit, select a new policy, or inspect additional test years. Run `conda run --no-capture-output -n xfund python -m scripts.diagnose_contract_q` to reproduce the diagnostics in `artifacts/contract_q/diagnosis/`.

## Current game

The contract implementation changed more than the execution data: it replaced the earlier discretized Q table with four ExtraTrees action-value regressors on 31 observable features. There are 90,774 historical training states, not an 81-row table. This architecture change confounds a clean proxy-versus-contract comparison.

| Position now | Available decisions | What happens next |
| --- | --- | --- |
| Flat | Wait; buy AMD calls/sell NVDA calls; reverse those legs | Select a common-expiry ATM pair using today's information; execute those exact IDs at tomorrow's bid/ask if eligible |
| Invested | Hold; exit | Keep the same IDs and quantities; hedge stock delta daily; exit at tomorrow's bid/ask |

There are four action codes because flat waiting and invested holding share code zero. All four are not simultaneously legal. Contract selection, maturity and size are prescribed by the engine; Q does not optimize them. Forced exits apply before expiry, at the holding limit and at other documented safeguards.

The reward is daily actual-account P&L above cash, including option prices, delta hedges, financing, spreads and fees. Historical future prices supply training outcomes, never decision-time features. Q is intended to propagate future rewards backward. It is not just a one-day predictor, although the current 12 synchronous sweeps and 0.97 discount make distant payoffs harder to learn. At 20 transitions the discount is about 0.54.

## Findings, separating evidence from explanation

**The learned cash values are wrong.** On all 2,964 eligible training dates, both Q variants prefer waiting. On all 244 eligible validation dates they do the same. For the unpenalized Q, the mean predicted value of waiting while flat is **−6.38 bp** in training and **−6.14 bp** in validation. The mean best-entry advantage relative to waiting is −15.03 bp and −6.05 bp respectively.

An optimal flat-state wait value cannot be negative: continuing to wait until the boundary earns exactly zero excess reward. A greedy all-wait policy also has exactly zero realized excess value. The approximation therefore fails an elementary value-function check. This does not alone prove that fixing it will produce profitable entries: entry values are also negative.

The shared wait/hold regressor fits zero-reward flat transitions together with risky invested transitions. With shallow trees and a minimum leaf size of 128, it can mix their targets; subsequent bootstrapping propagates errors. This is a credible mechanism, not an isolated causal estimate. The final sweep still changed values by up to 13.07 bp without the downside penalty and 22.79 bp with it. Neither convergence nor useful training-policy recovery was established. Both variants fail to enter, so the downside penalty is not the sole explanation.

**Costs impose a substantial hurdle.** A deterministic sample of every tenth training session attempted both entry directions, independent of the eventual outcome. Of 582 attempts, 544 entered; 38 were rejected by execution/sizing/notional gates. For those 544 hypothetical tickets held until their forced exit:

| Diagnostic | Result |
| --- | ---: |
| Mean excess P&L before adding back execution costs | −$3,785 |
| Mean option/stock execution costs | $3,723 |
| Mean excess P&L with those costs added back | −$63 |
| Profitable net round trips | 29.4% |
| Profitable net one-session round trips | 5.1% |
| Tickets with at least one profitable exit in hindsight | 51.1% |

These are overlapping hypothetical tickets, not portfolio returns or independent observations. Both opposing directions are included, so near-zero average P&L before execution costs is largely cancellation, not evidence that every conditional signal lacks an edge. Costs added back here are realized spreads, commissions and hedge trading costs; financing remains. This is an accounting decomposition, not a fresh frictionless strategy simulation.

The cost regime also varies materially: sampled mean execution cost was $9,279 in 2015 and $1,493 in 2022 for the same target dollar vega. Simply pooling those observations can obscure which current opportunities are economically feasible. A relative bid/ask-width input does not explicitly express the total round-trip hurdle per dollar of vega.

**Profitable paths exist; predictability remains unproved.** One illustrative training observation, deliberately selected after seeing its positive outcome:

| Five-session position: long AMD calls / short NVDA calls | Entry, 2022-03-30 | Exit, 2022-04-06 |
| --- | ---: | ---: |
| AMD stock | $119.22 | $103.67 |
| NVDA stock | $276.90 | $244.07 |
| Held AMD call IV | 49.17% | 59.40% |
| Held NVDA call IV | 45.47% | 50.74% |

Actual IDs are 145903787 and 145834559. With the engine's daily delta hedge, net excess P&L was **+$2,069**, after **$522** in execution costs. AMD IV rose about 10.23 points and NVDA about 5.27 points. This illustrates relative volatility exposure despite both stock prices falling; it does not attribute the entire profit to IV or establish an ex-ante entry signal. Gamma, theta, financing and changing vega also contribute. The downloaded paths and diagnostic CSV preserve both losing and winning outcomes.

## Proposed game: learn whether an actual round trip is worth buying

First repair the cash/inventory separation and establish a simpler entry-only control before introducing discretionary exits. Preserve the original saved model and results for comparison.

1. **Define the economic forecast.** Forecast the two assets' future realized variances jointly from past information: shared variance dynamics plus asset-specific deviations. Separately model each contract's expected IV change at a matched maturity/moneyness. Correlated stock returns or related RV do not establish mean reversion in the raw IV difference. Earnings/event exposure and maturity alignment matter. Fit all transforms and forecasts chronologically inside training; do not use forward RV as a state variable.
2. **Express opportunity in dollars.** Combine expected relative IV repricing and delta-hedged variance carry for the actual quantities, subtract estimated round-trip option/hedge costs and financing, and retain uncertainty. Supply separate per-leg forecasts, relative residual, recent residual changes, spread cost per vega, maturity, liquidity and event proximity. An IV premium on both names does not by itself identify which relative position has positive expected net P&L. Option time decay depends on maturity and other pricing variables ([OIC](https://www.optionseducation.org/advancedconcepts/theta)).
3. **Begin with fixed episodes.** At an eligible signal date, choose skip / long AMD-short NVDA / reverse for a predeclared short holding horizon, with the same next-session fills and safety exits. Skip has exact zero trade value. Supervise the two entry values on their fully realized, cost-inclusive round trips. This is a contextual-bandit control, explicitly not yet a full sequential Q policy. Use a bounded horizon grid selected only on validation; do not select each historical ticket's best hindsight exit as a forecast label.
4. **Then add Q for exit timing.** Keep separate entry and invested value functions. Within an episode, Q chooses hold or close using remaining time, residual convergence, exposure, current quotes and liquidation costs. Finite episodes use gamma=1 and terminal continuation zero. Reward cash-discounted wealth changes so cumulative reward equals cash-discounted net trade P&L, including entry and liquidation. No reward just for taking a position. If later optimizing a continuous stream of trades, model the opportunity value of waiting explicitly rather than silently treating an episode's skip as the continuous-game wait value.
5. **Validate learning before reporting performance.** Verify zero value for permanent cash, exact round-trip reward reconciliation, and recovery of a deliberately predictable synthetic profitable path after costs. Check conditional forecasts against realized outcomes and examine stability across training blocks. Compare fixed-horizon entry policy with Q-managed exits under identical contracts and fills, using purged temporal validation around overlapping labels. A Q exit model must improve net results over its simpler control; trading more is not an objective.

Batch Q is a sequence of fitted regression problems, so regression approximation and iteration behavior are part of the algorithm, not just implementation details ([Ernst, Geurts and Wehenkel, 2005](https://www.jmlr.org/papers/v6/ernst05a.html)). More iterations or an attractive payoff formula alone cannot establish a tradable edge.

This audit establishes a learning defect, a measurable execution hurdle, and examples of profitable historical paths. It does **not** establish how much of the no-entry result each cause explains, or that the proposed replacement will be profitable. The next experiment should change one component at a time on training/validation, preserving the previous test results.
