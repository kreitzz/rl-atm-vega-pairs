# Retraining for hold-to-expiry

Both models use regenerated actual-contract daily net excess P&L trajectories through each leg's expiry. Entry cost is 0.1 volatility point per leg plus $0.65 per contract; expiry settlement has no option spread or commission. Stock hedges cost 1 bp per executed notional, including liquidation; short stock borrow is 0.3% annually, with cash interest and a 1% borrowing premium on negative cash. These are the same stated costs as the frozen expiry replay, not a reduction selected for performance.

RL optimizes daily shadow-book excess Sharpe with a −1 to +1 ladder in 0.25 steps. NN predicts net outcomes and selects full-size long/short or flat. Independent pair models use the existing smile/term structure and other inputs. The commission state now describes entry only. Profitability memory is regenerated from completed expiry outcomes, available strictly after the later leg expires; RL retains positive 91/365-day gates and NN the positive 365-day gate.

Each month: fit five years ending before a separate one-year validation period; validate the filtered policy using the same expiry simulator and entry risk limits; permit trading only after positive validation excess Sharpe. Refit permitted specifications on the latest five years of completed outcomes before deployment. The gate validates the prior fitted proposal, not the refitted weights. Nothing from a still-open cohort enters training or memory. Validation and final entry cutoffs allow 50 calendar days of runoff.

Hold-to-expiry assumes cash intrinsic settlement and no early assignment. Daily marks remain observed mids; failed fitted Greeks use prior valid IV for that day's hedge. Existing risk breaches do not force liquidation; entry is blocked. Standalone training cohorts do not reproduce the overlapping portfolio's NAV sizing or risk entry rejections; validation and final evaluation apply those rules.

Attempted 358 pairs. Comparisons use 347 common complete broad pairs from the prior frozen-expiry roster, retaining cash-only sleeves. Incomplete histories are recorded, not assigned zero P&L. Complete-roster selection can be biased. All periods have already been inspected in research.

| Model | Policy | Period | Annualized excess Sharpe |
|---|---|---|---:|
| rl_sized | frozen | full | -0.040 |
| rl_sized | frozen | 2018_2021 | 0.084 |
| rl_sized | frozen | 2022_2025 | -0.141 |
| rl_sized | retrained | full | -0.207 |
| rl_sized | retrained | 2018_2021 | -0.296 |
| rl_sized | retrained | 2022_2025 | -0.111 |
| nn_fixed | frozen | full | -0.215 |
| nn_fixed | frozen | 2018_2021 | -0.228 |
| nn_fixed | frozen | 2022_2025 | -0.225 |
| nn_fixed | retrained | full | -0.492 |
| nn_fixed | retrained | 2018_2021 | -0.455 |
| nn_fixed | retrained | 2022_2025 | -0.531 |

Sharpe uses daily cash-excess returns and √252. 2025 ends in August.

Retraining did not improve full-period performance. On the matched 347-pair basket, RL entries rose from 410 to 547 while gross delta-hedged P&L fell from $85,625 to $15,123. NN entries rose from 334 to 746 and gross P&L fell from $25,400 to −$66,496. Thus more trading did not produce enough gross edge to cover costs; this result cannot be explained solely by obsolete option exit fees.

| Retrained policy | Gross hedged P&L | Option entry costs | Stock execution costs | Stock borrow | Financing versus cash | Net excess P&L |
|---|---:|---:|---:|---:|---:|---:|
| RL | $15,123 | $66,644 | $47,215 | $14,477 | −$16,273 | −$129,486 |
| NN | −$66,496 | $189,566 | $143,405 | $43,088 | −$59,921 | −$502,476 |

Gross includes hedge dividends. Financing versus cash also includes foregone interest from earlier losses; it is not solely an explicit funding charge. The basket starts with $347m ($1m per pair, including cash-only sleeves). Percentage total returns in the annual CSV include cash interest and should not be interpreted as strategy excess returns.

The saved-artifact audit passed for all 358 attempted pairs and 597,598 label rows. It checked sampled profitability-memory dates against completed outcomes, action ladders, option entry commissions, zero expiry charges, closed inventories and code fingerprints. There are 3,248 unresolved label histories, retained as unknown outcomes. Final NN evaluations are unresolved for BA/GE (invalid held quote on January 4, 2023) and CCL/RCL (missing held quote on March 13, 2020); comparisons preserve the prior common roster and omit CCL/RCL on both sides.

NN optimizer warnings remain material: 1,562 of 4,020 validation-stage fits and 583 of 1,603 recent refits did not converge within the fixed optimizer limits. These results describe that fitted benchmark, not a proven optimal NN. No optimizer or cost tuning was performed after viewing these results.

All 141 tests passed. An additional 180 real-contract/action trajectories matched the simulator to at most 1.3×10⁻¹² per dollar of target vega. Skipping fits when both directions are blocked throughout validation preserved identical actions on all 19 completed pairs from the initial attempt. The optimized full run took 28.9 minutes with two workers, peaking at 0.80 GiB per worker.

[Cost attribution](expiry_trained/cost_attribution.csv) · [Artifact audit](expiry_trained/audit.json)

[Comparison](expiry_trained/comparison.csv) · [Annual returns](expiry_trained/annual.csv) · [Run status](expiry_trained/status.csv) · [Fit audit](expiry_trained/fits.csv)
