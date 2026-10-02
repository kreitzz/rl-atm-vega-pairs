# Retraining for hold-to-expiry

Both models use regenerated actual-contract daily net excess P&L trajectories through each leg's expiry. Entry cost is 0.1 volatility point per leg plus $0.65 per contract; expiry settlement has no option spread or commission. Stock hedges cost 1 bp per executed notional, including liquidation; short stock borrow is 0.3% annually, with cash interest and a 1% borrowing premium on negative cash. These are the same stated costs as the frozen expiry replay, not a reduction selected for performance.

RL optimizes daily shadow-book excess Sharpe with a −1 to +1 ladder in 0.25 steps. NN predicts net outcomes and selects full-size long/short or flat. Independent pair models use the existing smile/term structure and other inputs. The commission state now describes entry only. Profitability memory is regenerated from completed expiry outcomes, available strictly after the later leg expires; RL retains positive 91/365-day gates and NN the positive 365-day gate.

Each month: fit five years ending before a separate one-year validation period; validate the filtered policy using the same expiry simulator and entry risk limits; permit trading only after positive validation excess Sharpe. Refit permitted specifications on the latest five years of completed outcomes before deployment. The gate validates the prior fitted proposal, not the refitted weights. Nothing from a still-open cohort enters training or memory. Validation and final entry cutoffs allow 50 calendar days of runoff.

Hold-to-expiry assumes cash intrinsic settlement and no early assignment. Daily marks remain observed mids; failed fitted Greeks use prior valid IV for that day's hedge. Existing risk breaches do not force liquidation; entry is blocked. Standalone training cohorts do not reproduce the overlapping portfolio's NAV sizing or risk entry rejections; validation and final evaluation apply those rules.

Attempted 1 pairs. Comparisons use 1 common complete broad pairs from the prior frozen-expiry roster, retaining cash-only sleeves. Incomplete histories are recorded, not assigned zero P&L. Complete-roster selection can be biased. All periods have already been inspected in research.

| Model | Policy | Period | Annualized excess Sharpe |
|---|---|---|---:|
| rl_sized | frozen | full | -0.270 |
| rl_sized | frozen | 2018_2021 | 0.001 |
| rl_sized | frozen | 2022_2025 | -0.614 |
| rl_sized | retrained | full | -0.397 |
| rl_sized | retrained | 2018_2021 | -0.549 |
| rl_sized | retrained | 2022_2025 | nan |
| nn_fixed | frozen | full | -0.248 |
| nn_fixed | frozen | 2018_2021 | -0.274 |
| nn_fixed | frozen | 2022_2025 | -0.224 |
| nn_fixed | retrained | full | 0.258 |
| nn_fixed | retrained | 2018_2021 | 0.515 |
| nn_fixed | retrained | 2022_2025 | -0.399 |

Sharpe uses daily cash-excess returns and √252. 2025 ends in August.

[Comparison](expiry_training_pilot/comparison.csv) · [Annual returns](expiry_training_pilot/annual.csv) · [Run status](expiry_training_pilot/status.csv) · [Fit audit](expiry_training_pilot/fits.csv)
