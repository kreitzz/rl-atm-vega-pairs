# Retraining for hold-to-expiry

Pair eligibility: monthly preceding-126-session first-leg-on-second-leg beta 0.85–1.15, with return correlation at least 0.5. Applied to training, validation and new entries; existing positions remain held through expiry.

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
| rl_sized | retrained | full | 0.086 |
| rl_sized | retrained | 2018_2021 | -0.087 |
| rl_sized | retrained | 2022_2025 | 0.179 |
| nn_fixed | frozen | full | -0.215 |
| nn_fixed | frozen | 2018_2021 | -0.228 |
| nn_fixed | frozen | 2022_2025 | -0.225 |
| nn_fixed | retrained | full | -0.097 |
| nn_fixed | retrained | 2018_2021 | -0.263 |
| nn_fixed | retrained | 2022_2025 | -0.029 |

Sharpe uses daily cash-excess returns and √252. 2025 ends in August.

[Comparison](expiry_trained_beta_085_115/comparison.csv) · [Annual returns](expiry_trained_beta_085_115/annual.csv) · [Run status](expiry_trained_beta_085_115/status.csv) · [Fit audit](expiry_trained_beta_085_115/fits.csv)
