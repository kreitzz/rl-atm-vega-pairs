# NN fixed size versus RL sizing at matched training hurdles

Attempted all six hurdles across all 346 original pairs. The comparison uses the same 345 complete pairs for both models, all costs and both periods. Excluded pairs: DVN/MRO. Any unresolved NN book removes that pair from every matched comparison, including RL; it is never assigned cash or zero P&L. This is an ex-post data-completeness subset and can be biased. A full-roster comparison spanning all six settings is unavailable.

| Training hurdle | NN full Sharpe | RL full Sharpe | NN recent Sharpe | RL recent Sharpe | NN full entries | RL full entries |
|---|---:|---:|---:|---:|---:|---:|
| $0.03 | -0.170 | 0.113 | -0.168 | 0.181 | 268 | 306 |
| $0.15 | -0.161 | 0.139 | -0.226 | 0.177 | 240 | 272 |
| $0.30 | -0.104 | 0.270 | 0.055 | 0.285 | 227 | 261 |
| $0.65 | -0.097 | 0.318 | -0.099 | 0.347 | 221 | 224 |
| $1.00 | -0.104 | 0.248 | -0.010 | 0.340 | 172 | 193 |
| $1.50 | -0.088 | 0.117 | -0.169 | 0.193 | 148 | 158 |

RL exceeds this fixed-size NN at every hurdle in both periods. The $0.65 hurdle has the highest RL Sharpe in both periods on this matched roster. Raising it to $1.00 or $1.50 reduces full-period Sharpe. At $0.65, the models make almost the same number of entries (221 NN versus 224 RL), so fewer trades alone does not explain the difference; direction, timing and sizing can all contribute. The previously reported full-roster RL $0.65 Sharpe was 0.350; excluding DVN/MRO from both models reduces it to 0.318.

Full: January 2018–August 2025. Recent: January 2022–August 2025, starting flat. Daily cash-excess Sharpes are annualized with √252. Same $1m per retained pair, including complete cash sleeves. Actual execution is $0.03 per contract for every model and hurdle; option entry penalty, stock hedging, borrowing, financing, lifecycle and risk limits match.

NN: one 16-unit tanh hidden layer, L-BFGS, alpha=1, maximum 1000 iterations, train-only scaling. It predicts the two net full-size outcomes and chooses the larger positive prediction or cash. RL: the saved eight-leaf direct Sharpe policy search with the signed 25/50/75/100% ladder or cash. Architecture, learning objective and sizing change together, so this is not a pure RL-effect ablation.

Identical feature columns, cost-rebuilt completed-outcome memory, 91/365-day directional gates, beta/correlation eligibility, monthly schedule, five-year fitting windows, outcome purging and prior-year positive-Sharpe validation. Validation enables the proposal, followed by the inherited recent-five-year refit; the newly refitted weights do not have an independent validation test. NN convergence warnings remain in the fit audit. This is exploratory research on previously examined history.

NN optimization hit its iteration limit in 5,474 of 12,561 proposal fits (43.6%) and 1,860 of 4,377 refits (42.5%). Those fits remain in the results. This substantially limits a conclusion about RL versus a fully optimized NN; the table compares the implemented models.

The final audit passed: 4,150 complete NN sleeve windows, two explicitly unresolved windows, 840 matching state/gate combinations, fixed-size NN actions only, actual commission $0.03, and maximum ledger reconciliation error below $6e-10. The four targeted NN and roster regression tests passed. The fitting run used one worker, a 50% duty-cycle cap on one CPU core and a 3 GiB memory stop.

The original NN source and fitting hashes remain intact. A separately recorded continuation catches unresolved evaluation paths, attempts the other windows, and reports matching without changing prices, trades, rewards, fees or Greeks. The DVN/MRO $0.30 NN path first failed on 2023-02-14 with unsupported distribution for option 152049246.

The [full 346-pair supplement](atm_nn_cost_hurdles/full_roster_comparison.csv) preserves results where every sleeve completed. Only the $0.30 NN rows are unavailable. RL exceeds NN at all five fully observed settings in both periods. Full-roster RL at $0.65 is 0.350 full / 0.396 recent, compared with NN at −0.127 / −0.146. At $1.00, full-roster NN is −0.003 full / 0.150 recent versus RL at 0.286 / 0.397. These rows include DVN/MRO and should not be substituted into the 345-pair matched table above.

[Comparison](atm_nn_cost_hurdles/comparison.csv) · [Annual results](atm_nn_cost_hurdles/annual.csv) · [Pair results](atm_nn_cost_hurdles/pairs.csv) · [Fit audit](atm_nn_cost_hurdles/fits.csv) · [Unresolved cases](atm_nn_cost_hurdles/unresolved.csv) · [Matched roster](atm_nn_cost_hurdles/matched_roster.csv) · [Checks](atm_nn_cost_hurdles/audit.json)
