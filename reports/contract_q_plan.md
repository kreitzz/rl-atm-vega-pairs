# Contract-based Q learner: implementation plan

Implemented and tested; see [execution details and first results](contract_q.md). The roadmap below records the design, with additional joint-RV forecasting, sizing, and contract-selection work still separate from the first implementation.

## Objective

Train on the cash P&L of identifiable option contracts and stock hedges, rather than on a synthetic 30-day IV payoff. Start with AMD/NVDA and one open pair at a time. This is an offline research/backtest and order-decision architecture, not a broker-connected live bot.

## Reuse and change

- Reuse the contract pilot's quote normalization, standard-call convention, integer vega sizing, side-of-market fills, commissions, stock hedge costs, financing, and strict handling of unresolved held positions.
- Select a common expiration for the two legs using only information available on the decision date. Prefer approximately 30 days, within 21–45 days, and near-ATM strikes. Keep actual contract IDs until exit; do not roll silently into a constant-maturity surface.
- Make selection deterministic initially. Q learns whether to enter, direction, and when to exit. Letting Q choose from thousands of strikes and expirations at once would confound signal learning with contract selection and greatly reduce observations per action.
- Use a lagged open-interest field for signal-time eligibility. Execute an order generated on one session at the following session's quote. Recheck that exact contract; reject an unfillable entry without substituting a different ID.

## Models to compare under the same execution rules

1. Cash.
2. Short the higher-IV contract / buy the lower-IV contract, with a fixed holding horizon.
3. Relative trailing variance-carry direction, with a fixed holding horizon.
4. A fixed-hold tree policy search, with tree states fitted on net, fully resolved contract outcomes and policies scored on daily contract P&L. This separates entry quality from exit control.
5. Fitted Q iteration, allowing hold/exit decisions after entry. Fit shared regression models over continuous state inputs instead of growing a sparse table by multiplying contract IDs, ages, and P&L buckets.

The existing cross-entropy policy search is a useful conceptual baseline, but its old IV-proxy score is not comparable. A same-execution fixed-hold learned policy is the initial control; a contract-trained CEM comparison can be layered on after the contract data and accounting pass validation.

## State, actions, and reward

The flat state describes both selected contracts: IV, dollar vega, delta, moneyness, days to expiration, relative bid/ask widths, and lagged liquidity, plus trailing realized volatility and return correlation. The invested state describes the actual held contracts, direction, holding age, fixed quantities, entry-to-current IV displacement, hedge exposure, and marked position value.

No option identifier, future RV, future quote availability, realized trade outcome, or calendar year is a predictor. Identifiers belong in the execution ledger, not as predictive features.

Flat actions are wait, enter long A / short B, or enter short A / long B. Invested actions are hold or exit. Position sizes are fixed, integer, initially matched by dollar vega. Known expiry and dividend safeguards override the policy.

The initial reward is daily dollar P&L above cash, scaled into basis points of fixed initial capital. Option P&L, stock-hedge P&L, spreads, commissions, financing, borrow, and dividend cash flows belong in the same ledger. A separately labeled downside-penalty ablation may change training utility; evaluation always reports unshaped cash P&L. This penalty is not a claim to optimize Sharpe exactly.

Each hypothetical ticket has its own financing account; reserve capital remains in a cash benchmark. Ticket P&L above that benchmark can be combined into an account NAV without inventing a surface payoff. The implementation must reconcile that accounting numerically.

## Historical learning and timing

Build action-conditioned transitions for both directions and every feasible entry date. A held-contract trajectory supplies the hold transition and the counterfactual cost of exiting at each following close. Different entries are dependent observations of the same market history, not independent extra samples.

Fit only on training transitions whose next observation is inside the training period. Fit transforms on training only. Select a small fixed grid on a later validation block; freeze the policy before test. Restart books flat at split boundaries and charge all final liquidation costs. Reserve at least several years for training when contract data begin in 2010.

An entry rejected at the next session is a genuine no-fill transition. A missing/invalid held price is different: it is an unresolved outcome, not zero P&L, an ex-ante action mask, or permission to delete a losing trade. Record censored training transitions and halt any evaluation that encounters an unresolved held position, retaining the partial ledger.

## Data and validation gates

The repository described a large licensed contract cache that was absent from this local checkout. A new pair-specific AMD/NVDA download has now been completed using existing WRDS authentication. Passwords were not written to scripts or logs. Licensed rows remain in ignored data directories with bounded request/checksum manifests.

Required data: actual call quotes and Greeks, contract identity/adjustment fields, underlying closes and adjustment factors, rates, and distribution announcements. The current cache format supports calls only; do not label the strategy a straddle.

Tests must cover next-session fills, unchanged contract identity, integer quantities, spread/commission round trips, delta hedge offset, financing/NAV reconciliation, future-data invariance of current decisions, terminal costs, no cross-split targets, legal actions, and unresolved held quotes.

The initial engine does not provide broker margin, depth/queue fills, contract participation limits, locate availability, or arbitrary American-option assignment. Do not interpret EOD bid/ask simulation as a verified executable return. Adjusted contracts require an explicit supported transformation or an unresolved-position failure, never a guessed mark.

## Research basis

- OptionMetrics describes actual EOD option quotes, IVs and Greeks: [IvyDB US](https://optionmetrics.com/united-states/).
- Batch fitted Q iteration is a standard way to learn from historical transition tuples with regression: [Ernst, Geurts and Wehenkel (2005)](https://www.jmlr.org/papers/v6/ernst05a.html). Its use here is a design choice, not evidence of profitable option trading.
- Short equity options can be assigned early, particularly around dividends; contract adjustments change deliverables: [OIC assignment reference](https://www.optionseducation.org/referencelibrary/faq/options-assignment) and [corporate-action reference](https://www.optionseducation.org/referencelibrary/faq/splits-mergers-spinoffs-bankruptcies).
