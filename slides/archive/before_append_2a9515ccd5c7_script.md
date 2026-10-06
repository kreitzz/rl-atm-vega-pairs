# Presentation script: equity pairs to options relative value

Updated research deck. Slides 1–20 form the main narrative; slides 21–22 are appendix and sources. Press N in the HTML deck to show notes.

## Slide 1 — From equity pairs to options relative value

The project begins with an equity pairs paper. Our extension asks whether learned decisions improve volatility relative value. The headline is a selected historical cash-excess Sharpe, not a promised live result.

## Slide 2 — Ning & Lee: formation and trading are separate problems

EMRT measures how quickly significant historical deviations return to the mean. The paper then learns trading decisions instead of fixing a threshold. We borrow that separation, rather than replicate its formation rule.

## Slide 3 — What the Q-learner learns

Explain state, action and reward briefly. The paper uses simulated paths to learn because its reward needs the true mean. Its definition of four move bins with a four-step lookback implies 256 combinations; a later statement of 16 states needs clarification.

## Slide 4 — The paper reports higher daily Sharpe in all ten pairs

These are arithmetic means of the published daily Sharpes, not annualized account results. Both formation and trading change, so the table does not isolate RL from EMRT.

## Slide 5 — What we need to test in the volatility extension

The point is not that the paper is wrong. It gives us a useful framework, but options demand a different payoff and risk representation. This motivates our actual-contract simulator and fixed-size controls.

## Slide 6 — ATM proxy signal did not transfer unchanged

The old 1.461 proxy number remains historical evidence, not our current performance. Silver dominates that basket and the full-size Sharpe policy was stronger than sizing there. The current research found a more useful signal in downside puts. These rows are not controlled ablations of the same strategy.

## Slide 7 — Broad research universe; sparse eligible trades

The universe was expanded substantially beyond AMD and NVDA. But the retained entry rule is selective and formation calendars are retrospective. Full proposal coverage means every qualified state is kept, not that every option pair trades.

## Slide 8 — Approximately 30-day, 25-delta puts

Contracts are selected at the previous close and checked on the next session. Execution deltas can drift. Vega neutrality does not equal gamma or jump neutrality, and vega is not continually rebalanced.

## Slide 9 — Account for actual positions and costs

We use midpoint plus a penalty rather than full bid-ask crossing. The option penalty is charged on entry for expiry-held positions, not daily. Stricter training costs and execution costs differ. The risk limits are inherited research assumptions.

## Slide 10 — Keep relative value separate from size allocation

The learner adjusts size; it does not invent the signal or reverse its direction. There is no learned skip action in the successful retained candidate. Risk and quote checks can still reject a proposed entry.

## Slide 11 — Relative expected variance premium improves the state

These are backward observable forecast proxies, not future RV. Missing features are neutral-filled and flagged. We have not separately proven this forecast dominates HAR or GARCH.

## Slide 12 — A contextual sizing policy optimized for daily Sharpe

Use precise terminology. In our project this sits under the RL research direction, but the retained model is contextual tree plus CEM. It has no within-trade Bellman updates. State-dependent sizing already changes portfolio admission and overlap.

## Slide 13 — Sizing benefit appears before and after 2018

The first January 2018 fit uses 74 completed examples from July 2016 onward, ending in December 2017. Training improves from 1.573 to 2.015. The subsequent monthly walk-forward result at ten million improves from .769 to 1.003. Training gains are expected under direct optimization. The stronger evidence is walk-forward improvement; it is researched history, not an untouched test.

## Slide 14 — Learned sizing improves the fixed-vega benchmarks

All three see the same qualified proposals and contracts. Their executed portfolios differ because integer sizes and risk admission change overlapping inventory. At ten million, the learner earns about 323 thousand dollars above cash and cash-relative CAGR is .342 percent.

## Slide 15 — Profits vary sharply across regimes

2022 is the strongest year. 2019, 2021 and 2024 have negative excess Sharpe. Total funded returns can still be positive because cash earns interest. 2025 is partial. The annual table is in the appendix and the paper.

## Slide 16 — Convexity creates both major wins and major losses

About 55 percent of total net excess dollars arrive in 2022. Long SNAP puts benefit from the guidance shock; short AAL puts are hurt by a different guidance shock in 2024. UPST losses include expensive long carry. Several cohorts share events, so entries are not independent observations. Do not turn hindsight winners and losers into name bans.

## Slide 17 — More signals did not generally mean higher Sharpe

Variance-premium features improve .962 to 1.006. The other added blocks do not improve the prior learned baseline. This includes many alternative-data and smile ideas. The chart preserves negative findings. The small improvement over prior learned sizing is statistically uncertain.

## Slide 18 — Simple scaling beats the first capital-aware pilot

At ten million, average reserve is about two percent and peak eighteen percent. Doubling size and per-pair allowance roughly doubles cash-relative CAGR while keeping about one Sharpe. Four times size lowers Sharpe. The capital-aware overlay gets .854; this is a limited pilot, not proof that RL cannot manage portfolios.

## Slide 19 — One Sharpe is a research finding, not a live guarantee

Bootstrap intervals refer to the original research reference, not newly calculated ten-million intervals. The same-state profit-only policy is close at .979. We should claim adaptive sizing value, not universal RL superiority.

## Slide 20 — Preserve the signal; test sequential exits separately

The narrow scope is a strength: identify the economic signal first, then test whether a richer control policy adds value. Keep costs and executable marks common. Do not widen instruments, features and actions all at once.

## Slide 21 — Returns and Sharpe by year at $10m

Use this appendix for detailed return questions. Funded return, above-cash percentage points and excess Sharpe are distinct. A positive funded year can underperform cash.

## Slide 22 — Sources and reproducibility

The paper records exact strategy assumptions and the source files for every figure. Data ends in August 2025, even though this draft is dated October 2026. All source data remain subject to existing licenses.
