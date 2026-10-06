# Presentation Script (Slides 1–7)

Terse version, read word-for-word. Trimmed from the full version to leave time for the "our extension" slides (8+).

**Estimated duration: ~2.6–3 minutes** (394 words, at a presentation pace of 130–150 words/minute).

---

## Slide 1 — Title

This project is based on Ning and Lee's 2024 paper, *Advanced Statistical Arbitrage with Reinforcement Learning*, which replaces two hand-tuned pieces of classic pairs trading with learned ones. We'll cover the paper first, then our own extension to vega-neutral options pairs.

## Slide 2 — The Paper

Pairs trading has two questions: which spread to trade, and when to trade it. Traditionally both are fixed, hand-tuned rules — the Distance Method fixes the hedge ratio at one, OU fits it by maximum likelihood, and you trade at a fixed threshold. This paper learns both instead: EMRT picks the hedge ratio, Q-learning picks the timing. Tested on ten US pairs, formed in 2022, traded in 2023, against those two benchmarks.

## Slide 3 — EMRT

The spread is X equals S1 minus B times S2 — different B gives a different reversion speed. EMRT times how long the spread takes to snap back after a big swing, at least two standard deviations from the mean, and averages those times. Lower EMRT means faster reversion, and B is chosen to minimize it.

## Slide 4 — MDP

Now, when to trade. The state is the last four daily moves of the spread, bucketed as big or small, up or down — just the shape of recent momentum, not the price level. The action is buy, sell, or hold one unit. The reward pays for trading toward the mean, and charges a small cost per trade.

## Slide 5 — Q-learning

Q-learning learns one number per state-action pair — the expected future payoff — and updates it a little after every trade, using its own next-state estimate. It's trained with a learning rate of 0.1, a discount factor of 0.99, and epsilon-greedy exploration that fades to zero at test time.

## Slide 6 — Why simulate

One catch: the reward needs the spread's true mean, which you never know in real markets. So they train on 10,000 simulated years with a known mean, then trade the real 2023 data with that trained table. Only training is synthetic — every result you'll see is on real data.

## Slide 7 — Results

These are the paper's actual published tables — ten pairs, comparing Distance Method, OU, and RL on daily Sharpe and other metrics. RL wins on Sharpe in all ten pairs — mean Sharpe 0.024, 0.047, and 0.100 for DM, OU, and RL. Best or tied-best drawdown in nine of ten.

---

### Duration note

- Word count: 394 words (slides 1–7 combined).
- At 130–150 words/minute: **2.6–3.0 minutes**, midpoint ~2.8 min at 140 wpm.
- Down from the full version's ~9.5 minutes — frees up roughly 6–7 minutes of a 10-minute talk for your own extension (slides 8+).
- What got cut: the full Q-learning derivation (one-step reward → G_t → bootstrapping) down to one sentence; the EMRT stopwatch walkthrough down to one sentence; the two MDP design-choice asides merged into a single clause; the Results slide's explanation of negative Sharpe values and the exact CSX-NSC tie-break numbers (both still visible on the slide/footnote for Q&A, just not spoken).


---

# Appended contract-level research

## Slide 24 — From the ATM proxy to actual option contracts

All original slides are preserved. Their proxy metrics describe the earlier experiment. The appended section reports the later contract-level work, with a different accounting convention. Do not compare the proxy and funded Sharpes as a controlled ablation.

## Slide 25 — Trade the contracts; measure returns against cash

The hedge, option positions, dividends and financing are reconciled in the actual ledger. Cash interest can make funded returns positive even in a negative excess-return year. Vega neutrality does not remove gamma, jump or changing volatility-factor exposure. Training commissions remain stricter at 65 cents; actual replay commissions are 3 cents.

## Slide 26 — Separate the trading signal from the allocation decision

The short strategy retains 345 candidate pair sleeves, 608 qualified proposals, and 33 model inputs. Its monthly fits use completed prior outcomes. The one-year study uses the same researched roster, but does not copy short-dated entry dates, directions, sizes or gates. Pair calendars and the research universe are retrospective.

## Slide 27 — Learn size within a focused strategy

The successful short-dated method is a contextual tree plus CEM optimized for portfolio Sharpe, not a neural sequential Q learner. The new native one-year method is a small-state contextual episodic direct policy optimization. The six-month fitted-Q experiment uses Bellman updates and is sequential. Claim adaptive policy value where the matched benchmark improves, rather than treating every method as the same RL algorithm.

## Slide 28 — Adaptive sizing improves the fixed-vega controls

The learned hold-to-expiry policy improves Sharpe from 0.769 to 1.003. In the initial 74-example pre-2018 training fit, fixed-full and learned Sharpes are 1.573 and 2.015. Training uses aggregated paths and stricter costs, so compare policies within each period; do not compare absolute train and funded-test levels. Sizing changes admission and overlapping inventory.

## Slide 29 — Exit decisions help some maturities more than others

These are separate within-study controls, not a single common entry policy across maturities. The 30-day managed checkpoint reaches 1.029. The locked 90-day management improves a negative hold strategy to about zero. Six-month conservative fitted Q improves .592 to .610; the 95% interval for that improvement includes zero. Aggressive management variants performed worse and remain in the reports.

## Slide 30 — A native long-horizon signal supports a small sizing gain

The independent study includes all 47,610 pair/month states and 26,910 evaluation states, with 763 eligible opportunities. It uses a quarterly pooled ridge variance forecast trained only on completed 252-session targets. Learning direction as well as size failed; freezing forecast direction and learning size was better. This replaces the legacy one-year component in the blend. The improvement is exploratory and statistically uncertain.

## Slide 31 — The downside-put signal does not transfer uniformly

These figures belong to the earlier smile-sizing report and its original funding and eligibility conventions. They are exploratory strategy comparisons, not common-universe strike ablations and not the current $10m maturity checkpoints. A 25-delta put is the downside wing; its comparable-strike call is approximately 75 delta. Higher-delta puts and lower-delta calls describe the opposite wing. Preserve the negative findings without claiming universal call/put asymmetry.

## Slide 32 — Profits and Sharpe vary materially across years

Do not suppress losing years. The 30-day managed policy is negative in 2019 and 2024 on a cash-excess basis; its 2022 gains are large. The native one-year policy has different weak years. These are separate researched strategies, so differences across columns are not a controlled maturity-only effect. Return percentage points above cash distinguish the trade signal from idle-cash yield.

## Slide 33 — More features usually failed to improve the retained signal

Only the expected-variance-premium addition improves the prior learned state in this experiment. Term structure, smile, volume/open-interest, employee sentiment and transcripts generally do not add Sharpe to this checkpoint. The .961745 to 1.006305 gain has an interval that includes zero. Do not confuse the feature study funding reference with the later $10m strategy, whose retained sizing Sharpe is 1.003.

## Slide 34 — Past-performance weighting trails simple equal weights

Positive-Sharpe weights normalize each sleeve’s positive trailing Sharpe and assign zero to nonpositive ones; all-nonpositive fits use equal weights. Quarter-start returns are excluded from all allocation fits, verified with future perturbation and truncation checks. These are blends of saved after-cost returns; shared positions, joint risk admission and additional reallocation costs are not replayed. Component selection is still retrospective.

## Slide 35 — Adaptive decisions help within specific strategies

The report should not claim that RL always wins, that all smile regions have the same signal, or that the contract strategy has a newly matched neural-network comparison. Price and Greek quality, American assignment, execution impact, jump risk, margin and retrospective research selection remain limitations. Vendor data remain subject to their existing licenses.

## Slide 36 — Keep the original study and the extension together

The paper is the previous short-dated write-up; this append does not silently rewrite it with the later long-tenor results. Source hashes record every numeric report used. No fitting or backtests were rerun to prepare these slides. The original proxy deck remains historical context, and its metric must not be conflated with the later funded contract accounting.

