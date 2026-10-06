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
