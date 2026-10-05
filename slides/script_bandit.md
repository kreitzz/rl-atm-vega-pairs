# Presentation Script (Slides 8–14) — Our Extension: Contextual Bandit

Continues directly from the paper script (slides 1–7), in the same terse, word-for-word style. Covers our main result, the vega-neutral IV-spread bandit, through its supervised-NN baseline. (Results I, Results II, Synthesis, and Honest Limitations are currently commented out of the deck, so this script skips them too.)

**Estimated duration: ~2.9–3.3 minutes** (435 words, at 130–150 words/minute).

---

## Slide 8 — The Bridge

We keep the paper's three-step skeleton, but move it to vega-neutral implied-vol spreads between two ATM straddles. State becomes regression-tree leaves instead of fixed return-direction buckets; action becomes a nine-level signed sizing ladder instead of buy, sell, or hold; and training becomes one-shot, a contextual bandit with cross-entropy search on real data, walk-forward, instead of sequential Q-learning on simulated paths. A sequential Q-learning variant is also explored later, as an alternative.

## Slide 9 — State: regression tree

Instead of four fixed return-direction buckets, we fit a small regression tree, on IV spread, both legs' earnings countdown, trailing volatility, and VIX, to predict the reward. Its leaves become the states, four to sixteen of them, chosen by validation Sharpe. Each leaf is then just one row in the bandit's policy table: one action to pick per state.

## Slide 10 — Action and reward

The reward is size times the forward relative-value payoff, minus a vega-scaled trading cost. Actions expand from buy, sell, or hold to a nine-level signed sizing ladder, zero up to full size, either direction. And because that reward is realized thirty days later, not estimated like the paper's theta, there's nothing to simulate: we train directly on real history.

## Slide 11 — Training: a contextual bandit

Every date is one independent decision, so there's no next state to bootstrap from, just cross-entropy search. Sample 256 full candidate policies, score each one's Sharpe across the whole training set at once, then reshape each state's action distribution toward whatever the top ten percent chose. Repeat eighty times. It has to be a search, not a per-state formula, because Sharpe's denominator, the policy's standard deviation, couples every state's choice to every other state's.

## Slide 12 — One search iteration, visually

This is one round of that search: 256 full policies sampled and sorted by Sharpe, the top ten percent kept as elites, and each state's action distribution nudged toward whatever those elites picked, from a near-uniform spread over nine actions per state, to a distribution sharpened around one dominant choice.

## Slide 13 — Train / validation / test

The paper fits once, on 2022, and trades a frozen table all through 2023. We instead use an expanding-window walk-forward: every fold re-fits the tree and re-runs the search, picks the best leaf-count by validation Sharpe, and scores that one frozen policy once on the next untouched slice before sliding forward, through August 2025.

## Slide 14 — The baseline: a small neural network

As a check on the bandit, we also built a small supervised baseline: the same seven features into one hidden layer of sixteen tanh units, trained to regress the forward reward directly by minimizing mean-squared error with weight decay, Sharpe never enters training. A threshold rule then turns that single number into a fixed plus-or-minus-one position, or flat, never the nine graded sizes the bandit gets to use.

---

### Duration note

- Word count: 435 words (slides 8–14 combined).
- At 130–150 words/minute: **2.9–3.3 minutes**, midpoint ~3.1 min at 140 wpm.
- Follows directly after the paper script (~2.8 min) — combined, roughly 5.9 minutes before the Q extension and closing.
