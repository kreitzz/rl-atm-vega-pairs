# Presentation Script (Slides 15–18) — Sequential Q Extension

A separate, more ambitious extension, presented after the bandit results: an agent that manages an open trade over time instead of re-deciding fresh every day. Closes with future work and thank-you. Same terse, word-for-word style as the earlier scripts.

**Estimated duration: ~1.7–2.0 minutes** (256 words, at 130–150 words/minute).

---

## Slide 15 — What the agent sees, and what it can do

This is a separate extension: instead of re-deciding fresh every day, this agent manages an open trade over time, so it has to remember more. State is the same tree bucket as the bandit, plus, if a trade is already open, its direction, how long it's been held, and whether it's currently working out. Actions split the same way: from flat, stay out or enter long or short; while holding, keep holding or exit. Never both at once.

## Slide 16 — How the table gets filled in

The reward is simple: entering or exiting costs a small fee, holding earns or loses that day's P&L, staying flat costs nothing. Every historical day that shared the same state and action is pooled into one average reward, R-hat of s, a. The Q-value then adds the discounted value of the best next action: Q(s,a) equals R-hat(s,a) plus gamma times the max over next actions. That equation is swept across the whole table repeatedly until nothing moves, solved directly, not nudged forward one experience at a time, and still walk-forward safe: training only ever sees data from before validation and test.

## Slide 17 — Future Work

A few natural next steps. On the formation side, an EMRT-style search over relative option notional, instead of a fixed one-to-one vega-neutral pairing. On the Q side, jointly forecasting both legs' realized variance and testing a risk-adjusted reward, these stay proposals, separate from our main bandit results. And more broadly: extending daily marked P&L to all four pairs, and exploring a joint capital-allocation policy across pairs instead of fixed independent budgets.

## Slide 18 — Thank you

Thank you, happy to take questions.

---

### Duration note

- Word count: 256 words (slides 15–18 combined).
- At 130–150 words/minute: **1.7–1.97 minutes**, midpoint ~1.8 min at 140 wpm.
- Combined with the paper script (2.8 min) and the bandit script (3.1 min), full talk runs roughly **7.8 minutes** at 140 wpm.
