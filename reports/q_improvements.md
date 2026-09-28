# Testing the diagnostic fixes on the AMD/NVDA Q-learner

These tests start from the Q-learner that sees whether its open trade is winning (Q+P&L), with spread inputs: Sharpe 0.24, 2010–2024. They try the five fixes suggested in [q_diagnostics.md](q_diagnostics.md) one at a time. Data, costs and folds match [q_learning.md](q_learning.md).

**The rule for keeping a change:** add one change at a time on top of the changes kept so far, and keep it only if it raises the average validation Sharpe across the 8 folds. Test results were never used to decide. They are shown for information only.

## Result

None of the fixes improves the Q-learner, and the stepwise procedure ends below the starting point on test. Every variant lands between −0.15 and 0.24 Sharpe, well within each one's ±0.45 uncertainty.

| Step | Change tested | Validation Sharpe | Kept? | Test Sharpe (95% CI) | Trades | Max DD |
| --- | --- | ---: | --- | --- | ---: | ---: |
| 0 | Baseline (Q+P&L, 5 seeds, grid) | 0.40 | — | 0.24 (−0.12, 0.66) | 229 | −120 |
| 1 | Stop-loss at 0.5 × training payoff sd | 0.31 | No | 0.07 (−0.34, 0.48) | 345 | −159 |
| 2 | 20 seeds instead of 5 | 0.37 | No | −0.15 (−0.60, 0.31) | 299 | −251 |
| 3 | Fewer states (drop the early/late split) | 0.67 | **Yes** | 0.05 (−0.36, 0.53) | 227 | −228 |
| 4 | + fixed settings (8 leaves, min leaf 252, γ = 0.9) | −0.17 | No* | 0.06 (−0.39, 0.42) | 311 | −215 |
| 5 | + 5-day minimum hold, 1.5× entry cost in training | 0.36 | No | 0.15 (−0.32, 0.67) | 155 | −209 |

*Fixed settings can't win on this rule: choosing the best of 8 settings on validation always inflates the grid's validation score. On test it makes no difference (0.06 vs 0.05).

Extra reference runs, not part of the procedure:
- 20 seeds + fewer states: validation 0.60, test 0.05
- Minimum hold alone: validation 0.61, test −0.13

## What this shows

1. **The baseline's 0.24 was partly luck.** Averaging 20 seeds instead of 5 is the more reliable version of the same model, and it scores −0.15. The original 5 seeds happened to land near the top of the range.
2. **Validation doesn't predict test for this model.** The change with the best validation score (fewer states, 0.67) did worse on test than the baseline (0.40 validation). Two-year validation windows are too short to separate real improvements from noise here.
3. **The stop-loss didn't fix the losing trades held to expiry.** Stopping out at a fixed loss added about 50% more trades, and costs grew with them. Many trades that were stopped out would have recovered.
4. **Trading less helps a little, but not enough.** The minimum hold cut trades by about a third (227 → 155) and raised test Sharpe from 0.05 to 0.15, but it lowered validation Sharpe.

## Conclusion

The AMD/NVDA Q-learner is at its noise floor. Its direction calls are better than random (see [q_diagnostics.md](q_diagnostics.md)), but after costs its daily P&L is too noisy for these structural tweaks to show a reliable gain. Its test Sharpe is between about −0.15 and 0.24 depending on seeds and settings, which is not reliably different from zero, or from the current tree + cross-entropy model (0.25).

Further gains are more likely to come from the signal and cost side than from the learner. Options:
- stronger or cleaner inputs
- far fewer, higher-conviction trades
- a longer history for validation

Note: the test period has now been viewed many times across these experiments, so none of these numbers is a clean out-of-sample estimate.

## Reproduce

```sh
python -m scripts.run_q_improvements --add baseline
python -m scripts.run_q_improvements --add stop_loss
python -m scripts.run_q_improvements --add seeds20
python -m scripts.run_q_improvements --add fewer_states
python -m scripts.run_q_improvements --add fixed_config --keep fewer_states
python -m scripts.run_q_improvements --add min_hold --keep fewer_states
```

Results go to `artifacts/q_improvements/AMD_NVDA/`. The summaries are in [q_improvements/](q_improvements/).
