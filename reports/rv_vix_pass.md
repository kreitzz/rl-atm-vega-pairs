# Trailing realized volatility and VIX state

The state now contains six inputs: AMD–NVDA 30-day IV spread, days until earnings for AMD and NVDA, trailing 30-calendar-day realized volatility for each stock, and the previous available VIX close.

## Results

All feature sets below use the same tree-state policy model, splits, dates, costs, and candidate budget. Sharpes are annualized 30-day net payoff ratios using `sqrt(365/30)`, including flat entry dates.

| Available features | Validation Sharpe | Test Sharpe | Test dates traded |
| --- | ---: | ---: | ---: |
| IV spread | 0.602 | 0.743 | 42.1% |
| IV spread + earnings | 0.812 | 0.195 | 5.4% |
| IV spread + earnings + trailing RV | 0.804 | 0.137 | 61.2% |
| IV spread + earnings + trailing RV + VIX | 0.804 | 0.137 | 61.2% |
| Short higher IV / long lower IV | — | 0.956 | 100% |

The full-input model has mean net payoff of 0.356 vol points per entry opportunity and a 95% block-bootstrap annualized Sharpe interval of **[-0.431, 0.693]**. Its selected tree uses both trailing RV inputs but no VIX split, producing identical actions to the RV model without VIX. All five selected-model seeds give identical test actions.

Validation would select the earnings-only feature set among these learned models. We report the full-input run because it is the requested experiment; no feature set or configuration was selected using test performance. The test period has been viewed repeatedly and these comparisons are exploratory.

## State construction

- Trailing RV: `100 * sqrt(252 * mean(log(1 + retx)^2))` over `(entry - 30 calendar days, entry]`, using CRSP split-adjusted ex-distribution daily returns. At least 15 observed returns and a full 30-calendar-day history are required; windows with missing returns are excluded. Future returns do not enter this feature.
- VIX: the close from the strictly preceding available session, with a maximum seven-calendar-day gap. This avoids assuming that the finalized entry-day VIX close was available at the option-entry close. Downloaded from [Cboe's official historical data](https://www.cboe.com/tradable-products/vix/vix-historical-data); the exact CSV hash is recorded in local `summary.json`.
- Earnings: actual historical countdowns under the agreed known-calendar assumption. Countdown values are capped at 31 days, so an exact date beyond the trading horizon does not affect the state. Within the horizon, the tree can use exact days rather than only an earnings-in-window flag. Same-day timing remains unknown.
- IV: the original 30-day ATM-proxy spread. No future RV, earnings return, or financial result is an input.

Training drops January 2002's 21 entries because the local return history starts that month. Every feature ablation uses this same reduced panel: training February 1, 2002–December 1, 2016 (3,735 entries), validation January 3, 2017–December 1, 2020 (969), and test January 4, 2021–August 29, 2025 (1,170). Outcome-boundary purging remains in place.

## Model

The [earnings audit](earnings_audit.md) found that independent table cells had as few as 13 training rows in the previously selected model. To share observations across inputs, this run uses a regression tree to partition states, then cross-entropy policy search to choose long AMD, short AMD, or flat for each leaf, maximizing annualized Sharpe after the 0.5-vol-point round-trip pair cost.

The tree partition is supervised: it predicts the long-AMD gross payoff using training data only. The final leaf actions are optimized against net Sharpe. This is a hybrid fitted-state/direct-policy-search model, not a neural RL agent. Candidate trees have at most 4, 8, or 16 leaves, each with at least 126 or 252 training rows. Five policy-search seeds per candidate use the same settings as before. Training Sharpe selects the seed; validation Sharpe selects tree size and minimum leaf size. The full-input run selects 8 leaves and a 126-row minimum.

These representation changes mean comparisons against the old 0.392 earnings-table result mix model and feature changes. The table above is the matched comparison. Larger state inputs can still overfit or fail across regimes; more features do not guarantee a better fitted policy. No further tuning on the observed test results was performed.

## Reproduce

```sh
curl -fL https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv -o data/raw/VIX_History.csv
python scripts/run_rv_vix_pass.py --source-root /path/to/convert-rv-rl/data/raw/wrds --cost 0.5
python -m unittest discover -s tests -v
```

Create `data/raw/` first if needed and install `requirements.txt`. The runner outputs feature data, predictions, candidate scores, tree rules, leaf policies, and settings to ignored `artifacts/rv_vix_pass/`. Ten tests pass, including trailing-window causality, prior-close VIX alignment, minimum leaf size, annualization, costs, and outcome separation. Market data remains local.
