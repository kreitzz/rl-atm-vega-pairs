# Tracking the 30-day IV index with real calls: data and rule-based baselines

Steps 1–2 of the tracking plan. The objective is **tracking, not P&L**: hold real listed calls whose implied vol moves with the index, and pay as little as possible to keep them there.

## What is being tracked

The index is OptionMetrics' volatility-surface point for each stock: **30 days, 50 delta, call** (`optionm.vsurfd`, `data/wrds_q/iv.csv`). It is a smoothed surface value, not a contract. The instruments are the real AMD and NVDA calls already downloaded in `data/q_contracts/AMD_NVDA` (all strikes, expiries up to 60 days, 2010–Aug 2025).

## How tracking and cost are measured

Each day the book holds contracts `n_i`. Holdings are frozen overnight, and

```
tracking_error[t+1] = Σ n_i · vega_i[t] · (IV_i[t+1] − IV_i[t]) / V  −  (Index[t+1] − Index[t])
```

`vega_i` is $ per vol point per contract; `V` is the target exposure ($10,000 per vol point per leg). The first term is the vol move the book is actually exposed to; the second is the index move. When the book rebalances, it trades at the day's quotes (buy at ask, sell at bid) plus $0.65 per contract:

```
cost[t] = Σ |change in n_i| · (half_spread_i + commission) / V
```

Both are in **vol points**, so cost and tracking error are directly comparable. At $10,000 of vega, 1 vol point = $10,000.

## Baselines

| Rule | Holds |
| --- | --- |
| `single` | One call nearest 30 days / 50 delta (what the existing contract engine holds). |
| `interp` | The two expiries around 30 days and, in each, the two strikes around 50 delta, weighted to land exactly on (30 days, 50 delta). Up to four contracts. This roughly mimics how the surface point itself is built. |
| `_monthly` | The same, using only standard monthly expiries. |

A **no-trade band** controls turnover. The book is rebalanced only when its vega-weighted maturity, delta or total vega drifts too far from (30 days, 0.50, target vega). For example, `10d_15` means ±10 days, ±0.15 delta, ±30% vega. "Daily" means no band. A held contract is always rolled before 7 days to expiry or if its quote disappears.

## Results, 2023–Aug 2025 (frozen, nothing fitted)

Per leg (AMD / NVDA), then the long-AMD / short-NVDA spread book:

| Rule | Daily TE (vol pts, RMSE) | Median abs daily TE | R² vs index moves | Cost per year (vol pts) | Days traded |
| --- | ---: | ---: | ---: | ---: | ---: |
| interp, daily | 0.96 / 0.90 | 0.31 / 0.42 | 0.86 / 0.92 | 169 / 144 | 100% |
| interp, band 3d / 0.05 | 0.97 / 0.92 | 0.32 / 0.43 | 0.86 / 0.92 | 130 / 108 | 61% |
| interp, band 10d / 0.15 | 1.05 / 1.01 | 0.37 / 0.43 | 0.84 / 0.90 | 52 / 41 | 21% |
| interp, band 15d / 0.20 | 1.09 / 1.10 | 0.43 / 0.48 | 0.82 / 0.88 | 34 / 31 | 15% |
| single, engine-like | 1.07 / 1.01 | 0.36 / 0.45 | 0.83 / 0.90 | 54 / 40 | 22% |
| single, monthly, wide | 1.36 / 1.23 | 0.43 / 0.50 | 0.72 / 0.85 | 25 / 23 | 15% |
| **Spread book** interp, band 10d / 0.15 | 1.03 | 0.58 | 0.84 | 93 | 36% |

Full results for every rule, book and period (2010–2020, 2021–22, 2010–22, 2023–25) are in `tracking/metrics.csv`.

## What this shows

1. **The tracking-vs-cost frontier is flat.** Going from daily rebalancing to a 10-day / 0.15-delta band cuts cost about **3.5×** (169 → 52 vol pts per year on AMD) while daily tracking error rises only from 0.96 to 1.05. Most of the tracking error is not caused by drifting away from 30 days / 50 delta, so rebalancing more often buys little.
2. **Four-contract interpolation barely beats one contract.** At similar cost, `interp` and `single` track almost identically. Landing exactly on (30d, 50Δ) is not where the error comes from.
3. **Where the error does come from** (these are the targets for the ML model):
   - **Stock moves (skew / sticky strike).** The index is fixed-delta; a held contract is fixed-strike. When the stock moves, the book's IV moves along the skew and the index doesn't. Daily tracking error correlates **0.35–0.51** with the stock's own return (AMD 2023–25: 0.35; NVDA 0.43; NVDA 2017–22: 0.51). This is predictable from information you have.
   - **Ageing drift.** Held contracts' IVs rise relative to the constant-30-day index by about **+15 to +30 vol pts per year per leg**. In the spread book this largely cancels (+1 to +5 per year).
   - **Surface smoothing and quote noise.** OptionMetrics' index is a smoothed fit; individual contract IVs are noisy. Some of this is irreducible.
4. **Costs are large relative to tracking error.** The median half-spread on a 30-day ATM call is about **0.4 vol points** (2023–25), so each full roll costs roughly 0.8 vol points. Even the cheapest reasonable rule spends 25–35 vol pts per year per leg. These are end-of-day quoted spreads; patient execution would pay less.
5. **Pre-2017 AMD is a different regime.** AMD traded at $2–10, with half-spreads of 1–5 vol points and very coarse strikes. Tracking there is poor (daily TE of roughly 3–8 vol points in 2012–2016) and costs are several times higher. Any model fitted on 2010–2016 AMD will be dominated by that regime.
6. **Over a month, tracking is loose.** The 21-day summed tracking error is 7–9 vol points against index 21-day moves of 12–15, mainly from the stock-move and ageing effects in item 3.

## What the ML step should aim at

Beat this frontier: lower tracking error at the same cost, or the same tracking at lower cost. The obvious inputs are the stock's move (skew exposure), the term-structure slope (ageing drift), each contract's spread in vol points and whether earnings fall before expiry. The obvious decision is **when** to rebalance and **which** contracts, rather than landing exactly on (30d, 50Δ) each day.

## Reproduce

```powershell
python -m scripts.tracking_data --pair AMD_NVDA
python -m scripts.run_tracking
python -m unittest tests.test_tracking
```

`tracking_data` writes the cleaned chain and index to `artifacts/tracking/`. `run_tracking` writes the per-day books to `artifacts/tracking/daily.csv` and the metrics to `reports/tracking/`. It takes about 2 minutes on 2 cores.

## Limits

- End-of-day quotes: trades fill at the day's closing bid/ask, with no depth or intraday timing.
- Holdings are continuous (fractional contracts). At $10,000 of vega per leg this is hundreds of contracts, so rounding is minor.
- A held contract that loses its quote is closed at its last seen half-spread and that day is excluded from tracking error. This happened on 4 or fewer days per leg in 2023–25.
- Calls only, because only calls were downloaded. This matches the index, which is a call-surface point.
