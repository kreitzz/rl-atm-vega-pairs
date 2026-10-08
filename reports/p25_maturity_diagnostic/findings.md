# Maturity diagnostic for the selected P25 policy

Descriptive inspection of saved trades for the 27-input policy associated with 1.263 excess Sharpe. No retraining, new order simulation, maturity restriction, or policy change.

## Maturity and holding periods

- All 608 proposals: actual intended-entry DTE ranges 21–45 days. Signal-date DTE is longer by the signal-to-entry calendar lag; it is not the holding period.
- All 90 saved tree training windows were reconstructed. Their union contains 845 distinct pair/date examples, with intended-entry leg DTE 21–45 days. Training examples overlap between monthly windows.
- 82 / 608 proposals have different leg expirations (13.5%); largest expiry gap 21 days.
- Selected policy: 207 executed pairs; leg entry DTE 21–45; actual leg holding 21–45 calendar days.
- 27 / 207 executed pairs have different expirations. The longest single-leg tail is 21 days; delta hedging continues but initial two-leg vega matching no longer applies once one leg settles.
- Every saved fill in the two arms is an entry or expiry settlement. Calendar expiry can be up to a few days after the final exchange-session settlement.

## Selected policy: 2022–August 2025

| Mean entry DTE | New pairs | Calendar net before shared financing | New-entry lifecycle win fraction | Mean chosen size |
|---|---:|---:|---:|---:|
| 21-27 | 14 | $63,462 | 71.4% | 71.4% |
| 28-34 | 77 | $117,420 | 64.9% | 69.8% |
| 35-45 | 19 | $30,381 | 63.2% | 68.4% |

Calendar P&L includes carried positions from before a period boundary. Lifecycle statistics describe new entries and can include their subsequent runoff. Shared cash financing is separate, not allocated arbitrarily to maturity groups. Stock execution and borrow costs use the existing absolute-hedge-change / short-hedge allocation. Different groups have different names, entry dates, sizes, and overlapping positions; this is not a controlled maturity experiment or a per-band Sharpe comparison.

## Entry gamma and theta

European Black–Scholes Greeks are computed from the saved entry quote, actual expiry, spot, rate, IV and declared-dividend escrow, following the repo pricing convention. Theta holds dividend escrow fixed and is dollars per calendar day. It is an instantaneous model sensitivity, not realized decay or isolated theta P&L.

Gamma is expressed as approximate option P&L for a 1% move in its own underlying, divided by entry vega in dollars per volatility point. Leg gamma is positive; pair gross gamma adds magnitudes rather than assuming separate stock shocks cancel. Pair theta signs reflect long versus short. Matching vega does not equate these exposures.

Among selected executed legs, the median gamma-per-vega measure is 1.26 times as high in the 21–27D band as in the 35–45D band; median absolute theta per vega is 2.12 times as high. These groups also have different IVs and names, so these ratios are descriptive rather than isolated causal maturity effects.

## Interpretation

The precise two-week-versus-one-month example is not present in the saved proposal pool: minimum entry DTE is 21. The training-window range above checks the learner's actual examples separately. Nevertheless, duration varies and different leg expirations can leave a meaningful single-leg tail. Maturity pooling remains an assumption to investigate, not an established model failure.

Compare exposure_by_leg_maturity.csv and period_summary.csv before proposing a restriction. Outcomes are confounded by IV, pair identity, market period, sizes and admission. An eventual narrow-DTE or same-expiry strategy must be refitted and replayed as a new controlled experiment; filtering these outcomes would not reproduce that strategy.

## Reproduction and audit

`python -m scripts.diagnose_p25_maturity` uses bundled sources, one numerical thread, 20% duty cycle, a 384 MiB RSS cap and macOS-only 35% memory-headroom floor. It reuses the existing shared-trade attribution, with daily option/stock/cost/P&L reconciliation and separate financing.

- `proposals.csv` / `proposal_summary.csv`: all qualified proposals, actual-entry maturity and sizing/admission.
- `training_maturity.csv`: all 90 original tree training populations and their maturity ranges.
- `executed_legs.csv` / `executed_pairs.csv`: holdings, actual entry exposures and lifecycle attribution.
- `period_summary.csv`: selected and fixed-full descriptions by maturity and expiry match.
- `reconciliation.csv`: attributed P&L plus financing reconciles to the funded portfolio.
- `exposure_by_leg_maturity.csv`: leg exposure distributions.
- `maturity_and_exposure.png` / `outcomes_by_maturity.png`: diagnostic figures.
- `protocol.json`: verified source hashes, accounting conventions and resource usage. Tracked CSVs with Windows CRLF are also verified against the manifest after LF canonicalization; data values are unchanged.
