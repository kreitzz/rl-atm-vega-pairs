# Audit of the AMD/NVDA capital exhaustion

The reported 100% drawdown is a cumulative account loss across years of overlapping positions. The simulator keeps a fixed dollar-vega budget as NAV falls and has no margin gate or NAV-linked reduction in sizing. The earlier results should have emphasized this limitation: they combine signal performance, substantial execution costs and an inadequately constrained capital model.

For the halfway-fill RL run, the account starts at $1 million in January 2010. Just before failing on July 18, 2014, it has **$56,911 of equity**, **18 active cohorts**, $10,442 gross vega per volatility point and $10.14 million of gross stock hedges. Stock hedge notional is financing exposure; modeled net delta is approximately zero. The fixed $450 maximum new-entry vega per leg is still sized from initial capital.

An independent reconstruction from signed option contracts and daily Greeks verifies the closing stock hedge on **all 1,142 sessions** before failure. Maximum residual is below 0.000000001 share. This audit finds no delta-sign or 100-share multiplier error in that path. The accounting checks also reconcile cash and position values. These checks do not validate observed quotes, model Greeks, or hypothetical fill availability against an independent data vendor.

| July 18, 2014 | Stock move | Option midpoint P&L on prior holdings | Stock hedge P&L | Combined |
| --- | ---: | ---: | ---: | ---: |
| AMD: short calls, long stock hedge | -16.19% | +$781,904 | -$930,196 | -$148,292 |
| NVDA: long calls, short stock hedge | -4.46% | -$142,912 | +$195,893 | +$52,981 |
| Total before execution/financing | | | | -$95,311 |
| Execution and financing that day | | | | -$3,916 |
| Total daily account loss | | | | **-$99,227** |

The position was short AMD convexity and long NVDA convexity. Daily delta hedging offsets local first-order exposure; it does not cancel the unequal large moves of the two stocks. The last-day loss is 9.92% of initial capital but exceeds the small remaining equity, taking NAV to -$42,316.

Accumulated option execution costs through failure are **$1,052,811**, including **$606,193 of commissions**. Historical low dollar vega requires hundreds or thousands of contracts per entry. Those costs repeatedly erode capital while the absolute risk budget stays fixed. Cost attribution is not a separately simulated no-cost result.

A realistic next capital experiment needs NAV-linked sizing and explicit gross-risk/margin limits. The existing bankruptcy dates are outcomes of the stated fixed-dollar model, not evidence that a single delta-hedged position lost its entire initial capital, nor a clean isolation of the signal's quality.

[Machine-readable audit](selected_pair_backtest/half_spread/drawdown_audit.json) · [Annual account/cost breakdown through failure](selected_pair_backtest/half_spread/drawdown_annual.csv).

Reproduce with `python -m scripts.audit_drawdown`.
