"""Causal profitability memory, the paper's 27 inputs, and entry qualification.

Historical outcomes may be computed offline, but enter memory only after both
legs have completed. Overlapping candidate trades are not independent samples.
The historical roster/formation calendar is an inherited input, not discovered
by these functions.
"""
import numpy as np
import pandas as pd

CORE = ["iv_mean", "oriented_iv_gap", "rank_strength"]
OPTIONAL = ["oriented_vrp_gap", "rv_mean", "beta", "correlation", "dte_mean",
            "oriented_dte_gap", "commission_vol_points", "vix", "profit_91d",
            "count_91d", "profit_365d", "count_365d"]
FEATURES = CORE + OPTIONAL + [name + "_available" for name in OPTIONAL]


def profitability_memory(dates, ends, rewards, entered):
    """Directional mean net profit per initial vega for ends in [t-window, t).

    Columns in rewards/entered are short-A/long-B, then long-A/short-B.
    ends is the later leg's completion session, not the entry date. The strict
    upper boundary also excludes a trade completing on today's signal date.
    """
    dates, ends = pd.DatetimeIndex(dates), pd.DatetimeIndex(ends)
    rewards, entered = np.asarray(rewards), np.asarray(entered, dtype=bool)
    if rewards.shape != entered.shape or rewards.shape != (len(ends), 2):
        raise ValueError("Expected two directional outcomes per historical cohort")
    result = pd.DataFrame(index=dates)
    for column, side in enumerate(("short", "long")):
        valid = entered[:, column] & np.isfinite(rewards[:, column]) & ends.notna()
        order = np.argsort(ends.asi8[valid], kind="stable")
        finish = ends.asi8[valid][order]
        cumulative = np.r_[0., np.cumsum(rewards[valid, column][order])]
        right = np.searchsorted(finish, dates.asi8, side="left")
        for days in (91, 365):
            left = np.searchsorted(finish, dates.asi8 - pd.Timedelta(days=days).value,
                                   side="left")
            count = right - left
            result[f"profit_{side}_{days}d"] = np.divide(
                cumulative[right] - cumulative[left], count,
                out=np.zeros(len(dates)), where=count > 0)
            result[f"count_{side}_{days}d"] = count
    return result


def beta_screen(returns, calendar):
    """126-session A-on-B beta/correlation, lagged and fixed monthly."""
    returns = returns.reindex(calendar)
    a, b = returns.iloc[:, 0], returns.iloc[:, 1]
    stats = pd.DataFrame({
        "beta": a.rolling(126, min_periods=126).cov(b)
                / b.rolling(126, min_periods=126).var(),
        "correlation": a.rolling(126, min_periods=126).corr(b),
    }).shift(1)
    stats["formation"] = stats.index.to_period("M")
    first = stats.groupby("formation", sort=False).head(1).set_index("formation")
    result = first.reindex(stats.formation).set_axis(calendar)
    result["eligible"] = result.beta.between(.85, 1.15) & result.correlation.ge(.5)
    return result


def build_state(selection_a, selection_b, returns, screen, calendar, memory, vix):
    """Selected signal-close contracts in; directional inputs and flags out.

    vix must already contain strictly prior closes. Returns are daily simple
    returns; RV is 252 times the mean squared return, without mean subtraction.
    Signal construction uses the original $0.65 training-cost proxy.
    """
    a = selection_a.set_index("date").reindex(calendar)
    b = selection_b.set_index("date").reindex(calendar)
    ia, ib = a.bs_iv.where(a.bs_iv.gt(0)), b.bs_iv.where(b.bs_iv.gt(0))
    gap, mean = ia - ib, (ia + ib) / 2
    rank = gap.rolling(252, min_periods=63).rank(pct=True)
    direction = pd.Series(np.where(rank.le(.2), 1, np.where(rank.gt(.8), -1, 0)),
                          index=calendar)
    rv2 = returns.reindex(calendar).pow(2).rolling(20, min_periods=10).mean() * 252
    days_a = (pd.to_datetime(a.expiry) - calendar.to_series()).dt.days
    days_b = (pd.to_datetime(b.expiry) - calendar.to_series()).dt.days
    result = pd.DataFrame({
        "observed": np.isfinite(ia) & np.isfinite(ib),
        "eligible": screen.eligible.fillna(False).astype(bool),
        "direction": direction, "rank": rank, "iv_mean": mean,
        "oriented_iv_gap": direction * gap / mean,
        "rank_strength": ((rank - .5).abs() * 2).fillna(0),
        "oriented_vrp_gap": direction * ((ia**2 - rv2.iloc[:, 0])
                                - (ib**2 - rv2.iloc[:, 1])) / mean**2,
        "rv_mean": np.sqrt(rv2).mean(axis=1, skipna=False),
        "beta": screen.beta, "correlation": screen.correlation,
        "dte_mean": (days_a + days_b) / 2,
        "oriented_dte_gap": direction * (days_a - days_b),
        "commission_vol_points": .65 / (100 * a.bs_vega) + .65 / (100 * b.bs_vega),
        "vix": vix.reindex(calendar) / 100,
    }, index=calendar)
    for days in (91, 365):
        count = np.where(direction.gt(0), memory[f"count_long_{days}d"],
                         memory[f"count_short_{days}d"])
        profit = np.where(direction.gt(0), memory[f"profit_long_{days}d"],
                          memory[f"profit_short_{days}d"]) / 100
        result[f"count_{days}d"] = count
        result[f"profit_{days}d"] = np.where(count > 0, profit, np.nan)
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    for name in OPTIONAL:
        result[name + "_available"] = np.isfinite(result[name]).astype(float)
        result[name] = result[name].fillna(0.)
    return result


def qualify(rows):
    """The common entry screen, before sizing and funded risk admission.

    context is the inherited formation/quote-coverage calendar, when supplied.
    Neither eventual profit nor eventual quote coverage can qualify today's row.
    """
    return (rows.observed & rows.eligible & rows.get("context", True)
            & rows.direction.ne(0) & rows.count_91d.ge(21) & rows.count_365d.ge(63)
            & rows.profit_91d.gt(0) & rows.profit_365d.gt(0))
