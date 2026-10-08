"""Direct volatility-rank threshold policies and causal funded-book selection.

Each candidate book must be a complete funded replay of its whole policy.
Selecting a policy for a month does not splice its counterfactual return series:
the selected decisions require a separate replay carrying actual open positions.
The VIX feature is the inherited prior-session value divided by 100.
"""
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


TAIL_GRID = (.05, .10, .15, .20, .25, .30)
CONTEXT_GRID = (.10, .20, .30)


@dataclass(frozen=True)
class ThresholdPolicy:
    key: str
    q_low: float
    q_high: float | None = None
    vix_cutoff: float = .20

    def __post_init__(self):
        if not self.key or not 0 < self.q_low < .5:
            raise ValueError("Policy needs a name and a tail threshold in (0,.5)")
        if self.q_high is not None and not 0 < self.q_high < .5:
            raise ValueError("High-VIX tail threshold must lie in (0,.5)")
        if not np.isfinite(self.vix_cutoff) or self.vix_cutoff <= 0:
            raise ValueError("VIX cutoff is expressed as a positive fraction")

    @property
    def tie_break(self):
        high = self.q_low if self.q_high is None else self.q_high
        # Favor proximity to inherited .20, then smaller tail(s), then stable key.
        return (round(abs(self.q_low-.20)+abs(high-.20), 12),
                self.q_low, high, self.key)


def policy_grid(include_context=True):
    policies = {f"q{int(round(q*100)):02d}": ThresholdPolicy(
        f"q{int(round(q*100)):02d}", q) for q in TAIL_GRID}
    if include_context:
        for low in CONTEXT_GRID:
            for high in CONTEXT_GRID:
                key = f"vix{int(round(low*100)):02d}_{int(round(high*100)):02d}"
                policies[key] = ThresholdPolicy(key, low, high)
    return policies


def policy_directions(frame, policy):
    """Signal-date direction: lower rank <=q, upper rank >1-q, else flat."""
    rank = pd.to_numeric(frame.iv_rank, errors="coerce").to_numpy(float)
    if policy.q_high is None:
        q = np.full(len(frame), policy.q_low)
        known = np.ones(len(frame), dtype=bool)
    else:
        vix = pd.to_numeric(frame.vix, errors="coerce").to_numpy(float)
        # The inherited optional-state convention uses zero for unavailable VIX.
        vix_known = np.isfinite(vix) & (vix > 0)
        q = np.where(vix >= policy.vix_cutoff, policy.q_high, policy.q_low)
        q = np.where(vix_known, q, .20)
        known = np.ones(len(frame), dtype=bool)
    known &= np.isfinite(rank) & (rank >= 0) & (rank <= 1)
    direction = np.where(rank <= q, 1, np.where(rank > 1-q, -1, 0))
    return pd.Series(np.where(known, direction, 0), index=frame.index, dtype=int)


def proposal_mask(frame, policy):
    """Evaluate direction-specific rows; caller supplies base-eligible dates.

    No target, end, future execution flag, or completed-label flag is consulted.
    Memory columns must be oriented to each candidate row's direction.
    """
    direction = policy_directions(frame, policy)
    return (direction.ne(0) & frame.direction.eq(direction) &
            frame.profit_91d.gt(0) & frame.count_91d.ge(21) &
            frame.profit_365d.gt(0) & frame.count_365d.ge(63))


def select_policy(books: Mapping, decisions: Mapping, policies: Mapping,
                  asof, lookback_months=18, start=None, fallback="q20",
                  minimum_sessions=252, minimum_entries=20):
    """Choose the highest prior funded excess-return Sharpe; return an audit.

    Books contain a Series or DataFrame.excess_return on a common daily calendar.
    Decisions need entry_date,status; only actually entered decisions count.
    For fixed pre-2022 selection use asof='2022-01-01', lookback_months=None.
    A rolling selection uses [asof-18 calendar months, asof), including cash days.
    The original book's first return is always excluded: it is an artificial
    zero at account initialization, not a measured holding-period return.
    """
    asof = pd.Timestamp(asof)
    if not policies or fallback not in policies:
        raise ValueError("Fallback must belong to the candidate policy family")
    if set(policies)-set(books) or set(policies)-set(decisions):
        raise ValueError("Every candidate needs a full funded book and decisions")
    if minimum_sessions < 2 or minimum_entries < 1:
        raise ValueError("Minimum sessions/entries must be positive and meaningful")
    if lookback_months is not None and lookback_months <= 0:
        raise ValueError("Lookback months must be positive or None")
    lower = (asof-pd.DateOffset(months=lookback_months)
             if lookback_months is not None else None)
    if start is not None:
        start = pd.Timestamp(start)
        lower = max(lower, start) if lower is not None else start
    records = []
    expected_index = None
    for key, policy in policies.items():
        if policy.key != key:
            raise ValueError("Policy dictionary key differs from policy.key")
        book = books[key]
        returns = book.excess_return if isinstance(book, pd.DataFrame) else book
        if not isinstance(returns.index, pd.DatetimeIndex):
            raise ValueError("Funded returns need a DatetimeIndex")
        if not returns.index.is_unique or not returns.index.is_monotonic_increasing:
            raise ValueError("Funded return dates must be unique and chronological")
        window = returns.loc[(returns.index < asof) &
            ((returns.index >= lower) if lower is not None else True) &
            (returns.index != returns.index[0] if len(returns) else True)]
        if expected_index is None:
            expected_index = window.index
        elif not expected_index.equals(window.index):
            raise ValueError("Candidate books must use the same historical calendar")
        values = window.to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError("Unavailable historical funded returns cannot be imputed")
        entries = decisions[key]
        entry_date = pd.to_datetime(entries.entry_date)
        in_window = (entry_date < asof)
        if lower is not None:
            in_window &= entry_date >= lower
        count = int((in_window & entries.status.eq("entered")).sum())
        enough = len(window) >= minimum_sessions and count >= minimum_entries
        sd = float(values.std(ddof=1)) if len(values) >= 2 else np.nan
        score = float(np.sqrt(252)*values.mean()/sd) if enough and sd > 1e-12 else None
        reason = ("eligible" if score is not None else
                  "insufficient_sessions" if len(window) < minimum_sessions else
                  "insufficient_entries" if count < minimum_entries else "zero_return_variance")
        records.append(dict(policy=key, q_low=policy.q_low, q_high=policy.q_high,
            sessions=len(window), entries=count, funded_excess_sharpe=score,
            eligible=score is not None, reason=reason,
            first_return=window.index.min().isoformat() if len(window) else None,
            latest_return=window.index.max().isoformat() if len(window) else None))
    eligible = [r for r in records if r["eligible"]]
    chosen = (min(eligible, key=lambda r: (-r["funded_excess_sharpe"],
                                         policies[r["policy"]].tie_break))["policy"]
              if eligible else fallback)
    return dict(policy=chosen, status="selected" if eligible else "fallback",
                asof=asof.isoformat(), lower=lower.isoformat() if lower is not None else None,
                lookback_months=lookback_months, fallback=fallback,
                minimum_sessions=minimum_sessions, minimum_entries=minimum_entries,
                latest_return=max((r["latest_return"] for r in records if r["latest_return"]),
                                  default=None),
                candidates=records)
