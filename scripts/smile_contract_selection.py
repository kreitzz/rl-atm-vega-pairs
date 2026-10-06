"""Common causal expiry-first selection for the three smile silos."""
import numpy as np
import pandas as pd

KEYS = ['date', 'secid', 'bucket']
SORT = KEYS + ['tenor_distance', 'expiry', 'distance', 'relative_width', 'open_interest', 'optionid']
ASCENDING = [True] * 7 + [False, True]


def select(q, following, cp):
    """Signal-close quotes only; next session is the known exchange calendar.

ATM keeps its closest-strike definition. Tail deltas use the same +/-0.075
tolerance. Every bucket chooses its own closest usable expiry to 30D.
"""
    q = q.copy()
    q['entry_date'] = q.date.map(following)
    q['entry_dte'] = (q.expiry - q.entry_date).dt.days
    q['relative_width'] = (q.ask - q.bid) / ((q.ask + q.bid) / 2)
    valid = (q.bs_status.eq('ok') & q.bs_vega.gt(0) & q.standard &
             q.multiplier.eq(100) & q.cp_flag.eq(cp) & q.bid.gt(0) &
             q.ask.ge(q.bid) & q.open_interest.ge(10) & q.relative_width.le(.25) &
             q.entry_dte.between(21, 45) & q.close.gt(0))
    q = q.loc[valid].copy()
    q['tenor'], q['tenor_distance'] = 30, q.entry_dte.sub(30).abs()
    parts = []
    for bucket in (['atm'] if cp == 'C' else ['25', '75']):
        if bucket == 'atm':
            distance = (q.strike / q.close - 1).abs()
            good = distance.le(.1)
        else:
            distance = q.bs_delta.abs().sub(int(bucket) / 100).abs()
            good = distance.le(.075)
        p = q.loc[good].copy()
        p['bucket'], p['distance'] = bucket, distance.loc[good]
        parts.append(p)
    return reduce_choices(pd.concat(parts, ignore_index=True))


def reduce_choices(q):
    return q.sort_values(SORT, ascending=ASCENDING).drop_duplicates(KEYS).reset_index(drop=True)
