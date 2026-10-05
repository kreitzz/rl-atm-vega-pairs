"""Step 2: rule-based baselines for tracking the 30-day 50-delta IV index with real calls.

Objective is tracking, not P&L. Each day t (end of day):
  1. Check the book: if its vega-weighted maturity, delta or total vega has drifted outside
     a no-trade band around (30 days, 0.50, target vega) -- or a held contract is expiring
     or has no quote -- trade to the rule's target at today's quotes (buy at ask, sell at
     bid, plus a per-contract commission).
  2. Holdings are then frozen until t+1, and we measure how the book's IV moved against
     the index:

        tracking_error[t+1] = sum_i n_i * vega_i[t] * (iv_i[t+1] - iv_i[t]) / V  -  (I[t+1] - I[t])

     n_i = contracts, vega_i = $ per vol point per contract, V = target $ vega. The first
     term is the vol-point move the book is exposed to; the second is the index move. Both
     are in vol points, so tracking error and trading cost share one unit.

Trading cost of a rebalance, in vol points of the target exposure:
        cost[t] = sum_i |change in n_i| * (half_spread_i + commission) / V

Target rules (vega weights summing to 1)
  single   one call nearest 30 days / 50 delta (what the existing contract engine holds).
  interp   the two expiries around 30 days and, within each, the two strikes around
           50 delta, weighted linearly to land exactly on (30 days, 50 delta).
Either can be restricted to standard monthly expiries. Band() of zeros = rebalance daily.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Settings:
    target_vega: float = 10_000.0     # $ per vol point per leg
    commission: float = 0.65          # $ per contract per side
    min_hold_dte: int = 7             # held contracts are rolled before this
    target_dte: int = 30
    target_delta: float = 0.50
    single_dte_range: tuple = (21, 45)   # where 'single' looks for its one contract


# ------------------------------------------------------------------ target rules
def delta_point(expiry_rows, target_delta):
    """Weights on (at most) two strikes of one expiry that interpolate to target delta."""
    r = expiry_rows.sort_values('delta')
    d = r.delta.to_numpy(); ids = r.optionid.to_numpy()
    below = np.flatnonzero(d < target_delta); above = np.flatnonzero(d >= target_delta)
    if len(below) and len(above):
        lo, hi = below[-1], above[0]
        w_hi = (target_delta - d[lo]) / (d[hi] - d[lo])
        return {ids[lo]: 1 - w_hi, ids[hi]: w_hi}
    k = int(np.argmin(abs(d - target_delta)))
    return {ids[k]: 1.0}


def interp_target(cands, s, monthly_only=False):
    """Vega weights landing on (target_dte, target_delta) by linear interpolation."""
    c = cands[cands.dte >= s.min_hold_dte]
    if monthly_only:
        c = c[c.monthly]
    if c.empty:
        return None
    exps = c.groupby('exdate').dte.first().sort_values()
    near = exps[exps <= s.target_dte]; far = exps[exps > s.target_dte]
    if len(near) and len(far):
        e1, e2 = near.index[-1], far.index[0]
        t1, t2 = near.iloc[-1], far.iloc[0]
        w2 = (s.target_dte - t1) / (t2 - t1)
        legs = [(e1, 1 - w2), (e2, w2)]
    else:
        e = (exps - s.target_dte).abs().idxmin()
        legs = [(e, 1.0)]
    out = {}
    for e, we in legs:
        for oid, wk in delta_point(c[c.exdate == e], s.target_delta).items():
            out[oid] = out.get(oid, 0.0) + we * wk
    return out


def single_choice(cands, s):
    lo, hi = s.single_dte_range
    c = cands[cands.dte.between(lo, hi)]
    if c.empty:
        c = cands[cands.dte >= s.min_hold_dte]
    if c.empty:
        return None
    key = pd.DataFrame({'t': (c.dte - s.target_dte).abs(), 'd': (c.delta - s.target_delta).abs(),
                        'oid': c.optionid})
    return int(c.optionid.loc[key.sort_values(['t', 'd', 'oid']).index[0]])


# ------------------------------------------------------------------ simulator
@dataclass
class Band:
    """No-trade band on the book's exposure characteristics (vega-weighted).

    Rebalance to the target only when the book's average maturity, average delta or total
    vega has drifted further than these tolerances from (target_dte, target_delta, 1).
    All zeros = rebalance to the target every day.
    """
    dte: float = 0.0
    delta: float = 0.0
    vega: float = 0.0


def book_profile(held, marks, V):
    vw = {o: n * marks.loc[o, 'vega'] for o, n in held.items()}
    tot = sum(vw.values())
    if tot <= 0:
        return None
    return {'vega_ratio': tot / V,
            'dte': sum(vw[o] * marks.loc[o, 'dte'] for o in vw) / tot,
            'delta': sum(vw[o] * marks.loc[o, 'delta'] for o in vw) / tot,
            'iv': sum(vw[o] * marks.loc[o, 'iv'] for o in vw) / tot}


def make_target(cands, s, method, monthly_only):
    if method == 'interp':
        return interp_target(cands, s, monthly_only)
    if method == 'single':
        c = cands[cands.monthly] if monthly_only else cands
        o = single_choice(c, s)
        return None if o is None else {o: 1.0}
    raise ValueError(method)


def simulate(chain, index, ticker, method, band=Band(), s=Settings(), monthly_only=False):
    """Daily book for one ticker. Returns a per-day DataFrame of tracking and cost."""
    days = list(index.index)
    by_day = {d: g.set_index('optionid') for d, g in chain[chain.ticker == ticker].groupby('date')}
    I = index[ticker]
    V = s.target_vega
    held, last_half_spread, rows = {}, {}, []
    for t in range(len(days) - 1):
        d, d1 = days[t], days[t+1]
        marks = by_day.get(d)
        if marks is None or not np.isfinite(I.loc[d]):
            continue
        # held contracts with no usable quote today are closed at their last seen quote
        missing = [o for o in held if o not in marks.index]
        cost = sum(abs(held.pop(o)) * (last_half_spread.get(o, 0.0) + s.commission) for o in missing)
        expiring = any(marks.loc[o, 'dte'] < s.min_hold_dte for o in held)
        prof = book_profile(held, marks, V) if held else None
        off = (prof is None or expiring or bool(missing)
               or abs(prof['dte'] - s.target_dte) > band.dte + 1e-9
               or abs(prof['delta'] - s.target_delta) > band.delta + 1e-9
               or abs(prof['vega_ratio'] - 1) > band.vega + 1e-9)
        traded, turnover = False, 0.0
        if off:
            tw = make_target(marks[marks.candidate].reset_index(), s, method, monthly_only)
            if tw is not None:
                new = {o: w * V / marks.loc[o, 'vega'] for o, w in tw.items() if w > 1e-12}
                for o in set(held) | set(new):
                    dn = abs(new.get(o, 0.0) - held.get(o, 0.0))
                    if dn > 1e-12:
                        cost += dn * (marks.loc[o, 'half_spread'] + s.commission)
                        turnover += dn * marks.loc[o, 'vega'] / V
                traded = turnover > 0
                held = new
        for o in held:
            last_half_spread[o] = marks.loc[o, 'half_spread']
        prof = book_profile(held, marks, V) if held else None

        # exposure over t -> t+1 with holdings frozen
        nxt = by_day.get(d1)
        gap = (nxt is None or not held or any(o not in nxt.index for o in held)
               or not np.isfinite(I.loc[d1]))
        move = np.nan
        if not gap:
            move = sum(n * marks.loc[o, 'vega'] * (nxt.loc[o, 'iv'] - marks.loc[o, 'iv'])
                       for o, n in held.items()) / V
        di = I.loc[d1] - I.loc[d]
        rows.append({
            'date': d, 'ticker': ticker, 'index': I.loc[d], 'index_change': di,
            'basket_iv': prof['iv'] if prof else np.nan, 'basket_change': move,
            'tracking_error': move - di, 'cost': cost / V, 'turnover': turnover,
            'traded': traded, 'forced_closes': len(missing), 'gap': gap,
            'contracts_held': len(held),
            'vega_ratio': prof['vega_ratio'] if prof else np.nan,
            'avg_dte': prof['dte'] if prof else np.nan,
            'avg_delta': prof['delta'] if prof else np.nan,
        })
    out = pd.DataFrame(rows)
    out['level_error'] = out.basket_iv - out['index']
    return out


# ------------------------------------------------------------------ metrics
def metrics(daily, window=21):
    """Tracking quality and cost for one daily result frame (one ticker or a pair spread)."""
    x = daily.dropna(subset=['tracking_error'])
    years = len(daily) / 252
    te, di, db = x.tracking_error, x.index_change, x.basket_change
    blocks = x.tracking_error.groupby(np.arange(len(x)) // window).sum()
    beta = float(np.cov(db, di)[0, 1] / np.var(di, ddof=1)) if len(x) > 2 else np.nan
    return {
        'days': int(len(daily)), 'measured_days': int(len(x)), 'gaps': int(daily.gap.sum()),
        'te_daily_rmse': float(np.sqrt((te**2).mean())),
        'te_median_abs': float(te.abs().median()),
        'te_p95_abs': float(te.abs().quantile(.95)),
        'te_mean_per_year': float(te.mean() * 252),
        'index_21d_sd': float(x.index_change.groupby(np.arange(len(x)) // window).sum().std()),
        'te_21d_rmse': float(np.sqrt((blocks**2).mean())),
        'index_change_sd': float(di.std()),
        'r2': float(1 - (te**2).sum() / ((di - di.mean())**2).sum()),
        'corr': float(np.corrcoef(db, di)[0, 1]), 'beta': beta,
        'level_error_mean': float(daily.level_error.mean()),
        'level_error_mae': float(daily.level_error.abs().mean()),
        'mean_vega_ratio': float(daily.vega_ratio.mean()),
        'cost_per_year': float(daily.cost.sum() / years),
        'turnover_per_year': float(daily.turnover.sum() / years),
        'trade_day_fraction': float(daily.traded.mean()),
        'forced_closes': int(daily.forced_closes.sum()),
    }


def pair_daily(a, b):
    """Long-A / short-B spread book: tracking errors subtract, costs add."""
    m = a.merge(b, on='date', suffixes=('_a', '_b'))
    return pd.DataFrame({
        'date': m.date,
        'index_change': m.index_change_a - m.index_change_b,
        'basket_change': m.basket_change_a - m.basket_change_b,
        'tracking_error': m.tracking_error_a - m.tracking_error_b,
        'level_error': m.level_error_a - m.level_error_b,
        'vega_ratio': (m.vega_ratio_a + m.vega_ratio_b) / 2,
        'cost': m.cost_a + m.cost_b, 'turnover': m.turnover_a + m.turnover_b,
        'traded': m.traded_a | m.traded_b, 'gap': m.gap_a | m.gap_b,
        'forced_closes': m.forced_closes_a + m.forced_closes_b,
    })
