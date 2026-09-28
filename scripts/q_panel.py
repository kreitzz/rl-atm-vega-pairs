"""Daily pair panel for the Q-learning experiment, built from data/wrds_q/*.csv.

Column names follow the repo convention (first leg = 'amd', second = 'nvda')
so the existing tree/policy-search code can be reused unchanged.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_first_pass import forward_rv
from scripts.run_rv_vix_pass import trailing_rv

HOLD = 21            # trading days held (~30 calendar days)
COST_RATE = 0.025    # round-trip pair cost = 2.5% of average IV (repo's current rule)


def load_raw(root=Path('data/wrds_q')):
    iv = pd.read_csv(root/'iv.csv', parse_dates=['date']).pivot(index='date', columns='ticker', values='iv')
    ret = pd.read_csv(root/'returns.csv', parse_dates=['date']).pivot(index='date', columns='ticker', values='retx')
    earn = pd.read_csv(root/'earnings.csv', parse_dates=['rdq'])
    sup = Path('config/earnings_supplement.csv')   # repo's manual fix for a missing NVDA date
    if sup.exists():
        s = pd.read_csv(sup, parse_dates=['earnings_date'])
        earn = pd.concat([earn, s.rename(columns={'earnings_date': 'rdq'})[['ticker', 'rdq']]])
    return iv.sort_index(), ret.sort_index(), earn


def build_pair(a, b, root=Path('data/wrds_q')):
    iv, ret, earn = load_raw(root)
    r = ret[[a, b]].rename(columns={a: 'AMD', b: 'NVDA'}).dropna()
    p = iv[[a, b]].rename(columns={a: 'iv_amd', b: 'iv_nvda'})
    p = p.where(p > 0).dropna()
    p = p.loc[p.index.isin(r.index)]                       # trading days with returns
    p = p.join(forward_rv(r, p.index), how='left')          # repo's 30-cal-day forward RV (for the original proxy)
    p = p.join(trailing_rv(r))
    for ticker, prefix in [(a, 'amd'), (b, 'nvda')]:
        dates = pd.DatetimeIndex(earn.loc[earn.ticker.eq(ticker), 'rdq'].dropna().unique()).sort_values()
        pos = dates.searchsorted(p.index, side='left')
        nxt = pd.Series([dates[i] if i < len(dates) else pd.NaT for i in pos], index=p.index)
        p[prefix+'_earnings_days_capped'] = (nxt - pd.Series(p.index, index=p.index)).dt.days.clip(upper=31)
    p['spread'] = p.iv_amd - p.iv_nvda
    p['iv_ratio'] = p.iv_amd / p.iv_nvda
    p['long_reward_orig'] = (p.rv_amd - p.iv_amd) - (p.rv_nvda - p.iv_nvda)
    # next-day log returns (return realised from close t to close t+1)
    lr = np.log1p(r.reindex(p.index))
    p['lr_amd_next'] = lr['AMD'].shift(-1)
    p['lr_nvda_next'] = lr['NVDA'].shift(-1)
    p['cost'] = COST_RATE * (p.iv_amd + p.iv_nvda) / 2     # full round trip, per $1 vega per leg
    feats = ['spread', 'amd_earnings_days_capped', 'nvda_earnings_days_capped', 'trailing_rv_amd', 'trailing_rv_nvda']
    p = p.dropna(subset=feats + ['lr_amd_next', 'lr_nvda_next'])
    p['long_reward'] = full_hold_payoff(p)
    p['end'] = pd.Series(p.index, index=p.index).shift(-HOLD)
    return p


def leg_step(s0, iv_t, iv_t1, lr1, k):
    """P&L (vol points per $1 initial vega) of one delta-hedged ATM leg, day k -> k+1 of the hold.

    carry: equal-weighted variance carry; summed over HOLD days it equals
           100*(RV^2 - IV0^2)/(2*IV0) ~= RV - IV0, i.e. the repo's payoff.
    mark:  remaining vega sqrt((HOLD-k)/HOLD) times the IV move since entry,
           so an early exit carries the IV mark and a full hold carries none.
    """
    s = s0 / 100.0
    carry = 100.0 * (lr1 * lr1 - s * s / 252.0) / (2.0 * s * HOLD / 252.0)
    w0 = np.sqrt((HOLD - k) / HOLD)
    w1 = np.sqrt((HOLD - k - 1) / HOLD)
    mark = w1 * (iv_t1 - s0) - w0 * (iv_t - s0)
    return carry + mark


def full_hold_payoff(p):
    """Long-A/short-B payoff of a full HOLD-day trade entered each day (before costs); NaN near the end."""
    ia, ib = p.iv_amd.to_numpy(), p.iv_nvda.to_numpy()
    la, lb = p.lr_amd_next.to_numpy(), p.lr_nvda_next.to_numpy()
    n = len(p)
    out = np.full(n, np.nan)
    for e in range(n - HOLD):
        tot = 0.0
        for k in range(HOLD):
            t = e + k
            tot += leg_step(ia[e], ia[t], ia[t+1], la[t], k) - leg_step(ib[e], ib[t], ib[t+1], lb[t], k)
        out[e] = tot
    return out
