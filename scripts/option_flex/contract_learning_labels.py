"""Bounded, independent shadow-cohort labels from observed contract marks.

These are standalone cash accounts, not attribution from the overlapping book.
Risk limits and NAV sizing are applied later by contract_engine.simulate.
"""
from dataclasses import replace
import numpy as np
import pandas as pd

from scripts.option_flex.contract_engine import valid_quote, valid_greeks, integer_size, verified_integer_split, execution_terms

LADDER = np.arange(-1., 1.01, .25)


class ShadowLabels:
    def __init__(self, quotes, stocks, selections, calendar, secids, cfg, dividends):
        if not cfg.hold_to_expiry:
            raise ValueError('The option-flex label experiment supports hold-to-expiry only')
        self.cfg = replace(cfg, risk_controls=False)
        self.calendar = pd.DatetimeIndex(calendar)
        self.secids = secids
        self.quotes = {(r['date'], r['optionid']): r for r in quotes.to_dict('records')}
        self.stocks = {(r['date'], r['secid']): r for r in stocks.to_dict('records')}
        self.selections = {(r.date, r.secid): r.optionid for r in selections.itertuples()}
        self.events = {}
        self.cash_dividends = {}
        for i, d in enumerate(self.calendar[:-1]):
            nxt = self.calendar[i+1]
            self.events[d] = not dividends[(dividends.declare_date <= d) &
                                           (dividends.ex_date > d) & (dividends.ex_date <= nxt)].empty
            ev = dividends[(dividends.ex_date > d) & (dividends.ex_date <= nxt)]
            self.cash_dividends[nxt] = np.array([ev.loc[ev.secid.eq(s), 'amount'].sum() for s in secids])

    def label(self, signal_index):
        """All nine integer-sized actions; daily excess dollars / base target vega.

        A known rejected entry is zero. An unresolved held trajectory is NaN,
        never treated as a lossless trade. No future status enters prediction.
        """
        if self.cfg.hold_to_expiry:
            from scripts.option_flex.batched_contract_labels import label_batch
            daily,ends,vegas,statuses=label_batch(self,[signal_index])
            return dict(daily=daily[0],end_index=int(ends[0]),entry_vega=vegas[0],status=str(statuses[0]))
        c = self.cfg; cal = self.calendar; i = signal_index
        zero = np.zeros((len(LADDER), 32))
        result = dict(end_index=min(i+1, len(cal)-1), status='no_entry',
                      daily=zero, entry_vega=np.zeros(len(LADDER)))
        if i+2 >= len(cal):
            result.update(status='insufficient_runoff', daily=np.full_like(zero, np.nan)); return result
        signal, entry = cal[i], cal[i+1]
        old, live = [], []
        for s in self.secids:
            oid = self.selections.get((signal, s))
            a = self.quotes.get((signal, oid)); b = self.quotes.get((entry, oid))
            stock = self.stocks.get((signal, s))
            if a is None or b is None or stock is None: return result
            dte = (a['expiry']-entry).days
            if not (valid_quote(a, True, c) and c.min_dte <= dte <= c.max_dte and
                    abs(a['strike']/stock['close']-1) <= c.max_moneyness_distance): return result
            check = dict(b, open_interest=a['open_interest'])
            if not valid_quote(check, True, c) or any(a[k] != b[k] for k in ['secid','strike','expiry','cfadj']): return result
            old.append(a); live.append(b)
        target = min(entry+pd.Timedelta(days=c.holding_days), min(q['expiry'] for q in old)-pd.Timedelta(days=1))
        end = cal.searchsorted(target, side='right')-1
        if target > cal[-1] or end <= i+1:
            result.update(status='insufficient_runoff', daily=np.full_like(zero, np.nan)); return result
        if self.events.get(entry, False): return result
        units = np.zeros((len(LADDER), 2))
        v = np.array([q['vega']*100 for q in live])
        for j, action in enumerate(LADDER):
            if action == 0: continue
            n = integer_size(v, c.entry_vega_per_leg*abs(action), c.max_entry_vega_imbalance)
            if n is not None and (n*v).sum() <= c.max_gross_vega:
                units[j] = n*np.sign(action)*np.array([1., -1.])
        if not np.any(units): return result
        result['entry_vega'] = (abs(units)*v).mean(axis=1)
        cash = np.full(len(LADDER), c.capital, dtype=float); benchmark = float(c.capital)
        hedge = np.zeros_like(units); previous_nav = cash.copy(); previous_benchmark = benchmark
        previous_spots = None; previous_adjustments = None; previous_rate = 0.
        last_vegas = v/100
        daily = zero.copy()
        for t in range(i, end+1):
            date = cal[t]
            sr = [self.stocks.get((date, s)) for s in self.secids]
            if any(r is None or not np.isfinite([r['close'], r['cfadj'], r['rate']]).all() or r['close'] <= 0 for r in sr):
                result.update(status='unresolved_underlying', daily=np.full_like(zero, np.nan), end_index=t); return result
            spots = np.array([r['close'] for r in sr]); adj = np.array([r['cfadj'] for r in sr])
            if t > i+1 and np.any(adj != previous_adjustments):
                for leg in np.flatnonzero(adj != previous_adjustments):
                    ratio = adj[leg]/previous_adjustments[leg]
                    new = self.quotes.get((date, old[leg]['optionid']))
                    if not (c.support_integer_splits and new is not None and verified_integer_split(old[leg],new,ratio)):
                        result.update(status='unresolved_corporate_action', daily=np.full_like(zero, np.nan), end_index=t); return result
                    ratio=int(round(ratio)); units[:,leg]*=ratio; hedge[:,leg]*=ratio
                    last_vegas[leg]/=ratio
                    previous_spots[leg]/=ratio; old[leg]=new
            if t > i:
                elapsed = (date-cal[t-1]).days/365
                cash += cash*(previous_rate+np.where(cash < 0, c.borrowing_spread, 0))*elapsed
                cash -= (np.maximum(-hedge, 0)*previous_spots).sum(axis=1)*c.stock_borrow_rate*elapsed
                cash += hedge @ self.cash_dividends.get(date, np.zeros(2))
                benchmark *= 1+previous_rate*elapsed
            active = t >= i+1
            close = t == end
            qr = []
            if active:
                for a in old:
                    q = self.quotes.get((date, a['optionid']))
                    if q is not None and c.repair_crossed_held_quotes and 0 <= q['ask'] < q['bid']:
                        q = dict(q, bid=q['ask'], ask=q['bid'])
                    if q is None or not valid_quote(q) or any(q[k] != a[k] for k in ['strike','expiry','cfadj']):
                        result.update(status='unresolved_contract', daily=np.full_like(zero, np.nan), end_index=t); return result
                    qr.append(q)
                if t > i+1:
                    close |= not all(valid_greeks(q) for q in qr) or self.events.get(date, False)
                mid = np.array([(q['ask']+q['bid'])/2 for q in qr])
                for leg,q in enumerate(qr):
                    if np.isfinite(q['vega']) and q['vega'] >= 0:last_vegas[leg]=q['vega']
                half = np.array([execution_terms(q,1,c,last_vegas[leg])[2]/100 for leg,q in enumerate(qr)])
                if t == i+1 or close:
                    qty = units if t == i+1 else -units
                    cash -= (qty*100*mid + abs(qty)*100*half + abs(qty)*c.option_commission).sum(axis=1)
                target_hedge = np.zeros_like(units) if close else -units*100*np.array([q['delta'] for q in qr])
                change = target_hedge-hedge
                cash -= (change*spots + abs(change)*spots*c.stock_cost_bps/10000).sum(axis=1)
                hedge = target_hedge
                option_value = np.zeros(len(LADDER)) if close else units @ (100*mid)
            else:
                option_value = np.zeros(len(LADDER))
            nav = cash+(hedge*spots).sum(axis=1)+option_value
            if t > i:
                daily[:, t-i-1] = (nav-previous_nav-(benchmark-previous_benchmark))/c.entry_vega_per_leg
            previous_nav = nav; previous_benchmark = benchmark
            previous_spots = spots; previous_adjustments = adj; previous_rate = np.mean([r['rate'] for r in sr])
            if active and close:
                result.update(status='complete', daily=daily, end_index=t); return result
        raise AssertionError('Shadow position did not close')


def profitability_memory(dates, ends, rewards, entered):
    """Three/twelve calendar months of completed directional shadow trades.

    rewards: net dollars / actual mean initial leg vega for [-1,+1].
    Overlapping cohorts are not independent observations; no cohort Sharpe.
    """
    states = pd.DataFrame(index=dates)
    for days in (91, 365):
        for j, name in enumerate(('short', 'long')):
            means, counts = [], []
            for d in dates:
                use = (ends < d) & (ends >= d-pd.Timedelta(days=days)) & entered[:, j] & np.isfinite(rewards[:, j])
                means.append(float(rewards[use, j].mean()) if use.any() else 0.)
                counts.append(int(use.sum()))
            states[f'profit_{name}_{days}d'] = means
            states[f'count_{name}_{days}d'] = counts
    return states


def beta_screen(returns, calendar, window=126, correlation_floor=.5, beta_tolerance=.25):
    """A-on-B beta; screen on preceding close, then freeze for each month."""
    r = returns.reindex(calendar)
    a, b = r.iloc[:, 0], r.iloc[:, 1]
    beta = a.rolling(window, min_periods=window).cov(b)/b.rolling(window, min_periods=window).var()
    corr = a.rolling(window, min_periods=window).corr(b)
    stats = pd.DataFrame({'beta': beta, 'correlation': corr}, index=calendar).shift(1)
    stats['formation'] = stats.index.to_period('M')
    # nth keeps NaN at formation rather than silently taking a later estimate.
    first = stats.groupby('formation', sort=False).head(1)
    lookup = first.set_index('formation')
    out = lookup.reindex(stats.formation).set_axis(calendar)
    out['eligible'] = out.beta.between(1-beta_tolerance,1+beta_tolerance) & (out.correlation >= correlation_floor)
    return out
