"""One-ticket, next-session actual-call execution for offline Q learning.

Calls, not straddles. Quotes are normalized as in run_contract_pilot.
The ticket has a segregated financing account; reserve capital earns cash.
Contract IDs never enter predictive features and never roll while held.
"""
from dataclasses import dataclass, replace
import numpy as np
import pandas as pd

from scripts.contract_engine import Settings, UnresolvedPosition, integer_size, valid_quote, valid_greeks

WAIT, LONG, SHORT, EXIT = 0, 1, 2, 3
ACTION_NAMES = ('wait_or_hold', 'enter_long', 'enter_short', 'exit')
FEATURE_NAMES = (
    'eligible', 'iv_a', 'iv_b', 'relative_carry', 'rv_a', 'rv_b', 'return_corr',
    'moneyness_a', 'moneyness_b', 'dte_a', 'dte_b', 'delta_a', 'delta_b',
    'vega_a', 'vega_b', 'width_a', 'width_b', 'log_oi_a', 'log_oi_b',
    'direction', 'age_days', 'entry_iv_a', 'entry_iv_b', 'signed_iv_displacement',
    'book_value_bps', 'financing_cash_fraction', 'gross_hedge_fraction',
    'vega_a_fraction', 'vega_b_fraction', 'dividend_days_a', 'dividend_days_b',
)


@dataclass
class Ticket:
    entry: int
    direction: int
    optionids: tuple
    quantity: np.ndarray
    entry_quotes: tuple
    cash: float
    hedge: np.ndarray
    value: float


def common_expiry_pair(chain, spots, next_date, secids, cfg):
    """Selection only uses the supplied signal-date chain; same expiry on both legs."""
    candidates = []
    for i, sid in enumerate(secids):
        q = chain[chain.secid.eq(sid)].copy()
        if q.empty:
            return None
        # OI must have been published before the decision, not filled using future rows.
        q['open_interest'] = q.known_open_interest
        q['dte'] = (q.expiry-next_date).dt.days
        q['distance'] = abs(q.strike/spots[i]-1)
        ok = q.apply(lambda r: valid_quote(r, True, cfg), axis=1)
        q = q[ok & q.impl_volatility.gt(0) & np.isfinite(q.impl_volatility) &
              q.dte.between(cfg.min_dte, cfg.max_dte) & q.distance.le(cfg.max_moneyness_distance)]
        if q.empty:
            return None
        candidates.append(q)
    expiries = sorted(set(candidates[0].expiry).intersection(candidates[1].expiry),
                      key=lambda d: (abs((d-next_date).days-cfg.target_dte), d))
    if not expiries:
        return None
    expiry = expiries[0]
    chosen = [q[q.expiry.eq(expiry)].sort_values(
        ['distance', 'open_interest', 'optionid'], ascending=[True, False, True]).iloc[0]
        for q in candidates]
    return tuple(int(q.optionid) for q in chosen)


class ContractMarket:
    def __init__(self, quotes, stocks, calendar, secids, cfg=None, dividends=None, candidates=None):
        self.cfg = cfg or Settings()
        if self.cfg.repair_crossed_held_quotes:
            raise ValueError('Contract Q uses strict held quotes; no automatic bid/ask repair')
        self.dates = pd.DatetimeIndex(calendar)
        self.secids = tuple(int(s) for s in secids)
        if len(self.secids)!=2 or len(set(self.secids))!=2:
            raise ValueError('Need two different underlyings')
        if len(self.dates)<2 or not self.dates.is_monotonic_increasing or self.dates.has_duplicates:
            raise ValueError('Need a sorted unique session calendar')
        self.n = len(self.dates)
        if quotes.duplicated(['date','optionid']).any() or stocks.duplicated(['date','secid']).any():
            raise ValueError('Duplicate quote or stock keys')
        q = quotes.copy()
        if 'known_open_interest' not in q:
            q = q.sort_values(['optionid','date'])
            q['known_open_interest'] = q.groupby('optionid').open_interest.shift(1)
        self.quotes = {(pd.Timestamp(r['date']), int(r['optionid'])): r for r in q.to_dict('records')}
        s = stocks.copy()
        s['date'] = pd.to_datetime(s.date)
        idx = pd.MultiIndex.from_product([self.dates, self.secids], names=['date','secid'])
        s = s.set_index(['date','secid']).reindex(idx)
        cols = ['close','cfadj','rate']
        if not np.isfinite(s[cols].to_numpy(float)).all() or (s.close<=0).any():
            raise ValueError('Missing or invalid underlying/rate observations; do not compress gaps')
        self.spots = s.close.to_numpy(float).reshape(-1,2)
        self.adjustments = s.cfadj.to_numpy(float).reshape(-1,2)
        self.rates = s.rate.to_numpy(float).reshape(-1,2).mean(axis=1)
        self.rv = s[['rv']].to_numpy(float).reshape(-1,2) if 'rv' in s else np.full((self.n,2), np.nan)
        self.corr = s['corr'].to_numpy(float).reshape(-1,2).mean(axis=1) if 'corr' in s else np.full(self.n,np.nan)
        self.dividends = [] if dividends is None else dividends.to_dict('records')
        for event in self.dividends:
            event['ex_date'] = pd.Timestamp(event['ex_date'])
            event['declare_date'] = pd.Timestamp(event['declare_date'])
        self._div_days = np.full((self.n,2),60.)
        self._before_dividend = np.zeros(self.n,bool)
        self._cash_events = [[] for _ in range(self.n)]
        for t,date in enumerate(self.dates):
            for event in self.dividends:
                if event['declare_date']<=date<event['ex_date']:
                    i=self.secids.index(int(event['secid']))
                    self._div_days[t,i]=min(self._div_days[t,i],(event['ex_date']-date).days)
                    if t+1<self.n and event['ex_date']<=self.dates[t+1]:
                        self._before_dividend[t]=True
                if t and self.dates[t-1]<event['ex_date']<=date:
                    self._cash_events[t].append(event)
        if candidates is None:
            chains = {d:g for d,g in q.groupby('date')}
            self.candidates = [common_expiry_pair(chains.get(d, q.iloc[:0]), self.spots[t],
                self.dates[min(t+1,self.n-1)], self.secids, self.cfg) for t,d in enumerate(self.dates)]
        else:
            self.candidates = [candidates.get(d) for d in self.dates]

    def quote(self, t, oid, entry=None):
        q = self.quotes.get((self.dates[t], int(oid)))
        if q is None or not valid_quote(q):
            raise UnresolvedPosition(f'{self.dates[t].date()}: missing/invalid held quote {oid}')
        if entry is not None and any(q[k] != entry[k] for k in ('strike','expiry','cfadj','secid')):
            raise UnresolvedPosition(f'{self.dates[t].date()}: adjusted held contract {oid}')
        return q

    def known_dividend_days(self, t, sid):
        return self._div_days[t,self.secids.index(sid)]

    def before_dividend(self, t):
        return bool(self._before_dividend[t])

    def legal(self, t, ticket):
        if ticket is not None:
            return (WAIT, EXIT)
        # End-of-sample entries are not opened merely to be immediately liquidated.
        ok = (t < self.n-2 and self.candidates[t] is not None and
              np.isfinite(self.rv[t]).all() and np.isfinite(self.corr[t]) and not self.before_dividend(t))
        return (WAIT, LONG, SHORT) if ok else (WAIT,)

    def features(self, t, ticket):
        ids = self.candidates[t] if ticket is None else ticket.optionids
        if ids is None:
            x = np.zeros(len(FEATURE_NAMES))
            x[4:7] = np.nan_to_num([*self.rv[t],self.corr[t]])
            return x
        qs = [self.quote(t, oid, None if ticket is None else ticket.entry_quotes[i])
              for i,oid in enumerate(ids)]
        iv = np.array([q['impl_volatility']*100 for q in qs],float)
        vega = np.array([q['vega']*100 for q in qs])
        rv = np.nan_to_num(self.rv[t])
        carry = (rv**2-iv**2)/(2*np.maximum(iv,.01))
        mid = np.array([(q['bid']+q['ask'])/2 for q in qs])
        width = np.array([(q['ask']-q['bid'])/max(m,.01) for q,m in zip(qs,mid)])
        entry_iv = iv if ticket is None else np.array([q['impl_volatility']*100 for q in ticket.entry_quotes])
        direction = 0 if ticket is None else ticket.direction
        quantities = np.zeros(2) if ticket is None else ticket.quantity
        x = [float(ticket is not None or LONG in self.legal(t,None)), *iv, carry[0]-carry[1], *rv,
             self.corr[t], *[q['strike']/s-1 for q,s in zip(qs,self.spots[t])],
             *[(pd.Timestamp(q['expiry'])-self.dates[t]).days for q in qs],
             *[q['delta'] for q in qs], *vega, *width,
             *[np.log1p(max(0, q.get('known_open_interest',0))) for q in qs],
             direction, 0 if ticket is None else (self.dates[t]-self.dates[ticket.entry]).days,
             *entry_iv, direction*((iv[0]-iv[1])-(entry_iv[0]-entry_iv[1])),
             0 if ticket is None else 10000*ticket.value/self.cfg.capital,
             0 if ticket is None else ticket.cash/self.cfg.capital,
             0 if ticket is None else sum(abs(ticket.hedge)*self.spots[t])/self.cfg.capital,
             *list(quantities*vega/self.cfg.entry_vega_per_leg),
             *[self.known_dividend_days(t,s) for s in self.secids]]
        return np.nan_to_num(np.asarray(x,float), nan=0., posinf=0., neginf=0.)

    def step(self, t, ticket, action):
        """Action observed at t executes at t+1. Return next ticket, excess $, ledger."""
        if t>=self.n-1 or action not in self.legal(t,ticket):
            raise ValueError('Illegal action or no next execution session')
        j = t+1
        cfg = self.cfg
        elapsed = (self.dates[j]-self.dates[t]).days/365
        cash_return = self.rates[t]*elapsed
        info = dict(date=self.dates[j], signal_date=self.dates[t], action=ACTION_NAMES[action],
                    status='flat', cash_return=cash_return, option_mark_pnl=0., hedge_pnl=0.,
                    interest=0., borrow=0., dividend_cash=0., option_cost=0., hedge_cost=0.,
                    optionid_a=None, optionid_b=None, qty_a=0, qty_b=0,
                    trade_qty_a=0,trade_qty_b=0,fill_price_a=None,fill_price_b=None,
                    hedge_a=0.,hedge_b=0.,hedge_trade_a=0.,hedge_trade_b=0.,
                    stock_close_a=float(self.spots[j,0]),stock_close_b=float(self.spots[j,1]))
        if ticket is None:
            if action==WAIT:
                return None, 0., info
            old = [self.quote(t, oid) for oid in self.candidates[t]]
            live = [self.quotes.get((self.dates[j],q['optionid'])) for q in old]
            good = []
            for prior,q in zip(old,live):
                if q is None:
                    break
                check = dict(q,open_interest=prior['known_open_interest'])
                if (not valid_quote(check,True,cfg) or not np.isfinite(q['impl_volatility']) or
                    q['impl_volatility']<=0 or any(q[k]!=prior[k] for k in ('strike','expiry','cfadj','secid'))):
                    break
                good.append(q)
            if len(good)!=2 or not np.array_equal(self.adjustments[t],self.adjustments[j]) or self.before_dividend(j):
                info['status']='entry_rejected'
                return None,0.,info
            qty = integer_size([q['vega']*100 for q in live],cfg.entry_vega_per_leg,cfg.max_entry_vega_imbalance)
            if qty is None or sum(qty[i]*live[i]['vega']*100 for i in range(2))>cfg.max_gross_vega:
                info['status']='size_rejected'
                return None,0.,info
            direction = 1 if action==LONG else -1
            qty = qty*np.array([direction,-direction])
            hedge = -qty*100*np.array([q['delta'] for q in live])
            # Fixed research exposure cap; not a broker margin calculation.
            if sum(abs(hedge)*self.spots[j])>cfg.capital:
                info['status']='hedge_notional_rejected'
                return None,0.,info
            prices = np.array([q['ask'] if n>0 else q['bid'] for q,n in zip(live,qty)])
            mids = np.array([(q['ask']+q['bid'])/2 for q in live])
            fees = sum(abs(qty))*cfg.option_commission
            cash = -sum(qty*100*prices)-fees
            hedge_cost = sum(abs(hedge)*self.spots[j])*cfg.stock_cost_bps/10000
            cash -= sum(hedge*self.spots[j])+hedge_cost
            value = cash+sum(qty*100*mids)+sum(hedge*self.spots[j])
            info.update(status='entered', option_cost=float(sum(abs(qty)*100*abs(prices-mids))+fees),
                        hedge_cost=float(hedge_cost), optionid_a=live[0]['optionid'],optionid_b=live[1]['optionid'],
                        qty_a=int(qty[0]),qty_b=int(qty[1]),ticket_cash=float(cash),ticket_value=float(value),
                        trade_qty_a=int(qty[0]),trade_qty_b=int(qty[1]),
                        fill_price_a=float(prices[0]),fill_price_b=float(prices[1]),
                        hedge_a=float(hedge[0]),hedge_b=float(hedge[1]),
                        hedge_trade_a=float(hedge[0]),hedge_trade_b=float(hedge[1]))
            return Ticket(j,direction,tuple(q['optionid'] for q in live),qty,tuple(live),float(cash),hedge,float(value)),float(value),info
        # Held positions may not disappear from the accounting when prices fail.
        if not np.array_equal(self.adjustments[t],self.adjustments[j]):
            raise UnresolvedPosition(f'{self.dates[j].date()}: held underlying corporate action')
        old = [self.quote(t,oid,ticket.entry_quotes[i]) for i,oid in enumerate(ticket.optionids)]
        live = [self.quote(j,oid,ticket.entry_quotes[i]) for i,oid in enumerate(ticket.optionids)]
        interest = ticket.cash*(self.rates[t]+(cfg.borrowing_spread if ticket.cash<0 else 0))*elapsed
        borrow = sum(np.maximum(-ticket.hedge,0)*self.spots[t])*cfg.stock_borrow_rate*elapsed
        div_cash = 0.
        for d in self._cash_events[j]:
            if not d.get('ordinary',True):
                raise UnresolvedPosition(f'{self.dates[j].date()}: unsupported held distribution')
            div_cash += ticket.hedge[self.secids.index(int(d['secid']))]*float(d['amount'])
        cash = ticket.cash+interest-borrow+div_cash
        expiry = min(pd.Timestamp(q['expiry']) for q in live)
        if self.dates[j]>=expiry:
            raise UnresolvedPosition('A position reached expiry without a supported settlement')
        forced = ('terminal_exit' if j==self.n-1 else
                  'expiry_exit' if self.dates[j+1]>=expiry else
                  'horizon_exit' if (self.dates[j]-self.dates[ticket.entry]).days>=cfg.holding_days else
                  'dividend_exit' if self.before_dividend(j) else
                  'greeks_exit' if not all(valid_greeks(q) and np.isfinite(q['impl_volatility']) and q['impl_volatility']>0 for q in live) else
                  'vega_limit_exit' if sum(abs(ticket.quantity[i])*100*live[i]['vega'] for i in range(2))>cfg.max_gross_vega else '')
        closing = bool(forced) or action==EXIT
        mid_old = np.array([(q['bid']+q['ask'])/2 for q in old])
        mid_new = np.array([(q['bid']+q['ask'])/2 for q in live])
        option_pnl = sum(ticket.quantity*100*(mid_new-mid_old))
        stock_pnl = sum(ticket.hedge*(self.spots[j]-self.spots[t]))
        option_cost = 0.
        if closing:
            prices = np.array([q['bid'] if n>0 else q['ask'] for q,n in zip(live,ticket.quantity)])
            fees = sum(abs(ticket.quantity))*cfg.option_commission
            cash += sum(ticket.quantity*100*prices)-fees
            option_cost = sum(abs(ticket.quantity)*100*abs(prices-mid_new))+fees
            target = np.zeros(2)
            option_value = 0.
        else:
            target = -ticket.quantity*100*np.array([q['delta'] for q in live])
            option_value = sum(ticket.quantity*100*mid_new)
        change = target-ticket.hedge
        hedge_cost = sum(abs(change)*self.spots[j])*cfg.stock_cost_bps/10000
        cash -= sum(change*self.spots[j])+hedge_cost
        value = cash+option_value+sum(target*self.spots[j])
        excess = value-ticket.value*(1+cash_return)
        components = option_pnl+stock_pnl+interest-borrow+div_cash-option_cost-hedge_cost
        if not np.isclose(value-ticket.value,components,atol=1e-6,rtol=1e-9):
            raise AssertionError('Ticket accounting did not reconcile')
        info.update(status=forced or ('policy_exit' if closing else 'held'), option_mark_pnl=float(option_pnl),
                    hedge_pnl=float(stock_pnl),interest=float(interest),borrow=float(borrow),
                    dividend_cash=float(div_cash),option_cost=float(option_cost),hedge_cost=float(hedge_cost),
                    optionid_a=ticket.optionids[0],optionid_b=ticket.optionids[1],
                    qty_a=int(ticket.quantity[0]),qty_b=int(ticket.quantity[1]),
                    ticket_cash=float(cash),ticket_value=float(value),prior_ticket_value=ticket.value,
                    trade_qty_a=int(-ticket.quantity[0]) if closing else 0,
                    trade_qty_b=int(-ticket.quantity[1]) if closing else 0,
                    fill_price_a=float(prices[0]) if closing else None,
                    fill_price_b=float(prices[1]) if closing else None,
                    hedge_a=float(target[0]),hedge_b=float(target[1]),
                    hedge_trade_a=float(change[0]),hedge_trade_b=float(change[1]))
        nxt = None if closing else replace(ticket,cash=float(cash),hedge=target,value=float(value))
        return nxt,float(excess),info


def evaluate(m, policy):
    """Frozen policy, real quotes, complete daily NAV or explicit partial failure."""
    nav = benchmark = m.cfg.capital
    ticket = None
    rows = [dict(date=m.dates[0],nav=nav,cash_benchmark=benchmark,excess_return=0.,status='initial')]
    for t in range(m.n-1):
        try:
            action = int(policy(m,t,ticket))
            ticket,excess,info = m.step(t,ticket,action)
            new_nav = nav*(1+info['cash_return'])+excess
            benchmark *= 1+info['cash_return']
            info.update(nav=new_nav,cash_benchmark=benchmark,excess_dollars=excess,
                        excess_return=excess/nav,return_=new_nav/nav-1,
                        position=0 if ticket is None else ticket.direction)
            rows.append(info)
            nav = new_nav
            if nav<=0:
                raise UnresolvedPosition('Capital exhausted')
        except UnresolvedPosition as exc:
            return pd.DataFrame(rows).set_index('date'), str(exc)
    if ticket is not None:
        raise AssertionError('Evaluation ended with an open ticket')
    return pd.DataFrame(rows).set_index('date'), None
