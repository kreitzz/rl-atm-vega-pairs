"""Daily cash-account simulation of initially vega-matched, delta-hedged calls.

Normalized quotes use dollar strike, dollar premium per share and dollar vega
per SHARE per ONE volatility point. Selection occurs on the signal session;
entry is next session. All hedges are executed at the daily stock mark plus
explicit costs. This is an EOD execution assumption, not an intraday fill model.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


class UnresolvedPosition(RuntimeError):
    """An existing position cannot be valued/hedged; do not drop its outcome."""


@dataclass
class Settings:
    capital: float = 1_000_000
    entry_vega_per_leg: float = 450
    max_gross_vega: float = 20_000
    max_entry_vega_imbalance: float = .10
    option_commission: float = .65
    stock_cost_bps: float = 1
    borrowing_spread: float = .01
    stock_borrow_rate: float = .003
    min_open_interest: int = 100
    max_relative_spread: float = .25
    max_moneyness_distance: float = .10
    min_dte: int = 21
    max_dte: int = 45
    target_dte: int = 30
    holding_days: int = 30
    repair_crossed_held_quotes: bool = False

    def __post_init__(self):
        if min(self.capital, self.entry_vega_per_leg, self.max_gross_vega) <= 0:
            raise ValueError('Capital and vega limits must be positive')
        if min(self.option_commission, self.stock_cost_bps,
               self.borrowing_spread, self.stock_borrow_rate) < 0:
            raise ValueError('Costs must be nonnegative')


def valid_greeks(q):
    return np.isfinite([q['delta'], q['vega']]).all() and 0 <= q['delta'] <= 1 and q['vega'] >= 0


def valid_quote(q, entry=False, cfg=None):
    fields = ['bid', 'ask', 'strike', 'multiplier', 'cfadj']
    if not all(np.isfinite(float(q[x])) for x in fields):
        return False
    if not (0 <= q['bid'] <= q['ask'] and q['ask'] > 0 and q['strike'] > 0):
        return False
    if q['multiplier'] != 100 or not q['standard'] or q['cp_flag'] != 'C':
        return False
    if entry:
        mid = (q['bid'] + q['ask']) / 2
        return (valid_greeks(q) and q['bid'] > 0 and q['vega'] > 0 and
                q['open_interest'] >= cfg.min_open_interest and
                (q['ask'] - q['bid']) / mid <= cfg.max_relative_spread)
    return True


def choose_contract(chain, spot, entry_date, cfg):
    """Known prior-close chain; maturity measured at next session's entry."""
    if chain.empty:
        return None
    q = chain.copy()
    q['dte'] = (q.expiry - entry_date).dt.days
    q['distance'] = abs(q.strike / spot - 1)
    q = q[q.apply(lambda x: valid_quote(x, True, cfg), axis=1) &
          q.dte.between(cfg.min_dte, cfg.max_dte) &
          q.distance.le(cfg.max_moneyness_distance)]
    if q.empty:
        return None
    q['tenor_distance'] = abs(q.dte - cfg.target_dte)
    return q.sort_values(['tenor_distance', 'distance', 'open_interest', 'optionid'],
                         ascending=[True, True, False, True]).iloc[0]


def integer_size(vegas, target, tolerance):
    """Floor each leg to its budget; reject excessive residual vega."""
    v = np.asarray(vegas, dtype=float)
    if not np.isfinite(v).all() or (v <= 0).any():
        return None
    n = np.floor(target / v).astype(int)
    actual = n * v
    if (n < 1).any() or abs(actual[0]-actual[1]) / actual.max() > tolerance:
        return None
    return n


def simulate(quotes, stocks, signals, calendar, secids, cfg, dividends=None):
    """Return daily NAV, immutable trade records, and every signal disposition.

    stocks: date,secid,close,cfadj,rate (decimal, known as of that close).
    dividends: secid,ex_date,declare_date,amount (ordinary USD cash dividends).
    No price filling or hindsight deletion. Any unsupported held-contract
    adjustment or missing mark raises, leaving the run incomplete.
    """
    calendar = pd.DatetimeIndex(calendar).sort_values()
    if calendar.has_duplicates or len(calendar) < 2:
        raise ValueError('Need a unique trading calendar with runoff sessions')
    if len(secids) != 2 or len(set(secids)) != 2:
        raise ValueError('Exactly two distinct underlying IDs are required')
    if signals.index.has_duplicates or not signals.index.isin(calendar).all():
        raise ValueError('Signals must be unique and on the supplied calendar')
    if len(signals) and signals.index.max() >= calendar[-1]:
        raise ValueError('Every signal needs a subsequent execution session')
    if not np.isfinite(signals).all() or not signals.between(-1, 1).all():
        raise ValueError('Actions must be finite and in [-1, 1]')
    if quotes.duplicated(['date', 'optionid']).any() or stocks.duplicated(['date', 'secid']).any():
        raise ValueError('Duplicate market-data keys')
    qindex = quotes.set_index(['date', 'optionid'])
    sindex = stocks.set_index(['date', 'secid'])
    dividends = (pd.DataFrame(columns=['secid', 'ex_date', 'declare_date', 'amount'])
                 if dividends is None else dividends.copy())
    cash = cfg.capital
    hedge = {s: 0. for s in secids}
    positions, trades, daily, decisions = [], [], [], []
    previous_spots, previous_adjustments = {}, {}
    previous_rate = 0.
    previous_nav = cfg.capital
    previous_date = None
    repaired_quotes = set()

    def get_quote(date, oid, position=None):
        try:
            q = qindex.loc[(date, oid)]
        except KeyError as exc:
            raise UnresolvedPosition(f'{date.date()}: missing held option {oid}') from exc
        if cfg.repair_crossed_held_quotes and 0 <= q.ask < q.bid:
            q = q.copy()
            q['bid'], q['ask'] = q.ask, q.bid
            repaired_quotes.add((date, oid))
        if not valid_quote(q):
            raise UnresolvedPosition(f'{date.date()}: invalid held quote {oid}')
        if position is not None and (q.cfadj != position['cfadj'] or
                                    q.strike != position['strike'] or
                                    q.expiry != position['expiry']):
            raise UnresolvedPosition(f'{date.date()}: adjusted held contract {oid}')
        return q

    for i, date in enumerate(calendar):
        spots, stock_adjustments, rates = {}, {}, []
        for s in secids:
            try:
                row = sindex.loc[(date, s)]
            except KeyError as exc:
                raise UnresolvedPosition(f'{date.date()}: missing underlying {s}') from exc
            if not np.isfinite([row.close, row.cfadj, row.rate]).all() or row.close <= 0:
                raise UnresolvedPosition(f'{date.date()}: invalid underlying {s}')
            spots[s], stock_adjustments[s] = float(row.close), float(row.cfadj)
            if s in previous_adjustments and row.cfadj != previous_adjustments[s] and (
                    hedge[s] != 0 or any(p['secid'] == s for p in positions)):
                raise UnresolvedPosition(f'{date.date()}: held underlying corporate action {s}')
            rates.append(float(row.rate))
        interest, borrow, dividend_cash = 0., 0., 0.
        if previous_date is not None:
            elapsed = (date - previous_date).days / 365
            interest = cash * (previous_rate + (cfg.borrowing_spread if cash < 0 else 0)) * elapsed
            borrow = sum(max(-hedge[s], 0) * previous_spots[s] for s in secids) * cfg.stock_borrow_rate * elapsed
            events = dividends[(dividends.ex_date > previous_date) & (dividends.ex_date <= date)]
            dividend_cash = sum(hedge[int(d.secid)] * d.amount for d in events.itertuples())
            cash += interest - borrow + dividend_cash
        option_cost = 0.
        # Close both legs before either expires, or before a known ex-dividend date.
        # This reduces assignment exposure but does not model arbitrary early exercise.
        close_groups = {}
        for p in positions:
            held_quote = get_quote(date, p['optionid'], p)
            # Existing prices remain usable when vendor inversion fails. Close
            # the WHOLE pair at quoted sides instead of imputing its Greeks.
            if not valid_greeks(held_quote):
                close_groups[p['cohort']] = 'missing_greeks_exit'
            if date >= p['exit_date']:
                close_groups.setdefault(p['cohort'], 'scheduled_exit')
            if i + 1 < len(calendar):
                ev = dividends[(dividends.secid == p['secid']) &
                               (dividends.declare_date <= date) &
                               (dividends.ex_date > date) &
                               (dividends.ex_date <= calendar[i+1])]
                if not ev.empty:
                    close_groups[p['cohort']] = 'pre_ex_dividend_exit'
        kept = []
        for p in positions:
            if p['cohort'] not in close_groups:
                kept.append(p)
                continue
            q = get_quote(date, p['optionid'], p)
            qty = -p['quantity']
            price = float(q.ask if qty > 0 else q.bid)
            fee = abs(qty) * cfg.option_commission
            cash -= qty * 100 * price + fee
            option_cost += abs(qty)*100*(q.ask-q.bid)/2 + fee
            trades.append(dict(date=date, cohort=p['cohort'], secid=p['secid'],
                               optionid=p['optionid'], quantity=qty, price=price,
                               commission=fee, reason=close_groups[p['cohort']]))
        positions = kept
        # Lagged signals select lagged contracts; the current quote is only a
        # fill/eligibility/sizing observation, never a replacement signal.
        if i > 0 and calendar[i-1] in signals.index:
            signal_date = calendar[i-1]
            action = float(signals.loc[signal_date])
            decision = dict(signal_date=signal_date, entry_date=date, action=action,
                            status='flat')
            if action != 0:
                selected = []
                decision['status'] = 'no_eligible_contract'
                for s in secids:
                    chain = quotes[(quotes.date == signal_date) & (quotes.secid == s)]
                    chosen = choose_contract(chain, previous_spots[s], date, cfg)
                    if chosen is None:
                        break
                    selected.append(chosen)
                if len(selected) == 2:
                    live = []
                    decision['status'] = 'entry_quote_unavailable'
                    for q in selected:
                        key = (date, q.optionid)
                        if key not in qindex.index:
                            break
                        x = qindex.loc[key]
                        # Today's reported OI is published the next morning;
                        # only prior-session OI is available at today's entry.
                        eligibility = x.copy()
                        eligibility['open_interest'] = q.open_interest
                        if (not valid_quote(eligibility, True, cfg) or x.cfadj != q.cfadj or
                                x.strike != q.strike or x.expiry != q.expiry or
                                x.secid != q.secid):
                            break
                        live.append(x)
                    if len(live) == 2:
                        target_exit = min(date + pd.Timedelta(days=cfg.holding_days),
                                          min(q.expiry for q in selected) - pd.Timedelta(days=1))
                        exits = calendar[(calendar > date) & (calendar <= target_exit)]
                        decision['status'] = 'insufficient_runoff'
                        if len(exits) and target_exit <= calendar[-1]:
                            exit_date = exits[-1]
                            size = integer_size([q.vega*100 for q in live],
                                                cfg.entry_vega_per_leg*abs(action),
                                                cfg.max_entry_vega_imbalance)
                            decision['status'] = 'integer_vega_mismatch'
                            if size is not None:
                                gross = sum(abs(p['quantity'])*100*get_quote(date, p['optionid'], p).vega for p in positions)
                                added = sum(n*100*q.vega for n, q in zip(size, live))
                                # No entry immediately before a known ex-dividend date.
                                events = dividends[(dividends.declare_date <= date) &
                                                   (dividends.ex_date > date) &
                                                   (dividends.ex_date <= calendar[i+1])]
                                decision['status'] = 'gross_vega_limit'
                                if not events.empty:
                                    decision['status'] = 'pre_ex_dividend'
                                elif gross + added <= cfg.max_gross_vega:
                                    decision['status'] = 'entered'
                                    direction = int(np.sign(action))
                                    for leg, (s, old, q, n) in enumerate(zip(secids, selected, live, size)):
                                        qty = int(n)*direction*(1 if leg == 0 else -1)
                                        price = float(q.ask if qty > 0 else q.bid)
                                        fee = abs(qty)*cfg.option_commission
                                        cash -= qty*100*price + fee
                                        option_cost += abs(qty)*100*(q.ask-q.bid)/2 + fee
                                        position = dict(cohort=str(signal_date.date()), secid=s,
                                                        optionid=int(old.optionid), quantity=qty,
                                                        expiry=q.expiry, strike=q.strike, cfadj=q.cfadj,
                                                        exit_date=exit_date)
                                        positions.append(position)
                                        trades.append(dict(date=date, cohort=position['cohort'], secid=s,
                                                           optionid=position['optionid'], quantity=qty,
                                                           price=price, commission=fee, reason='entry'))
                                    decision.update(vega_a=size[0]*100*live[0].vega,
                                                    vega_b=size[1]*100*live[1].vega,
                                                    optionid_a=int(selected[0].optionid),
                                                    optionid_b=int(selected[1].optionid), exit_date=exit_date)
            decisions.append(decision)
        option_value, net_vega, gross_vega = 0., 0., 0.
        target_hedge = {s: 0. for s in secids}
        for p in positions:
            q = get_quote(date, p['optionid'], p)
            option_value += p['quantity']*100*(q.bid+q.ask)/2
            target_hedge[p['secid']] -= p['quantity']*100*q.delta
            v = p['quantity']*100*q.vega
            net_vega += v
            gross_vega += abs(v)
        stock_cost, stock_turnover = 0., 0.
        for s in secids:
            change = target_hedge[s] - hedge[s]
            notional = change*spots[s]
            cost = abs(notional)*cfg.stock_cost_bps/10_000
            cash -= notional + cost
            stock_cost += cost
            stock_turnover += abs(notional)
            hedge[s] = target_hedge[s]
        stock_value = sum(hedge[s]*spots[s] for s in secids)
        nav = cash + stock_value + option_value
        if nav <= 0:
            raise UnresolvedPosition(f'{date.date()}: nonpositive NAV; capital exhausted')
        daily.append(dict(date=date, nav=nav, pnl=nav-previous_nav,
                          return_=nav/previous_nav-1, cash=cash, option_value=option_value,
                          stock_value=stock_value, net_vega=net_vega, gross_vega=gross_vega,
                          gross_vega_limit_breached=gross_vega > cfg.max_gross_vega,
                          active_cohorts=len(set(p['cohort'] for p in positions)),
                          option_execution_cost=option_cost, hedge_cost=stock_cost,
                          hedge_turnover=stock_turnover, interest=interest,
                          stock_borrow_cost=borrow, dividend_cash=dividend_cash,
                          repaired_held_quotes=sum(d == date for d, _ in repaired_quotes),
                          **{f'hedge_{s}': hedge[s] for s in secids}))
        previous_spots, previous_adjustments = spots, stock_adjustments
        previous_nav, previous_date, previous_rate = nav, date, float(np.mean(rates))
    if positions:
        raise UnresolvedPosition('Unclosed positions at end of data')
    return pd.DataFrame(daily).set_index('date'), pd.DataFrame(trades), pd.DataFrame(decisions)
