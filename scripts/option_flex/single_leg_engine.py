"""Isolated action-menu extension of the frozen audited hold-to-expiry engine.

Pair actions retain the original inventory, execution and risk accounting.
Single legs have their own hedge and expiry and an entry-only net-vega cap.
"""
import numpy as np
import pandas as pd
from scripts.option_flex.contract_engine import (PreparedMarket,UnresolvedPosition,valid_quote,
    valid_greeks,choose_contract,verified_integer_split,integer_size,execution_terms)
from scripts.curve_trading.config import LADDER
WEIGHTS=[(0.,0.)]+[(s,-s) for s in LADDER]+[(s,0.) for s in LADDER]+[(0.,s) for s in LADDER]


def simulate(quotes, stocks, signals, calendar, secids, cfg, dividends=None, selections=None, prepared=None, max_single_net_vega=900.):
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
    if not np.isfinite(signals).all() or not signals.isin(range(len(WEIGHTS))).all():
        raise ValueError('Actions must be integer codes in the declared menu')
    if prepared is None:prepared=PreparedMarket(quotes,stocks,selections)
    elif any(a is not b for a,b in zip(prepared.sources,(quotes,stocks,selections))):
        raise ValueError('Prepared market belongs to different input frames')
    qindex=prepared.quotes;sindex=prepared.stocks;selection_ids=prepared.selection_ids
    dividends = (pd.DataFrame(columns=['secid', 'ex_date', 'declare_date', 'amount'])
                 if dividends is None else dividends.copy())
    cash = cfg.capital
    hedge = {s: 0. for s in secids}
    positions, trades, daily, decisions = [], [], [], []
    previous_spots, previous_adjustments = {}, {}
    previous_rate = 0.
    previous_nav = cfg.capital
    previous_date = None
    previous_unavailable = set()
    repaired_quotes = set()
    greek_fallbacks = set()
    # These events are fixed input data, but only declarations known on each
    # day are admitted. Precomputing avoids repeated scans per open cohort.
    next_dividend={}
    cash_events={}
    for event in dividends.itertuples():
        j=calendar.searchsorted(event.ex_date)
        if 0<j<len(calendar):
            previous,date=calendar[j-1],calendar[j]
            cash_events.setdefault(date,[]).append((int(event.secid),event.amount))
            if event.declare_date<=previous:
                next_dividend.setdefault(previous,set()).add(int(event.secid))
    if cfg.risk_controls:
        from scripts.option_flex.contract_risk import book_risk, risk_breaches

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
        if cfg.hold_to_expiry and position is not None:
            if q.get('bs_status') == 'unsupported_distribution':
                raise UnresolvedPosition(f'{date.date()}: unsupported distribution for {oid}')
            sigma = float(q.get('bs_iv', np.nan))
            if valid_greeks(q) and np.isfinite(sigma) and sigma > 0:
                position['last_iv'] = sigma
            elif not valid_greeks(q) or not np.isfinite(sigma) or sigma <= 0:
                from scripts.option_flex.black_scholes import option_values
                sigma = position.get('last_iv', np.nan)
                rate = float(q.get('bs_rate', np.nan))
                pv = float(q.get('known_dividend_pv', np.nan))
                if not np.isfinite([sigma, rate, pv]).all() or sigma <= 0:
                    raise UnresolvedPosition(f'{date.date()}: cannot construct causal hedge for {oid}')
                if spots[position['secid']]-pv <= 0 or q.expiry <= date:
                    raise UnresolvedPosition(f'{date.date()}: invalid hedge model inputs for {oid}')
                _, delta, gamma, vega = option_values(spots[position['secid']]-pv,
                    q.strike, (q.expiry-date).days/365, rate, sigma,q.cp_flag)
                q = q.copy()
                q.update(delta=float(delta), vega=float(vega), bs_iv=sigma, bs_gamma=float(gamma))
                greek_fallbacks.add((date, oid))
        if position is not None and np.isfinite(q.vega) and q.vega >= 0:
            position['last_vega'] = float(q.vega)
        return q

    for i, date in enumerate(calendar):
        scenario_cache={}
        spots, stock_adjustments, rates = {}, {}, []
        unavailable = set()
        split_factors = {s: 1. for s in secids}
        for s in secids:
            try:
                row = sindex.loc[(date, s)]
            except KeyError as exc:
                raise UnresolvedPosition(f'{date.date()}: missing underlying {s}') from exc
            if not np.isfinite([row.close, row.cfadj, row.rate]).all() or row.close <= 0:
                if (cfg.allow_missing_flat_underlying and np.isfinite(row.rate) and hedge[s] == 0 and
                        not any(p['secid'] == s for p in positions)):
                    unavailable.add(s)
                    # Bookkeeping placeholders only while entirely flat in this
                    # name. Entry is blocked; no held price is filled forward.
                    spots[s]=previous_spots.get(s,0.);stock_adjustments[s]=previous_adjustments.get(s,1.)
                    rates.append(float(row.rate));continue
                raise UnresolvedPosition(f'{date.date()}: invalid underlying {s}')
            spots[s], stock_adjustments[s] = float(row.close), float(row.cfadj)
            if s in previous_adjustments and row.cfadj != previous_adjustments[s] and (
                    hedge[s] != 0 or any(p['secid'] == s for p in positions)):
                held = [p for p in positions if p['secid'] == s]
                ratio = row.cfadj/previous_adjustments[s]
                marks = [get_quote(date,p['optionid']) for p in held]
                if not (cfg.support_integer_splits and held and
                        all(verified_integer_split(p,q,ratio) for p,q in zip(held,marks))):
                    raise UnresolvedPosition(f'{date.date()}: held underlying corporate action {s}')
                ratio = int(round(ratio)); split_factors[s] = ratio
                for p,q in zip(held,marks):
                    # A zero-cash inventory adjustment, not an option execution.
                    trades.append(dict(date=date,cohort=p['cohort'],secid=s,optionid=p['optionid'],
                                       quantity=p['quantity']*(ratio-1),price=0.,commission=0.,execution_penalty=0.,reason='split_adjustment'))
                    p['quantity'] *= ratio; p['strike'] = q.strike; p['cfadj'] = q.cfadj
                    p['last_vega'] /= ratio
                hedge[s] *= ratio
                previous_spots[s] /= ratio
            rates.append(float(row.rate))
        interest, borrow, dividend_cash = 0., 0., 0.
        if previous_date is not None:
            elapsed = (date - previous_date).days / 365
            interest = cash * (previous_rate + (cfg.borrowing_spread if cash < 0 else 0)) * elapsed
            borrow = sum(max(-hedge[s], 0) * previous_spots[s] for s in secids) * cfg.stock_borrow_rate * elapsed
            dividend_cash = sum(hedge[s] * amount for s,amount in cash_events.get(date,[]))
            cash += interest - borrow + dividend_cash
        option_cost = 0.
        option_model_penalty = 0.
        if cfg.hold_to_expiry:
            kept = []
            for p in positions:
                if date < p['exit_date']:
                    kept.append(p)
                    continue
                price = max((spots[p['secid']]-p['strike'])*(1 if p.get('cp_flag','C')=='C' else -1), 0.)
                cash += p['quantity']*100*price
                trades.append(dict(date=date, cohort=p['cohort'], secid=p['secid'],
                    optionid=p['optionid'], quantity=-p['quantity'], price=price,
                    commission=0., execution_penalty=0., reason='expiry_settlement'))
            positions = kept
        # Close both legs before either expires, or before a known ex-dividend date.
        # This reduces assignment exposure but does not model arbitrary early exercise.
        close_groups = {}
        for p in positions:
            if cfg.hold_to_expiry:
                continue
            held_quote = get_quote(date, p['optionid'], p)
            # Existing prices remain usable when vendor inversion fails. Close
            # the WHOLE pair at quoted sides instead of imputing its Greeks.
            if not valid_greeks(held_quote):
                close_groups[p['cohort']] = 'missing_greeks_exit'
            if date >= p['exit_date']:
                close_groups.setdefault(p['cohort'], 'scheduled_exit')
            if i + 1 < len(calendar):
                if p['secid'] in next_dividend.get(date,set()):
                    close_groups[p['cohort']] = 'pre_ex_dividend_exit'
        kept = []
        for p in positions:
            if p['cohort'] not in close_groups:
                kept.append(p)
                continue
            q = get_quote(date, p['optionid'], p)
            qty = -p['quantity']
            price, penalty, slippage = execution_terms(q,qty,cfg,p.get('last_vega'))
            fee = abs(qty) * cfg.option_commission
            cash -= qty * 100 * price + fee + penalty
            option_cost += slippage + fee
            option_model_penalty += penalty
            trades.append(dict(date=date, cohort=p['cohort'], secid=p['secid'],
                               optionid=p['optionid'], quantity=qty, price=price,
                               commission=fee, execution_penalty=penalty, reason=close_groups[p['cohort']]))
        positions = kept
        risk_exit_count=0
        pre_risk_breach=False

        def assess(book, cash_balance):
            metrics=book_risk(book,lambda p:get_quote(date,p['optionid'],p),spots,date,cfg,scenario_cache)
            hedge_cost_estimate=sum(abs(metrics['target_hedge'][s]-hedge[s])*spots[s] for s in secids)*cfg.stock_cost_bps/10_000
            marked=cash_balance+sum(hedge[s]*spots[s] for s in secids)+metrics['option_value']
            return metrics,marked-hedge_cost_estimate

        if cfg.risk_controls:
            # Reassess existing positions using current marks, before new risk.
            # Whole-cohort liquidation preserves both legs; hedge rebalance below
            # nets all today's changes and charges stock costs once.
            while positions:
                metrics,projected_nav=assess(positions,cash)
                if not risk_breaches(metrics,projected_nav,cfg):break
                pre_risk_breach=True
                if cfg.hold_to_expiry:
                    break
                oldest=positions[0]['cohort']
                for p in [p for p in positions if p['cohort']==oldest]:
                    q=get_quote(date,p['optionid'],p);qty=-p['quantity']
                    price,penalty,slippage=execution_terms(q,qty,cfg,p.get('last_vega'))
                    fee=abs(qty)*cfg.option_commission
                    cash-=qty*100*price+fee+penalty
                    option_cost+=slippage+fee
                    option_model_penalty+=penalty
                    trades.append(dict(date=date,cohort=oldest,secid=p['secid'],optionid=p['optionid'],
                                       quantity=qty,price=price,commission=fee,execution_penalty=penalty,reason='risk_limit_exit'))
                positions=[p for p in positions if p['cohort']!=oldest]
                risk_exit_count+=1
            _,sizing_nav=assess(positions,cash)
        else:
            sizing_nav=cfg.capital
        # Lagged signals select lagged contracts; the current quote is only a
        # fill/eligibility/sizing observation, never a replacement signal.
        if i > 0 and calendar[i-1] in signals.index:
            signal_date = calendar[i-1]
            action = int(signals.loc[signal_date])
            weights = WEIGHTS[action]
            active_secids = [s for s,w in zip(secids,weights) if w]
            directions = [int(np.sign(w)) for w in weights if w]
            level = max(abs(w) for w in weights)
            decision = dict(signal_date=signal_date, entry_date=date, action=action,
                            status='flat')
            if action != 0 and (unavailable or previous_unavailable):
                decision['status']='underlying_unavailable_at_entry'
            elif action != 0:
                selected = []
                decision['status'] = 'no_eligible_contract'
                for s in active_secids:
                    if selections is None:
                        chain = quotes[(quotes.date == signal_date) & (quotes.secid == s)]
                    else:
                        oid = selection_ids.get((signal_date,s))
                        key = (signal_date,oid)
                        if oid is None or key not in qindex.index:
                            break
                        row = qindex.loc[key]
                        if row.secid != s:
                            raise ValueError('Selected contract belongs to another underlying')
                        dte=(row.expiry-date).days
                        chosen=row if (valid_quote(row,True,cfg) and cfg.min_dte<=dte<=cfg.max_dte and
                                       abs(row.strike/previous_spots[s]-1)<=cfg.max_moneyness_distance) else None
                    if selections is None:chosen = choose_contract(chain, previous_spots[s], date, cfg)
                    if chosen is None:
                        break
                    selected.append(chosen)
                if len(selected) == len(active_secids):
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
                    if len(live) == len(active_secids):
                        target_exit = min(date + pd.Timedelta(days=cfg.holding_days),
                                          min(q.expiry for q in selected) - pd.Timedelta(days=1))
                        if cfg.hold_to_expiry:
                            target_exit = max(q.expiry for q in selected)
                        exits = calendar[(calendar > date) & (calendar <= target_exit)]
                        decision['status'] = 'insufficient_runoff'
                        # A later session is needed to establish the final trading
                        # session for weekend/holiday expirations without guessing.
                        if len(exits) and target_exit <= calendar[-1]:
                            exit_date = exits[-1]
                            target = cfg.entry_vega_per_leg*level*max(0.,sizing_nav)/cfg.capital
                            size = integer_size([q.vega*100 for q in live],target,cfg.max_entry_vega_imbalance) if len(live)==2 else [int(np.floor(target/(live[0].vega*100)))]
                            if size is not None and min(size)<1:size=None
                            decision['sizing_nav']=sizing_nav
                            decision['status'] = 'integer_vega_mismatch'
                            if size is not None:
                                gross = sum(abs(p['quantity'])*100*get_quote(date, p['optionid'], p).vega for p in positions)
                                added = sum(n*100*q.vega for n, q in zip(size, live))
                                # No entry immediately before a known ex-dividend date.
                                decision['status'] = 'gross_vega_limit'
                                entry_risk=[]
                                if cfg.risk_controls:
                                    trial=positions.copy();trial_cash=cash
                                    for leg,(s,old,q,n) in enumerate(zip(active_secids,selected,live,size)):
                                        qty=int(n)*directions[leg]
                                        price,penalty,_=execution_terms(q,qty,cfg)
                                        trial_cash-=qty*100*price+abs(qty)*cfg.option_commission+penalty
                                        trial.append(dict(cohort=str(signal_date.date()),secid=s,optionid=int(old.optionid),quantity=qty,
                                                          expiry=q.expiry,strike=q.strike,cfadj=q.cfadj,cp_flag=q.cp_flag,exit_date=exit_date,single_leg=len(live)==1))
                                    metrics,projected_nav=assess(trial,trial_cash)
                                    entry_risk=risk_breaches(metrics,projected_nav,cfg)
                                current_net=sum(p['quantity']*100*get_quote(date,p['optionid'],p).vega for p in positions)
                                proposed_net=current_net+sum(n*100*q.vega*direction for n,q,direction in zip(size,live,directions))
                                if len(live)==1 and abs(proposed_net)>max_single_net_vega*max(0.,sizing_nav)/cfg.capital+1e-8:
                                    decision['status']='single_net_vega_limit'
                                elif risk_exit_count or (cfg.hold_to_expiry and pre_risk_breach):
                                    decision['status']='risk_deleveraging'
                                elif next_dividend.get(date,set()) and not cfg.hold_to_expiry:
                                    decision['status'] = 'pre_ex_dividend'
                                elif entry_risk:
                                    decision['status']='risk_limit_entry'
                                    decision['risk_limits']=','.join(entry_risk)
                                elif cfg.risk_controls or gross + added <= cfg.max_gross_vega:
                                    decision['status'] = 'entered'
                                    direction = directions[0]
                                    for leg, (s, old, q, n) in enumerate(zip(active_secids, selected, live, size)):
                                        qty = int(n)*directions[leg]
                                        price,penalty,slippage=execution_terms(q,qty,cfg)
                                        fee = abs(qty)*cfg.option_commission
                                        cash -= qty*100*price + fee + penalty
                                        option_cost += slippage + fee
                                        option_model_penalty += penalty
                                        position = dict(cohort=str(signal_date.date()), secid=s,
                                                        optionid=int(old.optionid), quantity=qty,
                                                        expiry=q.expiry, strike=q.strike, cfadj=q.cfadj, cp_flag=q.cp_flag,
                                                        exit_date=exit_date,last_vega=float(q.vega),single_leg=len(live)==1)
                                        if cfg.hold_to_expiry:
                                            position['exit_date'] = calendar[calendar <= q.expiry][-1]
                                            position['last_iv'] = float(q.get('bs_iv', np.nan))
                                        positions.append(position)
                                        trades.append(dict(date=date, cohort=position['cohort'], secid=s,
                                                           optionid=position['optionid'], quantity=qty,
                                                           price=price, commission=fee, execution_penalty=penalty, reason='entry'))
                                    entry_vegas={s:n*100*q.vega for s,n,q in zip(active_secids,size,live)}
                                    entry_ids={s:int(q.optionid) for s,q in zip(active_secids,selected)}
                                    decision.update(vega_a=entry_vegas.get(secids[0],0.),vega_b=entry_vegas.get(secids[1],0.),
                                        optionid_a=entry_ids.get(secids[0]),optionid_b=entry_ids.get(secids[1]),exit_date=exit_date,
                                        entry_kind='pair' if len(live)==2 else 'single_a' if active_secids[0]==secids[0] else 'single_b',
                                        entry_sign=directions[0],proposed_net_vega=proposed_net)
            decisions.append(decision)
        option_value, net_vega, gross_vega = 0., 0., 0.
        single_net_vega=single_gross_vega=pair_net_vega=0.
        target_hedge = {s: 0. for s in secids}
        for p in positions:
            q = get_quote(date, p['optionid'], p)
            option_value += p['quantity']*100*(q.bid+q.ask)/2
            target_hedge[p['secid']] -= p['quantity']*100*q.delta
            v = p['quantity']*100*q.vega
            net_vega += v
            gross_vega += abs(v)
            if p.get('single_leg'):
                single_net_vega+=v;single_gross_vega+=abs(v)
            else:pair_net_vega+=v
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
            failure = UnresolvedPosition(f'{date.date()}: nonpositive NAV; capital exhausted')
            failure.diagnostics = dict(date=str(date.date()), nav=float(nav), cash=float(cash),
                                       stock_value=float(stock_value), option_value=float(option_value),
                                       option_execution_cost=float(sum(x['option_execution_cost'] for x in daily)+option_cost),
                                       hedge_cost=float(sum(x['hedge_cost'] for x in daily)+stock_cost),
                                       stock_borrow_cost=float(sum(x['stock_borrow_cost'] for x in daily)+borrow))
            failure.partial_nav = pd.DataFrame(daily).set_index('date')
            failure.trades = pd.DataFrame(trades)
            raise failure
        risk_stats={}
        if cfg.risk_controls:
            metrics=book_risk(positions,lambda p:get_quote(date,p['optionid'],p),spots,date,cfg,scenario_cache)
            violations=risk_breaches(metrics,nav,cfg)
            if violations and not cfg.hold_to_expiry:
                raise AssertionError('Post-trade risk limits violated: '+','.join(violations))
            risk_stats=dict(sizing_nav=sizing_nav,risk_exit_cohorts=risk_exit_count,
                            post_trade_risk_breach=bool(violations),
                            pre_trade_risk_breach=pre_risk_breach,
                            gross_hedge_notional=metrics['gross_hedge_notional'],
                            capital_reserve=metrics['capital_reserve'],stress_loss=metrics['stress_loss'])
        daily.append(dict(date=date, nav=nav, pnl=nav-previous_nav,
                          return_=nav/previous_nav-1, cash=cash, option_value=option_value,
                          stock_value=stock_value, net_vega=net_vega, gross_vega=gross_vega,
                          single_net_vega=single_net_vega,single_gross_vega=single_gross_vega,pair_net_vega=pair_net_vega,
                          gross_vega_limit_breached=gross_vega > cfg.max_gross_vega*(nav/cfg.capital if cfg.risk_controls else 1),
                          active_cohorts=len(set(p['cohort'] for p in positions)),
                          option_execution_cost=option_cost, option_model_penalty=option_model_penalty, hedge_cost=stock_cost,
                          hedge_turnover=stock_turnover, interest=interest,
                          stock_borrow_cost=borrow, dividend_cash=dividend_cash,
                          repaired_held_quotes=sum(d == date for d, _ in repaired_quotes),
                          greek_fallbacks=sum(d == date for d, _ in greek_fallbacks),
                          **risk_stats,
                          **{f'split_factor_{s}': split_factors[s] for s in secids},
                          **{f'hedge_{s}': hedge[s] for s in secids}))
        previous_spots, previous_adjustments = spots, stock_adjustments
        previous_unavailable = unavailable
        previous_nav, previous_date, previous_rate = nav, date, float(np.mean(rates))
    if positions:
        raise UnresolvedPosition('Unclosed positions at end of data')
    return pd.DataFrame(daily).set_index('date'), pd.DataFrame(trades), pd.DataFrame(decisions)
