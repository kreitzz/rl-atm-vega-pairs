"""Isolated ATM residual-study ledger fork.

Financial formulas copied from year_prefix_engine. Standalone entries also
permit shorts; an optional immutable schedule cache speeds checkpointed replay.
Paired accounting and risk behavior are identical.
"""
from collections import defaultdict
from dataclasses import replace
from copy import deepcopy

import numpy as np
import pandas as pd

from scripts.option_flex.contract_engine import (MarketRow, UnresolvedPosition,
    valid_quote, valid_greeks, integer_size, verified_integer_split, execution_terms)
from scripts.option_flex.contract_risk import book_risk as ordinary_book_risk, risk_breaches
from scripts.option_flex.black_scholes import option_values


def cash_book_risk(positions,quotes,spots,date,cfg,scenario_cache=None,base_risk=ordinary_book_risk):
    """A cash-only option deliverable has no remaining spot or volatility risk."""
    cash_positions=[p for p in positions if 'cash_deliverable' in p]
    normal=[p for p in positions if 'cash_deliverable' not in p]
    result=base_risk(normal,quotes,spots,date,cfg,scenario_cache)
    value=sum(p['quantity']*100*p['cash_deliverable'] for p in cash_positions)
    premium=sum(abs(p['quantity'])*100*p['cash_deliverable'] for p in cash_positions)
    result['option_value']+=value
    result['gross_option_premium']+=premium
    result['capital_reserve']+=premium
    return result


book_risk=cash_book_risk


def inventory_features(pair_metrics,global_metrics,pair_capital,capital,max_vega):
    return dict(pair_inventory_vega=pair_metrics.get('gross_vega',0.)/(max_vega*pair_capital/capital),
                pair_inventory_hedge=pair_metrics.get('gross_hedge_notional',0.)/(2*pair_capital),
                pair_inventory_cohorts=pair_metrics.get('active_cohorts',0.)/10,
                global_inventory_vega=global_metrics.get('gross_vega',0.)/max_vega,
                global_inventory_hedge=global_metrics.get('gross_hedge_notional',0.)/(2*capital))


def simulate(market,signals,calendar,cfg,pair_capital,policy=None,corporate_events=(),wing_capital=None,scale_policy=None,scale_pair_budget=False,management_policy=None,locked_entry_counts=None,require_runoff=True,settlement_calendar=None,standalone=False,resume_state=None,through_date=None,max_stale_quote_sessions=None,prepared_schedule=None,straddle=False):
    """Signals are previous-close rows with fixed pair, wing and option IDs.

    Sizes are absolute fractions of entry_vega_per_leg. Policy uses only its
    signal-close state and that close's ACTUAL shared inventory. Current entry
    quotes enter eligibility/integer sizing/risk checks only. Held cohorts keep
    their own expiry. Shared cash, stock costs, borrow and dividends are netted.
    Optional locked_entry_counts is a counterfactual attribution control only:
    original execution counts, with all current risk checks still enforced.
    Standalone mode permits either direction, with secid_b=secid_a and no second option.
    Each name has one signal per date; there is no entry-vega matching.
    Straddle mode requires one call and put with identical underlying, strike,
    expiry and deliverable. Both trade in the same direction, with equal integer
    counts and a TOTAL vega budget across the two options.
    Optional chunks retain the full known calendar and resume the exact cash,
    inventory, pending next-session orders and previous-close hedge state.
    """
    if not require_runoff and settlement_calendar is None:
        raise ValueError('Training prefix requires a full known settlement calendar')
    cal = pd.DatetimeIndex(calendar)
    if not cfg.hold_to_expiry or not cfg.risk_controls:
        raise ValueError('Shared test requires the frozen expiry and risk conventions')
    if cal.has_duplicates or len(cal)<2:
        raise ValueError('Unique calendar with runoff required')
    if signals.duplicated(['date','pair','wing']).any():
        raise ValueError('Duplicate pair/wing signal')
    if not pd.DatetimeIndex(signals.date).isin(cal[:-1]).all():
        raise ValueError('Every signal needs a subsequent execution session')
    if not signals.base_level.isin([1,2,3,4]).all() or not signals.direction.isin([-1,1]).all():
        raise ValueError('Only frozen nonzero direction proposals are executable')
    if standalone and (not signals.secid_a.eq(signals.secid_b).all()
                       or signals.duplicated(['date','secid_a']).any() or signals.optionid_b.notna().any()):
        raise ValueError('Standalone entries require unique names and an absent second option')
    if straddle and (standalone or not signals.secid_a.eq(signals.secid_b).all()
                      or signals.duplicated(['date','secid_a']).any() or signals.optionid_b.isna().any()):
        raise ValueError('Straddles require unique names and two options on the same underlying')
    secids = sorted(set(signals.secid_a)|set(signals.secid_b))
    pair_members={r.pair:{int(r.secid_a),int(r.secid_b)} for r in signals.itertuples()}
    pair_cfg = replace(cfg,capital=pair_capital,max_gross_vega=cfg.max_gross_vega*pair_capital/cfg.capital)
    wing_cfg=None if wing_capital is None else replace(cfg,capital=wing_capital,max_gross_vega=cfg.max_gross_vega*wing_capital/cfg.capital)
    schedule = prepared_schedule
    if schedule is None:
        schedule = {pd.Timestamp(d):g.sort_values(['pair','wing'],kind='stable') for d,g in signals.groupby('date',sort=False)}
    if prepared_schedule is None and (standalone or straddle) and 'priority' in signals:
        schedule={pd.Timestamp(d):g.sort_values(['priority','pair'],ascending=[False,True],kind='stable')
                  for d,g in signals.groupby('date',sort=False)}
    events = defaultdict(set)
    cash_mergers=defaultdict(dict)
    entry_blocks={int(e['secid']):max(pd.Timestamp(e['no_entry_from']),pd.Timestamp(e['announced']))
                  for e in corporate_events if 'no_entry_from' in e}
    for event in corporate_events:
        if 'cash_merger_price' in event:
            after=cal[cal>=max(pd.Timestamp(event['effective']),pd.Timestamp(event['announced']))]
            if len(after):cash_mergers[after[0]][int(event['secid'])]=event
            continue
        if event.get('exit_same_day',False):
            after=cal[cal>=pd.Timestamp(event['effective'])]
            exit_day=after[0] if len(after) else None
        else:
            before=cal[cal<pd.Timestamp(event['effective'])]
            exit_day=before[-1] if len(before) else None
        if exit_day is not None and pd.Timestamp(event['announced'])<=exit_day:
            events[exit_day].add(int(event['secid']))
            # Prior-close exits must not immediately reopen adjusted
            # deliverables on the known effective session.
            effective=cal[cal>=pd.Timestamp(event['effective'])]
            if len(effective):events[effective[0]].add(int(event['secid']))
    cash_events = defaultdict(list)
    for event in market.divs.itertuples():
        j = cal.searchsorted(event.ex_date)
        if 0<j<len(cal):
            cash_events[cal[j]].append((int(event.secid),float(event.amount)))
    cash = cfg.capital; hedge = {s:0. for s in secids}; positions = []
    trades = []; decisions = []; daily = []
    previous_spots = {}; previous_adjustments = {}; previous_date = None; previous_unavailable=set()
    previous_nav = cfg.capital; previous_option_value = 0.; previous_rate = 0.
    lag_pairs = {}; lag_global = {}; max_ledger_error = 0.
    # Optional capital policy sees only the preceding completed session.
    lag_cash = cash; lag_names = {}; benchmark = cfg.capital; relative_peak = 1.
    pending_management = {}; management_log = []
    resume_date=None
    if resume_state is not None:
        state=deepcopy(resume_state)
        if set(state['hedge'])!=set(secids):raise ValueError('Resume inventory belongs to a different universe')
        cash=state['cash'];hedge=state['hedge'];positions=state['positions']
        previous_spots=state['previous_spots'];previous_adjustments=state['previous_adjustments']
        previous_date=state['previous_date'];previous_unavailable=state['previous_unavailable']
        previous_nav=state['previous_nav'];previous_option_value=state['previous_option_value'];previous_rate=state['previous_rate']
        lag_pairs=state['lag_pairs'];lag_global=state['lag_global'];max_ledger_error=state['max_ledger_error']
        lag_cash=state['lag_cash'];lag_names=state['lag_names'];benchmark=state['benchmark'];relative_peak=state['relative_peak']
        pending_management=state['pending_management'];resume_date=previous_date
        if resume_date not in cal:raise ValueError('Resume date must belong to the full trading calendar')
        for p in positions:p.setdefault('last_observed_quote_date',previous_date)
    if through_date is not None:through_date=pd.Timestamp(through_date)
    for i,date in enumerate(cal):
        if resume_date is not None and date<=resume_date:continue
        if through_date is not None and date>through_date:break
        repaired = fallbacks = modeled = 0; scenario_cache = {}; mark_cache = {}
        spots = {}; adjustments = {}; unavailable = set(); split_factors = {s:1. for s in secids}
        for sid in secids:
            spot,adjust = market.stock(date,sid)
            if sid in cash_mergers.get(date,{}) and (hedge[sid] or any(p['secid']==sid for p in positions)):
                spot=float(cash_mergers[date][sid]['cash_merger_price'])
                adjust=previous_adjustments.get(sid,adjust)
            cash_held=next((p for p in positions if p['secid']==sid and 'cash_merger_price' in p),None)
            if cash_held is not None:
                spot=cash_held['cash_merger_price'];adjust=cash_held['cash_merger_stock_adjustment']
            if not np.isfinite([spot,adjust]).all() or spot<=0:
                if hedge[sid] or any(p['secid']==sid for p in positions):
                    raise UnresolvedPosition(f'{date}: missing held underlying {sid}')
                unavailable.add(sid); spot=previous_spots.get(sid,0.);adjust=previous_adjustments.get(sid,1.)
            spots[sid]=float(spot);adjustments[sid]=float(adjust)

        def quote(oid,p=None):
            nonlocal repaired,fallbacks,modeled
            key = (int(oid),None if p is None else p['cohort'],None if p is None else p.get('last_iv'))
            if key in mark_cache:
                return mark_cache[key]
            if p is not None and 'cash_deliverable' in p:
                return MarketRow(secid=p['secid'],optionid=int(oid),date=date,strike=p['strike'],
                    expiry=p['expiry'],cp_flag=p['cp_flag'],cfadj=p['cfadj'],multiplier=100.,standard=True,
                    bid=p['cash_deliverable'],ask=p['cash_deliverable'],delta=0.,vega=0.,
                    bs_iv=p['last_iv'],bs_rate=market.rate(date),known_dividend_pv=0.,bs_status='cash_merger_deliverable')
            raw = market.quote(date,int(oid))
            if raw is None:
                if p is None:
                    return None
                stale=i-cal.get_loc(p['last_observed_quote_date'])
                if max_stale_quote_sessions is not None and stale>max_stale_quote_sessions:
                    raise UnresolvedPosition(f'{date}: held quote gap exceeds {max_stale_quote_sessions} sessions for {oid}; restore observed history before continuing')
                sigma = p.get('last_iv',np.nan)
                if not np.isfinite(sigma) or sigma<=0 or date>=p['expiry']:
                    raise UnresolvedPosition(f'{date}: no causal held mark for {oid}')
                rate = market.rate(date); pv = market.dividend_pv(date,p['secid'],p['expiry'],rate)
                price,delta,_,vega = option_values(spots[p['secid']]-pv,p['strike'],(p['expiry']-date).days/365,rate,sigma,p['cp_flag'])
                q = MarketRow(secid=p['secid'],optionid=int(oid),date=date,strike=p['strike'],expiry=p['expiry'],
                              cp_flag=p['cp_flag'],cfadj=p['cfadj'],multiplier=100.,standard=True,
                              bid=float(price),ask=float(price),delta=float(delta),vega=float(vega),
                              bs_iv=sigma,bs_rate=rate,known_dividend_pv=pv,bs_status='causal_held_mark')
                modeled += 1
            else:
                q = raw.copy()
            if cfg.repair_crossed_held_quotes and 0<=q.ask<q.bid:
                q['bid'],q['ask']=q.ask,q.bid;repaired+=1
            if not valid_quote(q):
                raise UnresolvedPosition(f'{date}: invalid held quote {oid}')
            if p is not None:
                if q.cfadj!=p['cfadj'] or q.strike!=p['strike'] or q.expiry!=p['expiry']:
                    raise UnresolvedPosition(f'{date}: unsupported option adjustment {oid}')
                if q.get('bs_status')=='unsupported_distribution' and p['pair'] not in event_pairs:
                    raise UnresolvedPosition(f'{date}: unsupported distribution {oid}')
                sigma = float(q.get('bs_iv',np.nan))
                if not valid_greeks(q) or not np.isfinite(sigma) or sigma<=0:
                    sigma=p.get('last_iv',np.nan);rate=q.get('bs_rate',np.nan);pv=q.get('known_dividend_pv',np.nan)
                    if not np.isfinite([sigma,rate,pv]).all() or sigma<=0 or spots[p['secid']]-pv<=0:
                        raise UnresolvedPosition(f'{date}: no causal hedge for {oid}')
                    _,delta,_,vega=option_values(spots[p['secid']]-pv,p['strike'],(p['expiry']-date).days/365,rate,sigma,p['cp_flag'])
                    q.update(delta=float(delta),vega=float(vega),bs_iv=float(sigma));fallbacks+=1
                p['last_iv']=float(q.bs_iv);p['last_vega']=float(q.vega)
                if raw is not None:p['last_observed_quote_date']=date
            mark_cache[key]=q
            return q

        # Apply verified stock/standard-option split adjustments before all marks.
        for sid in secids:
            if sid not in previous_adjustments or adjustments[sid]==previous_adjustments[sid]:
                continue
            held=[p for p in positions if p['secid']==sid]
            if not held and not hedge[sid]:
                continue
            ratio=adjustments[sid]/previous_adjustments[sid]
            raw=[market.quote(date,p['optionid']) for p in held]
            if not held or not cfg.support_integer_splits or any(q is None or not verified_integer_split(p,q,ratio) for p,q in zip(held,raw)):
                raise UnresolvedPosition(f'{date}: unverified held split {sid}')
            ratio=int(round(ratio));split_factors[sid]=ratio;hedge[sid]*=ratio
            for p,q in zip(held,raw):
                trades.append(dict(date=date,cohort=p['cohort'],pair=p['pair'],wing=p['wing'],secid=sid,optionid=p['optionid'],quantity=p['quantity']*(ratio-1),price=0.,commission=0.,execution_penalty=0.,reason='split_adjustment'))
                p['quantity']*=ratio;p['strike']=q.strike;p['cfadj']=q.cfadj;p['last_vega']/=ratio
                if management_policy is not None:p['initial_quantity']*=ratio
        adjusted_prior = {s:previous_spots.get(s,spots[s])/split_factors[s] for s in secids}
        hedge_pnl = sum(hedge[s]*(spots[s]-adjusted_prior[s]) for s in secids) if previous_date is not None else 0.
        interest=borrow=dividend=option_cost=option_penalty=0.
        if previous_date is not None:
            elapsed=(date-previous_date).days/365
            interest=cash*(previous_rate+(cfg.borrowing_spread if cash<0 else 0))*elapsed
            borrow=sum(max(-hedge[s],0)*adjusted_prior[s] for s in secids)*cfg.stock_borrow_rate*elapsed
            dividend=sum(hedge.get(s,0.)*amount for s,amount in cash_events.get(date,[]))
            cash+=interest-borrow+dividend
        option_cash=0.; kept=[]
        # Merger cash replaces the stock hedge; remaining options become fixed
        # cash claims and settle on the accelerated OCC expiration. Settlement
        # lag and voluntary early exercise are not modeled, as at ordinary expiry.
        for sid,event in cash_mergers.get(date,{}).items():
            if sid not in hedge:continue
            cash+=hedge[sid]*spots[sid];hedge[sid]=0.
            for p in positions:
                if p['secid']!=sid:continue
                p['cash_merger_price']=spots[sid]
                p['cash_merger_stock_adjustment']=adjustments[sid]
                p['cash_deliverable']=max((spots[sid]-p['strike'])*(1 if p['cp_flag']=='C' else -1),0.)
                accelerated=pd.Timestamp(event['accelerated_expiry'])
                p['exit_date']=min(p['exit_date'],cal[cal<=accelerated][-1])
        for p in positions:
            if date>=p['exit_date']:
                price=max((spots[p['secid']]-p['strike'])*(1 if p['cp_flag']=='C' else -1),0.)
                amount=p['quantity']*100*price;cash+=amount;option_cash+=amount
                trades.append(dict(date=date,cohort=p['cohort'],pair=p['pair'],wing=p['wing'],secid=p['secid'],optionid=p['optionid'],quantity=-p['quantity'],price=price,commission=0.,execution_penalty=0.,reason='cash_merger_settlement' if 'cash_deliverable' in p else 'expiry_settlement'))
            else:
                kept.append(p)
        positions=kept
        event_pairs={p['pair'] for p in positions if pair_members[p['pair']]&events.get(date,set())}
        # This flag was computed using declarations known on the quote date.
        # Exit both legs at the observed close before attempting daily hedges
        # under an unsupported future deliverable. Last IV is used only for
        # exit slippage when the flagged quote lacks fitted Greeks.
        for p in positions:
            if 'cash_deliverable' in p:continue
            raw=market.quote(date,p['optionid'])
            if raw is not None and raw.get('bs_status')=='unsupported_distribution':
                event_pairs.add(p['pair'])
        kept=[]
        for p in positions:
            if p['pair'] not in event_pairs:
                kept.append(p);continue
            q=quote(p['optionid'],p);qty=-p['quantity']
            price,penalty,slippage=execution_terms(q,qty,cfg,p.get('last_vega'));fee=abs(qty)*cfg.option_commission
            amount=-qty*100*price-fee-penalty;cash+=amount;option_cash+=amount
            option_cost+=slippage+fee;option_penalty+=penalty
            trades.append(dict(date=date,cohort=p['cohort'],pair=p['pair'],wing=p['wing'],secid=p['secid'],optionid=p['optionid'],quantity=qty,price=price,commission=fee,execution_penalty=penalty,reason='announced_corporate_action_exit'))
        positions=kept

        # Prior-close instructions fill only at the next observed session.
        # Modeled held marks are allowed for NAV, never for an elective fill.
        if management_policy is not None:
            grouped_management=defaultdict(list)
            for p in positions:grouped_management[p['cohort']].append(p)
            for cohort,instruction in pending_management.items():
                held=grouped_management.get(cohort,[])
                if not held:continue
                target,signal_date=instruction
                raw=[market.quote(date,p['optionid']) for p in held]
                if any(q is None or not valid_quote(q) or not valid_greeks(q) or q.bid<=0 or
                       q.strike!=p['strike'] or q.cfadj!=p['cfadj'] or q.expiry!=p['expiry']
                       for p,q in zip(held,raw)):
                    management_log.append(dict(cohort=cohort,signal_date=signal_date,date=date,target=target,status='no_executable_exit_quote'))
                    continue
                if target==0:
                    remaining=[0]*len(held)
                elif len(held)==2 or (standalone and len(held)==1):
                    # Scale both legs proportionally to their original counts.
                    remaining=[int(np.floor(abs(p['initial_quantity'])*target)) for p in held]
                    if any(n<1 or n>abs(p['quantity']) for n,p in zip(remaining,held)) or all(n==abs(p['quantity']) for n,p in zip(remaining,held)):
                        management_log.append(dict(cohort=cohort,signal_date=signal_date,date=date,target=target,status='integer_trim_unavailable'))
                        continue
                    fractions=[n/abs(p['initial_quantity']) for n,p in zip(remaining,held)]
                    if max(fractions)-min(fractions)>.1:
                        management_log.append(dict(cohort=cohort,signal_date=signal_date,date=date,target=target,status='unbalanced_integer_trim'))
                        continue
                else:continue
                for p,q,n in zip(held,raw,remaining):
                    qty=int(np.sign(p['quantity'])*n-p['quantity'])
                    if not qty:continue
                    price,penalty,slippage=execution_terms(q,qty,cfg);fee=abs(qty)*cfg.option_commission
                    amount=-qty*100*price-fee-penalty;cash+=amount;option_cash+=amount
                    option_cost+=slippage+fee;option_penalty+=penalty;p['quantity']+=qty
                    trades.append(dict(date=date,cohort=cohort,pair=p['pair'],wing=p['wing'],secid=p['secid'],optionid=p['optionid'],quantity=qty,price=price,commission=fee,execution_penalty=penalty,reason='learned_policy_exit' if target==0 else 'learned_policy_trim',exit_signal_date=signal_date))
                for p in held:p['management_fraction']=target
                management_log.append(dict(cohort=cohort,signal_date=signal_date,date=date,target=target,status='filled'))
            positions=[p for p in positions if p['quantity']]
            pending_management={}

        def assess(book,cash_balance):
            metrics=book_risk(book,lambda p:quote(p['optionid'],p),spots,date,cfg,scenario_cache)
            hc=sum(abs(metrics['target_hedge'][s]-hedge[s])*spots[s] for s in secids)*cfg.stock_cost_bps/10000
            nav=cash_balance+sum(hedge[s]*spots[s] for s in secids)+metrics['option_value']-hc
            return metrics,nav

        global_metrics,sizing_nav=assess(positions,cash)
        pre_global=risk_breaches(global_metrics,sizing_nav,cfg)
        if i>0 and cal[i-1] in schedule:
            signal_date=cal[i-1]
            for row in schedule[signal_date].to_dict('records'):
                sid_a,sid_b=int(row['secid_a']),int(row['secid_b']);pair=row['pair'];wing=row['wing']
                context=inventory_features(lag_pairs.get(pair,{}),lag_global,pair_capital,cfg.capital,cfg.max_gross_vega)
                level=int(row['base_level']) if policy is None else int(policy(row,context))
                if level not in range(5):
                    raise ValueError('Sizing must use the declared 0/25/50/75/100 ladder')
                action=row['direction']*level/4
                requested_budget_scale=1.
                if scale_policy is not None:
                    context.update(capital_context(row,lag_pairs.get(pair,{}),lag_global,
                                   lag_names,previous_nav,lag_cash,benchmark,relative_peak,cfg,pair_capital))
                    scale=float(scale_policy(row,context))
                    if not np.isfinite(scale) or scale not in [.5,1.,2.,4.]:
                        raise ValueError('Capital policy must use the declared .5/1/2/4 size multipliers')
                    action*=scale
                    context['trade_scale']=scale
                    if scale_pair_budget:
                        requested_budget_scale=scale
                        context['pair_budget_scale']=scale
                decision=dict(signal_date=signal_date,entry_date=date,pair=pair,wing=wing,base_level=int(row['base_level']),level=level,action=action,status='flat',**context)
                if not level:
                    decisions.append(decision);continue
                if {sid_a,sid_b}&events.get(date,set()):
                    decision['status']='announced_corporate_action_no_entry';decisions.append(decision);continue
                if any(sid in entry_blocks and date>=entry_blocks[sid] for sid in {sid_a,sid_b}):
                    decision['status']='public_listing_risk_no_entry';decisions.append(decision);continue
                if {sid_a,sid_b}&(unavailable|previous_unavailable):
                    decision['status']='underlying_unavailable_at_entry';decisions.append(decision);continue
                selected=[];live=[]
                requested_legs=[(sid_a,row['optionid_a'])] if standalone else [(sid_a,row['optionid_a']),(sid_b,row['optionid_b'])]
                for sid,oid in requested_legs:
                    if not np.isfinite(oid):
                        break
                    old=market.quote(signal_date,int(oid));now=market.quote(date,int(oid))
                    if old is None or now is None:
                        break
                    if sid not in previous_spots or previous_spots[sid]<=0:
                        break
                    dte=(old.expiry-date).days
                    if old.secid!=sid or not valid_quote(old,True,cfg) or not cfg.min_dte<=dte<=cfg.max_dte or abs(old.strike/previous_spots[sid]-1)>cfg.max_moneyness_distance:
                        break
                    eligible=now.copy();eligible['open_interest']=old.open_interest
                    if not valid_quote(eligible,True,cfg) or now.strike!=old.strike or now.cfadj!=old.cfadj or now.expiry!=old.expiry or now.secid!=sid:
                        break
                    selected.append(old);live.append(now)
                if len(live)!=len(requested_legs):
                    decision['status']='no_eligible_contract';decisions.append(decision);continue
                if straddle and (selected[0].cp_flag!='C' or selected[1].cp_flag!='P'
                    or selected[0].strike!=selected[1].strike or selected[0].expiry!=selected[1].expiry
                    or selected[0].cfadj!=selected[1].cfadj):
                    decision['status']='not_matching_straddle';decisions.append(decision);continue
                target_exit=max(q.expiry for q in selected)
                if require_runoff and (target_exit>cal[-1] or not (cal[(cal>date)&(cal<=target_exit)]).size):
                    decision['status']='insufficient_runoff';decisions.append(decision);continue
                cohort=f'{signal_date.date()}|{pair}|{wing}'
                if locked_entry_counts is None:
                    budget=cfg.entry_vega_per_leg*abs(action)*max(0,sizing_nav)/cfg.capital
                    if straddle:
                        count=int(np.floor(budget/(100*sum(q.vega for q in live))))
                        size=(count,count) if count>=1 else None
                    elif standalone:
                        count=int(np.floor(budget/(100*live[0].vega)))
                        size=(count,) if count>=1 else None
                    else:
                        size=integer_size([100*q.vega for q in live],budget,cfg.max_entry_vega_imbalance)
                else:
                    size=locked_entry_counts[cohort]
                    if len(size)!=len(requested_legs) or any(not np.isfinite(n) or n<1 or int(n)!=n for n in size):
                        raise ValueError('Locked counts must contain one positive integer per leg')
                    vegas=[n*100*q.vega for n,q in zip(size,live)]
                    if straddle and size[0]!=size[1]:
                        raise ValueError('Locked straddle counts must match')
                    if not standalone and not straddle and abs(vegas[0]-vegas[1])/max(vegas)>cfg.max_entry_vega_imbalance:
                        raise ValueError('Locked benchmark counts violate entry vega tolerance')
                if size is None:
                    decision['status']='integer_vega_mismatch';decisions.append(decision);continue
                current_pair=[p for p in positions if p['pair']==pair]
                pm=book_risk(current_pair,lambda p:quote(p['optionid'],p),spots,date,cfg,scenario_cache)
                pair_nav=pair_capital*requested_budget_scale*max(0,sizing_nav)/cfg.capital
                if pre_global or risk_breaches(pm,pair_nav,pair_cfg):
                    decision['status']='risk_deleveraging';decisions.append(decision);continue
                current_wing=[p for p in current_pair if p['wing']==wing]
                if wing_cfg is not None:
                    wm=book_risk(current_wing,lambda p:quote(p['optionid'],p),spots,date,cfg,scenario_cache)
                    if risk_breaches(wm,wing_capital*requested_budget_scale*max(0,sizing_nav)/cfg.capital,wing_cfg):
                        decision['status']='wing_risk_deleveraging';decisions.append(decision);continue
                additions=[];trial_cash=cash
                fills=[]
                for leg,(sid,old,q,n) in enumerate(zip([sid_a,sid_b],selected,live,size)):
                    qty=int(n)*int(np.sign(action))*(1 if leg==0 or straddle else -1)
                    price,penalty,slippage=execution_terms(q,qty,cfg);fee=abs(qty)*cfg.option_commission
                    trial_cash-=qty*100*price+fee+penalty
                    expiry_calendar=cal if require_runoff else pd.DatetimeIndex(settlement_calendar)
                    if not require_runoff and (settlement_calendar is None or expiry_calendar[-1]<q.expiry):
                        raise ValueError('Training prefix needs full known settlement calendar through expiry')
                    p=dict(cohort=cohort,pair=pair,wing=wing,secid=sid,optionid=int(old.optionid),quantity=qty,expiry=q.expiry,
                           strike=q.strike,cfadj=q.cfadj,cp_flag=q.cp_flag,exit_date=expiry_calendar[expiry_calendar<=q.expiry][-1],last_iv=float(q.bs_iv),last_vega=float(q.vega),last_observed_quote_date=date)
                    if scale_pair_budget:p['budget_scale']=requested_budget_scale
                    if management_policy is not None:
                        p.update(entry_date=date,initial_quantity=qty,entry_iv=float(q.bs_iv),management_fraction=1.)
                    additions.append(p);fills.append((price,penalty,slippage,fee))
                gm,projected_nav=assess(positions+additions,trial_cash)
                pm=book_risk(current_pair+additions,lambda p:quote(p['optionid'],p),spots,date,cfg,scenario_cache)
                breaches=risk_breaches(gm,projected_nav,cfg)+['pair_'+x for x in risk_breaches(pm,pair_capital*requested_budget_scale*projected_nav/cfg.capital,pair_cfg)]
                if wing_cfg is not None:
                    wm=book_risk(current_wing+additions,lambda p:quote(p['optionid'],p),spots,date,cfg,scenario_cache)
                    breaches+=['wing_'+x for x in risk_breaches(wm,wing_capital*requested_budget_scale*projected_nav/cfg.capital,wing_cfg)]
                if breaches:
                    decision.update(status='risk_limit_entry',risk_limits=','.join(breaches));decisions.append(decision);continue
                for p,(price,penalty,slippage,fee) in zip(additions,fills):
                    amount=-p['quantity']*100*price-fee-penalty;cash+=amount;option_cash+=amount
                    option_cost+=slippage+fee;option_penalty+=penalty;positions.append(p)
                    trades.append(dict(date=date,cohort=cohort,pair=pair,wing=wing,secid=p['secid'],optionid=p['optionid'],quantity=p['quantity'],price=price,commission=fee,execution_penalty=penalty,reason='entry'))
                decision.update(status='entered',vega_a=size[0]*100*live[0].vega,
                                vega_b=0. if standalone else size[1]*100*live[1].vega)
                decisions.append(decision)
        global_metrics,unused=assess(positions,cash)
        stock_cost=turnover=0.
        for sid in secids:
            change=global_metrics['target_hedge'][sid]-hedge[sid];notional=change*spots[sid]
            fee=abs(notional)*cfg.stock_cost_bps/10000;cash-=notional+fee;stock_cost+=fee;turnover+=abs(notional)
            hedge[sid]=global_metrics['target_hedge'][sid]
        option_value=global_metrics['option_value'];stock_value=sum(hedge[s]*spots[s] for s in secids)
        nav=cash+option_value+stock_value
        option_pnl=option_cash+option_value-previous_option_value
        expected=option_pnl+hedge_pnl-stock_cost+interest-borrow+dividend
        error=nav-previous_nav-expected;max_ledger_error=max(max_ledger_error,abs(error))
        if abs(error)>1e-5 or nav<=0:
            raise AssertionError(f'{date}: shared ledger error {error}, NAV {nav}')
        grouped=defaultdict(list)
        for p in positions:
            grouped[p['pair']].append(p)
        lag_pairs={}
        for pair,held in grouped.items():
            m=book_risk(held,lambda p:quote(p['optionid'],p),spots,date,cfg,scenario_cache)
            m['active_cohorts']=len({p['cohort'] for p in held});lag_pairs[pair]=m
        lag_global=global_metrics
        if management_policy is not None:
            cohorts=defaultdict(list)
            for p in positions:cohorts[p['cohort']].append(p)
            for cohort,held in cohorts.items():
                context=[dict(p,quote=dict(quote(p['optionid'],p)),spot=spots[p['secid']]) for p in held]
                target=float(management_policy(date,context))
                if target not in [0.,.5,1.]:raise ValueError('Management action must retain0/50/100% of initial inventory')
                current=min(p['management_fraction'] for p in held)
                if target<current:pending_management[cohort]=(target,date)
        daily.append(dict(date=date,nav=nav,pnl=nav-previous_nav,return_=nav/previous_nav-1,cash=cash,option_value=option_value,stock_value=stock_value,
                          gross_vega=global_metrics['gross_vega'],net_vega=sum(p['quantity']*100*quote(p['optionid'],p).vega for p in positions),
                          gross_hedge_notional=global_metrics['gross_hedge_notional'],capital_reserve=global_metrics['capital_reserve'],stress_loss=global_metrics['stress_loss'],
                          active_cohorts=len({p['cohort'] for p in positions}),active_pairs=len(grouped),option_execution_cost=option_cost,option_model_penalty=option_penalty,
                          hedge_cost=stock_cost,hedge_turnover=turnover,interest=interest,stock_borrow_cost=borrow,dividend_cash=dividend,
                          option_pnl=option_pnl,hedge_gross_pnl=hedge_pnl,ledger_error=error,repaired_held_quotes=repaired,greek_fallbacks=fallbacks,causal_held_mark_fallbacks=modeled,
                          post_trade_risk_breach=bool(risk_breaches(global_metrics,nav,cfg)),
                          pair_risk_breaches=sum(bool(risk_breaches(m,pair_capital*nav/cfg.capital*
                              (max(p.get('budget_scale',1.) for p in grouped[pair]) if scale_pair_budget else 1.),pair_cfg))
                              for pair,m in lag_pairs.items())))
        if scale_policy is not None:
            if previous_date is not None:
                benchmark*=1+previous_rate*(date-previous_date).days/365
            relative_peak=max(relative_peak,nav/benchmark)
            lag_cash=cash;lag_names=defaultdict(float)
            for p in positions:
                lag_names[p['secid']]+=abs(p['quantity'])*100*quote(p['optionid'],p).vega
        previous_spots=spots;previous_adjustments=adjustments;previous_date=date;previous_nav=nav;previous_option_value=option_value;previous_rate=market.rate(date);previous_unavailable=unavailable
    final_chunk=through_date is None or through_date>=cal[-1]
    if require_runoff and final_chunk and (positions or abs(option_value)+abs(stock_value)>1e-7):
        raise AssertionError('Runoff must close all inventory')
    audit=dict(max_ledger_error=max_ledger_error,shared_cash=True,shared_stock_hedges=True)
    if management_policy is not None:audit['management_log']=management_log
    if through_date is not None:
        audit['resume_state']=dict(cash=cash,hedge=hedge,positions=positions,
            previous_spots=previous_spots,previous_adjustments=previous_adjustments,previous_date=previous_date,
            previous_unavailable=previous_unavailable,previous_nav=previous_nav,previous_option_value=previous_option_value,
            previous_rate=previous_rate,lag_pairs=lag_pairs,lag_global=lag_global,max_ledger_error=max_ledger_error,
            lag_cash=lag_cash,lag_names=lag_names,benchmark=benchmark,relative_peak=relative_peak,
            pending_management=pending_management)
    return pd.DataFrame(daily).set_index('date'),pd.DataFrame(trades),pd.DataFrame(decisions),audit


def capital_context(row,pair_metrics,global_metrics,names,equity,cash,benchmark,relative_peak,cfg,pair_capital):
    """Completed-session account state; reserve is an internal margin proxy."""
    if equity<=0 or benchmark<=0 or relative_peak<=0:
        raise ValueError('Capital state requires positive equity and benchmark')
    pair_equity=pair_capital*equity/cfg.capital
    unit=cfg.entry_vega_per_leg*equity/cfg.capital
    return dict(prior_equity=equity,prior_cash=cash,
        reserve_utilization=global_metrics.get('capital_reserve',0.)/(cfg.max_reserve_fraction*equity),
        stress_utilization=global_metrics.get('stress_loss',0.)/(cfg.max_stress_loss_fraction*equity),
        hedge_utilization=global_metrics.get('gross_hedge_notional',0.)/(cfg.max_hedge_leverage*equity),
        pair_stress_utilization=pair_metrics.get('stress_loss',0.)/(cfg.max_stress_loss_fraction*pair_equity),
        name_inventory_units=max(names.get(int(row['secid_a']),0.),names.get(int(row['secid_b']),0.))/unit,
        excess_drawdown_fraction=max(0.,1-equity/benchmark/relative_peak),
        borrowing_fraction=max(0.,-cash/equity))
