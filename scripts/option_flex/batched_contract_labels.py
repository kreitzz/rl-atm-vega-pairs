"""Bounded vectorized execution of the standalone ShadowLabels trajectories.

Same observed marks, integer sizing, costs, hedge, lifecycle and missing-data
rules; vectorization is across independent entry dates and ladder actions.
"""
import numpy as np
import pandas as pd

from scripts.option_flex.contract_engine import valid_quote, valid_greeks, integer_size, verified_integer_split
from scripts.option_flex.contract_learning_labels import LADDER
from scripts.option_flex.black_scholes import option_values


def label_batch(maker,positions):
    c=maker.cfg;cal=maker.calendar;positions=np.asarray(positions,int);n=len(positions)
    width=c.max_dte+2 if c.hold_to_expiry else 32
    daily=np.zeros((n,9,width));ends=np.minimum(positions+1,len(cal)-1)
    leg_ends=np.zeros((n,2),int)
    vegas=np.zeros((n,9));statuses=np.full(n,'no_entry',dtype='U40')
    units=np.zeros((n,9,2));old=[None]*n
    for j,i in enumerate(positions):
        if i+2>=len(cal):statuses[j]='insufficient_runoff';daily[j]=np.nan;continue
        signal,entry=cal[i],cal[i+1];a=[];live=[]
        for sid in maker.secids:
            oid=maker.selections.get((signal,sid));prior=maker.quotes.get((signal,oid));now=maker.quotes.get((entry,oid))
            stock=maker.stocks.get((signal,sid))
            if prior is None or now is None or stock is None:break
            dte=(prior['expiry']-entry).days
            if not (valid_quote(prior,True,c) and c.min_dte<=dte<=c.max_dte and
                    abs(prior['strike']/stock['close']-1)<=c.max_moneyness_distance):break
            if not valid_quote(dict(now,open_interest=prior['open_interest']),True,c):break
            if any(prior[k]!=now[k] for k in ['secid','strike','expiry','cfadj']):break
            a.append(prior);live.append(now)
        if len(a)!=2:continue
        target=min(entry+pd.Timedelta(days=c.holding_days),min(q['expiry'] for q in a)-pd.Timedelta(days=1))
        if c.hold_to_expiry:
            target=max(q['expiry'] for q in a)
            leg_ends[j]=[cal.searchsorted(q['expiry'],side='right')-1 for q in a]
        end=cal.searchsorted(target,side='right')-1
        if target>cal[-1] or end<=i+1:statuses[j]='insufficient_runoff';daily[j]=np.nan;continue
        if maker.events.get(entry,False) and not c.hold_to_expiry:continue
        v=np.array([q['vega']*100 for q in live])
        for k,action in enumerate(LADDER):
            if action==0:continue
            size=integer_size(v,c.entry_vega_per_leg*abs(action),c.max_entry_vega_imbalance)
            if size is not None and (size*v).sum()<=c.max_gross_vega:
                units[j,k]=size*np.sign(action)*np.array([1.,-1.])
        if not units[j].any():continue
        old[j]=a;vegas[j]=(abs(units[j])*v).mean(axis=1);ends[j]=end;statuses[j]='pending'
    cash=np.full((n,9),c.capital,dtype=float);benchmark=np.full(n,c.capital,dtype=float)
    hedge=np.zeros_like(units);previous_nav=cash.copy();previous_benchmark=benchmark.copy()
    previous_spots=np.zeros((n,2));previous_adj=np.ones((n,2));previous_rate=np.zeros(n)
    last_vega=np.zeros((n,2))
    last_iv=np.full((n,2),np.nan)
    for j in np.flatnonzero(statuses=='pending'):
        last_vega[j]=[maker.quotes[(cal[positions[j]+1],q['optionid'])]['vega'] for q in old[j]]
        last_iv[j]=[maker.quotes[(cal[positions[j]+1],q['optionid'])].get('bs_iv',np.nan) for q in old[j]]

    def fail(j,status,idx):
        statuses[j]=status;ends[j]=idx;daily[j]=np.nan

    for step in range(width+1):
        pending=np.flatnonzero(statuses=='pending')
        if not len(pending):break
        kept=[];stock_rows=[];quote_rows=[];closes=[]
        for j in pending:
            i=positions[j]+step;date=cal[i]
            sr=[maker.stocks.get((date,s)) for s in maker.secids]
            if any(r is None or not np.isfinite([r['close'],r['cfadj'],r['rate']]).all() or r['close']<=0 for r in sr):
                fail(j,'unresolved_underlying',i);continue
            spots=np.array([r['close'] for r in sr]);adj=np.array([r['cfadj'] for r in sr])
            bad=False
            if step>1:
                for leg in np.flatnonzero(adj!=previous_adj[j]):
                    if c.hold_to_expiry and not units[j,:,leg].any() and not hedge[j,:,leg].any():continue
                    ratio=adj[leg]/previous_adj[j,leg]
                    new=maker.quotes.get((date,old[j][leg]['optionid']))
                    if not (c.support_integer_splits and new is not None and verified_integer_split(old[j][leg],new,ratio)):
                        fail(j,'unresolved_corporate_action',i);bad=True;break
                    ratio=int(round(ratio));units[j,:,leg]*=ratio;hedge[j,:,leg]*=ratio
                    last_vega[j,leg]/=ratio;previous_spots[j,leg]/=ratio;old[j][leg]=new
            if bad:continue
            qr=[];close=i==ends[j]
            if step:
                for leg,a in enumerate(old[j]):
                    if c.hold_to_expiry and i>=leg_ends[j,leg]:
                        price=max((spots[leg]-a['strike'])*(1 if a['cp_flag']=='C' else -1),0.) if i==leg_ends[j,leg] else 0.
                        qr.append(dict(bid=price,ask=price,delta=0.,vega=0.))
                        continue
                    q=maker.quotes.get((date,a['optionid']))
                    if q is not None and c.repair_crossed_held_quotes and 0<=q['ask']<q['bid']:
                        q=dict(q,bid=q['ask'],ask=q['bid'])
                    if q is None or not valid_quote(q) or any(q[k]!=a[k] for k in ['strike','expiry','cfadj']):
                        fail(j,'unresolved_contract',i);bad=True;break
                    if c.hold_to_expiry:
                        sigma=float(q.get('bs_iv',np.nan))
                        if q.get('bs_status')=='unsupported_distribution':
                            fail(j,'unresolved_distribution',i);bad=True;break
                        if valid_greeks(q) and np.isfinite(sigma) and sigma>0:last_iv[j,leg]=sigma
                        else:
                            sigma=last_iv[j,leg];rate=q.get('bs_rate',np.nan);pv=q.get('known_dividend_pv',np.nan)
                            if not np.isfinite([sigma,rate,pv]).all() or sigma<=0 or spots[leg]-pv<=0:
                                fail(j,'unresolved_hedge',i);bad=True;break
                            _,delta,_,vega=option_values(spots[leg]-pv,q['strike'],(q['expiry']-date).days/365,rate,sigma,q['cp_flag'])
                            q=dict(q,delta=float(delta),vega=float(vega))
                    qr.append(q)
                if bad:continue
                if step>1 and not c.hold_to_expiry:close|=not all(valid_greeks(q) for q in qr) or maker.events.get(date,False)
            kept.append(j);stock_rows.append(sr);quote_rows.append(qr);closes.append(close)
        if not kept:continue
        k=np.asarray(kept);date_idx=positions[k]+step
        spots=np.array([[r['close'] for r in sr] for sr in stock_rows])
        adj=np.array([[r['cfadj'] for r in sr] for sr in stock_rows])
        rates=np.array([[r['rate'] for r in sr] for sr in stock_rows]).mean(axis=1)
        close=np.array(closes,bool)
        if step:
            elapsed=(cal[date_idx]-cal[date_idx-1]).days.to_numpy()/365
            cash[k]+=cash[k]*(previous_rate[k,None]+np.where(cash[k]<0,c.borrowing_spread,0))*elapsed[:,None]
            cash[k]-=(np.maximum(-hedge[k],0)*previous_spots[k,None,:]).sum(axis=2)*c.stock_borrow_rate*elapsed[:,None]
            distributions=np.array([maker.cash_dividends.get(cal[i],np.zeros(2)) for i in date_idx])
            cash[k]+=(hedge[k]*distributions[:,None,:]).sum(axis=2)
            benchmark[k]*=1+previous_rate[k]*elapsed
            mid=np.array([[(q['bid']+q['ask'])/2 for q in qr] for qr in quote_rows])
            current_vega=np.array([[q['vega'] for q in qr] for qr in quote_rows])
            last_vega[k]=np.where(np.isfinite(current_vega)&(current_vega>=0),current_vega,last_vega[k])
            if c.option_penalty_vol_points is None:
                half=np.array([[(q['ask']-q['bid'])/2*c.option_spread_fraction for q in qr] for qr in quote_rows])
            else:half=last_vega[k]*c.option_penalty_vol_points
            if c.hold_to_expiry:
                settling=(date_idx[:,None]>=leg_ends[k]) if step>1 else np.zeros((len(k),2),bool)
                transact=np.ones((len(k),2),bool) if step==1 else settling
                qty=units[k]*(1 if step==1 else -1)*transact[:,None,:]
                fee=100*half+c.option_commission if step==1 else np.zeros_like(half)
                cash[k]-=(qty*100*mid[:,None,:]+abs(qty)*fee[:,None,:]).sum(axis=2)
                units[k]*=(~settling)[:,None,:]
            else:
                transact=np.ones(len(k),bool) if step==1 else close
                qty=units[k]*(1 if step==1 else -1)*transact[:,None,None]
                cash[k]-=(qty*100*mid[:,None,:]+abs(qty)*(100*half[:,None,:]+c.option_commission)).sum(axis=2)
            deltas=np.array([[q['delta'] for q in qr] for qr in quote_rows])
            target=-units[k]*100*np.where(close[:,None],0,deltas)[:,None,:]
            change=target-hedge[k]
            cash[k]-=(change*spots[:,None,:]+abs(change)*spots[:,None,:]*c.stock_cost_bps/10000).sum(axis=2)
            hedge[k]=target
            option_value=np.where(close[:,None],0,(units[k]*100*mid[:,None,:]).sum(axis=2))
        else:option_value=np.zeros((len(k),9))
        nav=cash[k]+(hedge[k]*spots[:,None,:]).sum(axis=2)+option_value
        if step:
            daily[k,:,step-1]=(nav-previous_nav[k]-(benchmark[k]-previous_benchmark[k])[:,None])/c.entry_vega_per_leg
        previous_nav[k]=nav;previous_benchmark[k]=benchmark[k]
        previous_spots[k]=spots;previous_adj[k]=adj;previous_rate[k]=rates
        if step:
            statuses[k[close]]='complete';ends[k[close]]=date_idx[close]
    if (statuses=='pending').any():raise AssertionError('Batched shadow did not close')
    return daily,ends,vegas,statuses


def build_labels(maker,positions,batch_size=256):
    pieces=[label_batch(maker,positions[i:i+batch_size]) for i in range(0,len(positions),batch_size)]
    if not pieces:return np.empty((0,9,maker.cfg.max_dte+2 if maker.cfg.hold_to_expiry else 32)),np.empty(0,int),np.empty((0,9)),np.empty(0,dtype='U40')
    return tuple(np.concatenate([p[j] for p in pieces]) for j in range(4))
