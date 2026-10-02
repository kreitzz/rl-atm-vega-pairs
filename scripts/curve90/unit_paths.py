"""Vectorized per-contract cash flows, combined with exact pair-level financing.

Flows are per initial contract for long and short ownership, excluding cash
interest. This preserves integer sizing, split factors and sign-specific borrow
costs; pair financing is applied after combining legs, including negative cash.
"""
import numpy as np
import pandas as pd
from scripts.option_flex.black_scholes import option_values
from scripts.curve90.config import SLOTS,COMBOS,LADDER

class UnitPaths:
    def __init__(self,q,stock,selection,calendar,dates,divs,cfg):
        assert cfg.hold_to_expiry and cfg.option_spread_fraction==0, 'Unit paths implement hold-to-expiry midpoint plus vol-point penalty'
        self.calendar=pd.DatetimeIndex(calendar);self.dates=pd.DatetimeIndex(dates);self.cfg=cfg
        self.positions=self.calendar.get_indexer(self.dates);assert (self.positions>=0).all()
        self.width=cfg.max_dte+2
        s=stock.set_index('date').reindex(self.calendar)
        self.stock_valid=np.isfinite(s[['close','cfadj','rate']]).all(axis=1).to_numpy()&s.close.gt(0).to_numpy()
        self.rates=s.rate.to_numpy(float)
        self.stock_bad_prefix=np.r_[0,np.cumsum(~self.stock_valid)]
        slots={tuple(x):i for i,x in enumerate(SLOTS)}
        sel=selection[selection.date.isin(self.dates)].copy()
        sel['slot']=[slots[(int(t),cp,int(b) if str(b)!='atm' else 'atm')] for t,cp,b in zip(sel.tenor,sel.cp_flag,sel.bucket)]
        assert not sel.duplicated(['date','slot']).any()
        sel=sel.sort_values(['date','slot']).reset_index(drop=True)
        n=len(sel);w=self.width
        self.lookup=np.full((len(dates),len(SLOTS)),-1,int)
        self.lookup[self.dates.get_indexer(sel.date),sel.slot]=np.arange(n)
        self.flow_long=np.zeros((n+1,w));self.flow_short=np.zeros((n+1,w));self.mark=np.zeros((n+1,w))
        self.entry_vega=np.zeros(n+1);self.entered=np.zeros(n+1,bool);self.resolved=np.ones(n+1,bool)
        self.ends=np.zeros(n+1,int)
        if not n:return
        q=q.reset_index(drop=True).copy();assert not q.duplicated(['date','optionid']).any()
        index=pd.MultiIndex.from_frame(q[['date','optionid']])
        # A final sentinel row permits bounded vector gathers for missing quotes.
        q=pd.concat([q,pd.DataFrame([{}])],ignore_index=True)
        def numbers(col,default=np.nan):
            return pd.to_numeric(q[col],errors='coerce').to_numpy(float) if col in q else np.full(len(q),default)
        arr={k:numbers(k) for k in ['bid','ask','strike','multiplier','cfadj','delta','vega','open_interest','bs_iv','bs_rate','known_dividend_pv']}
        standard=q.standard.eq(True).to_numpy()
        calls=q.cp_flag.eq('C').to_numpy();puts=q.cp_flag.eq('P').to_numpy()
        expiry=pd.to_datetime(q.expiry).to_numpy('datetime64[ns]')
        unsupported=q.bs_status.eq('unsupported_distribution').to_numpy() if 'bs_status' in q else np.zeros(len(q),bool)
        def locate(positions,ids):
            good=(positions>=0)&(positions<len(self.calendar))
            ds=self.calendar[np.clip(positions,0,len(self.calendar)-1)]
            r=index.get_indexer(pd.MultiIndex.from_arrays([ds,ids]));r[~good]=-1
            return np.where(r>=0,r,len(q)-1)
        def greeks(rows):
            d,v=arr['delta'][rows],arr['vega'][rows]
            return np.isfinite(d+v)&(v>=0)&np.where(calls[rows],(d>=0)&(d<=1),(d>=-1)&(d<=0))
        def valid(rows,entry=False):
            b,a=arr['bid'][rows],arr['ask'][rows]
            finite=np.ones(len(rows),bool)
            for k in ['bid','ask','strike','multiplier','cfadj']:finite &= np.isfinite(arr[k][rows])
            good=finite&(b>=0)&(b<=a)&(a>0)&(arr['strike'][rows]>0)&(arr['multiplier'][rows]==100)&standard[rows]&(calls[rows]|puts[rows])
            if entry:
                with np.errstate(divide='ignore',invalid='ignore'):
                    good &= greeks(rows)&(b>0)&(arr['vega'][rows]>0)&(arr['open_interest'][rows]>=cfg.min_open_interest)&((a-b)/((a+b)/2)<=cfg.max_relative_spread)
            return good
        pos=self.calendar.get_indexer(sel.date);ids=sel.optionid.to_numpy()
        prior=locate(pos,ids);entry=locate(pos+1,ids)
        end=self.calendar.searchsorted(expiry[prior],side='right')-1
        self.ends[:n]=np.clip(end,0,len(self.calendar)-1)
        dte=(expiry[prior]-self.calendar[np.minimum(pos+1,len(self.calendar)-1)].values)/np.timedelta64(1,'D')
        with np.errstate(divide='ignore',invalid='ignore'):
            entry_width=(arr['ask'][entry]-arr['bid'][entry])/((arr['ask'][entry]+arr['bid'][entry])/2)
        # Today's OI is not known at entry; use signal-close OI only.
        live=valid(entry)&greeks(entry)&(arr['bid'][entry]>0)&(arr['vega'][entry]>0)&(entry_width<=cfg.max_relative_spread)
        same=(arr['strike'][prior]==arr['strike'][entry])&(arr['cfadj'][prior]==arr['cfadj'][entry])&(expiry[prior]==expiry[entry])
        prior_spot=s.close.to_numpy(float)[pos]
        available=valid(prior,True)&live&same&(dte>=cfg.min_dte)&(dte<=cfg.max_dte)&(abs(arr['strike'][prior]/prior_spot-1)<=cfg.max_moneyness_distance)
        runoff=(expiry[prior]<=self.calendar[-1].to_datetime64())&(end>pos+1)&(pos+2<len(self.calendar))
        self.entered[:n]=available
        self.resolved[:n]=~available|runoff
        self.entry_vega[:n]=np.where(available,arr['vega'][entry]*100,0.)
        active=available&runoff
        factor=np.ones(n);hedge=np.zeros(n);last_iv=arr['bs_iv'][entry].copy()
        strike=arr['strike'][prior].copy();adjustment=arr['cfadj'][prior].copy()
        spot=s.close.to_numpy(float);stock_adj=s.cfadj.to_numpy(float)
        previous_spot=spot[pos].copy();previous_adj=stock_adj[pos].copy()
        previous_pos=pos.copy()
        dividend=np.zeros(len(self.calendar))
        if len(divs):
            indexes=self.calendar.searchsorted(pd.to_datetime(divs.ex_date),side='left')
            use=(indexes>0)&(indexes<len(self.calendar))
            np.add.at(dividend,indexes[use],divs.loc[use,'amount'].to_numpy(float))
        max_steps=min(w,int(np.max(np.where(active,end-pos,0),initial=0)))
        for step in range(1,max_steps+1):
            k=np.flatnonzero(active)
            if not len(k):break
            now=pos[k]+step;rows=locate(now,ids[k]);settle=now>=end[k]
            ok=self.stock_valid[now]&self.stock_valid[pos[k]]
            ratio=stock_adj[now]/previous_adj[k]
            changed=(ratio!=1)&(step>1)
            if changed.any():
                good_split=(cfg.support_integer_splits&np.isfinite(ratio)&(ratio>=2)&np.isclose(ratio,np.round(ratio))&standard[rows]&
                    (arr['multiplier'][rows]==100)&(calls[rows]==calls[prior[k]])&(expiry[rows]==expiry[prior[k]])&
                    np.isclose(arr['cfadj'][rows]/adjustment[k],ratio)&np.isclose(arr['strike'][rows],strike[k]/ratio,rtol=0,atol=.00500001))
                ok &= ~changed|good_split
                split=changed&good_split
                kk=k[split];rr=np.round(ratio[split]);factor[kk]*=rr;hedge[kk]*=rr;previous_spot[kk]/=rr
                strike[kk]=arr['strike'][rows[split]];adjustment[kk]=arr['cfadj'][rows[split]]
            bid=arr['bid'][rows].copy();ask=arr['ask'][rows].copy()
            crossed=(ask>=0)&(ask<bid)
            if cfg.repair_crossed_held_quotes:
                bid[crossed],ask[crossed]=ask[crossed].copy(),bid[crossed].copy()
            good_quote=(np.isfinite(bid+ask+arr['strike'][rows]+arr['cfadj'][rows])&(bid>=0)&(ask>=bid)&(ask>0)&
                (arr['strike'][rows]>0)&standard[rows]&(arr['multiplier'][rows]==100)&(calls[rows]|puts[rows]))
            unchanged=(arr['strike'][rows]==strike[k])&(arr['cfadj'][rows]==adjustment[k])&(expiry[rows]==expiry[prior[k]])
            ok &= settle|(good_quote&unchanged&~unsupported[rows])
            good_greeks=greeks(rows)&np.isfinite(arr['bs_iv'][rows])&(arr['bs_iv'][rows]>0)
            update=good_greeks&~settle;last_iv[k[update]]=arr['bs_iv'][rows[update]]
            delta=arr['delta'][rows].copy();vega=arr['vega'][rows].copy()
            fallback=~good_greeks&~settle
            rate=arr['bs_rate'][rows];pv=arr['known_dividend_pv'][rows]
            model_ok=np.isfinite(last_iv[k]+rate+pv)&(last_iv[k]>0)&(spot[now]-pv>0)&(expiry[rows]>self.calendar[now].values)
            ok &= ~fallback|model_ok
            f=fallback&model_ok&ok
            if f.any():
                _,dd,_,vv=option_values(spot[now[f]]-pv[f],strike[k[f]],
                    (expiry[rows[f]]-self.calendar[now[f]].values)/np.timedelta64(1,'D')/365,rate[f],last_iv[k[f]],np.where(calls[rows[f]],'C','P'))
                delta[f]=dd;vega[f]=vv
            failed=k[~ok];self.resolved[failed]=False;active[failed]=False
            k=k[ok];now=now[ok];rows=rows[ok];settle=settle[ok];bid=bid[ok];ask=ask[ok];delta=delta[ok];vega=vega[ok]
            if not len(k):continue
            mid=(bid+ask)/2
            mid=np.where(settle,np.maximum((spot[now]-strike[k])*np.where(calls[prior[k]],1.,-1.),0.),mid)
            delta=np.where(settle,0.,delta)
            target=-factor[k]*100*delta
            change=target-hedge[k]
            elapsed=(self.calendar[now]-self.calendar[previous_pos[k]]).days.to_numpy()/365
            long=-change*spot[now]-abs(change)*spot[now]*cfg.stock_cost_bps/10000
            short=change*spot[now]-abs(change)*spot[now]*cfg.stock_cost_bps/10000
            long-=np.maximum(-hedge[k],0)*previous_spot[k]*cfg.stock_borrow_rate*elapsed
            short-=np.maximum(hedge[k],0)*previous_spot[k]*cfg.stock_borrow_rate*elapsed
            long+=hedge[k]*dividend[now];short-=hedge[k]*dividend[now]
            if step==1:
                penalty=vega*100*cfg.option_penalty_vol_points+cfg.option_commission
                long-=factor[k]*(100*mid+penalty);short+=factor[k]*(100*mid-penalty)
            else:
                long+=factor[k]*100*mid*settle;short-=factor[k]*100*mid*settle
            hedge[k]=target
            self.flow_long[k,step-1]=long;self.flow_short[k,step-1]=short
            self.mark[k,step-1]=np.where(settle,0.,factor[k]*100*mid)+hedge[k]*spot[now]
            previous_spot[k]=spot[now];previous_adj[k]=stock_adj[now];previous_pos[k]=now
            active[k[settle]]=False
        self.resolved[:n] &= ~active
        bad=~self.resolved
        self.flow_long[bad]=np.nan;self.flow_short[bad]=np.nan;self.mark[bad]=np.nan

    def ids(self,tenor,cp,bucket):return self.lookup[:,SLOTS.index((tenor,cp,bucket))]


def pair_labels(a,b,combo,cfg):
    """All eight nonflat sizes, exact standalone funding, no portfolio risk overlay."""
    assert a.dates.equals(b.dates) and a.calendar.equals(b.calendar) and a.width==b.width
    tenor,cp,ba,bb=combo;ia=a.ids(tenor,cp,ba);ib=b.ids(tenor,cp,bb)
    n=len(a.dates);w=a.width;va=a.entry_vega[ia];vb=b.entry_vega[ib]
    exists=(ia>=0)&(ib>=0)&a.entered[ia]&b.entered[ib]
    sizes=np.array(LADDER);targets=cfg.entry_vega_per_leg*abs(sizes)
    with np.errstate(divide='ignore',invalid='ignore'):
        qa=np.floor(targets[None,:]/va[:,None]);qb=np.floor(targets[None,:]/vb[:,None])
        v_a=qa*va[:,None];v_b=qb*vb[:,None]
        accept=exists[:,None]&(qa>=1)&(qb>=1)&(abs(v_a-v_b)/np.maximum(v_a,v_b)<=cfg.max_entry_vega_imbalance)&((v_a+v_b)<=cfg.max_gross_vega)
    qa=np.where(accept,qa,0.);qb=np.where(accept,qb,0.)
    entered=accept.any(axis=1);vegas=np.where(accept,(v_a+v_b)/2,0.)
    ends=np.maximum(a.ends[ia],b.ends[ib]);ends=np.where(entered,ends,np.minimum(a.positions+1,len(a.calendar)-1))
    valid=~entered|(a.resolved[ia]&b.resolved[ib])
    # Match existing shadow-label convention: both stock histories through later expiry.
    for unit in [a,b]:valid &= ~entered|((unit.stock_bad_prefix[ends+1]-unit.stock_bad_prefix[a.positions])==0)
    daily=np.zeros((n,len(sizes),w));cash=np.full((n,len(sizes)),cfg.capital,dtype=float)
    benchmark=np.full(n,cfg.capital,dtype=float);previous=cash.copy();previous_benchmark=benchmark.copy()
    positions=a.positions
    max_steps=int(np.max(np.where(entered,ends-positions,0),initial=0))
    sign=np.sign(sizes)[None,:]
    for step in range(1,min(w,max_steps)+1):
        now=np.minimum(positions+step,len(a.calendar)-1);prev=np.minimum(positions+step-1,len(a.calendar)-1)
        elapsed=(a.calendar[now]-a.calendar[prev]).days.to_numpy()/365
        rate=(a.rates[prev]+b.rates[prev])/2
        cash+=cash*(rate[:,None]+np.where(cash<0,cfg.borrowing_spread,0))*elapsed[:,None]
        benchmark*=1+rate*elapsed
        al=np.nan_to_num(a.flow_long[ia,step-1]);ash=np.nan_to_num(a.flow_short[ia,step-1])
        bl=np.nan_to_num(b.flow_long[ib,step-1]);bsh=np.nan_to_num(b.flow_short[ib,step-1])
        cash+=qa*np.where(sign>0,al[:,None],ash[:,None])+qb*np.where(sign>0,bsh[:,None],bl[:,None])
        marked=qa*sign*np.nan_to_num(a.mark[ia,step-1])[:,None]-qb*sign*np.nan_to_num(b.mark[ib,step-1])[:,None]
        nav=cash+marked
        daily[:,:,step-1]=np.where((entered&(now<=ends))[:,None],(nav-previous-(benchmark-previous_benchmark)[:,None])/cfg.entry_vega_per_leg,0.)
        previous=nav;previous_benchmark=benchmark.copy()
    daily[~valid]=np.nan
    status=np.where(~entered,'no_entry',np.where(valid,'complete','unresolved'))
    return daily,ends,vegas,status
