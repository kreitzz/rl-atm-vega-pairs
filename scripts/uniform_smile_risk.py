"""Memoize individual marked contributions without changing portfolio limits."""
import numpy as np
from scripts.option_flex.black_scholes import option_values


def book_risk(positions,quotes,spots,date,cfg,scenario_cache=None):
    cache={} if scenario_cache is None else scenario_cache
    record_cache=cache.setdefault(('uniform_risk_records',date),{})
    moves=np.repeat(np.array([-cfg.spot_stress,0.,cfg.spot_stress]),3)
    vol_moves=np.tile(np.array([-cfg.vol_stress,0.,cfg.vol_stress]),3)
    sids=list(spots);lookup={s:i for i,s in enumerate(sids)};records=[];indices=[]
    for p in positions:
        identity=(p['secid'],p.get('cohort'),p.get('optionid'),p['quantity'],p['strike'],p['cfadj'],p['expiry'])
        record=record_cache.get(identity)
        if record is None:
            q=quotes(p);qty=p['quantity']*100;sid=p['secid']
            delta=float(q.delta);vega=float(q.vega);sigma=float(q.get('bs_iv',np.nan));rate=float(q.get('bs_rate',np.nan));pv=float(q.get('known_dividend_pv',np.nan))
            if not np.isfinite([delta,vega,sigma,rate,pv]).all() or sigma<=0:raise ValueError('Invalid risk Greeks')
            t=(q.expiry-date).days/365;spot=spots[sid]
            if t<=0 or spot-pv<=0:raise ValueError('Invalid risk maturity or effective spot')
            key=(sid,p.get('optionid'),float(q.strike),t,rate,sigma,pv,spot,cfg.spot_stress,cfg.vol_stress)
            values=cache.get(key)
            if values is None:
                model=option_values(spot-pv,q.strike,t,rate,sigma,q.cp_flag)[0]
                stressed=option_values(np.maximum(spot*(1+moves)-pv,1e-8),q.strike,t,rate,np.maximum(sigma+vol_moves,.0001),q.cp_flag)[0]
                values=model,stressed;cache[key]=values
            model,stressed=values;mid=(q.bid+q.ask)/2
            record=np.r_[abs(qty)*vega,abs(qty)*mid,qty*mid,-qty*delta,qty*(stressed-model)]
            record_cache[identity]=record
        records.append(record);indices.append(lookup[p['secid']])
    sums=np.zeros((len(sids),13))
    if records:np.add.at(sums,np.asarray(indices),np.asarray(records))
    hedge=sums[:,3];scenario=sums[:,4:]+hedge[:,None]*np.array([spots[s] for s in sids])[:,None]*moves
    stress=float(np.maximum(0.,-scenario.min(axis=1)).sum()) if len(sids) else 0.
    gross_hedge=float(sum(abs(h)*spots[s] for s,h in zip(sids,hedge)))
    gross_vega,premium,value=sums[:,:3].sum(axis=0)
    return dict(target_hedge=dict(zip(sids,hedge)),option_value=float(value),gross_vega=float(gross_vega),
                gross_hedge_notional=gross_hedge,gross_option_premium=float(premium),
                capital_reserve=cfg.stock_reserve_fraction*gross_hedge+float(premium),stress_loss=stress)
