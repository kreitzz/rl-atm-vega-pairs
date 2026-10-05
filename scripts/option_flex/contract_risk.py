"""Transparent portfolio risk overlay; capital reserve is not broker margin."""
import numpy as np
from scripts.option_flex.black_scholes import option_values


def book_risk(positions, quotes, spots, date, cfg, scenario_cache=None):
    hedge={sid:0. for sid in spots}
    scenario={sid:np.zeros(9) for sid in spots}
    gross_vega=gross_premium=option_value=0.
    moves=np.repeat(np.array([-cfg.spot_stress,0.,cfg.spot_stress]),3)
    vol_moves=np.tile(np.array([-cfg.vol_stress,0.,cfg.vol_stress]),3)
    for p in positions:
        q=quotes(p); sid=p['secid']; qty=p['quantity']*100
        mid=(q.bid+q.ask)/2
        delta=float(q.delta); vega=float(q.vega)
        sigma=float(q.get('bs_iv',np.nan));rate=float(q.get('bs_rate',np.nan))
        pv=float(q.get('known_dividend_pv',np.nan))
        if not np.isfinite([delta,vega,sigma,rate,pv]).all() or sigma<=0:
            raise ValueError('Risk overlay requires valid fitted IV, rates, dividends and Greeks')
        t=(q.expiry-date).days/365
        if t<=0:raise ValueError('Risk overlay cannot price expired positions')
        spot=spots[sid];effective=spot-pv
        if effective<=0:raise ValueError('Nonpositive dividend-adjusted spot')
        key=(sid,p.get('optionid'),float(q.strike),t,rate,sigma,pv,spot,cfg.spot_stress,cfg.vol_stress)
        cached=None if scenario_cache is None else scenario_cache.get(key)
        if cached is None:
            model_mid=option_values(effective,q.strike,t,rate,sigma,q.cp_flag)[0]
            stressed=option_values(np.maximum(spot*(1+moves)-pv,1e-8),q.strike,t,rate,
                                 np.maximum(sigma+vol_moves,.0001),q.cp_flag)[0]
            if scenario_cache is not None:scenario_cache[key]=(model_mid,stressed)
        else:model_mid,stressed=cached
        scenario[sid]+=qty*(stressed-model_mid)
        hedge[sid]-=qty*delta
        gross_vega+=abs(qty)*vega
        gross_premium+=abs(qty)*mid
        option_value+=qty*mid
    for sid in spots:scenario[sid]+=hedge[sid]*spots[sid]*moves
    # Independent worst shocks for each name, never credit pair correlation.
    stress_loss=sum(max(0.,-float(values.min())) for values in scenario.values())
    gross_hedge=sum(abs(hedge[sid])*spots[sid] for sid in spots)
    return dict(target_hedge=hedge,option_value=option_value,gross_vega=gross_vega,
                gross_hedge_notional=gross_hedge,gross_option_premium=gross_premium,
                capital_reserve=cfg.stock_reserve_fraction*gross_hedge+gross_premium,
                stress_loss=stress_loss)


def risk_breaches(metrics, nav, cfg):
    if not np.isfinite(nav) or nav<=0:return ['nonpositive_nav']
    limits={'gross_vega':cfg.max_gross_vega*nav/cfg.capital,
            'gross_hedge_notional':cfg.max_hedge_leverage*nav,
            'capital_reserve':cfg.max_reserve_fraction*nav,
            'stress_loss':cfg.max_stress_loss_fraction*nav}
    return [name for name,limit in limits.items() if metrics[name]>limit+1e-8]
