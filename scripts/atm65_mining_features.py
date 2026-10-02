"""Causal candidate states from signal-close quotes and trailing stock returns."""
import numpy as np
import pandas as pd

DYNAMICS=['iv_ratio_log','variance_value_5','variance_value_21','variance_value_63',
          'rv_acceleration_5_21','rv_acceleration_21_63','iv_change_5','iv_change_21',
          'spread_z_21','spread_z_63','iv_correlation_63','return_correlation_63',
          'correlation_gap','beta_scaled_iv_ratio']
GAMMA=['gamma_risk_gap','gamma_ratio_log','gamma_value_5','gamma_value_21','gamma_value_63',
       'relative_width_a','relative_width_b','width_difference','width_vol_a','width_vol_b',
       'log_open_interest_a','log_open_interest_b','dte_gap']

def names(values):
    return [f'mine_{v}{ending}' for v in values for ending in ['', '_available']]

def engineer(panel,quotes,stocks,selections,secids):
    index=pd.DatetimeIndex(sorted(stocks.date.unique()))
    legs=[]
    for sid in secids:
        stock=stocks[stocks.secid.eq(sid)].set_index('date').reindex(index)
        sel=selections[selections.secid.eq(sid)].set_index('date').reindex(index)
        quote=quotes[quotes.secid.eq(sid)].set_index(['date','optionid'])
        key=pd.MultiIndex.from_arrays([index,sel.optionid])
        q=quote.reindex(key).set_axis(index)
        adjusted=stock.close*stock.cfadj
        ret=np.log(adjusted/adjusted.shift(1))
        dte=(pd.to_datetime(q.expiry)-index.to_series()).dt.days
        # The compact cache omits gamma. Recover its exact BS ratio from the
        # stored fitted IV and dividend-adjusted spot; vega is per vol point.
        effective=stock.close-q.get('known_dividend_pv',0.)
        gamma_per_vega=100/(q.bs_iv*dte/365)*(stock.close/effective)**2
        leg=pd.DataFrame(dict(iv=q.bs_iv*100,ret=ret,
            gamma_per_vega=gamma_per_vega.where((q.vega>0)&(effective>0)&(dte>0)),
            relative_width=(q.ask-q.bid)/((q.ask+q.bid)/2),
            width_vol=(q.ask-q.bid)/q.vega.where(q.vega>0),
            log_open_interest=np.log1p(q.open_interest.clip(lower=0)),
            dte=dte),index=index)
        for days in [5,21,63]:leg[f'rv{days}']=100*np.sqrt(252*ret.pow(2).rolling(days,min_periods=days).mean())
        legs.append(leg)
    a,b=legs;spread=a.iv-b.iv
    out=pd.DataFrame(index=index)
    out['iv_ratio_log']=np.log(a.iv/b.iv)
    out['beta_scaled_iv_ratio']=np.log(a.iv/(b.iv*panel.beta.reindex(index).abs().where(lambda x:x>0)))
    for n in [5,21,63]:
        out[f'variance_value_{n}']=(a[f'rv{n}']**2-a.iv**2)/(2*a.iv)-(b[f'rv{n}']**2-b.iv**2)/(2*b.iv)
        ga=.5*a.gamma_per_vega*((a[f'rv{n}']/100)**2-(a.iv/100)**2)*a.dte/365
        gb=.5*b.gamma_per_vega*((b[f'rv{n}']/100)**2-(b.iv/100)**2)*b.dte/365
        out[f'gamma_value_{n}']=ga-gb
    for short,long in [(5,21),(21,63)]:
        out[f'rv_acceleration_{short}_{long}']=np.log(a[f'rv{short}']/a[f'rv{long}'])-np.log(b[f'rv{short}']/b[f'rv{long}'])
    for n in [5,21]:out[f'iv_change_{n}']=spread.diff(n)
    for n in [21,63]:
        prior=spread.shift(1).rolling(n,min_periods=(n+1)//2)
        out[f'spread_z_{n}']=(spread-prior.mean())/prior.std().where(lambda x:x>0)
    # Fixed calendar-session windows; sparse valid quotes are not forward-filled.
    out['iv_correlation_63']=a.iv.rolling(63,min_periods=32).corr(b.iv)
    out['return_correlation_63']=a.ret.rolling(63,min_periods=63).corr(b.ret)
    out['correlation_gap']=out.iv_correlation_63-out.return_correlation_63
    out['gamma_risk_gap']=a.gamma_per_vega-b.gamma_per_vega
    out['gamma_ratio_log']=np.log(a.gamma_per_vega/b.gamma_per_vega)
    for side,leg in [('a',a),('b',b)]:
        for value in ['relative_width','width_vol','log_open_interest']:out[f'{value}_{side}']=leg[value]
    out['width_difference']=a.relative_width-b.relative_width
    out['dte_gap']=a.dte-b.dte
    assert set(out)==set(DYNAMICS+GAMMA)
    result=panel.copy()
    for value in DYNAMICS+GAMMA:
        observed=out[value].reindex(panel.index).replace([np.inf,-np.inf],np.nan)
        result[f'mine_{value}']=observed.fillna(0.)
        result[f'mine_{value}_available']=observed.notna().astype(float)
    pd.testing.assert_frame_equal(result[panel.columns],panel)
    return result

def thin_completed(dates,ends,eligible):
    chosen=[];last=None
    for i in np.flatnonzero(eligible):
        if last is None or dates[i]>last:chosen.append(i);last=ends[i]
    return np.asarray(chosen,dtype=int)
