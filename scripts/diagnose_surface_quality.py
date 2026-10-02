"""Join original surface quality fields and inspect two quote-chain examples."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.audit_contract_history import read_contracts
from scripts.run_contract_pilot import normalize_quotes
from scripts.black_scholes import fit_quotes

REPORT=Path('reports/expiry_signal_gap')
SOURCE=Path('/Users/adamkreitzman/Desktop/RL_Converts/convert-rv-rl/data/raw/wrds/optionmetrics/atm_iv_compact')


def main():
    legs=pd.read_csv(REPORT/'original_signal_legs.csv',parse_dates=['signal_date'])
    pieces=[]
    for month,g in legs.groupby(legs.signal_date.dt.to_period('M')):
        file=SOURCE/f'year={month.year}'/f'month={month.month:02}.parquet'
        q=pd.read_parquet(file,columns=['secid','date','impl_volatility','dispersion','impl_strike','impl_premium'],
            filters=[('secid','in',g.secid.unique().tolist()),('days','=',30),('delta','=',50),('cp_flag','=','C')])
        pieces.append(q)
    surface=pd.concat(pieces,ignore_index=True).drop_duplicates(['secid','date'])
    legs=legs.merge(surface,left_on=['secid','signal_date'],right_on=['secid','date'],how='left',validate='many_to_one')
    np.testing.assert_allclose(legs.standard_iv,100*legs.impl_volatility,rtol=0,atol=1e-10)
    legs['dispersion_vol_points']=100*legs.dispersion
    legs['old_proxy_cost_leg']=.0125*legs.standard_iv*legs.vega
    legs['old_net']=legs.old_proxy_gross-legs.old_proxy_cost_leg
    legs['corrected_net']=legs.corrected_proxy_gross-legs.old_proxy_cost_leg
    legs.to_csv(REPORT/'original_surface_quality_legs.csv',index=False)
    cohorts=legs.groupby(['pair','label','signal_date']).agg(old_net=('old_net','sum'),corrected_net=('corrected_net','sum'),
        max_iv=('standard_iv','max'),max_dispersion=('dispersion_vol_points','max'),
        max_tenor_error=('signal_dte',lambda x:abs(x-30).max()),max_delta_error=('signal_delta',lambda x:abs(x-.5).max()))
    cohorts['dispersion_bucket']=pd.cut(cohorts.max_dispersion,[-np.inf,5,10,25,np.inf],labels=['<=5','5-10','10-25','>25'])
    summary=cohorts.groupby('dispersion_bucket',observed=True).agg(cohorts=('old_net','size'),old_net=('old_net','sum'),corrected_net=('corrected_net','sum'))
    summary.to_csv(REPORT/'original_dispersion_groups.csv')
    days=pd.read_csv('reports/broad_contract_replay/gap_proxy_daily.csv',index_col=0,parse_dates=[0]).index
    quality=[]
    for name,use in [('all',np.ones(len(cohorts),bool)),('dispersion_at_most_5',cohorts.max_dispersion.le(5)),
        ('dispersion_at_most_10',cohorts.max_dispersion.le(10)),('dispersion_above_25',cohorts.max_dispersion.gt(25)),
        ('tenor_2_days_delta_45_55',cohorts.max_tenor_error.le(2)&cohorts.max_delta_error.le(.05))]:
        g=cohorts[use];row=dict(group=name,cohorts=len(g))
        for col in ['old_net','corrected_net']:
            daily=g.groupby('signal_date')[col].sum().reindex(days,fill_value=0.)
            row[col]=daily.sum();row[col+'_proxy_sharpe']=np.sqrt(365/30)*daily.mean()/daily.std()
        quality.append(row)
    pd.DataFrame(quality).to_csv(REPORT/'original_quality_proxy_scores.csv',index=False)
    samples=[];chain=[];root=Path('data/option_contracts')
    zero=pd.read_parquet(root/'zero_curve.parquet');divs=pd.read_parquet(root/'distributions.parquet')
    for sid,date in [(110366,pd.Timestamp('2021-03-09')),(106701,pd.Timestamp('2024-03-06'))]:
        raw=read_contracts(root,[sid],date,date)
        stocks=pd.read_parquet(root/'underlying'/f'year={date.year}.parquet',filters=[('secid','=',sid)])
        q=fit_quotes(normalize_quotes(raw),stocks,zero,divs)
        q['mid']=(q.bid+q.ask)/2;q['width']=(q.ask-q.bid)/q.mid
        q=q[(q.bid>0)&(q.ask>=q.bid)&q.width.le(.25)&q.open_interest.ge(100)&q.standard&q.multiplier.eq(100)&q.bs_iv.notna()]
        # Examine all valid calls with 25–45 days remaining and with
        # fitted call delta 0.35--0.65, rather than only our chosen strike.
        near=q[q.dte.between(25,45)&q.bs_delta.between(.35,.65)].copy()
        near['iv_percent']=100*near.bs_iv
        chain.append(near[['date','secid','optionid','expiry','dte','strike','close','bid','ask','open_interest','bs_delta','iv_percent']])
        original=surface[surface.secid.eq(sid)&surface.date.eq(date)].iloc[0]
        samples.append(dict(secid=sid,date=date,standard_iv=original.impl_volatility*100,dispersion=original.dispersion*100,
            near_30d_50delta_contracts=len(near),min_quote_iv=near.iv_percent.min(),max_quote_iv=near.iv_percent.max()))
    pd.concat(chain,ignore_index=True).to_csv(REPORT/'example_nearby_contracts.csv',index=False)
    pd.DataFrame(samples).to_csv(REPORT/'example_surface_gaps.csv',index=False)
    print(summary.to_string());print(pd.DataFrame(samples).to_string(index=False))


if __name__=='__main__':main()
