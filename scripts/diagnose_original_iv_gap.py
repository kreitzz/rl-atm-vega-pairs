"""Locate the standardized-versus-traded IV gap on old accepted RL signals."""
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
import pandas as pd

REPORT=Path('reports/expiry_signal_gap')


@lru_cache(maxsize=4)
def quotes(sid):
    cols=['date','optionid','expiry','bs_iv','vendor_impl_volatility','delta','strike','close','bid','ask']
    frames=[pd.read_parquet(p,columns=cols) for year in range(2018,2026)
        if (p:=Path('data/contract_history_shared')/str(year)/f'{sid}.parquet').exists()]
    return pd.concat(frames,ignore_index=True).set_index(['date','optionid'])


def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    base=Path('reports/broad_contract_replay')
    membership=pd.read_csv(base/'diagnostic_membership.csv')
    selected=set(membership.loc[membership.group.eq('broad')&membership.common_complete,'pair'])
    days=pd.read_csv(base/'gap_proxy_daily.csv',index_col=0,parse_dates=[0]).index
    rows=[]
    for d in json.loads((base/'universe.json').read_text()):
        if d['pair'] not in selected:continue
        try:p=pd.read_csv(Path('artifacts/contracts/broad_replay')/d['pair']/'rl_sized/decisions.csv',parse_dates=['signal_date','entry_date'])
        except pd.errors.EmptyDataError:continue
        p=p[p.status.eq('entered')&p.signal_date.isin(days)].copy()
        if not len(p):continue
        vega=(p.vega_a+p.vega_b)/2
        for side,sid,asset,orientation in zip(['a','b'],d['secids'],d['source_pair'].split('__'),[1,-1]):
            original=pd.read_parquet(Path('artifacts/broad_pairs/assets')/f'{asset}.parquet',columns=['iv','forward_rv'])
            old=original.reindex(pd.DatetimeIndex(p.signal_date))
            prior=quotes(sid).reindex(pd.MultiIndex.from_arrays([p.signal_date,p['optionid_'+side].astype(int)]))
            now=quotes(sid).reindex(pd.MultiIndex.from_arrays([p.entry_date,p['optionid_'+side].astype(int)]))
            sign=np.sign(p.action).to_numpy()*orientation;weight=vega.to_numpy()
            for i,r in enumerate(p.itertuples()):
                a=prior.iloc[i];b=now.iloc[i];o=old.iloc[i]
                rows.append(dict(pair=d['pair'],label=d['label'],signal_date=r.signal_date,entry_date=r.entry_date,
                    side=side,secid=sid,optionid=int(getattr(r,'optionid_'+side)),direction=sign[i],vega=weight[i],
                    standard_iv=o.iv,signal_contract_iv=a.bs_iv*100,entry_contract_iv=b.bs_iv*100,
                    vendor_iv=a.vendor_impl_volatility*100,signal_delta=a.delta,
                    signal_dte=(a.expiry-r.signal_date).days,entry_dte=(a.expiry-r.entry_date).days,
                    signal_moneyness=a.strike/a.close-1,signal_relative_spread=(a.ask-a.bid)/((a.bid+a.ask)/2),
                    rv=o.forward_rv,old_proxy_gross=sign[i]*weight[i]*(o.forward_rv-o.iv),
                    corrected_proxy_gross=sign[i]*weight[i]*(o.forward_rv-a.bs_iv*100),
                    iv_correction=sign[i]*weight[i]*(o.iv-a.bs_iv*100)))
    legs=pd.DataFrame(rows);legs.to_csv(REPORT/'original_signal_legs.csv',index=False)
    pairs=legs.groupby(['pair','label','signal_date']).agg(old_gross=('old_proxy_gross','sum'),
        corrected_gross=('corrected_proxy_gross','sum'),iv_correction=('iv_correction','sum'),
        max_tenor_error=('signal_dte',lambda x:abs(x-30).max()),
        max_delta_error=('signal_delta',lambda x:abs(x-.5).max()),max_standard_iv=('standard_iv','max'),
        max_abs_iv_gap=('standard_iv','max')).reset_index()
    gap=(legs.standard_iv-legs.signal_contract_iv).abs();legs['abs_iv_gap']=gap
    pairgap=legs.groupby(['pair','label','signal_date']).abs_iv_gap.max()
    pairs['max_abs_iv_gap']=pairgap.reindex(pd.MultiIndex.from_frame(pairs[['pair','label','signal_date']])).to_numpy()
    pairs.to_csv(REPORT/'original_signal_cohorts.csv',index=False)
    groups={'all':np.ones(len(pairs),bool),'tenor_within_2_days':pairs.max_tenor_error.le(2),
        'tenor_within_5_days':pairs.max_tenor_error.le(5),'both_delta_45_55':pairs.max_delta_error.le(.05),
        'tenor_2_days_and_delta_45_55':pairs.max_tenor_error.le(2)&pairs.max_delta_error.le(.05),
        'both_standard_iv_below_100':pairs.max_standard_iv.lt(100),
        'either_standard_iv_above_100':pairs.max_standard_iv.ge(100)}
    summary=[]
    for name,use in groups.items():
        g=pairs[use];summary.append(dict(group=name,cohorts=len(g),old_gross=g.old_gross.sum(),
            corrected_gross=g.corrected_gross.sum(),iv_correction=g.iv_correction.sum(),
            median_max_abs_iv_gap=g.max_abs_iv_gap.median()))
    result=pd.DataFrame(summary);result.to_csv(REPORT/'original_iv_gap_groups.csv',index=False)
    legs.assign(gap=legs.standard_iv-legs.signal_contract_iv).sort_values('iv_correction').head(50).to_csv(REPORT/'largest_original_iv_gaps.csv',index=False)
    print(result.to_string(index=False))


if __name__=='__main__':main()
