"""Replace only standardized entry IV with filled-contract IV in proxy rewards.

Dates, forward RV, proxy costs and accepted trade quantities stay fixed. This
isolates contract-mark plus next-session-entry differences, not true option P&L.
"""
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd

REPORT=Path('reports/broad_contract_replay')

@lru_cache(maxsize=4)
def marks(sid):
    paths=[Path('data/contract_history_shared')/str(y)/f'{sid}.parquet' for y in range(2018,2026)]
    return pd.concat([pd.read_parquet(p,columns=['date','optionid','bs_iv','vendor_impl_volatility']) for p in paths if p.exists()]).set_index(['date','optionid'])[['bs_iv','vendor_impl_volatility']]*100

def main():
    membership=pd.read_csv(REPORT/'diagnostic_membership.csv')
    selected=set(membership.loc[membership.group.eq('broad')&membership.common_complete,'pair'])
    days=pd.read_csv(REPORT/'gap_proxy_daily.csv',index_col=0,parse_dates=[0]).index
    defs=json.loads((REPORT/'universe.json').read_text());rows=[]
    daily=pd.DataFrame(0.,index=days,columns=['same_day_contract_correction','entry_day_contract_correction'])
    for d in defs:
        if d['pair'] not in selected:continue
        try:p=pd.read_csv(Path('artifacts/contracts/broad_replay')/d['pair']/'rl_sized/decisions.csv',parse_dates=['signal_date','entry_date'])
        except pd.errors.EmptyDataError:continue
        p=p[p.status.eq('entered')&p.signal_date.isin(days)].copy()
        if p.empty:continue
        for side,sid,asset in zip(['a','b'],d['secids'],d['source_pair'].split('__')):
            original=pd.read_parquet(Path('artifacts/broad_pairs/assets')/f'{asset}.parquet',columns=['iv']).iv
            p['proxy_iv_'+side]=original.reindex(pd.DatetimeIndex(p.signal_date)).to_numpy()
            p['contract_iv_'+side]=marks(sid).bs_iv.reindex(pd.MultiIndex.from_arrays([p.entry_date,p['optionid_'+side].astype(int)])).to_numpy()
            prior=marks(sid).reindex(pd.MultiIndex.from_arrays([p.signal_date,p['optionid_'+side].astype(int)]))
            p['prior_contract_iv_'+side]=prior.bs_iv.to_numpy()
            p['vendor_iv_'+side]=prior.vendor_impl_volatility.to_numpy()
        ivs=['proxy_iv_a','proxy_iv_b','contract_iv_a','contract_iv_b','prior_contract_iv_a','prior_contract_iv_b']
        assert np.isfinite(p[ivs]).all().all()
        vega=(p.vega_a+p.vega_b)/2
        correction=np.sign(p.action)*vega*((p.proxy_iv_a-p.contract_iv_a)-(p.proxy_iv_b-p.contract_iv_b))
        prior_correction=np.sign(p.action)*vega*((p.proxy_iv_a-p.prior_contract_iv_a)-(p.proxy_iv_b-p.prior_contract_iv_b))
        daily.loc[p.signal_date,'same_day_contract_correction']+=prior_correction.to_numpy()
        daily.loc[p.signal_date,'entry_day_contract_correction']+=correction.to_numpy()
        rows.append(dict(pair=d['pair'],label=d['label'],entries=len(p),iv_replacement_dollars=correction.sum(),
                         same_day_iv_replacement_dollars=prior_correction.sum(),
                         mean_abs_bs_vendor_iv_difference=(abs(p.prior_contract_iv_a-p.vendor_iv_a).mean()+abs(p.prior_contract_iv_b-p.vendor_iv_b).mean())/2,
                         mean_abs_leg_iv_difference=(abs(p.proxy_iv_a-p.contract_iv_a).mean()+abs(p.proxy_iv_b-p.contract_iv_b).mean())/2,
                         extreme_proxy_entries=int(p[['proxy_iv_a','proxy_iv_b']].max(axis=1).gt(100).sum())))
    result=pd.DataFrame(rows);result.to_csv(REPORT/'gap_entry_marks.csv',index=False)
    daily.to_csv(REPORT/'gap_entry_marks_daily.csv')
    print('IV-replacement dollars:',result.iv_replacement_dollars.sum())
    print('Same-day IV-replacement dollars:',result.same_day_iv_replacement_dollars.sum())
    print(result.nsmallest(10,'iv_replacement_dollars').to_string(index=False))

if __name__=='__main__':main()
