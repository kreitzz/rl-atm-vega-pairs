"""Inspect the actual quotes triggering missing-Greek forced exits."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scripts.diagnose_current_contract_losses import POLICIES, REPORT, read_csv


def main():
    pairs=pd.read_csv(REPORT/'pairs.csv');parts=[]
    for name,(root,subdir,*_) in POLICIES.items():
        for r in pairs[pairs.policy.eq(name)&pairs.entries.gt(0)].itertuples():
            trades=read_csv(root/r.pair/subdir/'trades.csv',['date'])
            t=trades[trades.reason.eq('missing_greeks_exit')].copy()
            if len(t):t['policy']=name;t['pair']=r.pair;t['label']=r.label;parts.append(t)
    requested=pd.concat(parts,ignore_index=True);quotes=[]
    columns=['date','secid','optionid','expiry','bid','ask','strike','close','dte','bs_rate','known_dividend_pv','delta','vega','bs_iv','bs_status','vendor_impl_volatility']
    for (year,sid),g in requested.groupby([requested.date.dt.year,'secid']):
        path=Path('data/contract_history_shared')/str(year)/f'{int(sid)}.parquet'
        q=pd.read_parquet(path,columns=columns,filters=[('optionid','in',g.optionid.unique().tolist()),('date','in',g.date.unique().tolist())])
        quotes.append(q)
    q=pd.concat(quotes,ignore_index=True).drop_duplicates(['date','optionid'])
    all_rows=requested.merge(q,on=['date','optionid','secid'],how='left',validate='many_to_one')
    assert all_rows.bs_status.notna().all()
    invalid=~(np.isfinite(all_rows.delta)&np.isfinite(all_rows.vega)&all_rows.delta.between(0,1)&all_rows.vega.ge(0))
    bad=all_rows[invalid].copy()
    bad['mid']=(bad.bid+bad.ask)/2
    bad['effective_spot']=bad.close-bad.known_dividend_pv
    bad['european_lower_bound']=np.maximum(bad.effective_spot-bad.strike*np.exp(-bad.bs_rate*bad.dte/365),0)
    bad['mid_minus_lower_bound']=bad.mid-bad.european_lower_bound
    bad['diagnosis']=np.select([
        bad.bs_status.eq('unsupported_distribution'),bad.bid.gt(bad.ask),
        bad.mid.le(bad.european_lower_bound+1e-8),bad.mid.ge(bad.effective_spot-1e-8)],
        ['unsupported_distribution','crossed_quote','mid_at_or_below_european_bound','mid_at_or_above_effective_spot'],default='other_inversion_or_input_failure')
    bad.to_csv(REPORT/'missing_greek_quotes.csv',index=False)
    summary=bad.groupby(['policy','diagnosis']).agg(leg_exits=('optionid','size'),median_dte=('dte','median'),
        median_bound_gap=('mid_minus_lower_bound','median')).reset_index()
    summary.to_csv(REPORT/'missing_greek_reasons.csv',index=False)
    counts=bad.groupby(['policy','pair','cohort']).size().rename('invalid_legs').reset_index()
    total=requested.groupby(['policy','pair','cohort']).size().reset_index()
    assert len(counts)==len(total)
    counts.to_csv(REPORT/'missing_greek_cohorts.csv',index=False)
    print(summary.to_string(index=False))


if __name__=='__main__':main()
