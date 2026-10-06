"""Descriptive entry-risk buckets for actual frozen 30D P25 trades."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

OUT = Path('reports/p25_risk_context')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    paths=[Path('reports/p25_recent_losses/legs.csv'),
           Path('artifacts/contracts/same_structure_smile/P25_rows.parquet')]
    legs=pd.read_csv(paths[0],parse_dates=['entry_date','exit_date'])
    legs=legs.loc[legs.model.eq('30D_floor_P25')]
    cohorts=legs.groupby('cohort').agg(pair=('pair','first'),label=('label','first'),
        entry_date=('entry_date','min'),exit_date=('exit_date','max'),
        net_pnl=('net_before_shared_financing','sum'),initial_vega=('initial_vega','mean'))
    cohorts['date']=pd.to_datetime(cohorts.index.str.split('|').str[0])
    cols=['date','pair','iv_mean','rv_mean','beta','correlation','oriented_vrp_gap',
          'oriented_iv_gap','rank_strength','profit_91d','profit_365d']
    state=pq.read_table(paths[1],columns=cols,use_threads=False).to_pandas()
    assert not state.duplicated(['date','pair']).any()
    x=cohorts.reset_index().merge(state,on=['date','pair'],how='left',validate='one_to_one',indicator=True)
    assert x._merge.eq('both').all()
    x=x.drop(columns='_merge')
    x['pnl_per_entry_vega']=x.net_pnl/x.initial_vega
    x['rv_iv_ratio']=x.rv_mean/x.iv_mean
    x['formation_vol_ratio']=x.beta/x.correlation
    x['entry_year']=x.entry_date.dt.year
    # Fixed round diagnostic buckets, not optimized thresholds or entry rules.
    x['correlation_bucket']=pd.cut(x.correlation,[0,.6,.75,1.00001],right=False,
                                 labels=['below0.60','0.60to0.75','atleast0.75'])
    x['rv_iv_bucket']=pd.cut(x.rv_iv_ratio,[0,1,1.25,np.inf],right=False,
                            labels=['below1.0','1.0to1.25','atleast1.25'])
    x['iv_bucket']=np.where(x.iv_mean>=.6,'atleast60pct','below60pct')
    x['low_corr_high_iv']=(x.correlation<.6)&(x.iv_mean>=.6)
    x['period']=np.where(x.entry_year<=2023,'2018to2023','2024to2025')
    records=[]
    for period in ['all','2018to2023','2024to2025']:
        own=x if period=='all' else x.loc[x.period.eq(period)]
        for field in ['correlation_bucket','rv_iv_bucket','iv_bucket','low_corr_high_iv']:
            for value,g in own.groupby(field,observed=True):
                p=g.pnl_per_entry_vega
                records.append(dict(period=period,feature=field,bucket=str(value),trades=len(g),
                    pairs=g.pair.nunique(),win_fraction=float(p.gt(0).mean()),
                    net_pnl=float(g.net_pnl.sum()),mean_pnl_per_entry_vega=float(p.mean()),
                    median_pnl_per_entry_vega=float(p.median()),
                    p05_pnl_per_entry_vega=float(p.quantile(.05)),worst_pnl_per_entry_vega=float(p.min())))
    summary=pd.DataFrame(records)
    x.to_csv(OUT/'trade_context.csv',index=False)
    summary.to_csv(OUT/'bucket_summary.csv',index=False)
    yearly=x.groupby(['entry_year','low_corr_high_iv']).agg(trades=('cohort','size'),
        net_pnl=('net_pnl','sum'),mean_pnl_per_entry_vega=('pnl_per_entry_vega','mean'))
    yearly.to_csv(OUT/'yearly_low_correlation_high_iv.csv')
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*paths,Path(__file__)]}
    (OUT/'audit.json').write_text(json.dumps(dict(trades=len(x),no_retraining=True,no_trade_changes=True,
        no_threshold_search=True,entry_states_observed_before_fill=True,
        source_hashes=sources,limitations=['Actual accepted trades only; frozen entry/risk selection',
        'Inspected history; no fresh holdout','Overlapping cohorts are not independent observations',
        'Trade P&L before shared financing; not portfolio return or Sharpe',
        'Latest study has May2025 entry cutoff','Round diagnostic thresholds are exploratory',
        'Recent losses are not used to remove names or change policy']),indent=2)+'\n')
    print(summary.to_string(index=False))


if __name__=='__main__':
    main()
