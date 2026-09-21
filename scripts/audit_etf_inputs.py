"""Quote-activity and extreme-surface attribution diagnostics; no policy retuning."""
import argparse
import json
from pathlib import Path
import pandas as pd
from scripts.run_first_pass import payoff_sharpe


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-root',type=Path,required=True);args=ap.parse_args()
    out=Path('reports/etf_exploration');out.mkdir(parents=True,exist_ok=True)
    config=json.loads(Path('config/etf_pairs.json').read_text());names=['SIL','HL','AG','GSLC','SPYM'];ids={config['assets'][t]['secid']:t for t in names}
    frames=[]
    for path in sorted((args.source_root/'optionmetrics/option_activity').glob('*.parquet')):
        d=pd.read_parquet(path);frames.append(d[d.secid.isin(ids)])
    d=pd.concat([x for x in frames if len(x)]);d['ticker']=d.secid.map(ids);d['date']=pd.to_datetime(d.date);d=d[d.date.between('2024-01-01','2025-08-29')]
    rows=[]
    for (ticker,cp),g in d.groupby(['ticker','cp_flag']):
        rows.append(dict(ticker=ticker,cp_flag=cp,observations=len(g),mean_volume=g.volume.mean(),median_volume=g.volume.median(),zero_volume_fraction=g.volume.eq(0).mean(),median_open_interest=g.open_interest.median()))
    pd.DataFrame(rows).to_csv(out/'activity_audit.csv',index=False)
    run=Path('artifacts/etf_exploration/run');panels=pd.read_parquet(run/'panels.parquet');rows=[]
    for pair in ['SIL_HL','SIL_AG']:
        p=pd.read_csv(run/f'{pair}.csv',index_col='date',parse_dates=['date']);q=panels.xs(pair,level='pair').loc[p.index];r=p.baseline_reward;high=q.iv_nvda>100
        rows.append(dict(pair=pair,threshold_vol_points=100,high_iv_entry_fraction=high.mean(),high_iv_net_pnl_fraction=r.loc[high].sum()/r.sum(),high_iv_net_pnl=r.loc[high].sum(),remaining_net_pnl=r.loc[~high].sum(),conditional_remaining_sharpe=payoff_sharpe(r.loc[~high]),full_calendar_skip_high_sharpe=payoff_sharpe(r.where(~high,0)),baseline_sharpe=payoff_sharpe(r),stock_iv_max=q.iv_nvda.max(),stock_iv_p99=q.iv_nvda.quantile(.99)))
    pd.DataFrame(rows).to_csv(out/'extreme_iv_attribution.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':main()
