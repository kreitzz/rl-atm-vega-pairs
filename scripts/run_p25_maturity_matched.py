"""Signal-quote matched-sample control for the 30D/90D transfer experiment."""
from dataclasses import replace
import gc
import json
from pathlib import Path
import resource
import time
import numpy as np
import pandas as pd

from scripts import uniform_smile_data as cached
from scripts import uniform_smile_engine as engine
from scripts.uniform_smile_risk import book_risk
from scripts.curve_trading.config import config
from scripts.run_atm65_ablation import write_json
from scripts.run_contract_walkforward import digest
from scripts.run_uniform_smile import verify,UNIFORM_CORPORATE_EVENTS
from scripts.run_aligned_contract_learning import mark_book
from scripts.run_p25_signal_transfer import Market
from scripts.p25_maturity_transfer import WINDOWS,period_stats
from scripts.p25_signal_transfer import WINGS

BASE=Path('artifacts/contracts/p25_maturity_transfer')
OUT=Path('artifacts/contracts/p25_maturity_matched')
REPORT=Path('reports/p25_maturity_matched')
PROTOCOL=(
    'Secondary signal-quote availability control for p25_maturity_transfer. '
    'For each wing, keep pair/date proposals only where both selected30D '
    'and90D contract pairs are present on the signal date. This is a '
    'point-in-time coverage predicate; no next-session execution quote, '
    'future holding path or outcome enters sample selection. Keep both '
    'maturities exact same pair/date/direction/frozenP25 sizing decisions '
    'and original contract IDs. No selection replacement or retraining. '
    'Retain primary full-proposal experiment separately; do not silently '
    'substitute this smaller sample. All costs, risk, daily hedging, '
    'expiry/corporate exits and common110day runoff unchanged. '
    '16 independent shared-cash replays: fixed100/frozenfloor,threewings '
    'and combined,each at30D/90D. Calendar slices2018-2020,2021-2023, '
    '2024-Aug2025 carry inventory across boundaries. No freshholdout. '
    'One worker,50%of onecore,3GiB RSS stop.'
)


def main():
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    sources={str(Path(__file__)):digest(Path(__file__))}
    rows={};coverage=[]
    for wing in WINGS:
        pieces={}
        for tenor in [30,90]:
            path=BASE/f'{tenor}D'/f'{wing}_rows.parquet'
            sources[str(path)]=digest(path)
            pieces[tenor]=pd.read_parquet(path)
        a,b=pieces[30],pieces[90]
        fields=['date','pair','wing','direction','base_level']
        pd.testing.assert_frame_equal(a[fields],b[fields])
        keep=a[['optionid_a','optionid_b']].notna().all(axis=1)&b[['optionid_a','optionid_b']].notna().all(axis=1)
        coverage.append(dict(wing=wing,primary_proposals=len(a),matched_signal_quote_proposals=int(keep.sum())))
        for tenor in [30,90]:
            rows[(tenor,wing)]=pieces[tenor].loc[keep].reset_index(drop=True)
            if tenor==90:
                rows[(tenor,wing)]['contract_source_a']='cached_90D'
                rows[(tenor,wing)]['contract_source_b']='cached_90D'
            rows[(tenor,wing)].to_parquet(OUT/f'{tenor}D_{wing}_rows.parquet',index=False)
    pd.DataFrame(coverage).to_csv(REPORT/'coverage.csv',index=False)
    write_json(OUT/'inputs.json',dict(protocol=PROTOCOL,coverage=coverage))
    print(pd.DataFrame(coverage).to_string(index=False),flush=True)
    results=[];periods=[];annual=[];books={}
    for tenor in [30,90]:
        cached.OUT=BASE/f'{tenor}D'
        for path in (cached.OUT/'market').glob('*.parquet'):
            sources[str(path)]=digest(path)
        market=Market()
        cal=market.rates.index[market.rates.index>='2018-01-01']
        for variant in ['fixed_100','floor']:
            for wing in ['P25','ATM','P75',None]:
                name=f'{tenor}D_{variant}'+('_'+wing if wing else '')
                folder=OUT/'replays'/name
                folder.mkdir(parents=True,exist_ok=True)
                proposal=pd.concat([rows[(tenor,target)] for target in ([wing] if wing else WINGS)],ignore_index=True)
                proposal=proposal.sort_values(['date','pair','wing'],kind='stable').reset_index(drop=True)
                if variant=='fixed_100':
                    proposal['base_level']=4
                proposal.to_parquet(folder/'scored_allocations.parquet',index=False)
                multiplier=3 if wing is None else 1
                cfg=replace(config(),option_commission=.03,capital=1_000_000*multiplier*345,max_gross_vega=20_000*multiplier*345)
                if tenor==90:
                    cfg=replace(cfg,target_dte=90,min_dte=75,max_dte=105)
                market.variant,market.started,market.reported_year='matched_'+name,time.monotonic(),None
                if (folder/'status.json').exists():
                    record=json.loads((folder/'status.json').read_text())
                    nav=pd.read_parquet(folder/'daily_nav.parquet')
                    decisions=pd.read_csv(folder/'decisions.csv',parse_dates=['signal_date','entry_date'])
                else:
                    print(f'Quote-matched replay {name}: {len(proposal)} proposals',flush=True)
                    saved=engine.book_risk
                    try:
                        engine.book_risk=book_risk
                        nav,trades,decisions,_=engine.simulate(market,proposal,cal,cfg,1_000_000*multiplier,
                            wing_capital=1_000_000,corporate_events=UNIFORM_CORPORATE_EVENTS)
                    finally:
                        engine.book_risk=saved
                    mark_book(nav,market.rates,cfg.capital)
                    audit=verify(nav,trades,decisions,proposal,cfg)
                    nav.to_parquet(folder/'daily_nav.parquet')
                    trades.to_csv(folder/'trades.csv',index=False)
                    decisions.to_csv(folder/'decisions.csv',index=False)
                    record=dict(status='complete',tenor=tenor,variant=variant,wing=wing,proposals=len(proposal),
                        net_excess_pnl=float(nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1]),**audit)
                    write_json(folder/'status.json',record)
                assert record['status']=='complete'
                results.append(dict(model=name,**record))
                books[name]=nav.excess_return.iloc[1:].to_numpy()
                for period,start,end in [('full','2018-01-01','2025-08-29'),*WINDOWS]:
                    periods.append(dict(model=name,tenor=tenor,variant=variant,wing=wing or 'combined',
                        period=period,**period_stats(nav,decisions,start,end)))
                for year in range(2018,2026):
                    annual.append(dict(model=name,year=year,**period_stats(nav,decisions,f'{year}-01-01',f'{year}-12-31')))
                write_json(OUT/'progress.json',dict(stage='replay',arm=name,models=len(results),
                    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
                print(f'{name}: Sharpe {record["annualized_excess_sharpe"]:.6f}, {record["entries"]} entries',flush=True)
        del market
        gc.collect()
    assert len(results)==16
    for wing in WINGS:
        pd.testing.assert_frame_equal(rows[(30,wing)][fields],rows[(90,wing)][fields])
    pd.DataFrame(results).to_csv(REPORT/'comparison.csv',index=False)
    pd.DataFrame(periods).to_csv(REPORT/'periods.csv',index=False)
    pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False)
    table=pd.DataFrame(periods).pivot(index=['tenor','wing','variant'],columns='period',values='annualized_excess_sharpe')
    table.to_csv(REPORT/'period_comparison.csv')
    pd.DataFrame(annual).pivot(index='year',columns='model',values='annualized_excess_sharpe').to_csv(REPORT/'annual_comparison.csv')
    # Paired calendar-block uncertainty, preserving daily cross-tenor pairing.
    uplift=[]
    def sharpe(x):
        return float(np.sqrt(252)*x.mean()/x.std(ddof=1)) if x.std(ddof=1)>1e-12 else 0.
    for wing in ['P25','ATM','P75',None]:
        suffix='_'+wing if wing else ''
        for variant in ['fixed_100','floor']:
            a,b=books[f'90D_{variant}{suffix}'],books[f'30D_{variant}{suffix}']
            rng=np.random.default_rng(2026);n=len(a);block=126;draws=[]
            for _ in range(1000):
                starts=rng.integers(0,n-block+1,size=int(np.ceil(n/block)))
                idx=(starts[:,None]+np.arange(block)).ravel()[:n]
                draws.append(sharpe(a[idx])-sharpe(b[idx]))
            uplift.append(dict(wing=wing or 'combined',variant=variant,sharpe_90_minus_30=sharpe(a)-sharpe(b),
                ci_lower=float(np.quantile(draws,.025)),ci_upper=float(np.quantile(draws,.975))))
    pd.DataFrame(uplift).to_csv(REPORT/'maturity_uplift.csv',index=False)
    for path,value in sources.items():
        assert digest(path)==value,path
    write_json(OUT/'sources.json',sources)
    write_json(REPORT/'audit.json',dict(status='complete',replays=16,signal_date_availability_only=True,
        no_future_path_or_outcome_selection=True,all_ledgers_verified=True,same_decisions_across_maturities=True,
        no_retraining=True,no_fresh_holdout=True,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    (REPORT/'findings.txt').write_text(PROTOCOL+'\n\n'+table.to_string()+'\n')
    write_json(REPORT/'output_hashes.json',{str(p):digest(p) for root in [OUT,REPORT] for p in root.rglob('*') if p.is_file() and p.name!='output_hashes.json'})
    print(table.to_string(),flush=True)


if __name__=='__main__':
    main()
