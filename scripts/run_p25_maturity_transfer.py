"""Matched 30D/90D expiry replay of the frozen P25 decision stream."""
from dataclasses import replace
import argparse
import gc
import json
from pathlib import Path
import resource
import shutil
import time
import unittest

import numpy as np
import pandas as pd

from scripts.p25_maturity_transfer import WINDOWS,remap,common_entries,period_stats
from scripts.p25_signal_transfer import WINGS
from scripts import uniform_smile_data as cached
from scripts import uniform_smile_engine as engine
from scripts.uniform_smile_risk import book_risk
from scripts.curve_trading.config import config
from scripts.run_atm65_ablation import write_json
from scripts.run_contract_walkforward import digest
from scripts.run_uniform_smile import verify,UNIFORM_CORPORATE_EVENTS
from scripts.run_aligned_contract_learning import mark_book
from scripts.run_p25_signal_transfer import Market as BaseMarket
from scripts.rebuild_smile_contracts import repaired_distributions
from scripts.option_flex.black_scholes import fit_quotes
from scripts.run_broad_contract_replay import ROOT

OUT=Path('artifacts/contracts/p25_maturity_transfer')
REPORT=Path('reports/p25_maturity_transfer')
SHORT=Path('artifacts/contracts/p25_signal_transfer')
LONG=Path('data/curve90')
PROTOCOL=(
    'Maturity transfer of the frozen 30D P25 pair/date, direction and monthly '
    '50/75/100 sizing stream to 30D and90D ATM calls/P25 puts/P75 puts. '
    'No retraining and no new wing-native rank or profitability veto. '
    'Preserve existing30D mapped contracts. Use existing90D selections '
    '(OI100,ATM-expiry anchor, fitted delta tolerance.075) with no replacement '
    'or selection based on future quote paths. Missing selections reach '
    'execution explicitly. Both maturities use the same entry cutoff '
    '110 calendar days before Aug29,2025, allowing90D holding runoff. '
    'Original30D entry eligibility7-55D;90D75-105D. All other costs, '
    'risk and financing settings identical; actualcommission.03, '
    'mid+.1volpoint,stockhedge1bp,vegamatch<=10%,daily deltahedge, '
    'hold each leg to its own expiry with existing causal corporate exits. '
    'Fixed100 control versus frozen P25 sizing, separate wings and combined '
    'shared book. Report full2018-Aug2025, calendar2018-2020,2021-2023, '
    '2024-partial2025 and individualyears. Calendar returns include open '
    'inventory carried across boundaries; periods are not independent '
    'reset backtests or untouched holdouts. Report entry/coverage counts '
    'and reject all unresolved holdings rather than dropping outcomes. '
    'Full frozen30D baseline remains unchanged outside this matched cutoff '
    'study. One worker,50%of onecore,oneBLAS/Arrowthread,3GiB stop.'
)


def track(sources,path):
    path=Path(path)
    if str(path) not in sources:
        sources[str(path)]=digest(path)
    return path


def progress(stage,**details):
    write_json(OUT/'progress.json',dict(stage=stage,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,**details))


def prepare(sources):
    anchor=pd.read_parquet(track(sources,SHORT/'P25_rows.parquet'))
    last=pd.read_parquet(track(sources,SHORT/'market/rates.parquet')).index.max()
    anchor=common_entries(anchor,last)
    anchor.to_parquet(OUT/'anchor.parquet',index=False)
    sids=sorted(set(anchor.secid_a)|set(anchor.secid_b))
    choices={}
    for sid in sids:
        for wing,(cp,bucket) in WINGS.items():
            pieces=[]
            for year in range(2018,2026):
                path=LONG/'selection'/cp/str(year)/f'{sid}.parquet'
                if path.exists():
                    s=pd.read_parquet(track(sources,path))
                    pieces.append(s.loc[s.tenor.eq(90)&s.bucket.eq(bucket),['date','optionid']])
            choices[(sid,wing)]=pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame({'date':pd.to_datetime([]),'optionid':[]})
    coverage=[];long_needed={}
    for tenor in [30,90]:
        folder=OUT/f'{tenor}D'
        folder.mkdir(exist_ok=True)
        for wing in WINGS:
            if tenor==30:
                rows=common_entries(pd.read_parquet(track(sources,SHORT/f'{wing}_rows.parquet')),last)
                pd.testing.assert_frame_equal(rows[['date','pair','direction','base_level']],anchor[['date','pair','direction','base_level']])
            else:
                rows=remap(anchor,choices,wing)
                for side in ['a','b']:
                    for sid,g in rows.groupby('secid_'+side):
                        long_needed.setdefault(int(sid),set()).update(g['optionid_'+side].dropna().astype(int))
            rows.to_parquet(folder/f'{wing}_rows.parquet',index=False)
            coverage.append(dict(tenor=tenor,wing=wing,proposals=len(rows),
                signal_contract_pair_available=int(rows[['optionid_a','optionid_b']].notna().all(axis=1).sum())))
        market=folder/'market'
        market.mkdir(exist_ok=True)
        for kind in ['stocks','dividends','rates']:
            source=track(sources,SHORT/'market'/f'{kind}.parquet')
            shutil.copyfile(source,market/f'{kind}.parquet')
        if tenor==30:
            for year in range(2018,2026):
                source=track(sources,SHORT/'market'/f'{year}_quotes.parquet')
                shutil.copyfile(source,market/source.name)
    stocks=pd.read_parquet(OUT/'90D/market/stocks.parquet')
    zero=pd.read_parquet(track(sources,ROOT/'zero_curve.parquet'))
    distributions=repaired_distributions()
    track(sources,ROOT/'distributions.parquet')
    track(sources,Path('data/raw/opposite_wing_DEXCAUS.csv'))
    for year in range(2018,2026):
        target=OUT/'90D/market'/f'{year}_quotes.parquet'
        if target.exists():
            continue
        parts=[];repaired=0
        for sid,ids in long_needed.items():
            if not ids:
                continue
            for cp in ['C','P']:
                source=LONG/'history'/cp/str(year)/f'{sid}.parquet'
                if not source.exists():
                    continue
                q=pd.read_parquet(track(sources,source),filters=[('optionid','in',sorted(ids))])
                if not len(q):
                    continue
                # The same declared-distribution repairs already used by30D.
                # Keep unhandled cases unsupported for the causal exit logic.
                bad=q.bs_status.eq('unsupported_distribution')&q.secid.isin([108969,105003,127149])
                if bad.any():
                    fitted=fit_quotes(q.loc[bad],stocks.loc[stocks.secid.eq(sid)],zero,distributions)
                    for col in ['bs_iv','bs_rate','known_dividend_pv','bs_status']:
                        q.loc[bad,col]=fitted[col].to_numpy()
                    q.loc[bad,'delta']=fitted.bs_delta.to_numpy()
                    q.loc[bad,'vega']=fitted.bs_vega.to_numpy()
                    repaired+=int(bad.sum())
                parts.append(q)
        q=pd.concat(parts,ignore_index=True)
        assert not q.duplicated(['date','optionid']).any()
        q.to_parquet(target,index=False)
        print(f'90D replay cache {year}: {len(q):,} quotes, {repaired} known distribution repairs',flush=True)
        del q,parts
        gc.collect()
        progress('prepare',year=year)
        write_json(OUT/'sources.json',sources)
    pd.DataFrame(coverage).to_csv(REPORT/'coverage.csv',index=False)
    print(pd.DataFrame(coverage).to_string(index=False),flush=True)
    write_json(OUT/'preparation.json',dict(common_signal_end=str((last-pd.Timedelta(days=110)).date()),common_proposals=len(anchor)))


def allocations(tenor,variant,wing=None):
    pieces=[]
    for target in ([wing] if wing else WINGS):
        rows=pd.read_parquet(OUT/f'{tenor}D'/f'{target}_rows.parquet')
        if variant=='fixed_100':
            rows['base_level']=4
        elif variant!='floor':
            raise ValueError(variant)
        pieces.append(rows)
    return pd.concat(pieces,ignore_index=True).sort_values(['date','pair','wing'],kind='stable').reset_index(drop=True)


def replay():
    for tenor in [30,90]:
        cached.OUT=OUT/f'{tenor}D'
        market=BaseMarket()
        cal=market.rates.index[market.rates.index>='2018-01-01']
        for variant in ['fixed_100','floor']:
            for wing in ['P25','ATM','P75',None]:
                name=f'{tenor}D_{variant}'+('_'+wing if wing else '')
                folder=OUT/'replays'/name
                folder.mkdir(parents=True,exist_ok=True)
                if (folder/'status.json').exists():
                    continue
                rows=allocations(tenor,variant,wing)
                rows.to_parquet(folder/'scored_allocations.parquet',index=False)
                multiplier=3 if wing is None else 1
                cfg=replace(config(),option_commission=.03,capital=1_000_000*multiplier*345,max_gross_vega=20_000*multiplier*345)
                if tenor==90:
                    cfg=replace(cfg,target_dte=90,min_dte=75,max_dte=105)
                market.variant,market.started,market.reported_year=name,time.monotonic(),None
                print(f'Replay {name}: {len(rows):,} unchanged decision proposals',flush=True)
                saved=engine.book_risk
                try:
                    engine.book_risk=book_risk
                    nav,trades,decisions,_=engine.simulate(market,rows,cal,cfg,1_000_000*multiplier,
                        wing_capital=1_000_000,corporate_events=UNIFORM_CORPORATE_EVENTS)
                finally:
                    engine.book_risk=saved
                mark_book(nav,market.rates,cfg.capital)
                audit=verify(nav,trades,decisions,rows,cfg)
                if tenor==30:
                    reference=SHORT/'replays'/(variant+('_'+wing if wing else ''))
                    prior=pd.read_parquet(reference/'daily_nav.parquet')
                    cutoff=market.rates.index.max()-pd.Timedelta(days=110)
                    pd.testing.assert_frame_equal(nav.loc[:cutoff],prior.loc[:cutoff],
                        check_exact=False,rtol=1e-12,atol=1e-6)
                    print(f'30D reference prefix reproduced: {name}',flush=True)
                nav.to_parquet(folder/'daily_nav.parquet')
                trades.to_csv(folder/'trades.csv',index=False)
                decisions.to_csv(folder/'decisions.csv',index=False)
                write_json(folder/'status.json',dict(status='complete',tenor=tenor,variant=variant,wing=wing,
                    proposals=len(rows),net_excess_pnl=float(nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1]),**audit))
                progress('replay',arm=name,**audit)
                print(f'{name}: Sharpe {audit["annualized_excess_sharpe"]:.6f}, {audit["entries"]} entries',flush=True)
        del market
        gc.collect()


def summarize(sources):
    results=[];periods=[];annual=[]
    for folder in sorted((OUT/'replays').iterdir()):
        record=json.loads((folder/'status.json').read_text())
        assert record['status']=='complete'
        results.append(dict(model=folder.name,**record))
        nav=pd.read_parquet(folder/'daily_nav.parquet')
        decisions=pd.read_csv(folder/'decisions.csv',parse_dates=['signal_date','entry_date'])
        for name,start,end in [('full','2018-01-01','2025-08-29'),*WINDOWS]:
            periods.append(dict(model=folder.name,tenor=record['tenor'],variant=record['variant'],wing=record['wing'] or 'combined',
                period=name,**period_stats(nav,decisions,start,end)))
        for year in range(2018,2026):
            annual.append(dict(model=folder.name,year=year,**period_stats(nav,decisions,f'{year}-01-01',f'{year}-12-31')))
    assert len(results)==16
    for tenor in [30,90]:
        pd.testing.assert_frame_equal(allocations(tenor,'floor').drop(columns='base_level'),allocations(tenor,'fixed_100').drop(columns='base_level'))
    a=allocations(30,'floor');b=allocations(90,'floor')
    pd.testing.assert_frame_equal(a[['date','pair','wing','direction','base_level']],b[['date','pair','wing','direction','base_level']])
    pd.DataFrame(results).to_csv(REPORT/'comparison.csv',index=False)
    pd.DataFrame(periods).to_csv(REPORT/'periods.csv',index=False)
    pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False)
    pd.DataFrame(annual).pivot(index='year',columns='model',values='annualized_excess_sharpe').to_csv(REPORT/'annual_comparison.csv')
    table=pd.DataFrame(periods).pivot(index=['tenor','wing','variant'],columns='period',values='annualized_excess_sharpe')
    table.to_csv(REPORT/'period_comparison.csv')
    for path,value in sources.items():
        assert digest(path)==value,path
    write_json(REPORT/'audit.json',dict(status='complete',replays=16,no_retraining=True,no_fresh_holdout=True,
        same_pair_date_direction_and_sizing_across_maturities=True,same_control_proposals=True,
        no_forward_path_selection=True,all_ledgers_reconciled=True,all_inventory_closed=True,
        calendar_periods_include_carried_inventory=True,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    (REPORT/'findings.txt').write_text(PROTOCOL+'\n\n'+table.to_string()+'\n')
    write_json(REPORT/'output_hashes.json',{str(p):digest(p) for root in [OUT,REPORT] for p in root.rglob('*') if p.is_file() and p.name!='output_hashes.json'})
    print(table.to_string(),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=['test','prepare','replay','all'],default='all')
    args=p.parse_args()
    if args.phase=='test':
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests',pattern='test_p25_maturity_transfer.py'))
        if not result.wasSuccessful():
            raise SystemExit(1)
        return
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    path=OUT/'sources.json'
    sources=json.loads(path.read_text()) if path.exists() else {}
    for code in [Path(__file__),Path('scripts/p25_maturity_transfer.py'),Path('scripts/uniform_smile_engine.py'),Path('scripts/uniform_smile_risk.py'),Path('scripts/rebuild_smile_contracts.py')]:
        if str(code) in sources:
            assert digest(code)==sources[str(code)],code
        track(sources,code)
    write_json(OUT/'inputs.json',dict(protocol=PROTOCOL,windows=WINDOWS,signal_source=str(SHORT/'P25_rows.parquet'),no_retraining=True))
    if args.phase in ['prepare','all']:
        prepare(sources)
        write_json(OUT/'sources.json',sources)
    if args.phase in ['replay','all']:
        replay()
        summarize(sources)


if __name__=='__main__':
    main()
