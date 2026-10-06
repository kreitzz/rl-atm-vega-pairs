"""Bounded 180D/365D call/put selection and shared-ledger management."""
import argparse
from dataclasses import replace
import gc
import getpass
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import unittest

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.download_pair_contract_history import connection,QUOTE_SCHEMA,file_hash
from scripts.run_contract_pilot import normalize_quotes
from scripts.option_flex.black_scholes import fit_quotes
from scripts.rebuild_smile_contracts import repaired_distributions
from scripts.prepare_curve_contracts import HISTORY_COLUMNS
from scripts import run_p25_long_management as long
from scripts.run_p25_capital_sensitivity import annual_metrics

OUT=Path('artifacts/contracts/p25_long_expiries')
REPORT=Path('reports/p25_long_expiries')
DATA=Path('data/long_expiry_options')
PREVIOUS=long.OUT
ROOT=Path('data/option_contracts')
BOUNDS={30:(7,55),180:(150,210),365:(300,400)}
SELECTIONS=[('P',180),('P',365),('C',30),('C',180),('C',365)]
OLD_CONFIGURATION=long.configuration
GE_HEALTHCARE_EVENT=dict(secid=105169,announced='2022-11-30',effective='2023-01-04',reason='GE HealthCare spin-off after January 3 close changes option deliverable',source='https://www.ge.com/news/press-releases/ge-board-of-directors-approves-separation-of-ge-healthcare')


def verify_previous_sources():
    sources=json.loads(Path('reports/p25_long_management/source_manifest.json').read_text())
    for path,want in sources.items():
        assert file_hash(Path(path))==want,f'Original checkpoint source changed: {path}'
    return sources


def observed_peak():
    values=[resource.getrusage(resource.RUSAGE_SELF).ru_maxrss]
    for path,key in [(OUT/'resource_usage.json','observed_peak_rss_bytes'),(OUT/'P/progress.json','peak_rss_bytes'),(OUT/'C/progress.json','peak_rss_bytes')]:
        if path.exists():values.append(json.loads(path.read_text()).get(key,0))
    return max(values)


def configuration(tenor,training=False):
    low,high=BOUNDS[tenor]
    return replace(OLD_CONFIGURATION(30,training),target_dte=tenor,min_dte=low,max_dte=high)


def read_query(query,params,path):
    if path.exists() and path.with_suffix('.json').exists():
        note=json.loads(path.with_suffix('.json').read_text())
        assert note['sha256']==file_hash(path)
        return
    if shutil.disk_usage(DATA).free<10*1024**3:raise RuntimeError('Less than 10GiB disk space available')
    path.parent.mkdir(parents=True,exist_ok=True)
    db=connection(os.getenv('WRDS_USERNAME',getpass.getuser()))
    with db.cursor() as setup:setup.execute("SET statement_timeout = '15min'")
    writer=None;rows=0;temp=path.with_suffix('.partial')
    try:
        with db.cursor(name='long_expiry_stream') as cursor:
            cursor.itersize=4096;cursor.execute(query,params)
            while True:
                values=cursor.fetchmany(4096)
                if not values:break
                frame=pd.DataFrame.from_records(values,columns=QUOTE_SCHEMA.names)
                table=pa.Table.from_pandas(frame,schema=QUOTE_SCHEMA,preserve_index=False)
                if writer is None:writer=pq.ParquetWriter(temp,QUOTE_SCHEMA,compression='zstd')
                writer.write_table(table);rows+=len(frame)
                del frame,table,values
        if writer is None:pq.write_table(pa.Table.from_batches([],schema=QUOTE_SCHEMA),temp,compression='zstd')
        else:writer.close();writer=None
        db.commit();temp.replace(path)
        long.dump(path.with_suffix('.json'),dict(rows=rows,bytes=path.stat().st_size,sha256=file_hash(path),query=query,license='Licensed WRDS/OptionMetrics; local research cache'))
        print(f'Downloaded {path.name}: {rows:,} rows',flush=True)
    finally:
        if writer is not None:writer.close()


def select_long(q,following,tenor,cp='P'):
    cfg=configuration(tenor)
    q=q.copy();q['entry_date']=q.date.map(following)
    q['entry_dte']=(q.expiry-q.entry_date).dt.days
    q['relative_width']=(q.ask-q.bid)/((q.ask+q.bid)/2)
    q['delta_distance']=(q.bs_delta.abs()-.25).abs()
    valid=(q.cp_flag.eq(cp)&q.bs_status.eq('ok')&q.bs_vega.gt(0)&q.standard&q.multiplier.eq(100)&q.bid.gt(0)&q.ask.ge(q.bid)&q.open_interest.ge(cfg.min_open_interest)&q.relative_width.le(cfg.max_relative_spread)&q.entry_dte.between(cfg.min_dte,cfg.max_dte)&q.delta_distance.le(.075))
    q=q.loc[valid].copy();q['tenor_distance']=(q.entry_dte-tenor).abs()
    return q.sort_values(['date','secid','tenor_distance','expiry','delta_distance','relative_width','open_interest','optionid'],ascending=[True]*6+[False,True]).drop_duplicates(['date','secid'])


def prepare():
    if (OUT/'prepared.json').exists():return
    proposals=pd.read_parquet(long.REFERENCE/'replays/variance_premium/scored_allocations.parquet')
    cal=pd.read_parquet(long.base.TRANSFER/'market/rates.parquet').index
    cutoff=cal[-1]-pd.Timedelta(days=410)
    proposals=proposals.loc[proposals.date.le(cutoff)].copy()
    proposals.to_parquet(OUT/'anchor.parquet',index=False)
    print(f'Common entries: {len(proposals)} proposals through {cutoff.date()}; all maturities have full runoff',flush=True)
    requests=pd.concat([proposals[['date',f'secid_{side}']].rename(columns={f'secid_{side}':'secid'}) for side in ['a','b']]).drop_duplicates().sort_values(['date','secid'])
    following=pd.Series(cal[1:].to_numpy(),index=cal[:-1])
    zero=pd.read_parquet(ROOT/'zero_curve.parquet');divs=repaired_distributions()
    selected={key:[] for key in SELECTIONS}
    for year,g in requests.groupby(requests.date.dt.year):
        fields=', '.join('q.'+name for name in QUOTE_SCHEMA.names)
        query=f"select {fields} from optionm.opprcd{int(year)} q join unnest(%s::integer[],%s::date[]) wanted(secid,date) on q.secid=wanted.secid and q.date=wanted.date where q.cp_flag in ('P','C') and ((q.exdate between q.date+150 and q.date+213) or (q.exdate between q.date+300 and q.date+403) or (q.cp_flag='C' and q.exdate between q.date+7 and q.date+58))"
        path=DATA/'signal_chains'/f'{year}.parquet'
        read_query(query,(g.secid.astype(int).tolist(),g.date.dt.date.tolist()),path)
        stocks=pd.read_parquet(ROOT/'underlying'/f'year={year}.parquet',columns=['date','secid','close']);stocks.date=pd.to_datetime(stocks.date)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096,use_threads=False):
            q=normalize_quotes(batch.to_pandas())
            mid=(q.bid+q.ask)/2
            q=q.loc[q.standard&q.multiplier.eq(100)&q.bid.gt(0)&q.ask.ge(q.bid)&q.open_interest.ge(configuration(180).min_open_interest)&((q.ask-q.bid)/mid).le(configuration(180).max_relative_spread)]
            if q.empty:continue
            q=fit_quotes(q,stocks,zero,divs)
            for cp,tenor in selected:
                choice=select_long(q,following,tenor,cp)
                if len(choice):selected[cp,tenor].append(choice)
        print(f'Selected calls and puts for {year}',flush=True)
        long.progress('select',year=int(year));del stocks;gc.collect()
    coverage=[]
    for (cp,tenor),parts in selected.items():
        if not parts:raise RuntimeError(f'No eligible {cp} {tenor}D contracts')
        choice=pd.concat(parts,ignore_index=True)
        choice=select_long(choice,following,tenor,cp)
        folder=OUT/cp/f'{tenor}D';folder.mkdir(parents=True,exist_ok=True)
        choice.to_parquet(folder/'selected.parquet',index=False)
        rows=proposals.drop(columns=['optionid_a','optionid_b'])
        for side in ['a','b']:
            selection=choice[['date','secid','optionid']].rename(columns={'secid':f'secid_{side}','optionid':f'optionid_{side}'})
            rows=rows.merge(selection,on=['date',f'secid_{side}'],how='left',validate='many_to_one')
        assert len(rows)==len(proposals)
        rows['wing']=cp+'25'
        rows.to_parquet(folder/'scored_allocations.parquet',index=False)
        coverage.append(dict(option_type=cp,tenor=tenor,proposals=len(rows),signal_contract_pair_available=int(rows[['optionid_a','optionid_b']].notna().all(axis=1).sum()),selected_contracts=int(choice.optionid.nunique())))
        market=folder/'market';market.mkdir(parents=True,exist_ok=True)
        for kind in ['stocks','rates','dividends']:
            shutil.copyfile(long.base.TRANSFER/'market'/f'{kind}.parquet',market/f'{kind}.parquet')
    pd.DataFrame(coverage).to_csv(REPORT/'coverage.csv',index=False)
    long.dump(OUT/'prepared.json',dict(common_signal_cutoff=str(cutoff.date()),proposals=len(proposals),bounds=BOUNDS,selection='Signal-close expiry-first 25-delta calls and puts, fitted absolute delta tolerance .075, same original entry quote filters'))
    print(pd.DataFrame(coverage).to_string(index=False),flush=True)


def histories():
    choices=pd.concat([pd.read_parquet(OUT/cp/f'{tenor}D/selected.parquet') for cp,tenor in SELECTIONS],ignore_index=True)
    ids=choices.optionid.unique().astype(int).tolist()
    zero=pd.read_parquet(ROOT/'zero_curve.parquet');divs=repaired_distributions()
    for year in range(2018,2026):
        fitted=DATA/'fitted'/f'{year}.parquet'
        if not fitted.exists():
            raw=DATA/'history'/f'{year}.parquet'
            fields=', '.join(QUOTE_SCHEMA.names)
            query=f"select {fields} from optionm.opprcd{year} where optionid=ANY(%s) and cp_flag in ('P','C') and date between %s and %s"
            read_query(query,(ids,f'{year}-01-01',min(f'{year}-12-31','2025-08-29')),raw)
            stock=pd.read_parquet(ROOT/'underlying'/f'year={year}.parquet',columns=['date','secid','close']);stock.date=pd.to_datetime(stock.date)
            fitted.parent.mkdir(parents=True,exist_ok=True);temp=fitted.with_suffix('.partial');writer=None;count=0
            try:
                for batch in pq.ParquetFile(raw).iter_batches(batch_size=4096,use_threads=False):
                    q=fit_quotes(normalize_quotes(batch.to_pandas()),stock,zero,divs)
                    q['delta']=q.bs_delta;q['vega']=q.bs_vega
                    table=pa.Table.from_pandas(q[HISTORY_COLUMNS],preserve_index=False)
                    if writer is None:writer=pq.ParquetWriter(temp,table.schema,compression='zstd')
                    writer.write_table(table);count+=len(q)
                if writer is None:raise RuntimeError(f'No selected-contract history for {year}')
                writer.close();writer=None;temp.replace(fitted)
                long.dump(fitted.with_suffix('.json'),dict(rows=count,sha256=file_hash(fitted),source_sha256=file_hash(raw)))
            finally:
                if writer is not None:writer.close()
            print(f'Fitted {year}: {count:,} held quotes',flush=True);long.progress('history',year=year);del stock;gc.collect()
        for cp,tenor in SELECTIONS:
            target=OUT/cp/f'{tenor}D/market'/f'{year}_quotes.parquet'
            if not target.exists():os.link(fitted,target)


def market_for(tenor,cp):
    long.base.cached.OUT=long.base.TRANSFER if (tenor,cp)==(30,'P') else OUT/cp/f'{tenor}D'
    return long.base.Market()


def run_cases():
    rvframe=pd.read_parquet(PREVIOUS/'rv_states.parquet');rv={int(sid):g.set_index('date')[['rv20','rv60']] for sid,g in rvframe.groupby('secid')};del rvframe
    anchor=pd.read_parquet(OUT/'anchor.parquet')
    for cp in ['P','C']:
        long.OUT=OUT/cp;long.REPORT=REPORT/cp
        long.OUT.mkdir(parents=True,exist_ok=True);long.REPORT.mkdir(parents=True,exist_ok=True)
        for tenor in [30,180,365]:
            market=market_for(tenor,cp)
            rows=anchor if (tenor,cp)==(30,'P') else pd.read_parquet(OUT/cp/f'{tenor}D/scored_allocations.parquet')
            long.save_case(f'{tenor}D_hold',rows,market,tenor)
            records=long.management_fits(rows,market,tenor,36,rv)
            schedule={pd.Timestamp(r['fit']):r['selected'] for r in records}
            callback=lambda date,held:long.held_rule(date,held,rv,schedule[max(k for k in schedule if k<=date)])
            long.save_case(f'{tenor}D_manage_36M',rows,market,tenor,callback)
    long.OUT=OUT;long.REPORT=REPORT


def summarize():
    results=[];annual=[];decisions=[]
    for folder in sorted(OUT.glob('*/replays/*')):
        cp=folder.parent.parent.name
        record=json.loads((folder/'status.json').read_text());record['option_type']=cp;results.append(record);annual.append(pd.read_csv(folder/'annual.csv').assign(option_type=cp))
        counts=pd.read_csv(folder/'decisions.csv').status.value_counts()
        decisions.extend(dict(option_type=cp,case=folder.name,status=status,count=int(count)) for status,count in counts.items())
    result=pd.DataFrame(results);result.to_csv(REPORT/'comparison.csv',index=False)
    pd.concat(annual,ignore_index=True).to_csv(REPORT/'annual.csv',index=False)
    pd.DataFrame(decisions).to_csv(REPORT/'decision_counts.csv',index=False)
    yearly=[]
    for cp,tenor in SELECTIONS:
        rows=pd.read_parquet(OUT/cp/f'{tenor}D/scored_allocations.parquet')
        for year,g in rows.groupby(rows.date.dt.year):
            yearly.append(dict(option_type=cp,tenor=tenor,year=int(year),proposals=len(g),both_signal_contracts_available=int(g[['optionid_a','optionid_b']].notna().all(axis=1).sum())))
    pd.DataFrame(yearly).to_csv(REPORT/'coverage_by_year.csv',index=False)
    fits=[dict(json.loads(p.read_text()),option_type=p.parents[2].name) for p in OUT.glob('*/management_models/*/*.json')]
    assert all(r['training_last_outcome'] is None or pd.Timestamp(r['training_last_outcome'])<pd.Timestamp(r['fit']) for r in fits)
    long.dump(REPORT/'management_fits.json',fits)
    note=('180D AND 365D CALLS AND PUTS WITH MANAGEMENT\n\n'
          'Actual 25-delta calls and puts selected from signal-close WRDS option chains; closest eligible expiry first, then fitted absolute delta nearest .25 within .075. No subsequent quote-path or payoff filter. 180D accepts 150–210 days; 365D accepts 300–400. Calls matched with calls and puts with puts, integer vega matched at entry, daily delta hedge, shared cash and original risk limits.\n\n'
          'First controlled maturity/type transfer: freeze the retained 30D P25 pair/date/direction/initial-size signal stream. This is not native call or long-expiry entry-signal retraining. Current actual held-contract IV, variance-carry estimates and delta drift feed the management policy. Compare hold with the same six-rule management bank trained separately for each option type/tenor on completed 36M shared-ledger episodes, semiannual refits and hold fallback. Prior original-expiry purge retained even for early exits. Same $10m equity, $0.65 training commission, $0.03 actual commission, 0.1 vol point per entry/elective close per leg, and 1bp stock hedging. Closing instructions use prior close and next-session actual executable quotes only. Both legs trim proportionally; no continuous vega rebalance.\n\n'
          'Common entry cutoff 410 calendar days before 2025-08-29 gives all expiries runoff, with 30D matched-calendar call/put controls. Missing selected contracts remain explicit decisions. Raw licensed data streamed in 4096-row batches; only signal-day chains and selected-contract histories downloaded. Black–Scholes Greeks/IV retain the established declared-dividend approximations and corporate exit handling. American exercise/assignment, undeclared dividend forecasts and broker margin remain unmodeled, particularly relevant for long maturities. The calendar and entry signals were previously researched; these new contract paths do not provide a fresh calendar holdout. One worker, 50% of one core and 3GiB RSS stop.\n\n'+pd.read_csv(REPORT/'coverage.csv').to_string(index=False)+'\n\n'+result[['option_type','case','annualized_excess_sharpe','entries','trimmed_cohorts','closed_cohorts','relative_to_cash_cagr_pct']].to_string(index=False)+'\n')
    (REPORT/'findings.txt').write_text(note)
    assert len(result)==12
    for cp,tenor in SELECTIONS:
        selected=pd.read_parquet(OUT/cp/f'{tenor}D/selected.parquet')
        assert selected.cp_flag.eq(cp).all()
        assert selected.entry_dte.between(*BOUNDS[tenor]).all()
        assert selected.delta_distance.le(.075).all()
    protected=verify_previous_sources()
    sources=[Path(__file__),Path('scripts/run_p25_long_management.py'),Path('scripts/uniform_smile_engine.py'),OUT/'anchor.parquet',OUT/'prepared.json',OUT/'corporate_events.json',Path('tests/test_p25_long_expiries.py')]
    sources+=list(OUT.glob('*/management_models/*/*.json'))+list(OUT.glob('*/replays/*/status.json'))+list(OUT.glob('*/*/selected.parquet'))+list(OUT.glob('*/*/scored_allocations.parquet'))
    long.dump(REPORT/'source_manifest.json',{str(p):file_hash(p) for p in sources})
    long.dump(REPORT/'audit.json',dict(status='complete',cases=len(result),all_ledgers_verified=True,completed_outcome_purge=True,original_sources_unchanged=True,protected_source_count=len(protected),matched_option_types=True,peak_rss_bytes=observed_peak()))
    print(result[['option_type','case','annualized_excess_sharpe','entries']].to_string(index=False),flush=True)


def probe():
    db=connection(os.getenv('WRDS_USERNAME',getpass.getuser()))
    with db.cursor() as cursor:
        cursor.execute('select current_date')
        date=cursor.fetchone()[0]
    db.commit()
    print(f'WRDS read-only connection available; server date {date}',flush=True)


def inspect_data():
    for year in [2022,2023]:
        rows=pd.read_parquet(DATA/'fitted'/f'{year}.parquet',filters=[('optionid','=',141664753)])
        print(rows.loc[rows.date.between('2022-12-28','2023-01-10')].to_string(index=False),flush=True)
    bad=[]
    for year in range(2018,2026):
        for batch in pq.ParquetFile(DATA/'fitted'/f'{year}.parquet').iter_batches(columns=['date','secid','optionid','standard'],batch_size=4096,use_threads=False):
            q=batch.to_pandas();bad.append(q.loc[~q.standard])
    bad=pd.concat(bad,ignore_index=True)
    print(bad.groupby(['secid',bad.date.dt.year]).agg(first_date=('date','min'),contracts=('optionid','nunique'),rows=('optionid','size')).to_string(),flush=True)
    for path in OUT.glob('*/replays/*/trades.csv'):
        trades=pd.read_csv(path,parse_dates=['date'])
        g=trades.loc[trades.secid.eq(105169)&trades.date.le('2023-01-03')].groupby(['cohort','optionid']).quantity.sum()
        print(f'{path.parent.parent.parent.name}/{path.parent.name}: GE remaining shares-equivalent contracts {g.abs().sum()}',flush=True)
        assert g.eq(0).all(),'Previously completed case needs corporate-event replay'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['probe','test','prepare','histories','replay','summarize','inspect','all'],default='all');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    DATA.mkdir(parents=True,exist_ok=True)
    long.OUT=OUT;long.REPORT=REPORT;long.configuration=configuration
    long.base.UNIFORM_CORPORATE_EVENTS=[*long.base.UNIFORM_CORPORATE_EVENTS,GE_HEALTHCARE_EVENT]
    long.dump(OUT/'corporate_events.json',long.base.UNIFORM_CORPORATE_EVENTS)
    if args.phase=='probe':probe();return
    if args.phase=='inspect':inspect_data();return
    if args.phase=='test':
        suite=unittest.defaultTestLoader.loadTestsFromNames(['tests.test_p25_long_expiries'])
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        long.dump(REPORT/'tests.json',dict(tests_run=result.testsRun,successful=result.wasSuccessful()))
        if not result.wasSuccessful():raise SystemExit(1)
        return
    verify_previous_sources()
    if args.phase in ['prepare','all']:prepare()
    if args.phase in ['histories','all']:histories()
    if args.phase in ['replay','all']:run_cases()
    if args.phase in ['summarize','all']:summarize()
    long.dump(OUT/'resource_usage.json',dict(observed_peak_rss_bytes=observed_peak(),cpu_core_fraction=.5,workers=1,rss_limit_bytes=3*1024**3,last_completed_phase=args.phase))


if __name__=='__main__':main()
