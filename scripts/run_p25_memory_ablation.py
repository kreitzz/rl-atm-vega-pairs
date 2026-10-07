"""Controlled memory-gate ablations with bounded, disk-backed quote lookups."""
import argparse
from collections import OrderedDict
import gc
import json
from pathlib import Path
import resource
import sqlite3

import numpy as np
import pandas as pd

from scripts.submission import BUNDLE, FEATURE, CAPITAL, require_supervisor, digest
from scripts.run_p25_earnings import train_one, protected, write
from scripts.submission_market import FrozenMarket

OUT = Path('artifacts/contracts/p25_memory_ablation')
REPORT = Path('reports/p25_memory_ablation')
SOURCE = Path('artifacts/contracts/uniform_smile/market')
CASES = ['counts_only', 'no_memory_gate']
CK = Path('config/p25_submission.json')


def eligible(rows, case):
    base = rows.observed & rows.eligible & rows.context & rows.direction.ne(0)
    counts = rows.count_91d.ge(21) & rows.count_365d.ge(63)
    profits = rows.profit_91d.gt(0) & rows.profit_365d.gt(0)
    if case == 'retained_baseline':
        return base & counts & profits
    if case == 'counts_only':
        return base & counts
    if case == 'no_memory_gate':
        return base
    raise ValueError(case)


def prepare():
    if (OUT/'prepared.json').exists():
        return
    ck=json.loads(CK.read_text())
    meta=pd.read_parquet(BUNDLE/FEATURE/'training_rows.parquet')
    original=pd.read_parquet(BUNDLE/FEATURE/'signal_rows.parquet')
    pd.testing.assert_series_equal(eligible(meta,'retained_baseline'),meta.signal_eligible,check_names=False)
    evaluation=meta.date.ge(ck['evaluation_start']) & meta.date.le(ck['last_signal'])
    permitted=np.zeros(len(meta),bool)
    calendar_rows=[]
    calendar_sources={}
    saved_calendar=pd.read_csv(REPORT/'formation_calendar.csv',parse_dates=['date']) if (REPORT/'formation_calendar.csv').exists() else None
    for pair,idx in meta.groupby('pair',sort=False).groups.items():
        path=Path('artifacts/contracts/tail_put25_strategy')/pair/'rule/actions.csv'
        if path.exists():
            dates=pd.DatetimeIndex(pd.read_csv(path,index_col=0,parse_dates=True).index)
            permitted[idx]=pd.DatetimeIndex(meta.loc[idx,'date']).isin(dates)
            calendar_rows.extend(dict(pair=pair,date=str(date.date())) for date in dates)
            calendar_sources[str(path)]=digest(path)
        elif saved_calendar is not None:
            dates=pd.DatetimeIndex(saved_calendar.loc[saved_calendar.pair.eq(pair),'date'])
            permitted[idx]=pd.DatetimeIndex(meta.loc[idx,'date']).isin(dates)
            calendar_rows.extend(dict(pair=pair,date=str(date.date())) for date in dates)
    pd.DataFrame(calendar_rows).to_csv(REPORT/'formation_calendar.csv',index=False)
    evaluation &= permitted
    funnel=[]
    for case in ['retained_baseline']+CASES:
        mask=eligible(meta,case)
        proposals=meta.loc[mask & evaluation].copy()
        proposals['base_level']=4
        if case=='retained_baseline':
            print(f'Baseline selection: {len(proposals)} versus {len(original)}; archived signals {original.date.min()} to {original.date.max()}',flush=True)
            assert len(proposals)==len(original)==608
            common=['date','pair','direction','optionid_a','optionid_b']
            pd.testing.assert_frame_equal(proposals[common].sort_values(['date','pair']).reset_index(drop=True),original[common].sort_values(['date','pair']).reset_index(drop=True))
        else:
            proposals.to_parquet(OUT/f'{case}_signals.parquet',index=False)
        funnel.append(dict(case=case,proposals=len(proposals),pairs=int(proposals.pair.nunique()),
            all_historical_rows=int(mask.sum()),completed_training_rows=int((mask & meta.finite_label).sum())))
        print(f'{case}: {len(proposals)} proposals on {proposals.pair.nunique()} pairs',flush=True)
    pd.DataFrame(funnel).to_csv(REPORT/'proposal_funnel.csv',index=False)
    base=eligible(meta,'no_memory_gate') & evaluation
    group=meta.loc[base,['date','pair','profit_91d','profit_365d','count_91d','count_365d']].copy()
    group['passes_counts']=group.count_91d.ge(21)&group.count_365d.ge(63)
    group['passes_profits']=group.profit_91d.gt(0)&group.profit_365d.gt(0)
    group.to_csv(REPORT/'screen_audit.csv',index=False)
    broad=meta.loc[base]
    ids=sorted(set(broad.optionid_a.astype(int))|set(broad.optionid_b.astype(int)))
    write(OUT/'prepared.json',dict(rows=len(meta),features=ck['features'],quote_ids=ids,
        calendar_sources=calendar_sources,
        sources={str(p):digest(p) for p in [CK,BUNDLE/FEATURE/'training_rows.parquet',BUNDLE/FEATURE/'signal_rows.parquet']}))


def quotes():
    """Stream native yearly caches; keep only necessary IDs on disk, not in RAM."""
    import pyarrow.parquet as pq
    if (OUT/'quotes_complete.json').exists():
        return
    wanted=set(json.loads((OUT/'prepared.json').read_text())['quote_ids'])
    db=sqlite3.connect(OUT/'quotes.sqlite')
    db.execute('PRAGMA cache_size=-2048')
    db.execute('CREATE TABLE IF NOT EXISTS quotes(date TEXT, optionid INTEGER, payload TEXT, PRIMARY KEY(date,optionid)) WITHOUT ROWID')
    sources={}
    for year in range(2018,2026):
        source=SOURCE/f'{year}_quotes.parquet'
        sources[str(source)]=digest(source)
        status=OUT/f'quotes_{year}.json'
        if status.exists():
            continue
        count=0
        parquet=pq.ParquetFile(source)
        for batch in parquet.iter_batches(batch_size=4096,use_threads=False):
            frame=batch.to_pandas()
            frame=frame.loc[frame.optionid.isin(wanted)]
            records=[]
            for row in frame.to_dict('records'):
                for key,value in row.items():
                    if isinstance(value,pd.Timestamp):
                        row[key]=value.isoformat()
                    elif pd.isna(value):
                        row[key]=None
                records.append((str(pd.Timestamp(row['date']).date()),int(row['optionid']),json.dumps(row)))
            db.executemany('INSERT OR REPLACE INTO quotes VALUES(?,?,?)',records)
            count+=len(records)
            del frame,batch,records
        db.commit()
        write(status,dict(year=year,rows=count,source_sha256=sources[str(source)]))
        print(f'Quote cache {year}: {count} rows, streamed in batches of 4096',flush=True)
        del parquet
        gc.collect()
    totals=db.execute('SELECT COUNT(*),COUNT(DISTINCT optionid) FROM quotes').fetchone()
    db.close()
    write(OUT/'quotes_complete.json',dict(rows=totals[0],contracts=totals[1],requested_contracts=len(wanted),sources=sources))


def train(cases,limit):
    import joblib
    ck=json.loads(CK.read_text())
    meta=pd.read_parquet(BUNDLE/FEATURE/'training_rows.parquet')
    cal=pd.read_parquet(BUNDLE/'artifacts/contracts/uniform_smile/market/rates.parquet').index
    months=pd.period_range('2018-01-01',ck['last_signal'],freq='M')
    if limit is not None:
        months=months[:limit]
    for case in cases:
        valid=eligible(meta,case)&meta.finite_label
        folder=OUT/'models'/case;folder.mkdir(parents=True,exist_ok=True)
        for month in months:
            cutoff=month.start_time;stamp=str(cutoff.date())
            if (folder/f'{stamp}.json').exists():
                continue
            rows=meta.loc[valid & meta.date.ge(cutoff-pd.DateOffset(months=18)) & meta.date.lt(cutoff) & meta.end.lt(cutoff)].reset_index(drop=True)
            assert len(rows)==0 or rows.end.max()<cutoff
            model,details=train_one(rows,ck['features'],cal)
            model['fit']=stamp
            joblib.dump(model,folder/f'{stamp}.joblib')
            write(folder/f'{stamp}.json',dict(case=case,fit=stamp,training_rows=len(rows),
                latest_completed_outcome=str(rows.end.max().date()) if len(rows) else None,
                choices=model['choices'].tolist(),**details))
            print(f'{case} {stamp}: {len(rows)} completed examples',flush=True)
            del model,rows
            gc.collect()


def score(cases):
    import joblib
    for case in cases:
        folder=OUT/'replays'/case;folder.mkdir(parents=True,exist_ok=True)
        path=folder/'scored_allocations.parquet'
        if path.exists():
            continue
        rows=pd.read_parquet(OUT/f'{case}_signals.parquet')
        rows['learned_level']=4
        for month,idx in rows.groupby(rows.date.dt.to_period('M')).groups.items():
            model=joblib.load(OUT/'models'/case/f'{month.start_time.date()}.joblib')
            if model['tree'] is not None:
                states=model['tree'].apply(rows.loc[idx,model['features']])
                rows.loc[idx,'learned_level']=model['choices'][np.searchsorted(model['leaves'],states)]
        assert rows.learned_level.isin([2,3,4]).all()
        rows.to_parquet(path,index=False)
        print(f'{case}: scored {len(rows)} proposals',flush=True)
        del rows,model
        gc.collect()


class DiskMarket(FrozenMarket):
    def __init__(self, secids):
        folder=BUNDLE/'artifacts/contracts/p25_signal_transfer/market'
        ids=set(secids)
        if 101322 in ids:
            ids.add(102968)
        stock=pd.read_parquet(folder/'stocks.parquet',columns=['date','secid','close','cfadj'],filters=[('secid','in',sorted(ids))])
        self.spots=stock.pivot(index='date',columns='secid',values='close')
        self.adjustments=stock.pivot(index='date',columns='secid',values='cfadj')
        self.rates=pd.read_parquet(folder/'rates.parquet').rate
        self.divs=pd.read_parquet(folder/'dividends.parquet')
        self.reported_year=None
        self.db=sqlite3.connect(f'file:{OUT}/quotes.sqlite?mode=ro',uri=True)
        self.db.execute('PRAGMA cache_size=-2048')
        self.days=OrderedDict()
        del stock
        gc.collect()
        import pyarrow as pa
        pa.default_memory_pool().release_unused()

    def quote(self,date,oid):
        from scripts.option_flex.contract_engine import MarketRow
        if date not in self.days:
            found=self.db.execute('SELECT payload FROM quotes WHERE date=?',(str(date.date()),))
            daily={}
            for (payload,) in found:
                row=json.loads(payload)
                for key,value in row.items():
                    if key in ['date','expiry'] and value is not None:
                        row[key]=pd.Timestamp(value)
                    elif value is None:
                        row[key]=np.nan
                daily[int(row['optionid'])]=MarketRow(row)
            self.days[date]=daily
            while len(self.days)>2:
                self.days.popitem(last=False)
        self.days.move_to_end(date)
        return self.days[date].get(int(oid))


def replay(cases):
    from scripts import uniform_smile_engine as engine
    from scripts.uniform_smile_risk import book_risk
    from scripts.option_flex.contract_engine import Settings
    from scripts.submission_market import verify_ledger
    from scripts.cheap_vega_runtime import mark_book,metrics,annual_metrics
    ck=json.loads(CK.read_text())
    arms=list(cases)+[c+'_fixed100' for c in cases]
    if not (OUT/'baseline_replay_check.json').exists():
        arms=['retained_baseline']+arms
    for case in arms:
        folder=OUT/'replays'/case;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'status.json').exists():
            continue
        source=OUT/'replays'/case.removesuffix('_fixed100')/'scored_allocations.parquet'
        rows=pd.read_parquet(BUNDLE/FEATURE/'replays/variance_premium/scored_allocations.parquet' if case=='retained_baseline' else source)
        if case.endswith('_fixed100'):
            rows['learned_level']=4
        cfg=Settings(**json.loads((BUNDLE/CAPITAL/'replays/10m_learned/config.json').read_text()))
        market=DiskMarket(set(rows.secid_a.astype(int))|set(rows.secid_b.astype(int)))
        cal=market.rates.index[market.rates.index>='2018-01-01']
        saved=engine.book_risk
        try:
            engine.book_risk=book_risk
            nav,trades,decisions,ledger=engine.simulate(market,rows,cal,cfg,1_000_000,
                policy=lambda row,context:int(row['learned_level']),corporate_events=ck['corporate_events'],wing_capital=1_000_000)
        finally:
            engine.book_risk=saved
        mark_book(nav,market.rates,cfg.capital)
        audit=verify_ledger(nav,trades,decisions,rows)
        if case=='retained_baseline':
            reference=BUNDLE/CAPITAL/'replays/10m_learned'
            pd.testing.assert_frame_equal(nav,pd.read_parquet(reference/'daily_nav.parquet'),check_exact=False,rtol=1e-12,atol=1e-6)
            for frame,name,dates in [(trades,'trades.csv',['date']),(decisions,'decisions.csv',['signal_date','entry_date'])]:
                pd.testing.assert_frame_equal(frame.reset_index(drop=True),pd.read_csv(reference/name,parse_dates=dates),check_dtype=False,check_exact=False,rtol=1e-10,atol=1e-6)
            write(OUT/'baseline_replay_check.json',dict(exact_nav_trades_decisions=True,**audit))
        result=dict(status='complete',case=case,proposals=len(rows),**metrics(nav,cfg.capital),**audit,**ledger)
        nav.to_parquet(folder/'daily_nav.parquet')
        trades.to_csv(folder/'trades.csv',index=False)
        decisions.to_csv(folder/'decisions.csv',index=False)
        pd.DataFrame(annual_metrics(nav,case,cfg.capital)).to_csv(folder/'annual.csv',index=False)
        write(folder/'status.json',result)
        print(f'{case}: Sharpe {result["annualized_excess_sharpe"]:.6f}, {result["entries"]} entries',flush=True)
        market.db.close()
        del market,nav,trades,decisions,rows
        gc.collect()


def summarize():
    records=[];annual=[];decisions=[]
    reference=pd.read_parquet(BUNDLE/FEATURE/'signal_rows.parquet')
    reference_keys=set(map(tuple,reference[['date','pair']].to_numpy()))
    features=json.loads(CK.read_text())['features']
    original=reference.set_index(['date','pair'])[features].sort_index()
    for case in CASES:
        expanded=pd.read_parquet(OUT/f'{case}_signals.parquet').set_index(['date','pair'])
        pd.testing.assert_frame_equal(expanded.loc[original.index,features].sort_index(),original,check_dtype=False,check_exact=True)
    for case in ['retained_baseline']+CASES+[c+'_fixed100' for c in CASES]:
        folder=OUT/'replays'/case
        records.append(json.loads((folder/'status.json').read_text()))
        annual.append(pd.read_csv(folder/'annual.csv'))
        d=pd.read_csv(folder/'decisions.csv',parse_dates=['signal_date'])
        d['case']=case
        d['original_proposal']=[(date,pair) in reference_keys for date,pair in zip(d.signal_date,d.pair)]
        decisions.append(d)
    full_folder=BUNDLE/CAPITAL/'replays/10m_fixed100'
    full_nav=pd.read_parquet(full_folder/'daily_nav.parquet')
    from scripts.cheap_vega_runtime import metrics,annual_metrics
    full_decisions=pd.read_csv(full_folder/'decisions.csv')
    records.append(dict(case='retained_fixed100',proposals=len(full_decisions),entries=int(full_decisions.status.eq('entered').sum()),**metrics(full_nav,10_000_000)))
    annual.append(pd.DataFrame(annual_metrics(full_nav,'retained_fixed100',10_000_000)))
    comparison=pd.DataFrame(records)
    comparison.to_csv(REPORT/'comparison.csv',index=False)
    pd.concat(annual,ignore_index=True).to_csv(REPORT/'annual.csv',index=False)
    combined=pd.concat(decisions,ignore_index=True)
    combined.groupby(['case','original_proposal','status']).size().rename('count').reset_index().to_csv(REPORT/'admission.csv',index=False)
    fits=[]
    for path in sorted((OUT/'models').rglob('*.json')):
        r=json.loads(path.read_text())
        assert r['latest_completed_outcome'] is None or pd.Timestamp(r['latest_completed_outcome'])<pd.Timestamp(r['fit'])
        fits.append(r)
    assert len(fits)==180, 'Both complete monthly fit series are required'
    pd.DataFrame(fits).to_csv(REPORT/'fits.csv',index=False)
    from scripts.cheap_vega_runtime import sharpe
    baseline=pd.read_parquet(OUT/'replays/retained_baseline/daily_nav.parquet').excess_return.iloc[1:].to_numpy()
    intervals=[]
    for case in CASES:
        series=pd.read_parquet(OUT/'replays'/case/'daily_nav.parquet').excess_return.iloc[1:].to_numpy()
        rng=np.random.default_rng(2026)
        gains=[]
        for _ in range(1000):
            starts=rng.integers(0,len(baseline),size=int(np.ceil(len(baseline)/126)))
            idx=((starts[:,None]+np.arange(126)[None,:])%len(baseline)).ravel()[:len(baseline)]
            gains.append(sharpe(series[idx])-sharpe(baseline[idx]))
        intervals.append(dict(case=case,sharpe_change=sharpe(series)-sharpe(baseline),
            lower_95=float(np.quantile(gains,.025)),upper_95=float(np.quantile(gains,.975)),
            block_sessions=126,draws=1000,selection_adjusted=False))
    pd.DataFrame(intervals).to_csv(REPORT/'sharpe_change_intervals.csv',index=False)
    write(REPORT/'audit.json',dict(monthly_models=len(fits),completed_outcomes_only=True,
        evaluation_not_filtered_by_future_labels=True,baseline_full_ledger_exact=True,
        memory_features_retained=True,original_proposal_inputs_exact=True,
        no_checkpoint_promotion=True,original_checkpoint_unchanged=True))
    names={'retained_baseline':'Original gates, learned size','counts_only':'Counts only, learned size','no_memory_gate':'No memory gate, learned size',
        'retained_fixed100':'Original gates, fixed full size','counts_only_fixed100':'Counts only, fixed full size','no_memory_gate_fixed100':'No memory gate, fixed full size'}
    lines=['# Does the profitability-memory gate help?','',
        'Two declared ablations change entry qualification and retrain the same sizing policy monthly. Both keep the 33 inputs, including profitability memory and observation counts. Spread ranks, pair selection, contracts, costs, financing, sizing actions and account limits remain unchanged. Fixed-full-size controls isolate the screen change without a learned sizing change.','',
        '| Entry screen | Proposals | Trades | Annualized excess Sharpe | Profit above cash |',
        '| --- | ---: | ---: | ---: | ---: |']
    for r in comparison.itertuples():
        lines.append(f'| {names[r.case]} | {int(r.proposals)} | {int(r.entries)} | {r.annualized_excess_sharpe:.3f} | ${r.net_excess_pnl:,.0f} |')
    retained=comparison.loc[comparison.case.eq('retained_baseline')].iloc[0]
    broad=comparison.loc[comparison.case.eq('no_memory_gate')].iloc[0]
    counts=comparison.loc[comparison.case.eq('counts_only')].iloc[0]
    lines+=['',f'The full memory gate excludes {100*(1-retained.proposals/broad.proposals):.1f}% of otherwise qualified proposals. '
        f'The positive-profit tests alone exclude {100*(1-retained.proposals/counts.proposals):.1f}% of proposals that meet the count minimums. '
        'Restrictions therefore materially reduce trading opportunities; the result table separates profit from risk-adjusted performance.', '',
        '## Yearly annualized excess Sharpe','',
        '| Year | Original gates | Counts only | No memory gate |','| --- | ---: | ---: | ---: |']
    yearly=pd.concat(annual,ignore_index=True).pivot(index='year',columns='case',values='annualized_excess_sharpe')
    for year,row in yearly.iterrows():
        lines.append(f'| {int(year)}'+(' (partial)' if year==2025 else '')+f' | {row.retained_baseline:.3f} | {row.counts_only:.3f} | {row.no_memory_gate:.3f} |')
    lines+=['','The count-only variant keeps at least 21 completed observations over 91 days and 63 over 365 days. The ungated variant also removes those two minimums. Evaluation proposals never use future trade outcomes or future quote coverage as a filter. Monthly training uses only outcomes completed before the fit date.','',
        'The broader native quote cache is streamed in small batches into an indexed disk store. Its complete baseline NAV, decisions and trades reproduce the retained checkpoint. All new ledgers reconcile and finish with closed inventory. One worker remains capped at 20% of one core and 384 MiB RSS.','',
        'This is a comparison on previously researched 2018–2025 history, not a fresh holdout. More proposals can compete for the same inventory and risk budget; accepted portfolios therefore change. The learner can reduce a proposal to 50%, but its unchanged action menu cannot reject it. No threshold is tuned and the retained checkpoint is preserved.','',
        'This tests whether the gates help within the current architecture, not whether these exact cutoffs are optimal. Removing a hard gate while also introducing a learned zero-exposure action would be a separate experiment. Memory counts measure completed simulated daily candidate cohorts, often with overlapping holdings; they are not independent trades. Memory retains the original $0.65-per-contract label costs, while funded execution uses $0.03.','',
        'See [yearly results](annual.csv), [proposal funnel](proposal_funnel.csv), [screen audit](screen_audit.csv), [admission reasons](admission.csv), [fit dates](fits.csv), and [block-bootstrap Sharpe changes](sharpe_change_intervals.csv). The intervals do not adjust for strategy selection. 2025 ends on 29 August.']
    (REPORT/'findings.md').write_text('\n'.join(lines)+'\n')
    print(comparison[['case','proposals','entries','annualized_excess_sharpe','net_excess_pnl']].to_string(index=False),flush=True)


def package():
    import zipfile
    paths=[p for p in OUT.rglob('*') if p.is_file() and p.name not in ['STOP','progress.json'] and not p.name.endswith(('-wal','-shm'))]
    paths+=[REPORT/'formation_calendar.csv',REPORT/'protocol.json']
    target=Path('data/p25_memory_ablation_experiment.zip')
    write(REPORT/'bundle_manifest.json',dict(scope='Expanded proposals,180 monthly models,5 funded replays and indexed selected quote cache; original33-inputCSV remains unchanged',
        checkpoint_sha256=digest(CK),files={str(p):digest(p) for p in paths}))
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for path in paths:z.write(path,str(path))
    (REPORT/'reproduction.txt').write_text(
        'Use requirements.txt and data/submission, with the unchanged data/p25_variance_premium_inputs.csv.\n'
        'Restore the experiment: unzip -n data/p25_memory_ablation_experiment.zip\n'
        'Run phases separately: .venv/bin/python -m scripts.run_signal_discovery_guarded '
        '--module scripts.run_p25_memory_ablation --stop-file artifacts/contracts/p25_memory_ablation/STOP '
        '--timeout 1800 --duty-cycle 0.2 --rss-limit-mib 384 --min-free-percent 35 '
        '--warning-rss-limit-mib 384 --max-swap-growth-mib 64 --pause-low-headroom-seconds 60 '
        '-- --phase PHASE\n'
        'Phases: test,prepare,quotes,train,score,replay,summarize,package. '
        'Restored preparations,quotes,models and statuses resume without recomputation. '
        'To refit, use a separate checkout and move only restored experiment models away. '
        'To replay, move its saved status.json away. No WRDS access is needed for restored-cache replays. '
        'The exported formation-calendar CSV supports preparation without original action-date files. '
        'Building the disk quote cache anew requires local native uniform_smile yearly quote files; the archive already contains it.\n')
    print(f'Shared {len(paths)} experiment files, {target.stat().st_size/1024**2:.1f} MiB',flush=True)


def main():
    require_supervisor()
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['test','prepare','quotes','train','score','replay','summarize','package'],required=True)
    parser.add_argument('--case',choices=CASES)
    parser.add_argument('--limit-months',type=int)
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    before=protected()
    cases=[args.case] if args.case else CASES
    code=['scripts/run_p25_memory_ablation.py','scripts/run_p25_earnings.py',
        'scripts/uniform_smile_engine.py','scripts/uniform_smile_risk.py','scripts/submission_market.py','scripts/cheap_vega_runtime.py']
    write(REPORT/'protocol.json',dict(cases=CASES,base_checkpoint=str(CK),features='All original 33, including memory',
        code_sha256={p:digest(p) for p in code},
        unchanged='Spread rule, formation/context, contracts, costs, monthly18m completed-only fits,8leaf CEM50x192seed2026,50/75/100 sizing, expiry/dailyhedges,10m capital and inherited risk limits',
        changes='counts_only removes two profit>0 tests; no_memory_gate also removes count91>=21/count365>=63. Both include fixed100 sizing controls.',
        evaluation='2018-01-02 to2025-08-29;last_signal2025-06-30;never gate on finite_label/end or future coverage'))
    if args.phase=='test':
        import unittest
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests',pattern='test_p25_memory_ablation.py'))
        if not result.wasSuccessful():raise SystemExit(1)
    elif args.phase=='prepare':prepare()
    elif args.phase=='quotes':quotes()
    elif args.phase=='train':train(cases,args.limit_months)
    elif args.phase=='score':score(cases)
    elif args.phase=='replay':replay(cases)
    elif args.phase=='summarize':summarize()
    elif args.phase=='package':package()
    assert before==protected(),'Protected source changed'
    print(f'Peak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2:.1f} MiB',flush=True)


if __name__=='__main__':main()
