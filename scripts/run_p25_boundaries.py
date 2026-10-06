"""Separate spread qualification from inherited portfolio entry caps."""
import argparse
from dataclasses import asdict,replace
import gc
import json
from pathlib import Path
import resource
import shutil
import time
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from scripts import run_p25_action_windows as base
from scripts import p25_boundaries as policy
from scripts.p25_action_windows import fit_menu,score_model
from scripts.p25_recency import completed_months
from scripts.run_same_structure_calendar import formation_dates,filter_calendar

OUT=Path('artifacts/contracts/p25_boundaries')
REPORT=Path('reports/p25_boundaries')
NATIVE=Path('artifacts/contracts/uniform_smile')
REFERENCE=Path('artifacts/contracts/p25_recency/replays')
PROTOCOL=(
    'Declared spread rank20/80,30/70,40/60 boundaries x current inherited '
    'entry caps,stress cap disabled,and all discretionary entry caps disabled. '
    'Each has18-month monthly direct-Sharpe learned50/75/100 sizing and '
    'fixed100 control,18 replays. Same27 feature definitions,8leaf adaptive '
    'minleaf tree,CEM50x192 seed2026,fullsize anchor/fallback,.65 training '
    'labels,.03 actual commission,.1volpoint perleg penalty,1bp stock hedge '
    'execution,funding/borrow/dividends,daily delta hedge,own expiry,integer '
    'entry vega mismatch<=10%,contract validity and corporate exits. '
    'Only spread boundary changes signal qualification and its completed '
    'training universe;positive91/365day memory/count21/63,beta0.85-1.15, '
    'correlation>=.5 and inherited formation/calendar coverage remain fixed. '
    'Reorient features/targets/memory with the declared signal direction. '
    'No future label completeness filter on evaluated proposals. New contract '
    'IDs are drawn from cached original selections;existing quotes retain '
    'precedence. No new downloads or contract selection. Keep$345m funded '
    'portfolio denominator,$1m pair/wing sizing reference,$450 fullsize '
    'vega pervolpoint perleg. Disabling caps does not change capital or '
    'entry sizing;nonpositiveNAV checks remain and risk metrics are measured. '
    'Current20/80 learned andfixed replays/models must exactly reproduce '
    '.9617449 and.7569318 references. Select learning arm using2018-2023 '
    'only;report2024-partial2025,annuals and downside separately. Previously '
    'examined history/retrospective formation coverage,no new holdout or '
    'automatic checkpoint promotion. One worker,50%of onecore,3GiB RSS stop.'
)


def setup():base.OUT=OUT;base.REPORT=REPORT


def prepare(sources,cal):
    raw=pd.read_parquet(sources.track(base.SOURCE/'P25_rows.parquet'))
    calendars,hashes=formation_dates()
    for path,digest in hashes.items():
        sources.track(path);assert sources.hashes[path]==digest
    permitted=filter_calendar(raw,calendars);parts=[];counts=[]
    for boundary in policy.BOUNDARIES:
        rows=policy.states(raw,boundary)
        if boundary==20:pd.testing.assert_frame_equal(rows,raw)
        rows.to_parquet(OUT/f'{boundary}_training_rows.parquet',index=False)
        evaluated=rows.date.between('2018-01-01','2025-06-30')
        mask=evaluated&rows.signal_eligible&permitted
        proposals=rows.loc[mask].copy().sort_values(['date','pair','wing'],kind='stable').reset_index(drop=True)
        proposals['base_level']=4;proposals['contract_source_a']=proposals['contract_source_b']='frozen_P25'
        counts.append(dict(boundary=boundary,qualified_without_calendar=int((evaluated&rows.signal_eligible).sum()),
            proposals=len(proposals),proposal_pairs=int(proposals.pair.nunique()),
            unresolved_shadow_labels_retained=int((~proposals.finite_label).sum())))
        if boundary==20:
            frozen=pd.read_parquet(sources.track(base.TRANSFER/'P25_rows.parquet'))
            cols=['date','pair','direction','optionid_a','optionid_b','secid_a','secid_b']
            pd.testing.assert_frame_equal(proposals[cols],frozen[cols],check_dtype=False)
            assert len(proposals)==608
            proposals['base_level']=frozen.base_level.to_numpy()
        proposals.to_parquet(OUT/f'{boundary}_signal_rows.parquet',index=False)
        parts.append(proposals)
    counts=pd.DataFrame(counts);counts.to_csv(REPORT/'coverage.csv',index=False)
    assert counts.proposals.tolist()==[608,846,1127]
    needed=pd.concat(parts,ignore_index=True);folder=OUT/'market';folder.mkdir(exist_ok=True)
    for kind in ['stocks','dividends','rates']:
        shutil.copyfile(sources.track(base.TRANSFER/'market'/f'{kind}.parquet'),folder/f'{kind}.parquet')
    ids={}
    for side in ['a','b']:
        for sid,rows in needed.groupby('secid_'+side):ids.setdefault(int(sid),set()).update(rows['optionid_'+side].dropna().astype(int))
    # Stream at most one year and one asset at a time; original quote priority.
    for year in range(2018,2026):
        target=folder/f'{year}_quotes.parquet'
        if target.exists():continue
        old=pd.read_parquet(sources.track(base.TRANSFER/'market'/f'{year}_quotes.parquet'))
        pieces=[old]
        for sid,oids in ids.items():
            q=pd.read_parquet(sources.track(NATIVE/'market/assets'/f'{sid}_quotes.parquet'),filters=[
                ('date','>=',pd.Timestamp(year,1,1)),('date','<',pd.Timestamp(year+1,1,1)),('optionid','in',sorted(oids))])
            if len(q):pieces.append(q)
        quotes=pd.concat(pieces,ignore_index=True).drop_duplicates(['date','optionid'],keep='first')
        pd.testing.assert_frame_equal(quotes.iloc[:len(old)].reset_index(drop=True),old.reset_index(drop=True))
        quotes.to_parquet(target,index=False)
        print(f'Boundary quotes {year}: {len(quotes):,},original {len(old):,} preserved',flush=True)
        del quotes,pieces,old,q;gc.collect();sources.save();base.progress('prepare',year=year)
    sources.save();print(counts.to_string(index=False),flush=True)


def train(sources,cal):
    logs=[]
    for boundary in policy.BOUNDARIES:
        meta=pd.read_parquet(sources.track(OUT/f'{boundary}_training_rows.parquet'))
        for month in pd.period_range('2018-01-01',cal[-1]-pd.Timedelta(days=base.RUNOFF),freq='M'):
            cutoff=month.start_time;stamp=str(cutoff.date());folder=OUT/'models'/str(boundary);folder.mkdir(parents=True,exist_ok=True)
            path=folder/f'{stamp}.json'
            if path.exists():logs.append(json.loads(path.read_text()));continue
            rows=meta.loc[meta.signal_eligible&meta.finite_label&completed_months(meta.date,meta.end,cutoff,18)].reset_index(drop=True)
            assert len(rows)==0 or rows.end.max()<cutoff
            tree=None;leaves=np.array([0]);book=None
            if len(rows)>=16:
                tree=DecisionTreeRegressor(max_leaf_nodes=8,min_samples_leaf=max(8,min(63,len(rows)//8)),random_state=2026).fit(rows[base.FEATURES],rows.target)
                leaves=np.unique(tree.apply(rows[base.FEATURES]));states=np.searchsorted(leaves,tree.apply(rows[base.FEATURES]))
                first=int(rows.position.min()+1);last=int(cal.get_indexer(pd.DatetimeIndex(rows.end)).max());book=np.zeros((last-first+1,len(leaves)*5))
                for pair,idx in rows.groupby('pair',sort=False).groups.items():
                    idx=np.asarray(idx);disk=np.load(sources.track(NATIVE/'pairs'/pair/'P25_labels.npy'),mmap_mode='r')
                    labels=base.old.directional_paths(disk[rows.loc[idx,'label_row'].to_numpy(int)],rows.loc[idx,'direction'].to_numpy(int))
                    base.aggregate(book,states[idx],labels,rows.loc[idx,'position'].to_numpy(int),first)
                    del disk,labels
            choices,details=(np.array([4]),{}) if book is None else fit_menu(book,cal[first:last+1],[2,3,4])
            model=dict(tree=tree,leaves=leaves,choices=choices,fit=stamp)
            if boundary==20:
                previous=joblib.load(sources.track(REFERENCE.parent/'models/18M_floor'/f'{stamp}.joblib'))
                np.testing.assert_array_equal(choices,previous['choices']);np.testing.assert_array_equal(leaves,previous['leaves'])
                if tree is not None:
                    for field in ['feature','threshold','children_left','children_right','value']:
                        np.testing.assert_array_equal(getattr(tree.tree_,field),getattr(previous['tree'].tree_,field))
            record=dict(boundary=boundary,fit=stamp,training_rows=len(rows),training_pairs=int(rows.pair.nunique()),
                last_training_outcome=str(rows.end.max().date()) if len(rows) else None,choices=choices.tolist(),**details)
            joblib.dump(model,folder/f'{stamp}.joblib');base.write_json(path,record);logs.append(record)
            del rows,tree,book;gc.collect()
            if month.month in [1,7]:print(f'Boundary{boundary} fits {stamp},completed{len(logs)}',flush=True)
            sources.save();base.progress('train',boundary=boundary,fit=stamp,models_complete=len(logs))
        del meta
    assert len(logs)==270;pd.DataFrame(logs).to_csv(REPORT/'fits.csv',index=False)


def allocations(boundary,kind):
    rows=pd.read_parquet(OUT/f'{boundary}_signal_rows.parquet');rows['learned_level']=4
    if kind=='learned':
        for month,idx in rows.groupby(rows.date.dt.to_period('M')).groups.items():
            model=joblib.load(OUT/'models'/str(boundary)/f'{month.start_time.date()}.joblib')
            assert pd.Timestamp(model['fit'])<=rows.loc[idx,'date'].min()
            rows.loc[idx,'learned_level']=score_model(model,rows.loc[idx],base.FEATURES)
    assert rows.learned_level.isin([2,3,4]).all()
    return rows


def replay(sources,cal):
    base.cached.OUT=OUT;market=base.Market();calendar=cal[cal>='2018-01-01']
    cfg=replace(base.config(),option_commission=.03,capital=345_000_000,max_gross_vega=20_000*345)
    for risk_mode in policy.RISK_MODES:
        for boundary in policy.BOUNDARIES:
            for kind in ['learned','fixed100']:
                case=f'{boundary}_{risk_mode}_{kind}';folder=OUT/'replays'/case;folder.mkdir(parents=True,exist_ok=True)
                if (folder/'status.json').exists():continue
                rows=allocations(boundary,kind);rows.to_parquet(folder/'scored_allocations.parquet',index=False)
                market.variant=case;market.started=time.monotonic();market.reported_year=None
                print(f'Replay {case}:{len(rows)} proposals',flush=True)
                saved_risk=base.engine.book_risk;saved_breaches=base.engine.risk_breaches
                try:
                    base.engine.book_risk=base.book_risk
                    base.engine.risk_breaches=lambda metrics,nav,cfg:policy.breaches(metrics,nav,cfg,risk_mode)
                    nav,trades,decisions,ledger=base.engine.simulate(market,rows,calendar,cfg,1_000_000,
                        policy=lambda row,context:int(row['learned_level']),corporate_events=base.UNIFORM_CORPORATE_EVENTS,wing_capital=1_000_000)
                finally:base.engine.book_risk=saved_risk;base.engine.risk_breaches=saved_breaches
                base.mark_book(nav,market.rates,cfg.capital);audit=base.verify(nav,trades,decisions,rows,cfg)
                assert not decisions.level.eq(0).any()
                if risk_mode=='no_caps':assert not decisions.status.isin(['risk_limit_entry','risk_deleveraging','wing_risk_deleveraging']).any()
                if risk_mode=='current' and boundary==20:
                    reference=REFERENCE/('18M_floor' if kind=='learned' else 'fixed_100')
                    pd.testing.assert_frame_equal(nav,pd.read_parquet(sources.track(reference/'daily_nav.parquet')),rtol=1e-12,atol=1e-6)
                    for table,file in [(trades,'trades.csv'),(decisions,'decisions.csv')]:
                        pd.testing.assert_frame_equal(table.reset_index(drop=True),pd.read_csv(sources.track(reference/file),parse_dates=['date'] if file=='trades.csv' else ['signal_date','entry_date']),check_dtype=False,rtol=1e-12,atol=1e-7)
                    print(f'Frozen{kind} reference exactly reproduced',flush=True)
                nav.to_parquet(folder/'daily_nav.parquet');trades.to_csv(folder/'trades.csv',index=False);decisions.to_csv(folder/'decisions.csv',index=False)
                base.write_json(folder/'status.json',dict(status='complete',model=case,boundary=boundary,risk_mode=risk_mode,kind=kind,
                    proposals=len(rows),net_excess_pnl=float(nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1]),**audit,**ledger))
                sources.save();base.progress('replay',model=case,**audit)
                print(f'{case}:Sharpe{audit["annualized_excess_sharpe"]:.6f},entries{audit["entries"]}',flush=True)
                del nav,trades,decisions,rows;gc.collect()


def summarize(sources):
    rows=[];annual=[];statuses=[];books={}
    for folder in sorted((OUT/'replays').iterdir()):
        record=json.loads((folder/'status.json').read_text());assert record['status']=='complete'
        nav=pd.read_parquet(folder/'daily_nav.parquet');books[folder.name]=nav
        d=pd.read_csv(folder/'decisions.csv',parse_dates=['entry_date']);assert len(d)==record['proposals']
        excess=nav.nav-nav.cash_benchmark
        rows.append(dict(model=folder.name,boundary=record['boundary'],risk_mode=record['risk_mode'],kind=record['kind'],
            proposals=record['proposals'],entries=record['entries'],active_pairs=record['active_pairs'],
            annualized_excess_sharpe=record['annualized_excess_sharpe'],net_excess_pnl=record['net_excess_pnl'],
            selection_2018_2023_sharpe=base.sharpe(nav.loc[:'2023-12-31','excess_return'].iloc[1:]),
            recent_2024_2025_sharpe=base.sharpe(nav.loc['2024-01-01':,'excess_return']),
            worst_daily_excess_pnl=float((nav.pnl-nav.cash_benchmark.diff().fillna(0)).min()),
            max_excess_drawdown_dollars=float((excess-excess.cummax()).min()),peak_active_cohorts=int(nav.active_cohorts.max()),
            commission=record['commission'],mean_entered_size=float(d.loc[d.status.eq('entered'),'level'].mean()/4)))
        statuses.extend([dict(model=folder.name,status=status,count=int(count)) for status,count in d.status.value_counts().items()])
        for year,g in nav.groupby(nav.index.year):
            annual.append(dict(model=folder.name,year=int(year),annualized_excess_sharpe=base.sharpe(g.excess_return.iloc[1:] if year==2018 else g.excess_return),
                entries=int((d.status.eq('entered')&d.entry_date.dt.year.eq(year)).sum()),
                net_excess_pnl=float((nav.pnl-nav.cash_benchmark.diff().fillna(0)).loc[g.index].sum())))
    result=pd.DataFrame(rows).sort_values('annualized_excess_sharpe',ascending=False);assert len(result)==18
    result.to_csv(REPORT/'comparison.csv',index=False);pd.DataFrame(statuses).to_csv(REPORT/'decision_status.csv',index=False)
    yearly=pd.DataFrame(annual);yearly.to_csv(REPORT/'annual.csv',index=False)
    yearly.pivot(index='year',columns='model',values='annualized_excess_sharpe').to_csv(REPORT/'annual_sharpe.csv')
    controls=result.loc[result.kind.eq('fixed100')].set_index(['boundary','risk_mode']);comparisons=[]
    for row in result.loc[result.kind.eq('learned')].itertuples():
        control=controls.loc[(row.boundary,row.risk_mode)]
        comparisons.append(dict(boundary=row.boundary,risk_mode=row.risk_mode,learned_sharpe=row.annualized_excess_sharpe,
            fixed_sharpe=control.annualized_excess_sharpe,sharpe_difference=row.annualized_excess_sharpe-control.annualized_excess_sharpe,
            learned_entries=row.entries,fixed_entries=control.entries))
    pd.DataFrame(comparisons).to_csv(REPORT/'learning_comparison.csv',index=False)
    selected=result.loc[result.kind.eq('learned')].sort_values('selection_2018_2023_sharpe',ascending=False).iloc[0]
    base.write_json(REPORT/'selection.json',dict(selected_using='2018-2023 only',model=selected.model,
        full_sharpe=selected.annualized_excess_sharpe,recent_sharpe=selected.recent_2024_2025_sharpe,
        no_fresh_holdout=True,no_checkpoint_promotion=True))
    fits=pd.read_csv(REPORT/'fits.csv');assert len(fits)==270
    known=fits.last_training_outcome.notna();assert (pd.to_datetime(fits.loc[known,'last_training_outcome'])<pd.to_datetime(fits.loc[known,'fit'])).all()
    history=json.loads((OUT/'resource_history.json').read_text());peak=max([resource.getrusage(resource.RUSAGE_SELF).ru_maxrss]+[r['peak_rss_bytes'] for r in history])
    sources.check();sources.save()
    base.write_json(REPORT/'audit.json',dict(status='complete',replays=18,monthly_models=270,
        baseline90_models_and_two_replays_reproduced=True,no_future_outcome_filter_on_proposals=True,
        all_proposals_have_decisions=True,all_cash_ledgers_verified=True,all_inventory_closed=True,
        costs_and_capital_fixed=True,original_checkpoints_unchanged=True,no_fresh_holdout=True,peak_rss_bytes=peak))
    (REPORT/'findings.txt').write_text(result.to_string(index=False)+'\n\n'+pd.DataFrame(comparisons).to_string(index=False)+'\n\n'+PROTOCOL+'\n')
    print(result.to_string(index=False),flush=True);base.progress('complete')
    files=[p for root in [OUT,REPORT] for p in root.rglob('*') if p.is_file() and p.name!='output_hashes.json']
    base.write_json(REPORT/'output_hashes.json',{str(p):base.digest(p) for p in files})


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['test','prepare','train','replay','summarize','all'],default='all');args=p.parse_args()
    if args.phase=='test':
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests',pattern='test_p25_boundaries.py'))
        if not result.wasSuccessful():raise SystemExit(1)
        return
    setup();OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True);sources=base.Sources()
    sources.hashes[str(Path(__file__))]=base.digest(Path(__file__))
    for path in [Path(__file__),Path('scripts/p25_boundaries.py'),Path('tests/test_p25_boundaries.py'),
        Path('scripts/p25_action_windows.py'),Path('scripts/run_p25_action_windows.py'),Path('scripts/same_structure_smile.py'),
        Path('scripts/uniform_smile_engine.py'),Path('scripts/uniform_smile_risk.py'),Path('scripts/option_flex/contract_risk.py'),
        Path('reports/aligned_contract_learning/manifest.json'),Path('config/p25_recency_candidate.json')]:sources.track(path)
    cal=pd.read_parquet(sources.track(NATIVE/'market/rates.parquet')).index
    cfg=base.config()
    base.write_json(OUT/'inputs.json',dict(protocol=PROTOCOL,boundaries=policy.BOUNDARIES,risk_modes=policy.RISK_MODES,
        inherited_settings=asdict(cfg),risk_assumptions_source='reports/aligned_contract_learning/manifest.json',
        risk_assumptions_status='Implementation assumptions,not user-specified numeric limits',
        no_stress_disabled=['stress_loss'],no_caps_disabled=['gross_vega','gross_hedge_notional','capital_reserve','stress_loss']))
    if args.phase in ['prepare','all']:prepare(sources,cal)
    if args.phase in ['train','all']:train(sources,cal)
    if args.phase in ['replay','all']:replay(sources,cal)
    history_path=OUT/'resource_history.json';history=json.loads(history_path.read_text()) if history_path.exists() else []
    history.append(dict(phase=args.phase,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));base.write_json(history_path,history)
    if args.phase in ['summarize','all']:summarize(sources)
    sources.check();sources.save()


if __name__=='__main__':main()
