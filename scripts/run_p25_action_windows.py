"""Bounded action/window comparison on the frozen downside-put strategy."""
from dataclasses import replace
import argparse
import gc
import json
from pathlib import Path
import resource
import time
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from scripts import p25_action_windows as policy
from scripts import same_structure_smile as old
from scripts import uniform_smile_data as cached
from scripts import uniform_smile_engine as engine
from scripts.run_same_structure_smile import aggregate
from scripts.run_p25_signal_transfer import Market
from scripts.uniform_smile_risk import book_risk
from scripts.tail_put25_pooled import FEATURES,completed_mask
from scripts.curve_trading.config import config,RUNOFF
from scripts.run_aligned_contract_learning import mark_book
from scripts.run_atm65_ablation import write_json
from scripts.run_contract_walkforward import digest
from scripts.run_uniform_smile import verify,UNIFORM_CORPORATE_EVENTS

OUT=Path('artifacts/contracts/p25_action_windows')
REPORT=Path('reports/p25_action_windows')
SOURCE=Path('artifacts/contracts/same_structure_smile')
TRANSFER=Path('artifacts/contracts/p25_signal_transfer')
FROZEN=Path('artifacts/contracts/same_structure_calendar/replays')
CASES=[(5,'floor')]+[(years,menu) for years in policy.WINDOWS for menu in policy.MENUS if (years,menu)!=(5,'floor')]
PROTOCOL=(
    'Freeze the .8256229208692676 approximately30D25-delta put strategy: '
    'same608 pair/date/contract/direction proposals,original formation calendar, '
    'beta0.85-1.15/correlation>=0.5 and directional-profit screens,same27 '
    'features,monthly refits and8-leaf tree with same adaptive minleaf. '
    'Declared1/3/5-year completed-outcome windows x50/100,50/75/100, '
    '25/50/75/100,and0/25/50/75/100 action menus. Same direct-Sharpe CEM '
    '50x192 seed2026,full-size anchor,and full-size global fallback under16 '
    'training rows. Window cutoff uses signal date and outcomes must end '
    'strictly before fit. All5Y floor monthly trees/choices must reproduce '
    'frozen models;5Y floor NAV,trades,decisions must reproduce checkpoint. '
    'Learned zero actions are explicit decisions among all608 proposals, '
    'never silently removed. Saved0.65 training labels;actual0.03 commission, '
    'mid+.1volpoint,stockhedge1bp,financing/borrow/dividends,matched entry '
    'vega,daily deltahedges,own expiry,and all original risk budgets unchanged. '
    'Fixed100 control and one fixed100 same-learned-skip control for each '
    'flex window distinguish skip selection from sizing;entry risk eligibility '
    'can still differ by size. No new signals,pairs,wing,expiry,exit policy, '
    'inventory overlay,NN,or seed search. Predeclare selection on2018-2023 '
    'cash-excess Sharpe among the12 variants,then report2024-partial2025 '
    'and full history separately. All are previously inspected history, '
    'retrospective roster/calendar conditioning;no fresh holdout. '
    'One worker,50%of onecore,oneBLAS/Arrow thread,3GiB RSS stop.'
)


class Sources:
    def __init__(self):
        self.hashes=json.loads((OUT/'sources.json').read_text()) if (OUT/'sources.json').exists() else {}
        self.stats={}
    def track(self,path):
        path=Path(path);key=str(path)
        if path==Path(__file__):self.hashes[key]=digest(path)
        if key not in self.hashes:self.hashes[key]=digest(path)
        if key not in self.stats:self.stats[key]=(path.stat().st_size,path.stat().st_mtime_ns)
        return path
    def save(self):write_json(OUT/'sources.json',self.hashes)
    def check(self):
        for path,value in self.stats.items():
            p=Path(path);assert (p.stat().st_size,p.stat().st_mtime_ns)==value,path


def progress(stage,**details):
    write_json(OUT/'progress.json',dict(stage=stage,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,**details))


def name(years,menu):return f'{years}Y_{menu}'


def prepare(sources,cal):
    rows=pd.read_parquet(sources.track(SOURCE/'P25_rows.parquet'))
    assert (cal.get_indexer(pd.DatetimeIndex(rows.date))==rows.position.to_numpy()).all()
    proposals=pd.read_parquet(sources.track(TRANSFER/'P25_rows.parquet'))
    assert len(proposals)==608
    cols=['date','pair']+FEATURES
    joined=proposals.merge(rows[cols],on=['date','pair'],how='left',validate='one_to_one',indicator=True)
    assert joined._merge.eq('both').all()
    joined=joined.drop(columns='_merge')
    assert np.isfinite(joined[FEATURES]).all().all()
    joined.to_parquet(OUT/'signal_rows.parquet',index=False)
    pd.testing.assert_frame_equal(joined[proposals.columns],proposals)
    sources.save();progress('prepared',frozen_proposals=608)


def train(sources,cal):
    meta=pd.read_parquet(sources.track(SOURCE/'P25_rows.parquet'));logs=[]
    for month in pd.period_range('2018-01-01',cal[-1]-pd.Timedelta(days=RUNOFF),freq='M'):
        cutoff=month.start_time;stamp=str(cutoff.date())
        for years in policy.WINDOWS:
            rows=meta.loc[meta.signal_eligible&meta.finite_label&completed_mask(meta.date,meta.end,cutoff,years=years)].reset_index(drop=True)
            assert not len(rows) or (rows.end.max()<cutoff and rows.date.min()>=cutoff-pd.DateOffset(years=years))
            tree=None;leaves=np.array([0]);book=None
            pending=[menu for menu in policy.MENUS if not (OUT/'models'/name(years,menu)/f'{stamp}.json').exists()]
            if pending and len(rows)>=16:
                tree=DecisionTreeRegressor(max_leaf_nodes=8,min_samples_leaf=max(8,min(63,len(rows)//8)),random_state=2026)
                tree.fit(rows[FEATURES],rows.target)
                leaves=np.unique(tree.apply(rows[FEATURES]));states=np.searchsorted(leaves,tree.apply(rows[FEATURES]))
                first=int(rows.position.min()+1);last=int(cal.get_indexer(pd.DatetimeIndex(rows.end)).max())
                book=np.zeros((last-first+1,len(leaves)*5))
                for pair,idx in rows.groupby('pair',sort=False).groups.items():
                    idx=np.asarray(idx)
                    disk=np.load(sources.track(cached.OUT/'pairs'/pair/'P25_labels.npy'),mmap_mode='r')
                    labels=old.directional_paths(disk[rows.loc[idx,'label_row'].to_numpy(int)],rows.loc[idx,'direction'].to_numpy(int))
                    aggregate(book,states[idx],labels,rows.loc[idx,'position'].to_numpy(int),first)
                    del disk,labels
            for menu,levels in policy.MENUS.items():
                case=name(years,menu);folder=OUT/'models'/case;folder.mkdir(parents=True,exist_ok=True)
                path=folder/f'{stamp}.json'
                if path.exists():logs.append(json.loads(path.read_text()));continue
                choices,log=(np.array([4]),{}) if book is None else policy.fit_menu(book,cal[first:last+1],levels)
                assert np.isin(choices,levels).all()
                model=dict(tree=tree,leaves=leaves,choices=choices,fit=stamp)
                if years==5 and menu=='floor':
                    previous=joblib.load(sources.track(SOURCE/'models/P25/floor'/f'{stamp}.joblib'))
                    np.testing.assert_array_equal(choices,previous['choices']);np.testing.assert_array_equal(leaves,previous['leaves'])
                    if tree is not None:
                        for field in ['feature','threshold','children_left','children_right','value']:
                            np.testing.assert_array_equal(getattr(tree.tree_,field),getattr(previous['tree'].tree_,field))
                record=dict(model=case,years=years,menu=menu,fit=stamp,training_rows=len(rows),
                    training_pairs=int(rows.pair.nunique()),first_training_signal=str(rows.date.min().date()) if len(rows) else None,
                    last_training_outcome=str(rows.end.max().date()) if len(rows) else None,
                    leaves=len(leaves),choices=choices.tolist(),status='enabled' if tree is not None else 'original_fullsize_fallback',**log)
                joblib.dump(model,folder/f'{stamp}.joblib');write_json(path,record);logs.append(record)
            del rows,tree,book;gc.collect()
        if month.month in [1,4,7,10]:print(f'Completed action/window fits {stamp}: {len(logs)} models',flush=True)
        sources.save();progress('train',fit=stamp,models_complete=len(logs))
    assert len(logs)==1080
    pd.DataFrame(logs).to_csv(REPORT/'fits.csv',index=False)


def allocations(case):
    rows=pd.read_parquet(OUT/'signal_rows.parquet');rows['learned_level']=4
    for month,idx in rows.groupby(rows.date.dt.to_period('M')).groups.items():
        model=joblib.load(OUT/'models'/case/f'{month.start_time.date()}.joblib')
        assert pd.Timestamp(model['fit'])<=rows.loc[idx,'date'].min()
        rows.loc[idx,'learned_level']=policy.score_model(model,rows.loc[idx],FEATURES)
    return rows


def replay(sources,cal):
    cached.OUT=TRANSFER;market=Market();calendar=cal[cal>='2018-01-01']
    cfg=replace(config(),option_commission=.03,capital=1_000_000*345,max_gross_vega=20_000*345)
    arms=[(name(years,menu),'model') for years,menu in CASES]+[('fixed_100','fixed')]
    arms += [(name(years,'flex')+'_same_skip_full','same_skip') for years in policy.WINDOWS]
    for case,kind in arms:
        folder=OUT/'replays'/case;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'status.json').exists():continue
        if kind=='fixed':rows=pd.read_parquet(OUT/'signal_rows.parquet').assign(learned_level=4)
        else:
            parent=case.replace('_same_skip_full','');rows=allocations(parent)
            if kind=='same_skip':rows['learned_level']=np.where(rows.learned_level.eq(0),0,4)
        assert len(rows)==608 and not rows.duplicated(['date','pair','wing']).any()
        rows.to_parquet(folder/'scored_allocations.parquet',index=False)
        market.variant=case;market.started=time.monotonic();market.reported_year=None
        print(f'Replay {case}:608 proposals,{int(rows.learned_level.eq(0).sum())} explicit learned skips',flush=True)
        saved=engine.book_risk
        try:
            engine.book_risk=book_risk
            nav,trades,decisions,ledger=engine.simulate(market,rows,calendar,cfg,1_000_000,
                policy=lambda row,context:int(row['learned_level']),corporate_events=UNIFORM_CORPORATE_EVENTS,wing_capital=1_000_000)
        finally:engine.book_risk=saved
        mark_book(nav,market.rates,cfg.capital);audit=verify(nav,trades,decisions,rows,cfg)
        assert len(decisions)==608
        assert int(decisions.status.eq('flat').sum())==int(rows.learned_level.eq(0).sum())
        if case in ['5Y_floor','fixed_100']:
            reference=FROZEN/('floor_P25' if case=='5Y_floor' else 'fixed_100_P25')
            pd.testing.assert_frame_equal(nav,pd.read_parquet(sources.track(reference/'daily_nav.parquet')),rtol=1e-12,atol=1e-6)
            old_trades=pd.read_csv(sources.track(reference/'trades.csv'),parse_dates=['date'])
            pd.testing.assert_frame_equal(trades.reset_index(drop=True),old_trades.reset_index(drop=True),check_dtype=False,rtol=1e-12,atol=1e-7)
            if case=='5Y_floor':
                previous=pd.read_csv(sources.track(reference/'decisions.csv'),parse_dates=['signal_date','entry_date'])
                pd.testing.assert_frame_equal(decisions[previous.columns],previous,check_dtype=False,rtol=1e-12,atol=1e-7)
            print(f'Frozen reference reproduced for {case}',flush=True)
        nav.to_parquet(folder/'daily_nav.parquet');trades.to_csv(folder/'trades.csv',index=False)
        decisions.to_csv(folder/'decisions.csv',index=False)
        write_json(folder/'status.json',dict(status='complete',model=case,kind=kind,proposals=608,
            learned_skips=int(rows.learned_level.eq(0).sum()),net_excess_pnl=float(nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1]),
            elapsed_seconds=time.monotonic()-market.started,**audit,**ledger))
        sources.save();progress('replay',model=case,**audit)
        print(f'{case}:Sharpe {audit["annualized_excess_sharpe"]:.6f},{audit["entries"]} entries',flush=True)
        del nav,trades,decisions,rows;gc.collect()


def sharpe(r):return float(np.sqrt(252)*r.mean()/r.std(ddof=1)) if r.std(ddof=1)>1e-12 else 0.


def summarize(sources):
    comparison=[];annual=[];periods=[];books={}
    for folder in sorted((OUT/'replays').iterdir()):
        record=json.loads((folder/'status.json').read_text());assert record['status']=='complete'
        nav=pd.read_parquet(folder/'daily_nav.parquet');books[folder.name]=nav
        decisions=pd.read_csv(folder/'decisions.csv',parse_dates=['signal_date','entry_date'])
        if 'same_skip' in folder.name:
            parent=folder.name.replace('_same_skip_full','')
            other=pd.read_csv(OUT/'replays'/parent/'decisions.csv')
            np.testing.assert_array_equal(decisions.level.eq(0),other.level.eq(0))
        comparison.append(dict(model=folder.name,kind=record['kind'],annualized_excess_sharpe=record['annualized_excess_sharpe'],
            selection_2018_2023_sharpe=sharpe(nav.loc[:'2023-12-31','excess_return'].iloc[1:]),
            recent_2024_2025_sharpe=sharpe(nav.loc['2024-01-01':,'excess_return']),
            entries=record['entries'],active_pairs=record['active_pairs'],learned_skips=record['learned_skips'],
            mean_entered_size=float(decisions.loc[decisions.status.eq('entered'),'level'].mean()/4),
            net_excess_pnl=record['net_excess_pnl'],worst_daily_excess_pnl=float((nav.pnl-nav.cash_benchmark.diff().fillna(0)).min())))
        for year,g in nav.groupby(nav.index.year):
            annual.append(dict(model=folder.name,year=int(year),annualized_excess_sharpe=sharpe(g.excess_return.iloc[1:] if year==2018 else g.excess_return),
                entries=int((decisions.status.eq('entered')&decisions.entry_date.dt.year.eq(year)).sum()),
                net_excess_pnl=float((nav.pnl-nav.cash_benchmark.diff().fillna(0)).loc[g.index].sum())))
        for label,start,end in [('2018to2020','2018-01-01','2020-12-31'),('2021to2023','2021-01-01','2023-12-31'),('2024to2025','2024-01-01','2025-08-29')]:
            r=nav.loc[start:end,'excess_return'];r=r.iloc[1:] if start=='2018-01-01' else r
            periods.append(dict(model=folder.name,period=label,annualized_excess_sharpe=sharpe(r)))
    result=pd.DataFrame(comparison).sort_values('annualized_excess_sharpe',ascending=False)
    result.to_csv(REPORT/'comparison.csv',index=False)
    pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False)
    pd.DataFrame(annual).pivot(index='year',columns='model',values='annualized_excess_sharpe').to_csv(REPORT/'annual_sharpe.csv')
    pd.DataFrame(periods).to_csv(REPORT/'periods.csv',index=False)
    variants=result.loc[result.kind.eq('model')]
    selected=variants.sort_values('selection_2018_2023_sharpe',ascending=False).iloc[0]
    best_full=variants.iloc[0]
    write_json(REPORT/'selection.json',dict(predeclared_selection_period='2018-2023',selected_model=selected.model,
        selected_early_sharpe=selected.selection_2018_2023_sharpe,selected_recent_sharpe=selected.recent_2024_2025_sharpe,
        selected_full_sharpe=selected.annualized_excess_sharpe,descriptive_full_history_best=best_full.model,
        descriptive_full_best_sharpe=best_full.annualized_excess_sharpe,no_checkpoint_promotion=True,no_fresh_holdout=True))
    uplift=[];baseline=books['5Y_floor'].excess_return.iloc[1:].to_numpy()
    for case in variants.model:
        if case=='5Y_floor':continue
        a=books[case].excess_return.iloc[1:].to_numpy();assert books[case].index.equals(books['5Y_floor'].index)
        rng=np.random.default_rng(2026);n=len(a);block=126;draws=[]
        for _ in range(1000):
            starts=rng.integers(0,n-block+1,size=int(np.ceil(n/block)))
            idx=(starts[:,None]+np.arange(block)).ravel()[:n]
            draws.append(sharpe(a[idx])-sharpe(baseline[idx]))
        uplift.append(dict(model=case,sharpe_difference=sharpe(a)-sharpe(baseline),
            ci_lower=float(np.quantile(draws,.025)),ci_upper=float(np.quantile(draws,.975))))
    pd.DataFrame(uplift).to_csv(REPORT/'uplift.csv',index=False)
    fits=pd.read_csv(REPORT/'fits.csv');assert len(fits)==1080
    known=fits.last_training_outcome.notna()
    assert (pd.to_datetime(fits.loc[known,'last_training_outcome'])<pd.to_datetime(fits.loc[known,'fit'])).all()
    resources=json.loads((OUT/'resource_history.json').read_text()) if (OUT/'resource_history.json').exists() else []
    peak=max([resource.getrusage(resource.RUSAGE_SELF).ru_maxrss]+[r['peak_rss_bytes'] for r in resources])
    sources.check();sources.save()
    write_json(REPORT/'audit.json',dict(status='complete',declared_variants=12,replays=len(books),monthly_models=1080,
        original90_floor_models_reproduced=True,original_NAV_trades_decisions_reproduced=True,
        fixed100_control_reproduced=True,all608_signal_decisions_preserved=True,
        learned_skips_explicit=True,all_training_outcomes_before_fit=True,
        no_checkpoint_promotion=True,no_fresh_holdout=True,peak_rss_bytes=peak))
    (REPORT/'findings.txt').write_text(result.to_string(index=False)+'\n\n'+pd.DataFrame(periods).to_string(index=False)+'\n\n'+
        json.dumps(json.loads((REPORT/'selection.json').read_text()),indent=2)+'\n\n'+PROTOCOL+'\n')
    outputs=[p for root in [OUT,REPORT] for p in root.rglob('*') if p.is_file() and p.name!='output_hashes.json']
    write_json(REPORT/'output_hashes.json',{str(p):digest(p) for p in outputs})
    print(result.to_string(index=False),flush=True);progress('complete')


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['test','prepare','train','replay','summarize','all'],default='all');args=p.parse_args()
    if args.phase=='test':
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests',pattern='test_p25_action_windows.py'))
        if not result.wasSuccessful():raise SystemExit(1)
        return
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True);sources=Sources()
    for path in [Path(__file__),Path('scripts/p25_action_windows.py'),Path('tests/test_p25_action_windows.py'),
                 Path('scripts/uniform_smile_engine.py'),Path('scripts/run_tail_put25_residual_rl.py')]:sources.track(path)
    for path in (TRANSFER/'market').glob('*.parquet'):sources.track(path)
    cal=pd.read_parquet(sources.track(cached.OUT/'market/rates.parquet')).index
    write_json(OUT/'inputs.json',dict(protocol=PROTOCOL,windows=policy.WINDOWS,
        menus={k:v.tolist() for k,v in policy.MENUS.items()},seed=2026,features=FEATURES,
        evaluation_end='2025-08-29',signal_cutoff=str((cal[-1]-pd.Timedelta(days=RUNOFF)).date())))
    if args.phase in ['prepare','all']:prepare(sources,cal)
    if args.phase in ['train','all']:train(sources,cal)
    if args.phase in ['replay','all']:replay(sources,cal)
    history_path=OUT/'resource_history.json'
    history=json.loads(history_path.read_text()) if history_path.exists() else []
    history.append(dict(phase=args.phase,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    write_json(history_path,history)
    if args.phase in ['summarize','all']:summarize(sources)
    sources.check();sources.save()


if __name__=='__main__':main()
