"""Declared action/cap calibration with every constant-size control."""
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import resource
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from scripts import run_p25_action_windows as base
from scripts.p25_action_windows import fit_menu,score_model
from scripts.p25_recency import completed_months

OUT=Path('artifacts/contracts/p25_sharpe_search')
REPORT=Path('reports/p25_sharpe_search')
NATIVE=Path('artifacts/contracts/uniform_smile')
REFERENCE=Path('artifacts/contracts/p25_recency')
MENUS={'floor':[2,3,4],'binary':[2,4],'quarter':[1,2,3,4],'high_floor':[3,4]}
CAPS=[.025,.04,.05,.075,.10]
PROTOCOL=(
    'Declared four18-month monthly learned menus50/75/100,50/100,25/50/75/100, '
    '75/100 and fixed25/50/75/100 controls x fiveexplicitstresscaps2.5/4/5/7.5/10% '
    'of fundedNAV,40replays. Same608 frozenP25 proposals,direction/contracts, '
    '27features,profit/beta/correlation/formation screens,8leaf adaptive '
    'minleaf tree,CEM50x192 seed2026/fullsizeanchor/fallback. .65training '
    'labels,.03actualcommission,.1volpoint perleg,1bp stock,funding/dividend/ '
    'borrow/daily deltahedge/ownexpiry/riskreserve80%/hedge2x/grossvega unchanged. '
    'Capital$345m,pair/wing$1m and450fullvega perpoint perleg fixed;caps explicit '
    'research parameters,not user numerical requirements. No newdata/seed/ '
    'signal/exit change. Floor5% andfixed1005% reproduce original checkpoint. '
    'Rank all arms by2018-2023 first;report recent/annual/full separately. '
    'Best constant-size control at eachcap is selected on2018-2023,not full '
    'period. No learned skipping. Previously inspected history,no freshholdout '
    'or automaticcheckpoint promotion. One worker50%of onecore,3GiB stop.'
)


def setup():base.OUT=OUT;base.REPORT=REPORT


def train(sources,cal):
    meta=pd.read_parquet(sources.track(base.SOURCE/'P25_rows.parquet'));logs=[]
    for month in pd.period_range('2018-01-01',cal[-1]-pd.Timedelta(days=base.RUNOFF),freq='M'):
        cutoff=month.start_time;stamp=str(cutoff.date())
        rows=meta.loc[meta.signal_eligible&meta.finite_label&completed_months(meta.date,meta.end,cutoff,18)].reset_index(drop=True)
        pending=[menu for menu in MENUS if not (OUT/'models'/menu/f'{stamp}.json').exists()]
        tree=None;leaves=np.array([0]);book=None
        if pending and len(rows)>=16:
            tree=DecisionTreeRegressor(max_leaf_nodes=8,min_samples_leaf=max(8,min(63,len(rows)//8)),random_state=2026).fit(rows[base.FEATURES],rows.target)
            leaves=np.unique(tree.apply(rows[base.FEATURES]));states=np.searchsorted(leaves,tree.apply(rows[base.FEATURES]))
            first=int(rows.position.min()+1);last=int(cal.get_indexer(pd.DatetimeIndex(rows.end)).max());book=np.zeros((last-first+1,len(leaves)*5))
            for pair,idx in rows.groupby('pair',sort=False).groups.items():
                idx=np.asarray(idx);disk=np.load(sources.track(NATIVE/'pairs'/pair/'P25_labels.npy'),mmap_mode='r')
                labels=base.old.directional_paths(disk[rows.loc[idx,'label_row'].to_numpy(int)],rows.loc[idx,'direction'].to_numpy(int))
                base.aggregate(book,states[idx],labels,rows.loc[idx,'position'].to_numpy(int),first);del disk,labels
        for menu,levels in MENUS.items():
            folder=OUT/'models'/menu;folder.mkdir(parents=True,exist_ok=True);path=folder/f'{stamp}.json'
            if path.exists():logs.append(json.loads(path.read_text()));continue
            choices,details=(np.array([4]),{}) if book is None else fit_menu(book,cal[first:last+1],levels)
            model=dict(tree=tree,leaves=leaves,choices=choices,fit=stamp)
            if menu=='floor':
                previous=joblib.load(sources.track(REFERENCE/'models/18M_floor'/f'{stamp}.joblib'))
                np.testing.assert_array_equal(choices,previous['choices']);np.testing.assert_array_equal(leaves,previous['leaves'])
                if tree is not None:
                    for field in ['feature','threshold','children_left','children_right','value']:
                        np.testing.assert_array_equal(getattr(tree.tree_,field),getattr(previous['tree'].tree_,field))
            record=dict(menu=menu,fit=stamp,training_rows=len(rows),last_training_outcome=str(rows.end.max().date()) if len(rows) else None,
                choices=choices.tolist(),**details)
            joblib.dump(model,folder/f'{stamp}.joblib');base.write_json(path,record);logs.append(record)
        del rows,tree,book;gc.collect()
        if month.month in [1,7]:print(f'Sizing/cap fits{stamp}:models{len(logs)}',flush=True)
        sources.save();base.progress('train',fit=stamp,models_complete=len(logs))
    assert len(logs)==360;pd.DataFrame(logs).to_csv(REPORT/'fits.csv',index=False)


def allocations(menu):
    rows=pd.read_parquet(OUT/'signal_rows.parquet');rows['learned_level']=4
    if menu.startswith('fixed'):rows['learned_level']=int(menu[-1])
    else:
        for month,idx in rows.groupby(rows.date.dt.to_period('M')).groups.items():
            model=joblib.load(OUT/'models'/menu/f'{month.start_time.date()}.joblib')
            rows.loc[idx,'learned_level']=score_model(model,rows.loc[idx],base.FEATURES)
    assert len(rows)==608 and rows.learned_level.between(1,4).all()
    return rows


def name(cap,menu):return f'cap{int(round(cap*1000)):03d}_{menu}'


def replay(sources,cal):
    base.cached.OUT=base.TRANSFER;market=base.Market();calendar=cal[cal>='2018-01-01']
    for cap in [.05]+[c for c in CAPS if c!=.05]:
        cfg=replace(base.config(),option_commission=.03,capital=345_000_000,max_gross_vega=20_000*345,max_stress_loss_fraction=cap)
        for menu in list(MENUS)+[f'fixed{i}' for i in [1,2,3,4]]:
            case=name(cap,menu);folder=OUT/'replays'/case;folder.mkdir(parents=True,exist_ok=True)
            if (folder/'status.json').exists():continue
            rows=allocations(menu);rows.to_parquet(folder/'scored_allocations.parquet',index=False)
            market.variant=case;market.started=time.monotonic();market.reported_year=None
            print(f'Replay{case}',flush=True);saved=base.engine.book_risk
            try:
                base.engine.book_risk=base.book_risk
                nav,trades,decisions,ledger=base.engine.simulate(market,rows,calendar,cfg,1_000_000,
                    policy=lambda row,context:int(row['learned_level']),corporate_events=base.UNIFORM_CORPORATE_EVENTS,wing_capital=1_000_000)
            finally:base.engine.book_risk=saved
            base.mark_book(nav,market.rates,cfg.capital);audit=base.verify(nav,trades,decisions,rows,cfg)
            if cap==.05 and menu in ['floor','fixed4']:
                reference=REFERENCE/'replays'/('18M_floor' if menu=='floor' else 'fixed_100')
                pd.testing.assert_frame_equal(nav,pd.read_parquet(sources.track(reference/'daily_nav.parquet')),rtol=1e-12,atol=1e-6)
                for frame,file,dates in [(trades,'trades.csv',['date']),(decisions,'decisions.csv',['signal_date','entry_date'])]:
                    pd.testing.assert_frame_equal(frame.reset_index(drop=True),pd.read_csv(sources.track(reference/file),parse_dates=dates),check_dtype=False,rtol=1e-12,atol=1e-7)
            nav.to_parquet(folder/'daily_nav.parquet');trades.to_csv(folder/'trades.csv',index=False);decisions.to_csv(folder/'decisions.csv',index=False)
            base.write_json(folder/'status.json',dict(status='complete',model=case,cap=cap,menu=menu,kind='fixed' if menu.startswith('fixed') else 'learned',
                net_excess_pnl=float(nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1]),elapsed_seconds=time.monotonic()-market.started,**audit,**ledger))
            sources.save();base.progress('replay',model=case,**audit)
            print(f'{case}:Sharpe{audit["annualized_excess_sharpe"]:.6f},entries{audit["entries"]}',flush=True)
            del nav,trades,decisions,rows;gc.collect()


def summarize(sources):
    records=[];annual=[];books={}
    for folder in sorted((OUT/'replays').iterdir()):
        record=json.loads((folder/'status.json').read_text());assert record['status']=='complete'
        nav=pd.read_parquet(folder/'daily_nav.parquet');books[folder.name]=nav
        d=pd.read_csv(folder/'decisions.csv',parse_dates=['entry_date']);assert len(d)==608
        excess=nav.nav-nav.cash_benchmark
        records.append(dict(model=folder.name,cap=record['cap'],menu=record['menu'],kind=record['kind'],
            annualized_excess_sharpe=record['annualized_excess_sharpe'],entries=record['entries'],active_pairs=record['active_pairs'],net_excess_pnl=record['net_excess_pnl'],
            selection_2018_2023_sharpe=base.sharpe(nav.loc[:'2023-12-31','excess_return'].iloc[1:]),
            recent_2024_2025_sharpe=base.sharpe(nav.loc['2024-01-01':,'excess_return']),
            worst_daily_excess_pnl=float((nav.pnl-nav.cash_benchmark.diff().fillna(0)).min()),
            max_excess_drawdown_dollars=float((excess-excess.cummax()).min())))
        for year,g in nav.groupby(nav.index.year):
            annual.append(dict(model=folder.name,year=int(year),annualized_excess_sharpe=base.sharpe(g.excess_return.iloc[1:] if year==2018 else g.excess_return),
                entries=int((d.status.eq('entered')&d.entry_date.dt.year.eq(year)).sum()),net_excess_pnl=float((nav.pnl-nav.cash_benchmark.diff().fillna(0)).loc[g.index].sum())))
    result=pd.DataFrame(records).sort_values('annualized_excess_sharpe',ascending=False);assert len(result)==40
    result.to_csv(REPORT/'comparison.csv',index=False);pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False)
    pd.DataFrame(annual).pivot(index='year',columns='model',values='annualized_excess_sharpe').to_csv(REPORT/'annual_sharpe.csv')
    selection=result.sort_values('selection_2018_2023_sharpe',ascending=False).iloc[0]
    comparisons=[]
    for cap in CAPS:
        control=result.loc[result.cap.eq(cap)&result.kind.eq('fixed')].sort_values('selection_2018_2023_sharpe',ascending=False).iloc[0]
        for row in result.loc[result.cap.eq(cap)&result.kind.eq('learned')].itertuples():
            comparisons.append(dict(model=row.model,cap=cap,control=control.model,learned_sharpe=row.annualized_excess_sharpe,
                control_sharpe=control.annualized_excess_sharpe,sharpe_difference=row.annualized_excess_sharpe-control.annualized_excess_sharpe))
    pd.DataFrame(comparisons).to_csv(REPORT/'learning_comparison.csv',index=False)
    base.write_json(REPORT/'selection.json',dict(selected_using='2018-2023 only',model=selection.model,full_sharpe=selection.annualized_excess_sharpe,
        descriptive_best_full=result.iloc[0].model,no_fresh_holdout=True,no_checkpoint_promotion=True))
    fits=pd.read_csv(REPORT/'fits.csv');known=fits.last_training_outcome.notna();assert len(fits)==360
    assert (pd.to_datetime(fits.loc[known,'last_training_outcome'])<pd.to_datetime(fits.loc[known,'fit'])).all()
    history=json.loads((OUT/'resource_history.json').read_text());peak=max([resource.getrusage(resource.RUSAGE_SELF).ru_maxrss]+[r['peak_rss_bytes'] for r in history])
    sources.check();sources.save();base.write_json(REPORT/'audit.json',dict(status='complete',monthly_models=360,replays=40,
        baseline90_models_and_two_replays_exact=True,all608_decisions_preserved=True,completed_outcomes_before_fit=True,
        all_ledgers_and_inventory_verified=True,no_fresh_holdout=True,original_checkpoint_preserved=True,peak_rss_bytes=peak))
    (REPORT/'findings.txt').write_text(result.to_string(index=False)+'\n\n'+pd.DataFrame(comparisons).to_string(index=False)+'\n\n'+PROTOCOL+'\n')
    print(result.to_string(index=False),flush=True);base.progress('complete')
    paths=[p for root in [OUT,REPORT] for p in root.rglob('*') if p.is_file() and p.name!='output_hashes.json']
    base.write_json(REPORT/'output_hashes.json',{str(p):base.digest(p) for p in paths})


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prepare','train','replay','summarize','all'],default='all');args=p.parse_args()
    setup();OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True);sources=base.Sources()
    sources.hashes[str(Path(__file__))]=base.digest(Path(__file__))
    for path in [Path(__file__),Path('scripts/p25_action_windows.py'),Path('scripts/p25_recency.py'),Path('scripts/run_p25_action_windows.py'),
        Path('scripts/uniform_smile_engine.py'),Path('scripts/uniform_smile_risk.py'),Path('scripts/option_flex/contract_risk.py'),Path('config/p25_recency_candidate.json')]:sources.track(path)
    for path in (base.TRANSFER/'market').glob('*.parquet'):sources.track(path)
    cal=pd.read_parquet(sources.track(NATIVE/'market/rates.parquet')).index
    base.write_json(OUT/'inputs.json',dict(protocol=PROTOCOL,caps=CAPS,menus=MENUS,features=base.FEATURES))
    if args.phase in ['prepare','all']:base.prepare(sources,cal)
    if args.phase in ['train','all']:train(sources,cal)
    if args.phase in ['replay','all']:replay(sources,cal)
    history_path=OUT/'resource_history.json';history=json.loads(history_path.read_text()) if history_path.exists() else []
    history.append(dict(phase=args.phase,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));base.write_json(history_path,history)
    if args.phase in ['summarize','all']:summarize(sources)
    sources.check();sources.save()


if __name__=='__main__':main()
