"""Reward-only cost-hurdle sensitivity for the18-month P25 checkpoint."""
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import resource
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from scripts import run_p25_action_windows as base
from scripts.p25_action_windows import fit_menu,score_model
from scripts.p25_recency import completed_months
from scripts import p25_cost_hurdles as policy
from scripts.curve_trading.unit_paths import UnitPaths

OUT=Path('artifacts/contracts/p25_cost_hurdles')
REPORT=Path('reports/p25_cost_hurdles')
PREVIOUS=Path('artifacts/contracts/p25_recency')
NATIVE=Path('artifacts/contracts/uniform_smile')
PROTOCOL=(
    'Four training reward commissions:.65/.30/.15/.03,18-month completed-outcome '
    'windows,monthly refits,50/75/100 sizing,same8leaf/minleaf tree andCEM '
    '50x192seed2026/fullsize anchor/fallback. Rebuild standalone option cash '
    'flows and financing at each training commission;identical entry quantities, '
    'held paths and expiry. This is a reward-only ablation: all27 signal '
    'features,including original.65 commission feature and completed-shadow '
    'profitability memory,and original eligibility/formation calendars remain '
    'frozen. All608 signal proposals and direction/contracts unchanged;no '
    'learned skips. Training-target tree and Sharpe sizing objective are both '
    'rebuilt at the lower commission. Actual execution remains.03 commission, '
    'mid+.1volpoint perleg,1bp stock;financing/dividends/borrow,daily delta '
    'hedges,own expiry,entry vega matching and all risk limits unchanged. '
    'Lower rewards can alter sizing and risk-limit admission but cannot add '
    'signal proposals in this experiment. All required.65 label paths must '
    'reproduce original cached labels exactly;all90 baseline models and '
    'NAV/trades/decisions must reproduce18-month.9617449 candidate. '
    'Fixed100 control reproduced. Selection on2018-2023;report annuals, '
    '2024-partial2025 and full history. Previously inspected history and '
    'retrospective roster/calendar conditioning;no fresh holdout. One worker, '
    '50%of onecore,oneBLAS/Arrow thread,3GiB RSS stop. No automatic promotion.'
)


def configure():
    base.OUT=OUT;base.REPORT=REPORT
    base.CASES=[(policy.fee_name(fee),'') for fee in policy.FEES]
    base.name=lambda case,menu:case;base.policy.WINDOWS=[];base.allocations=allocations


def prepare(sources,cal):
    base.prepare(sources,cal)
    meta=pd.read_parquet(sources.track(base.SOURCE/'P25_rows.parquet'))
    last_cutoff=pd.period_range('2018-01-01',cal[-1]-pd.Timedelta(days=base.RUNOFF),freq='M')[-1].start_time
    needed=meta.loc[meta.signal_eligible&meta.finite_label&meta.date.ge(pd.Timestamp('2018-01-01')-pd.DateOffset(months=18))&meta.end.lt(last_cutoff)].copy()
    assert not needed.duplicated(['pair','date']).any()
    needed['source_label_row']=needed.label_row
    unit_folder=OUT/'units';unit_folder.mkdir(exist_ok=True)
    sids=sorted(set(needed.secid_a)|set(needed.secid_b));cfg=replace(base.config(),option_commission=.65)
    for n,sid in enumerate(sids,1):
        dest=unit_folder/f'{sid}.joblib'
        if dest.exists():continue
        dates=pd.DatetimeIndex(needed.loc[needed.secid_a.eq(sid)|needed.secid_b.eq(sid),'date'].unique())
        folder=NATIVE/'market/assets'
        sel=pd.read_parquet(sources.track(folder/f'{sid}_selection.parquet'))
        sel=sel.loc[sel.tenor.eq(30)&sel.cp_flag.eq('P')&sel.bucket.eq('25')&sel.date.isin(dates)]
        assert len(sel)>0
        q=pd.read_parquet(sources.track(folder/f'{sid}_quotes.parquet'),filters=[('optionid','in',sel.optionid.unique().tolist())])
        s=pd.read_parquet(sources.track(folder/f'{sid}_stocks.parquet'))
        divs=pd.read_parquet(sources.track(folder/f'{sid}_dividends.parquet'))
        unit=UnitPaths(q,s,sel,cal,cal,divs,cfg);joblib.dump(unit,dest)
        del unit,q,s,sel,divs;gc.collect()
        if n%20==0:print(f'Prepared bounded native units {n}/{len(sids)}',flush=True)
        sources.save();base.progress('units',assets_complete=n,assets=len(sids))
    prepared=[];checks=[]
    for n,(pair,rows) in enumerate(needed.groupby('pair',sort=False),1):
        folder=OUT/'pairs'/pair;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'prepared.json').exists():
            prepared.append(pd.read_parquet(folder/'rows.parquet'));checks.append(json.loads((folder/'prepared.json').read_text()));continue
        rows=rows.copy();a,b=[joblib.load(unit_folder/f'{int(rows[f"secid_{side}"].iloc[0])}.joblib') for side in ['a','b']]
        original=np.load(sources.track(NATIVE/'pairs'/pair/'P25_labels.npy'),mmap_mode='r')
        expected=original[rows.source_label_row.to_numpy(int)]
        for fee in policy.FEES:
            daily,ends,vegas,status=policy.reprice_labels(a,b,fee)
            positions=rows.position.to_numpy(int)
            labels=np.zeros((len(rows),9,daily.shape[-1]),np.float32);labels[:,1:]=daily[positions]
            np.testing.assert_array_equal(cal[ends[positions]],pd.DatetimeIndex(rows.end))
            assert (status[positions]=='complete').all() and np.isfinite(labels).all()
            if fee==.65:np.testing.assert_array_equal(labels,expected)
            np.save(folder/f'{policy.fee_name(fee)}_labels.npy',labels)
            del daily,labels
        rows['label_row']=np.arange(len(rows));rows.to_parquet(folder/'rows.parquet',index=False);prepared.append(rows)
        record=dict(pair=pair,rows=len(rows),original_labels_exact=True,fees=policy.FEES)
        base.write_json(folder/'prepared.json',record);checks.append(record)
        del a,b,original,expected,rows;gc.collect()
        if n%10==0:print(f'Prepared cost labels {n}/{needed.pair.nunique()} pairs',flush=True)
        sources.save();base.progress('labels',pairs_complete=n)
    # Preserve original source order for deterministic baseline tree fitting.
    joined=pd.concat(prepared).sort_index()
    pd.testing.assert_frame_equal(joined[needed.columns.difference(['label_row'])],needed[needed.columns.difference(['label_row'])])
    joined.to_parquet(OUT/'training_rows.parquet',index=False)
    pd.DataFrame(checks).to_csv(REPORT/'label_audit.csv',index=False)


def train(sources,cal):
    meta=pd.read_parquet(sources.track(OUT/'training_rows.parquet'));logs=[]
    for month in pd.period_range('2018-01-01',cal[-1]-pd.Timedelta(days=base.RUNOFF),freq='M'):
        cutoff=month.start_time;stamp=str(cutoff.date())
        original=pd.read_parquet(base.SOURCE/'P25_rows.parquet',columns=['date','end','pair','signal_eligible','finite_label'])
        expected=original.loc[original.signal_eligible&original.finite_label&completed_months(original.date,original.end,cutoff,18)]
        selection=meta.loc[completed_months(meta.date,meta.end,cutoff,18)].reset_index(drop=True)
        np.testing.assert_array_equal(selection[['date','pair']].to_numpy(),expected[['date','pair']].to_numpy())
        for fee in policy.FEES:
            case=policy.fee_name(fee);folder=OUT/'models'/case;folder.mkdir(parents=True,exist_ok=True)
            path=folder/f'{stamp}.json'
            if path.exists():logs.append(json.loads(path.read_text()));continue
            rows=selection.copy();tree=None;leaves=np.array([0]);book=None;labels_by_pair={}
            if len(rows)>=16:
                target=np.zeros(len(rows),np.float32)
                for pair,idx in rows.groupby('pair',sort=False).groups.items():
                    idx=np.asarray(idx);disk=np.load(sources.track(OUT/'pairs'/pair/f'{case}_labels.npy'),mmap_mode='r')
                    raw=disk[rows.loc[idx,'label_row'].to_numpy(int)]
                    target[idx]=policy.directional_targets(raw,rows.loc[idx,'direction'].to_numpy(int))
                    labels_by_pair[pair]=base.old.directional_paths(raw,rows.loc[idx,'direction'].to_numpy(int))
                    del disk,raw
                rows['target']=target
                tree=DecisionTreeRegressor(max_leaf_nodes=8,min_samples_leaf=max(8,min(63,len(rows)//8)),random_state=2026).fit(rows[base.FEATURES],rows.target)
                leaves=np.unique(tree.apply(rows[base.FEATURES]));states=np.searchsorted(leaves,tree.apply(rows[base.FEATURES]))
                first=int(rows.position.min()+1);last=int(cal.get_indexer(pd.DatetimeIndex(rows.end)).max())
                book=np.zeros((last-first+1,len(leaves)*5))
                for pair,idx in rows.groupby('pair',sort=False).groups.items():
                    idx=np.asarray(idx);base.aggregate(book,states[idx],labels_by_pair[pair],rows.loc[idx,'position'].to_numpy(int),first)
            choices,details=(np.array([4]),{}) if book is None else fit_menu(book,cal[first:last+1],[2,3,4])
            model=dict(tree=tree,leaves=leaves,choices=choices,fit=stamp)
            if fee==.65:
                previous=joblib.load(sources.track(PREVIOUS/'models/18M_floor'/f'{stamp}.joblib'))
                np.testing.assert_array_equal(choices,previous['choices']);np.testing.assert_array_equal(leaves,previous['leaves'])
                if tree is not None:
                    for field in ['feature','threshold','children_left','children_right','value']:
                        np.testing.assert_array_equal(getattr(tree.tree_,field),getattr(previous['tree'].tree_,field))
            record=dict(model=case,training_commission=fee,fit=stamp,training_rows=len(rows),training_pairs=int(rows.pair.nunique()),
                first_training_signal=str(rows.date.min().date()) if len(rows) else None,
                last_training_outcome=str(rows.end.max().date()) if len(rows) else None,choices=choices.tolist(),leaves=len(leaves),**details)
            joblib.dump(model,folder/f'{stamp}.joblib');base.write_json(path,record);logs.append(record)
            del rows,tree,book,labels_by_pair;gc.collect()
        if month.month in [1,4,7,10]:print(f'Completed cost-hurdle fits {stamp}: {len(logs)} models',flush=True)
        sources.save();base.progress('train',fit=stamp,models_complete=len(logs))
    assert len(logs)==360;pd.DataFrame(logs).to_csv(REPORT/'fits.csv',index=False)


def allocations(case):
    rows=pd.read_parquet(OUT/'signal_rows.parquet');rows['learned_level']=4
    for month,idx in rows.groupby(rows.date.dt.to_period('M')).groups.items():
        model=joblib.load(OUT/'models'/case/f'{month.start_time.date()}.joblib')
        rows.loc[idx,'learned_level']=score_model(model,rows.loc[idx],base.FEATURES)
    assert len(rows)==608 and rows.learned_level.isin([2,3,4]).all()
    return rows


def replay(sources,cal):
    base.replay(sources,cal)
    folder=OUT/'replays/train_065';previous=PREVIOUS/'replays/18M_floor'
    pd.testing.assert_frame_equal(pd.read_parquet(folder/'daily_nav.parquet'),pd.read_parquet(sources.track(previous/'daily_nav.parquet')),rtol=1e-12,atol=1e-6)
    for file in ['trades.csv','decisions.csv']:
        pd.testing.assert_frame_equal(pd.read_csv(folder/file),pd.read_csv(sources.track(previous/file)),rtol=1e-12,atol=1e-7)
    print('18-month .961745 checkpoint exactly reproduced',flush=True)


def summarize(sources):
    rows=[];annual=[];statuses=[];books={}
    for folder in sorted((OUT/'replays').iterdir()):
        record=json.loads((folder/'status.json').read_text());assert record['status']=='complete'
        nav=pd.read_parquet(folder/'daily_nav.parquet');books[folder.name]=nav
        decisions=pd.read_csv(folder/'decisions.csv',parse_dates=['entry_date']);assert len(decisions)==608
        rows.append(dict(model=folder.name,training_reward_commission=int(folder.name[-3:])/100 if folder.name.startswith('train_') else None,
            actual_commission=.03,annualized_excess_sharpe=record['annualized_excess_sharpe'],entries=record['entries'],
            active_pairs=record['active_pairs'],net_excess_pnl=record['net_excess_pnl'],
            selection_2018_2023_sharpe=base.sharpe(nav.loc[:'2023-12-31','excess_return'].iloc[1:]),
            recent_2024_2025_sharpe=base.sharpe(nav.loc['2024-01-01':,'excess_return']),
            mean_entered_size=float(decisions.loc[decisions.status.eq('entered'),'level'].mean()/4),
            worst_daily_excess_pnl=float((nav.pnl-nav.cash_benchmark.diff().fillna(0)).min())))
        statuses.extend([dict(model=folder.name,status=status,count=int(count)) for status,count in decisions.status.value_counts().items()])
        for year,g in nav.groupby(nav.index.year):
            annual.append(dict(model=folder.name,year=int(year),annualized_excess_sharpe=base.sharpe(g.excess_return.iloc[1:] if year==2018 else g.excess_return),
                entries=int((decisions.status.eq('entered')&decisions.entry_date.dt.year.eq(year)).sum()),
                net_excess_pnl=float((nav.pnl-nav.cash_benchmark.diff().fillna(0)).loc[g.index].sum())))
    result=pd.DataFrame(rows).sort_values('annualized_excess_sharpe',ascending=False)
    result.to_csv(REPORT/'comparison.csv',index=False);pd.DataFrame(statuses).to_csv(REPORT/'decision_status.csv',index=False)
    yearly=pd.DataFrame(annual);yearly.to_csv(REPORT/'annual.csv',index=False)
    yearly.pivot(index='year',columns='model',values='annualized_excess_sharpe').to_csv(REPORT/'annual_sharpe.csv')
    selected=result.loc[result.training_reward_commission.notna()].sort_values('selection_2018_2023_sharpe',ascending=False).iloc[0]
    base.write_json(REPORT/'selection.json',dict(selected_using='2018-2023 only',model=selected.model,
        annualized_excess_sharpe=selected.annualized_excess_sharpe,no_fresh_holdout=True,no_checkpoint_promotion=True))
    uplift=[];b=books['train_065'].excess_return.iloc[1:].to_numpy()
    for case in [policy.fee_name(fee) for fee in policy.FEES if fee!=.65]:
        a=books[case].excess_return.iloc[1:].to_numpy();assert books[case].index.equals(books['train_065'].index)
        rng=np.random.default_rng(2026);n=len(a);block=126;draws=[]
        for _ in range(1000):
            starts=rng.integers(0,n-block+1,size=int(np.ceil(n/block)));idx=(starts[:,None]+np.arange(block)).ravel()[:n]
            draws.append(base.sharpe(a[idx])-base.sharpe(b[idx]))
        uplift.append(dict(model=case,sharpe_difference=base.sharpe(a)-base.sharpe(b),ci_lower=float(np.quantile(draws,.025)),ci_upper=float(np.quantile(draws,.975))))
    pd.DataFrame(uplift).to_csv(REPORT/'uplift.csv',index=False)
    fits=pd.read_csv(REPORT/'fits.csv');assert len(fits)==360
    known=fits.last_training_outcome.notna();assert (pd.to_datetime(fits.loc[known,'last_training_outcome'])<pd.to_datetime(fits.loc[known,'fit'])).all()
    sources.check();sources.save()
    history=json.loads((OUT/'resource_history.json').read_text());peak=max([resource.getrusage(resource.RUSAGE_SELF).ru_maxrss]+[r['peak_rss_bytes'] for r in history])
    base.write_json(REPORT/'audit.json',dict(status='complete',label_paths_at_065_exact=True,baseline90_models_and_replay_exact=True,
        all608_proposals_preserved=True,actual_execution_unchanged=True,fixed100_exact=True,monthly_models=360,
        frozen_065_states_and_eligibility=True,reward_only_ablation=True,no_fresh_holdout=True,peak_rss_bytes=peak))
    (REPORT/'findings.txt').write_text(result.to_string(index=False)+'\n\n'+yearly.to_string(index=False)+'\n\n'+PROTOCOL+'\n')
    print(result.to_string(index=False),flush=True);base.progress('complete')
    paths=[p for root in [OUT,REPORT] for p in root.rglob('*') if p.is_file() and p.name!='output_hashes.json']
    base.write_json(REPORT/'output_hashes.json',{str(p):base.digest(p) for p in paths})


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['test','prepare','train','replay','summarize','all'],default='all');args=p.parse_args()
    if args.phase=='test':
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests',pattern='test_p25_cost_hurdles.py'))
        if not result.wasSuccessful():raise SystemExit(1)
        return
    configure();OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True);sources=base.Sources()
    sources.hashes[str(Path(__file__))]=base.digest(Path(__file__))
    for path in [Path(__file__),Path('scripts/p25_cost_hurdles.py'),Path('tests/test_p25_cost_hurdles.py'),
        Path('scripts/run_p25_action_windows.py'),Path('scripts/p25_action_windows.py'),Path('scripts/curve_trading/unit_paths.py'),
        Path('scripts/retrain_atm_commission.py'),Path('scripts/uniform_smile_engine.py'),Path('config/p25_recency_candidate.json')]:sources.track(path)
    for path in (base.TRANSFER/'market').glob('*.parquet'):sources.track(path)
    cal=pd.read_parquet(sources.track(NATIVE/'market/rates.parquet')).index
    base.write_json(OUT/'inputs.json',dict(protocol=PROTOCOL,training_reward_fees=policy.FEES,actual_commission=.03,
        training_window_months=18,features=base.FEATURES,evaluation_end='2025-08-29',signal_cutoff='2025-06-30'))
    if args.phase in ['prepare','all']:prepare(sources,cal)
    if args.phase in ['train','all']:train(sources,cal)
    if args.phase in ['replay','all']:replay(sources,cal)
    history_path=OUT/'resource_history.json';history=json.loads(history_path.read_text()) if history_path.exists() else []
    history.append(dict(phase=args.phase,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));base.write_json(history_path,history)
    if args.phase in ['summarize','all']:summarize(sources)
    sources.check();sources.save()


if __name__=='__main__':main()
