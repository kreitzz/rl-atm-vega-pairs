"""Bounded monthly RL feature ablations around the preserved .826 P25 book."""
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

from scripts import p25_risk_sizing as risk
from scripts import p25_inventory_engine as engine
from scripts import uniform_smile_data as cached
from scripts import same_structure_smile as old_policy
from scripts.run_same_structure_smile import aggregate
from scripts.run_p25_signal_transfer import Market
from scripts.uniform_smile_risk import book_risk
from scripts.tail_put25_pooled import completed_mask, FEATURES
from scripts.curve_trading.config import config,RUNOFF
from scripts.run_aligned_contract_learning import asset_paths,mark_book
from scripts.run_atm65_ablation import write_json
from scripts.run_contract_walkforward import digest
from scripts.run_uniform_smile import verify,UNIFORM_CORPORATE_EVENTS

OUT=Path('artifacts/contracts/p25_risk_sizing')
REPORT=Path('reports/p25_risk_sizing')
SOURCE=Path('artifacts/contracts/same_structure_smile')
TRANSFER=Path('artifacts/contracts/p25_signal_transfer')
FROZEN=Path('artifacts/contracts/same_structure_calendar/replays/floor_P25')
ARMS=['rv','relationship','both']
PROTOCOL=(
    'Preserve all608 original P25 pair/date/direction/contract proposals and '
    'original formation/calendar/positive directional-profit screens. '
    'Baseline .8256229208692676 must reproduce NAV,trades,decisions. '
    'Add causal long/short20-sessionRV-to-selected-putIV ratios; separately '
    'IV*(1-returnCorrelation) and absolute log126-session volatility ratio '
    'derived from beta/correlation; then both sets together. Optional missing '
    'inputs have explicit flags and zero neutral values; never a new veto. '
    'Same27 existing features, monthly5Y completed-outcome pooled fits, '
    '8-leaf tree with identical adaptive minleaf, CEM50x192 seed2026, '
    'full-size anchor,16-row global fallback,50/75/100 ladder. '
    'Same saved0.65 training-cost labels; actual0.03 commission,mid+.1volpoint, '
    '1bp stock hedge and original financing/borrow/dividends/risk limits. '
    'Fourth arm applies declared actual-prior-close same-name gross-vega '
    'overlay: nearest floor-ladder level to learned level/sqrt(1+max name '
    'inventory/current funded full-size per-leg vega budget),minimum50%. '
    'Overlay is fixed,not learned sequential inventory RL,and has no veto. '
    'Apply that same declared overlay to the frozen baseline as an isolated '
    'inventory ablation,at P25 and combined-book scope. '
    'Also replay both-feature P25 sizing transferred to all3 existing smile '
    'wings, with/without overlay, against the saved identical combined stream. '
    'Actual prior inventory is shared across names,pairs,and wings. '
    'No new pair exclusions,correlation cutoff,quote selection,or threshold '
    'search. Already inspected history and retrospective roster/calendar '
    'conditioning; this is research,not an untouched holdout. '
    'Single worker,50%of onecore,oneBLAS/Arrow thread,3GiB RSS stop.'
)


def progress(stage,**details):
    write_json(OUT/'progress.json',dict(stage=stage,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,**details))


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
    def verify_stats(self):
        for path,stat in self.stats.items():
            p=Path(path);assert (p.stat().st_size,p.stat().st_mtime_ns)==stat,path


def prepare(sources,cal):
    meta=pd.read_parquet(sources.track(SOURCE/'P25_rows.parquet'))
    definitions=json.loads(sources.track(Path('artifacts/contracts/tail_put25_pooled/inputs.json')).read_text())['definitions']
    definitions={d['pair']:d for d in definitions}
    assert (cal.get_indexer(pd.DatetimeIndex(meta.date))==meta.position.to_numpy()).all()
    ret_cache={};iv_cache={};parts=[]
    for n,(pair,rows) in enumerate(meta.groupby('pair',sort=False),1):
        d=definitions[pair];ivs=[];rvs=[]
        for sid,path in zip(d['secids'],asset_paths(d)):
            sid=int(sid)
            if sid not in iv_cache:
                selection=pd.read_parquet(sources.track(cached.OUT/'market/assets'/f'{sid}_selection.parquet'),
                    columns=['date','cp_flag','bucket','bs_iv'])
                selection=selection.loc[selection.cp_flag.eq('P')&selection.bucket.eq('25')]
                assert not selection.date.duplicated().any()
                iv_cache[sid]=selection.set_index('date').bs_iv.where(lambda x:x.gt(0))
            if str(path) not in ret_cache:
                if path.exists():
                    ret=pd.read_parquet(sources.track(path),columns=['ret']).ret.reindex(cal)
                else:
                    stock=pd.read_parquet(sources.track(cached.OUT/'market/assets'/f'{sid}_stocks.parquet'),
                        columns=['date','close','cfadj']).set_index('date').reindex(cal)
                    ret=(stock.close*stock.cfadj).pct_change(fill_method=None)
                ret_cache[str(path)]=np.sqrt(ret.pow(2).rolling(20,min_periods=10).mean()*252)
            ivs.append(iv_cache[sid].reindex(pd.DatetimeIndex(rows.date)).to_numpy())
            rvs.append(ret_cache[str(path)].reindex(pd.DatetimeIndex(rows.date)).to_numpy())
        good=np.isfinite(ivs[0])&np.isfinite(ivs[1])
        np.testing.assert_allclose((ivs[0][good]+ivs[1][good])/2,rows.iv_mean.to_numpy()[good],rtol=1e-8,atol=1e-9)
        known=np.isfinite(rvs[0])&np.isfinite(rvs[1])&rows.rv_mean_available.eq(1).to_numpy()
        np.testing.assert_allclose((rvs[0][known]+rvs[1][known])/2,rows.rv_mean.to_numpy()[known],rtol=1e-7,atol=1e-9)
        parts.append(risk.risk_features(rows,ivs[0],ivs[1],rvs[0],rvs[1]))
        if n%40==0:print(f'Risk state preparation {n} pairs',flush=True)
    prepared=pd.concat(parts).sort_index()
    pd.testing.assert_frame_equal(prepared[meta.columns],meta)
    prepared.to_parquet(OUT/'training_rows.parquet',index=False)
    proposals=pd.read_parquet(sources.track(TRANSFER/'P25_rows.parquet'))
    pd.testing.assert_frame_equal(proposals[['date','pair','direction','base_level']],
        pd.read_parquet(sources.track(FROZEN/'scored_allocations.parquet'),columns=['date','pair','direction','base_level']))
    assert len(proposals)==608
    cols=['date','pair']+FEATURES+risk.RV+risk.RELATION+[x+'_available' for x in risk.RV+risk.RELATION]
    features=prepared[cols]
    joined=proposals.merge(features,on=['date','pair'],how='left',validate='one_to_one',indicator=True)
    assert joined._merge.eq('both').all()
    joined.drop(columns='_merge').to_parquet(OUT/'signal_rows.parquet',index=False)
    for arm in ARMS:assert np.isfinite(prepared[risk.columns(arm)]).all().all()
    sources.save();progress('prepared',candidate_rows=len(meta),frozen_proposals=len(proposals))


def train(sources,cal):
    meta=pd.read_parquet(OUT/'training_rows.parquet');logs=[]
    for month in pd.period_range('2018-01-01',cal[-1]-pd.Timedelta(days=RUNOFF),freq='M'):
        cutoff=month.start_time;stamp=str(cutoff.date())
        rows=meta.loc[meta.signal_eligible&meta.finite_label&completed_mask(meta.date,meta.end,cutoff)].reset_index(drop=True)
        assert not len(rows) or rows.end.max()<cutoff
        for arm in ARMS:
            folder=OUT/'models'/arm;folder.mkdir(parents=True,exist_ok=True)
            info=folder/f'{stamp}.json'
            if info.exists():logs.append(json.loads(info.read_text()));continue
            tree=None;leaves=np.array([0]);choices=np.array([4]);details={}
            if len(rows)>=16:
                tree=DecisionTreeRegressor(max_leaf_nodes=8,min_samples_leaf=max(8,min(63,len(rows)//8)),random_state=2026)
                tree.fit(rows[risk.columns(arm)],rows.target)
                leaves=np.unique(tree.apply(rows[risk.columns(arm)]))
                states=np.searchsorted(leaves,tree.apply(rows[risk.columns(arm)]))
                first=int(rows.position.min()+1);last=int(cal.get_indexer(pd.DatetimeIndex(rows.end)).max())
                book=np.zeros((last-first+1,len(leaves)*5))
                for pair,idx in rows.groupby('pair',sort=False).groups.items():
                    idx=np.asarray(idx)
                    path=sources.track(cached.OUT/'pairs'/pair/'P25_labels.npy')
                    disk=np.load(path,mmap_mode='r')
                    labels=old_policy.directional_paths(disk[rows.loc[idx,'label_row'].to_numpy(int)],rows.loc[idx,'direction'].to_numpy(int))
                    aggregate(book,states[idx],labels,rows.loc[idx,'position'].to_numpy(int),first)
                    del disk,labels
                choices,details=old_policy.fit_sizes(book,cal[first:last+1],tree,leaves,'floor')
                del book
            assert np.isin(choices,[2,3,4]).all()
            record=dict(arm=arm,fit=stamp,training_rows=len(rows),training_pairs=int(rows.pair.nunique()),
                last_training_outcome=str(rows.end.max().date()) if len(rows) else None,
                leaves=len(leaves),choices=choices.tolist(),features=risk.columns(arm),
                status='enabled' if tree is not None else 'original_global_fullsize_fallback',**details)
            joblib.dump(dict(tree=tree,leaves=leaves,choices=choices,fit=stamp,features=risk.columns(arm)),folder/f'{stamp}.joblib')
            write_json(info,record);logs.append(record)
            del tree;gc.collect()
        if month.month in [1,4,7,10]:print(f'Completed risk fits {stamp}: {len(rows)} matured training rows',flush=True)
        sources.save();progress('train',fit=stamp,fits_complete=len(logs))
    pd.DataFrame(logs).to_csv(REPORT/'fits.csv',index=False)


def allocations(arm):
    rows=pd.read_parquet(OUT/'signal_rows.parquet')
    rows['base_level']=4
    for month,idx in rows.groupby(rows.date.dt.to_period('M')).groups.items():
        model=joblib.load(OUT/'models'/arm/f'{month.start_time.date()}.joblib')
        assert pd.Timestamp(model['fit'])<=rows.loc[idx,'date'].min()
        if model['tree'] is not None:
            leaves=model['tree'].apply(rows.loc[idx,model['features']])
            rows.loc[idx,'base_level']=model['choices'][np.searchsorted(model['leaves'],leaves)]
    return rows


def replay(sources,cal,baseline_only=False):
    cached.OUT=TRANSFER
    market=Market();calendar=cal[cal>='2018-01-01']
    arms=[('reference_P25',None,False,False)] if baseline_only else [
        ('rv_P25','rv',False,False),('relationship_P25','relationship',False,False),
        ('both_P25','both',False,False),('both_inventory_P25','both',True,False),
        ('both_combined','both',False,True),('both_inventory_combined','both',True,True),
        ('inventory_reference_P25','frozen',True,False),('inventory_reference_combined','frozen',True,True)]
    for name,arm,overlay,combined in arms:
        folder=OUT/'replays'/name;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'status.json').exists():continue
        if arm in [None,'frozen']:
            wings=['P25','ATM','P75'] if combined else ['P25']
            rows=pd.concat([pd.read_parquet(sources.track(TRANSFER/f'{wing}_rows.parquet')) for wing in wings],ignore_index=True)
        else:
            scored=allocations(arm)
            if combined:
                parts=[]
                for wing in ['P25','ATM','P75']:
                    part=pd.read_parquet(sources.track(TRANSFER/f'{wing}_rows.parquet'))
                    pd.testing.assert_frame_equal(part[['date','pair','direction']],scored[['date','pair','direction']])
                    part['base_level']=scored.base_level.to_numpy();parts.append(part)
                rows=pd.concat(parts,ignore_index=True)
            else:rows=scored
        rows=rows.sort_values(['date','pair','wing'],kind='stable').reset_index(drop=True)
        rows.to_parquet(folder/'scored_allocations.parquet',index=False)
        multiplier=3 if combined else 1
        cfg=replace(config(),option_commission=.03,capital=1_000_000*multiplier*345,max_gross_vega=20_000*multiplier*345)
        market.variant=name;market.started=time.monotonic();market.reported_year=None
        print(f'Replay {name}: {len(rows)} unchanged proposals',flush=True)
        original=engine.book_risk
        try:
            engine.book_risk=book_risk
            callback=(lambda row,context:risk.inventory_level(int(row['base_level']),context)) if overlay else None
            nav,trades,decisions,ledger=engine.simulate(market,rows,calendar,cfg,1_000_000*multiplier,
                policy=callback,corporate_events=UNIFORM_CORPORATE_EVENTS,wing_capital=1_000_000)
        finally:engine.book_risk=original
        mark_book(nav,market.rates,cfg.capital);audit=verify(nav,trades,decisions,rows,cfg)
        if arm is None:
            pd.testing.assert_frame_equal(nav,pd.read_parquet(sources.track(FROZEN/'daily_nav.parquet')),rtol=1e-12,atol=1e-6)
            for frame,file,dates in [(trades,'trades.csv',['date']),(decisions,'decisions.csv',['signal_date','entry_date'])]:
                old=pd.read_csv(sources.track(FROZEN/file),parse_dates=dates)
                pd.testing.assert_frame_equal(frame[old.columns].reset_index(drop=True),old.reset_index(drop=True),
                    check_dtype=False,rtol=1e-12,atol=1e-7)
            print('Original .825623 P25 NAV,trades,decisions reproduced',flush=True)
        if overlay:
            assert decisions.level.between(2,4).all()
            assert (decisions.level<=decisions.base_level).all()
            assert not decisions.status.eq('flat').any()
        nav.to_parquet(folder/'daily_nav.parquet');trades.to_csv(folder/'trades.csv',index=False)
        decisions.to_csv(folder/'decisions.csv',index=False)
        write_json(folder/'status.json',dict(status='complete',model=name,proposals=len(rows),
            no_new_signal_veto=True,inventory_overlay=overlay,combined=combined,
            inventory_reduced_proposals=int((decisions.level<decisions.base_level).sum()),
            elapsed_seconds=time.monotonic()-market.started,
            net_excess_pnl=float(nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1]),**audit,**ledger))
        progress('replay',model=name,**audit);sources.save()
        print(f'{name}: Sharpe {audit["annualized_excess_sharpe"]:.6f}, {audit["entries"]} entries',flush=True)
        del nav,trades,decisions,rows;gc.collect()


def summarize(sources):
    comparison=[];annual=[];periods=[];books={};details=[]
    folders={p.name:p for p in (OUT/'replays').iterdir()}
    folders['reference_combined']=TRANSFER/'replays/floor'
    for name,folder in folders.items():
        record=json.loads(sources.track(folder/'status.json').read_text())
        assert record['status']=='complete'
        nav=pd.read_parquet(sources.track(folder/'daily_nav.parquet'))
        decisions=pd.read_csv(sources.track(folder/'decisions.csv'),parse_dates=['signal_date','entry_date'])
        books[name]=nav
        comparison.append(dict(model=name,annualized_excess_sharpe=record['annualized_excess_sharpe'],
            net_excess_pnl=float(nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1]),entries=record['entries'],
            active_pairs=record['active_pairs'],mean_entered_size=float(decisions.loc[decisions.status.eq('entered'),'level'].mean()/4),
            inventory_reduced_proposals=int((decisions.level<decisions.base_level).sum()),
            worst_daily_excess_pnl=float((nav.pnl-nav.cash_benchmark.diff().fillna(0)).min())))
        for year,g in nav.groupby(nav.index.year):
            r=g.excess_return.iloc[1:] if year==2018 else g.excess_return
            annual.append(dict(model=name,year=int(year),annualized_excess_sharpe=float(np.sqrt(252)*r.mean()/r.std()),
                entries=int((decisions.status.eq('entered')&decisions.entry_date.dt.year.eq(year)).sum()),
                net_excess_pnl=float((nav.pnl-nav.cash_benchmark.diff().fillna(0)).loc[g.index].sum())))
        for label,start,end in [('2018to2020','2018-01-01','2020-12-31'),('2021to2023','2021-01-01','2023-12-31'),
                                ('2024to2025','2024-01-01','2025-08-29')]:
            r=nav.loc[start:end,'excess_return'];r=r.iloc[1:] if start=='2018-01-01' else r
            periods.append(dict(model=name,period=label,annualized_excess_sharpe=float(np.sqrt(252)*r.mean()/r.std())))
        key=decisions.loc[(decisions.pair.eq('214904__215427')&decisions.signal_date.between('2024-02-14','2024-02-15'))|
                          (decisions.pair.eq('125537__189845')&decisions.signal_date.isin(pd.to_datetime(['2024-05-01','2024-05-10'])))].copy()
        key['model']=name;details.append(key)
    pd.DataFrame(comparison).to_csv(REPORT/'comparison.csv',index=False)
    pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False)
    pd.DataFrame(annual).pivot(index='year',columns='model',values='annualized_excess_sharpe').to_csv(REPORT/'annual_sharpe.csv')
    pd.DataFrame(periods).to_csv(REPORT/'periods.csv',index=False)
    pd.concat(details,ignore_index=True).to_csv(REPORT/'key_risk_decisions.csv',index=False)
    # Paired daily block bootstrap: preserve overlapping holdings and calendars.
    uplift=[]
    for name,nav in books.items():
        if name.startswith('reference'):continue
        baseline=books['reference_combined' if name.endswith('combined') else 'reference_P25']
        a=nav.excess_return.iloc[1:].to_numpy();b=baseline.excess_return.iloc[1:].to_numpy()
        assert nav.index.equals(baseline.index)
        def score(r):return float(np.sqrt(252)*r.mean()/r.std(ddof=1)) if r.std(ddof=1)>1e-12 else 0.
        rng=np.random.default_rng(2026);draws=[];n=len(a);block=126
        for _ in range(1000):
            starts=rng.integers(0,n-block+1,size=int(np.ceil(n/block)))
            idx=(starts[:,None]+np.arange(block)).ravel()[:n]
            draws.append(score(a[idx])-score(b[idx]))
        uplift.append(dict(model=name,reference='reference_combined' if name.endswith('combined') else 'reference_P25',
            sharpe_difference=score(a)-score(b),ci_lower=float(np.quantile(draws,.025)),ci_upper=float(np.quantile(draws,.975))))
    pd.DataFrame(uplift).to_csv(REPORT/'uplift.csv',index=False)
    fits=pd.read_csv(REPORT/'fits.csv');assert len(fits)==270
    known=fits.last_training_outcome.notna()
    assert (pd.to_datetime(fits.loc[known,'last_training_outcome'])<pd.to_datetime(fits.loc[known,'fit'])).all()
    split_usage=[]
    for arm in ARMS:
        counts={name:0 for name in risk.columns(arm)}
        for path in (OUT/'models'/arm).glob('*.joblib'):
            model=joblib.load(path)
            if model['tree'] is None:continue
            for feature in model['tree'].tree_.feature:
                if feature>=0:counts[model['features'][int(feature)]]+=1
        split_usage.extend(dict(arm=arm,feature=feature,splits=count) for feature,count in counts.items())
    pd.DataFrame(split_usage).to_csv(REPORT/'feature_split_usage.csv',index=False)
    resources_path=OUT/'resource_history.json'
    resources=json.loads(resources_path.read_text()) if resources_path.exists() else []
    peak_rss=max([resource.getrusage(resource.RUSAGE_SELF).ru_maxrss]+[r['peak_rss_bytes'] for r in resources])
    sources.verify_stats();sources.save()
    write_json(REPORT/'audit.json',dict(status='complete',original_checkpoint_reproduced=True,models=270,
        training_labels_end_before_each_fit=True,same608_proposals=True,optional_missing_inputs_never_veto=True,
        overlay_uses_actual_previous_close_name_inventory=True,inventory_overlay_not_learned=True,
        no_fresh_holdout=True,seed=2026,peak_rss_bytes=peak_rss))
    (REPORT/'findings.txt').write_text(pd.DataFrame(comparison).to_string(index=False)+'\n\n'+
        pd.DataFrame(periods).to_string(index=False)+'\n\n'+pd.DataFrame(uplift).to_string(index=False)+'\n\n'+PROTOCOL+'\n')
    outputs=[p for root in [OUT,REPORT] for p in root.rglob('*') if p.is_file() and p.name!='output_hashes.json']
    write_json(REPORT/'output_hashes.json',{str(p):digest(p) for p in outputs})
    progress('complete');print(pd.DataFrame(comparison).to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['test','prepare','reference','train','replay','summarize','all'],default='all')
    args=p.parse_args()
    if args.phase=='test':
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests',pattern='test_p25_risk_sizing.py'))
        if not result.wasSuccessful():raise SystemExit(1)
        return
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    sources=Sources()
    for path in [Path(__file__),Path('scripts/p25_risk_sizing.py'),Path('scripts/p25_inventory_engine.py'),
                 Path('scripts/uniform_smile_engine.py'),Path('tests/test_p25_risk_sizing.py')]:sources.track(path)
    for path in (TRANSFER/'market').glob('*.parquet'):sources.track(path)
    rates=pd.read_parquet(sources.track(Path('artifacts/contracts/uniform_smile/market/rates.parquet')))
    cal=rates.index
    write_json(OUT/'inputs.json',dict(protocol=PROTOCOL,arms=ARMS,inventory_overlay='level/sqrt(1+name inventory units),rounded to50/75/100',
        reference_sharpe=.8256229208692676,features={arm:risk.columns(arm) for arm in ARMS},commission_training=.65,commission_execution=.03))
    if args.phase in ['prepare','all']:prepare(sources,cal)
    if args.phase in ['reference','all']:replay(sources,cal,True)
    if args.phase in ['train','all']:
        cached.OUT=Path('artifacts/contracts/uniform_smile');train(sources,cal)
    if args.phase in ['replay','all']:replay(sources,cal)
    if args.phase in ['summarize','all']:summarize(sources)
    sources.verify_stats();sources.save()


if __name__=='__main__':
    main()
