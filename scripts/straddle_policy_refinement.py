"""Regularized, recency-weighted contextual sizing under unchanged execution.

No proposal gates. Forecast contexts and action maps fit strictly completed
past trades. Every retained sizing policy is replayed by the funded simulator.
"""
import argparse
import gc
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from scripts import straddle_ensemble_search as replay
from scripts import straddle_performance_followup as source
from scripts.straddle_transfer_fit import scatter_levels
from scripts.straddle_context_followup import predict

CONFIG=Path('config/straddle_policy_refinement.json');CFG=json.loads(CONFIG.read_text())
OUT,REPORT=Path(CFG['output_root']),Path(CFG['report_root'])
DATA=Path(CFG['data_root']);LABELS=Path(CFG['label_root'])
FEATURES=dict(performance=source.FEATURES['performance'],
    minimal=['iv','residual_memory','vix','normalized_memory','variance_ratio','vix_change21',
             'benchmark_sharpe63','benchmark_vol63','benchmark_drawdown63'])
FACTORS=['spot_market_beta126','iv_vix_beta126','iv_spot_beta126','jump_share63']
FEATURES['market']=FEATURES['performance']+FACTORS+[f+'_available' for f in FACTORS]
FEATURES['dynamics']=FEATURES['minimal']+['momentum21','drawdown63','downside_share21','iv_change21','rv_acceleration','relative_iv']
ACTIONS=np.array([2,3,4])


def moments(book,weights=None):
    if weights is None:return book.mean(axis=0),np.atleast_2d(np.cov(book,rowvar=False,ddof=1))
    return np.average(book,axis=0,weights=weights),np.atleast_2d(np.cov(book,rowvar=False,ddof=1,aweights=weights))


def action_map(book,occupancy,spec,weights=None):
    n=len(occupancy);maps=np.array(list(itertools.product(range(3),repeat=n)),int)
    if spec.get('ordered'):maps=maps[(np.diff(maps,axis=1)>=0).all(axis=1)]
    columns=maps+3*np.arange(n)[None,:]
    mean,cov=moments(book,weights)
    means=mean[columns].sum(axis=1)
    variance=cov[columns[:,:,None],columns[:,None,:]].sum(axis=(1,2))
    scores=np.sqrt(252)*np.divide(means,np.sqrt(np.maximum(variance,0)),out=np.zeros(len(maps)),where=variance>1e-15)
    exposure=((ACTIONS[maps]-2)*occupancy).sum(axis=1)/(2*sum(occupancy))
    objective=scores-spec.get('exposure_penalty',0)*exposure
    best=np.flatnonzero(objective>=objective.max()-1e-8)
    winner=min(best,key=lambda i:(float((ACTIONS[maps[i]]*occupancy).sum()),int(i)))
    return ACTIONS[maps[winner]],float(scores[winner]),float(objective[winner])


def fit_model(meta,paths,cutoff,spec,calendar):
    cutoff=pd.Timestamp(cutoff)
    eligible=meta.finite_label&meta.date.ge(cutoff-pd.DateOffset(months=spec['months']))&meta.date.lt(cutoff)&meta.end.lt(cutoff)
    idx=np.flatnonzero(eligible.to_numpy());rows=meta.iloc[idx];names=FEATURES[spec['features']]
    model=dict(cutoff=str(cutoff.date()),specification=spec,features=names,training_samples=len(rows),choices=[2],
        last_completed=str(rows.end.max().date()) if len(rows) else None)
    if len(rows)<CFG['minimum_training_samples']:return model
    values=rows[names].to_numpy(float);target=paths[idx,2].sum(axis=1)
    if spec.get('target')=='risk':
        scale=np.sqrt((paths[idx,2]**2).sum(axis=1))
        target=np.clip(np.divide(target,scale,out=np.zeros(len(idx)),where=scale>1e-9),-3,3)
    if spec.get('winsor'):
        target=np.clip(target,*np.quantile(target,[spec['winsor'],1-spec['winsor']]))
    center,scale=np.median(values,axis=0),np.std(values,axis=0);scale=np.maximum(scale,.001)
    design=np.column_stack([np.ones(len(values)),np.clip((values-center)/scale,-5,5)])
    counts=rows.groupby('date').date.transform('size').to_numpy(float);weights=1/counts
    if spec.get('half_life_months'):
        age=(cutoff-rows.date).dt.days.to_numpy()/30.4375
        weights*=np.exp2(-age/spec['half_life_months'])
    weights*=len(weights)/weights.sum()
    penalty=np.eye(design.shape[1])*spec.get('alpha',100.);penalty[0,0]=0
    beta=np.linalg.solve(design.T@(weights[:,None]*design)+penalty,design.T@(weights*target))
    score=design@beta;cuts=np.quantile(score,np.arange(1,spec.get('states',3))/spec.get('states',3))
    states=np.searchsorted(cuts,score,side='left');occupancy=np.bincount(states)
    model.update(center=center.tolist(),scale=scale.tolist(),beta=beta.tolist(),cuts=cuts.tolist(),clip=5.)
    first,stop=calendar.searchsorted(cutoff-pd.DateOffset(months=spec['months'])),calendar.searchsorted(cutoff)
    book=np.zeros((stop-first,3*len(occupancy)))
    days=rows.entry_position.to_numpy(int)[:,None]+np.arange(paths.shape[2])[None]-first
    valid=(days>=0)&(days<len(book))
    for state in range(len(occupancy)):
        chosen=np.flatnonzero(states==state)
        for action in range(3):
            np.add.at(book[:,3*state+action],days[chosen][valid[chosen]],paths[idx[chosen],action][valid[chosen]])
    daily_weights=None
    if spec.get('half_life_months'):
        daily_weights=np.exp2(-(cutoff-calendar[first:stop]).days.to_numpy()/30.4375/spec['half_life_months'])
    choices,score,objective=action_map(book,occupancy,spec,daily_weights)
    model.update(choices=choices.tolist(),occupancy=occupancy.tolist(),training_shadow_sharpe=score,training_objective=objective)
    assert pd.Timestamp(model['last_completed'])<cutoff
    return model


def labels():
    replay.core.OUT=OUT;replay.core.DATA=LABELS
    return replay.core.load_labels()


def prepare():
    if (REPORT/'protocol.json').exists():return
    if (OUT/'data_manifest.json').exists():
        verify_inputs()
        replay.dump(OUT/'prepare_info.json',dict(proposals=35774,valid_rewards=4116,resumed_validated_data=True))
        return
    for path,signature in json.loads((LABELS/'training_manifest.json').read_text()).items():
        assert replay.digest(path)==signature,'Changed original reward input: '+path
    fields=list(dict.fromkeys(sum(FEATURES.values(),[])))
    base=list(dict.fromkeys(replay.unit.bandit.EXECUTION+['secid']+FEATURES['performance']))
    rows=pd.read_parquet(DATA/'proposals.parquet',columns=base,use_threads=False)
    for path,extras in [(Path('artifacts/contracts/straddle_overnight_study'),FACTORS+[f+'_available' for f in FACTORS]),
                        (Path('artifacts/contracts/straddle_context_followup'),[f for f in FEATURES['dynamics'] if f not in rows])]:
        extra=pd.read_parquet(path/'proposals.parquet',columns=['date','secid']+extras,use_threads=False)
        rows=rows.merge(extra,on=['date','secid'],how='left',validate='one_to_one')
        del extra;gc.collect();pa.default_memory_pool().release_unused()
    assert len(rows)==35774 and np.isfinite(rows[fields]).all().all()
    rows.to_parquet(OUT/'proposals.parquet',index=False)
    gc.collect();pa.default_memory_pool().release_unused()
    meta=pd.read_parquet(DATA/'training.parquet',use_threads=False)
    meta=meta.drop(columns=[c for c in fields if c in meta])
    meta=meta.merge(rows[['date','secid']+fields],on=['date','secid'],how='left',validate='many_to_one')
    assert int(meta.finite_label.sum())==4116 and np.isfinite(meta.loc[meta.finite_label,fields]).all().all()
    meta.to_parquet(OUT/'training.parquet',index=False)
    replay.dump(OUT/'data_manifest.json',{str(p):replay.digest(p) for p in [OUT/'proposals.parquet',OUT/'training.parquet']})
    p=OUT/'base_proposals.parquet'
    if not p.exists():p.symlink_to((DATA/'base_proposals.parquet').resolve())
    proposal_count=len(rows);valid_rewards=int(meta.finite_label.sum())
    # Benchmark CSVs contain35k string-heavy decision rows. Do not retain the
    # already-published preparation frames while loading those three ledgers.
    del rows,meta;gc.collect();pa.default_memory_pool().release_unused()
    replay.dump(OUT/'prepare_info.json',dict(proposals=proposal_count,valid_rewards=valid_rewards))


def references():
    verify_inputs()
    info=json.loads((OUT/'prepare_info.json').read_text())
    records=[]
    for name,folder in replay.FIXED.items():
        nav=pd.read_parquet(folder/'daily_nav.parquet',use_threads=False)
        decisions=pd.read_csv(folder/'decisions.csv',usecols=['entry_date','status'],parse_dates=['entry_date'])
        records += [dict(case=name,period=period,**replay.period_metrics(nav,decisions,start,end)) for period,start,end in replay.PERIODS]
        del nav,decisions;gc.collect();pa.default_memory_pool().release_unused()
    controls=pd.DataFrame(records);controls.to_csv(REPORT/'fixed_controls.csv',index=False)
    thresholds={p:float(controls.loc[controls.period.eq(p),'annualized_excess_sharpe'].max()) for p,_,_ in replay.PERIODS}
    replay.dump(REPORT/'protocol.json',dict(configuration=CFG,features=FEATURES,thresholds=thresholds,
        unique_cases=[s['name'] for s in CFG['specifications']],aliases={},proposals=info['proposals'],valid_rewards=info['valid_rewards'],
        forecast='Completed-trade profit/risk ridge contexts; equal date weights, optional recency. Action search uses full unmodified reward paths even when the partition target is winsorized.',
        objective='Daily aggregate training Sharpe; optional exposure regularization or ordered size maps. These are training choices, not changes to execution costs.',
        interpretation='Explicit historical search on both previously inspected periods. No new untouched evaluation.'))


def allocations(name,later):
    verify_inputs()
    spec=next(s for s in CFG['specifications'] if s['name']==name)
    marker=OUT/'cases'/name/('later_allocations.json' if later else 'validation_allocations.json')
    if marker.exists():return
    roster=pd.read_parquet(OUT/'base_proposals.parquet',use_threads=False)
    keys=pd.MultiIndex.from_frame(roster[['date','secid']]);assert keys.is_unique
    levels=np.full(len(roster),2,int);meta,paths=labels();calendar=replay.unit.base.calendar()
    end=CFG['evaluation_end'] if later else CFG['validation_end']
    for month in pd.period_range(roster.date.min(),min(pd.Timestamp(end),roster.date.max()),freq='M'):
        path=OUT/'models'/name/f'{month}.json'
        if path.exists():model=json.loads(path.read_text())
        else:model=fit_model(meta,paths,month.start_time,spec,calendar);replay.dump(path,model)
        frame=pd.read_parquet(OUT/'proposals.parquet',columns=['date','secid']+FEATURES[spec['features']],
            filters=[('date','>=',month.start_time),('date','<',(month+1).start_time)],use_threads=False)
        if len(frame):scatter_levels(keys,frame,predict(model,frame),levels)
        del frame;gc.collect();pa.default_memory_pool().release_unused()
    roster['learned_level']=levels
    folder=OUT/'cases'/name/'allocations';folder.mkdir(parents=True,exist_ok=True)
    roster.to_parquet(folder/'tree18_test.parquet',index=False)
    replay.dump(marker,dict(case=name,later=later,proposals=len(roster)))
    print(json.dumps(dict(phase='allocations',case=name,later=later)),flush=True)


def replication():
    verify_inputs()
    meta,paths=labels();calendar=replay.unit.base.calendar();records=[]
    root=Path('artifacts/contracts/straddle_performance_followup/models/performance_ridge48')
    spec=dict(name='replication',features='performance',months=48)
    for month in ['2019-01','2020-01','2021-01','2021-12','2025-01']:
        old=json.loads((root/f'{month}.json').read_text())
        new=fit_model(meta,paths,pd.Period(month,'M').start_time,spec,calendar)
        assert new['choices']==old['choices'] and new['training_samples']==old['training_samples']
        for field in ['beta','center','scale','cuts']:np.testing.assert_allclose(new[field],old[field],atol=1e-8,rtol=1e-10)
        records.append(dict(month=month,samples=new['training_samples'],unchanged_predictions=True))
    replay.dump(REPORT/'replication.json',records);print(json.dumps(records),flush=True)


def sources():
    files=[CONFIG,Path(__file__),Path(replay.__file__),Path(source.__file__),
        Path('scripts/straddle_context_followup.py'),Path(replay.unit.engine.__file__),Path('paper_code/execution.json'),
        DATA/'proposals.parquet',DATA/'training.parquet',LABELS/'training_manifest.json',
        Path('artifacts/contracts/straddle_overnight_study/proposals.parquet'),
        Path('artifacts/contracts/straddle_context_followup/proposals.parquet'),
        Path('scripts/straddle_transfer_fit.py'),Path('scripts/run_capped_straddle_book.py')]
    value={str(p):replay.digest(p) for p in files};path=OUT/'sources.json'
    if path.exists():assert json.loads(path.read_text())==value,'Changed sources; preserve this run and create a new root'
    else:replay.dump(path,value)


def verify_inputs():
    for path,signature in json.loads((OUT/'data_manifest.json').read_text()).items():
        assert replay.digest(path)==signature,'Changed prepared input: '+path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',required=True,choices=['test','prepare','references','replication','fit','book','score','shuffle'])
    p.add_argument('--case');p.add_argument('--later',action='store_true');p.add_argument('--months',type=int,default=8);p.add_argument('--seed',type=int);a=p.parse_args()
    if a.phase=='test':
        import unittest
        suite=unittest.defaultTestLoader.loadTestsFromNames(['tests.test_straddle_policy_refinement','tests.test_research_memory_guard','tests.test_research_atomic_io'])
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():raise SystemExit(1)
        return
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    replay.CFG=CFG;replay.OUT=OUT;replay.REPORT=REPORT;sources()
    if a.phase=='prepare':prepare()
    elif a.phase=='references':references()
    elif a.phase=='replication':replication()
    elif a.phase=='fit':allocations(a.case,a.later)
    elif a.phase=='book':replay.book(a.case,a.months,a.later)
    elif a.phase=='score':replay.score(a.case,a.later)
    else:replay.shuffled(a.case,a.seed)


if __name__=='__main__':main()
