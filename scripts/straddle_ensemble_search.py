"""Combine learned entry sizes and replay real integer-straddle portfolios.

No saved-return blends: each candidate issues one50/75/100% size per original
proposal. The same funded option/stock ledger handles all fills and admission.
Joint selection on the two historical periods is explicitly research selection.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import atm_rv_straddle_bandit as unit
from scripts import straddle_entry_refinement as core
from scripts.run_capped_straddle_book import native_ladder
from paper_code.metrics import period_metrics,paired_interval,sharpe

CONFIG=Path('config/straddle_ensemble_search.json');CFG=json.loads(CONFIG.read_text())
OUT,REPORT=Path(CFG['output_root']),Path(CFG['report_root'])
FIXED={
    'fixed50':Path('artifacts/contracts/atm_rv_straddle_fixed50/replays/tree18'),
    'fixed75':Path('artifacts/contracts/straddle_capped_books/fixed75/replays/tree18'),
    'fixed100':Path('artifacts/contracts/atm_rv_straddle_full/replays/straddle100')}
PERIODS=[('validation','2018-01-01',CFG['validation_end']),
    ('evaluation','2022-01-01',CFG['evaluation_end']),('full','2018-01-01',CFG['evaluation_end'])]


def dump(path,value):unit.base.dump(path,value)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def quantize(values):
    """Prefer the middle75% action at either equidistant boundary."""
    x=np.asarray(values,float)
    return np.where(x<2.5,2,np.where(x>3.5,4,3)).astype(int)


def combine(name,a,b,c,context):
    if name=='mean_ab':return quantize((a+b)/2)
    if name=='min_ab':return np.minimum(a,b)
    if name=='median_abc':return np.median(np.column_stack([a,b,c]),axis=1).astype(int)
    if name=='mean_abc':return quantize((a+b+c)/3)
    if name=='mean_ac':return quantize((a+c)/2)
    if name=='mean_ab_cap75':return np.minimum(3,combine('mean_ab',a,b,c,context))
    if name=='agree75':return np.where((a==4)&(b==4),3,2)
    if name=='max_ab':return np.maximum(a,b)
    if name.startswith('pick'):
        n=int(name[4:]);return np.where(context[f'sharpe_a_{n}']>=context[f'sharpe_b_{n}'],a,b)
    if name=='soft63':
        delta=np.clip(context.sharpe_a_63-context.sharpe_b_63,-5,5)
        weight=1/(1+np.exp(-delta))
        return quantize(weight*a+(1-weight)*b)
    if name=='risk63':
        va=context.vol_a_63.clip(lower=1e-5);vb=context.vol_b_63.clip(lower=1e-5)
        weight=vb/(va+vb);return quantize(weight*a+(1-weight)*b)
    raise ValueError(name)


def source_manifest():
    files=[CONFIG,Path(__file__),Path(unit.__file__),Path(unit.engine.__file__),
        Path(core.__file__),Path('scripts/run_capped_straddle_book.py'),Path('paper_code/execution.json')]
    for root in CFG['experts'].values():files += [Path(root)/'allocations/tree18_test.parquet',Path(root)/'replays/tree18/daily_nav.parquet']
    record={str(p):digest(p) for p in files};path=OUT/'sources.json'
    if path.exists():assert json.loads(path.read_text())==record,'Changed study sources; use a new root'
    else:dump(path,record)


def fixed_results():
    records=[]
    for name,folder in FIXED.items():
        nav=pd.read_parquet(folder/'daily_nav.parquet',use_threads=False)
        decisions=pd.read_csv(folder/'decisions.csv',parse_dates=['signal_date','entry_date'])
        records += [dict(case=name,period=label,**period_metrics(nav,decisions,start,end)) for label,start,end in PERIODS]
    table=pd.DataFrame(records);table.to_csv(REPORT/'fixed_controls.csv',index=False)
    return table


def expert_context(dates,navs):
    result=pd.DataFrame(index=pd.DatetimeIndex(dates))
    for expert,nav in navs.items():
        for n in [63,126]:
            r=nav.excess_return;minimum=42 if n==63 else 63
            mean=r.rolling(n,min_periods=minimum).mean();std=r.rolling(n,min_periods=minimum).std()
            result[f'sharpe_{expert}_{n}']=(np.sqrt(252)*mean/std.where(std>1e-10)).clip(-5,5).reindex(dates).fillna(0).to_numpy()
            result[f'vol_{expert}_{n}']=std.reindex(dates).fillna(1e-5).to_numpy()
    return result.reset_index(drop=True)


def prepare():
    marker=REPORT/'protocol.json'
    if marker.exists():return
    keys=None;arrays={};navs={};source_audits=[]
    for expert,folder in CFG['experts'].items():
        path=Path(folder)/'allocations/tree18_test.parquet'
        frame=pd.read_parquet(path,use_threads=False)
        current=pd.MultiIndex.from_frame(frame[['date','secid']]);assert current.is_unique
        if keys is None:keys=current;roster=frame.drop(columns=['learned_level']).copy()
        else:assert current.equals(keys)
        arrays[expert]=frame.learned_level.to_numpy(int);assert np.isin(arrays[expert],[2,3,4]).all()
        navs[expert]=pd.read_parquet(Path(folder)/'replays/tree18/daily_nav.parquet',columns=['excess_return'],use_threads=False)
        models=Path(folder).parent.parent/'models'/Path(folder).name
        assert len(list(models.glob('*.json')))==90
        for p in models.glob('*.json'):
            m=json.loads(p.read_text());assert m['last_completed'] is None or pd.Timestamp(m['last_completed'])<pd.Timestamp(m['cutoff'])
        source_audits.append(dict(expert=expert,monthly_models=90,strict_completion_cutoffs=True))
    assert len(roster)==35774
    context=expert_context(roster.date,navs);aliases={};unique={};sizes=[]
    for name in CFG['cases']:
        levels=combine(name,arrays['a'],arrays['b'],arrays['c'],context)
        assert np.isin(levels,[2,3,4]).all()
        signature=hashlib.sha256(levels.tobytes()).hexdigest()
        if signature in unique:aliases[name]=unique[signature];continue
        unique[signature]=name
        rows=roster.copy();rows['learned_level']=levels
        folder=OUT/'cases'/name/'allocations';folder.mkdir(parents=True,exist_ok=True)
        rows.to_parquet(folder/'tree18_test.parquet',index=False)
        sizes.append(dict(case=name,counts=pd.Series(levels).value_counts().to_dict()))
    controls=fixed_results()
    thresholds={label:float(controls.loc[controls.period.eq(label),'annualized_excess_sharpe'].max()) for label,_,_ in PERIODS}
    dump(marker,dict(configuration=CFG,unique_cases=list(unique.values()),aliases=aliases,thresholds=thresholds,
        expert_audits=source_audits,size_counts=sizes,
        causality='Each source policy refits from completed past rewards. Expert performance weighting uses only marks at or before the current signal close. No outcome-based proposal veto.',
        limitations='The source specifications were selected during earlier research. This new search explicitly screens both previously inspected periods. Actual funded replay, not fractional blends of source returns.'))
    print(json.dumps(dict(unique_cases=list(unique.values()),aliases=aliases,thresholds=thresholds)),flush=True)


def book(name,months,later):
    unit.OUT=OUT/'cases'/name
    if not later:
        done=len(list((unit.OUT/'replays/tree18/months').glob('????-??/checkpoint.joblib')))
        months=min(months,48-done)
        if months<=0:return
    original=unit.engine.simulate
    unit.engine.simulate=lambda *a,**k:native_ladder(original,*a,**k)
    try:unit.replay(months)
    finally:unit.engine.simulate=original


def score(name,later):
    core.OUT=OUT
    end=CFG['evaluation_end'] if later else CFG['validation_end']
    nav,decisions=core.funded_nav(name,end)
    records=[dict(case=name,period=label,**period_metrics(nav,decisions,start,stop))
        for label,start,stop in (PERIODS if later else PERIODS[:1])]
    folder=REPORT/'cases'/name;folder.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(folder/'results.csv',index=False)
    if later:
        annual=[dict(case=name,year=y,**period_metrics(nav,decisions,f'{y}-01-01',min(f'{y}-12-31',CFG['evaluation_end']))) for y in range(2018,2026)]
        pd.DataFrame(annual).to_csv(folder/'annual.csv',index=False)
        trades=pd.read_csv(OUT/'cases'/name/'replays/tree18/trades.csv',parse_dates=['date'])
        flow=(-100*trades.quantity*trades.price-trades.commission-trades.execution_penalty).groupby(trades.date).sum().reindex(nav.index,fill_value=0)
        expected=flow+nav.option_value.diff().fillna(nav.option_value)+nav.hedge_gross_pnl-nav.hedge_cost+nav.interest-nav.stock_borrow_cost+nav.dividend_cash
        error=float((expected-nav.pnl).abs().max());assert error<1e-5
        assert trades.groupby(['cohort','optionid']).quantity.sum().eq(0).all()
        assert abs(nav.option_value.iloc[-1])+abs(nav.stock_value.iloc[-1])<1e-7 and len(decisions)==35774
        intervals=[];exclusions=[]
        for control,root in FIXED.items():
            other=pd.read_parquet(root/'daily_nav.parquet',columns=['excess_return'],use_threads=False).excess_return
            for label,start,stop in PERIODS[:2]:
                a,b=nav.excess_return.loc[start:stop],other.loc[start:stop]
                if label=='validation':a,b=a.iloc[1:],b.iloc[1:]
                for block in [63,126]:intervals.append(dict(case=name,control=control,period=label,**paired_interval(a,b,block=block,draws=1000)))
            for year in range(2022,2026):
                a=nav.excess_return.loc['2022':];b=other.reindex(a.index);keep=a.index.year!=year
                exclusions.append(dict(case=name,control=control,excluded_year=year,learner=sharpe(a.loc[keep]),fixed=sharpe(b.loc[keep])))
        pd.DataFrame(intervals).to_csv(folder/'intervals.csv',index=False)
        pd.DataFrame(exclusions).to_csv(folder/'year_exclusions.csv',index=False)
        dump(folder/'audit.json',dict(proposals=len(decisions),cashflow_error=error,closed_inventory=True,unchanged_execution=True))
    print(json.dumps(dict(case=name,records=records)),flush=True)


def shuffled(name,seed):
    source=OUT/'cases'/name/'allocations/tree18_test.parquet'
    rows=pd.read_parquet(source,use_threads=False);rng=np.random.default_rng(seed)
    for _,idx in rows.groupby('date',sort=True).groups.items():
        rows.loc[idx,'learned_level']=rng.permutation(rows.loc[idx,'learned_level'].to_numpy())
    control=f'{name}_shuffle{seed}'
    path=OUT/'cases'/control/'allocations/tree18_test.parquet';path.parent.mkdir(parents=True,exist_ok=True)
    rows.to_parquet(path,index=False)
    # Same date counts are an observable, causal control, not future frequencies.
    original=pd.read_parquet(source,columns=['date','learned_level'],use_threads=False)
    pd.testing.assert_series_equal(rows.groupby(['date','learned_level']).size(),original.groupby(['date','learned_level']).size())
    print(json.dumps(dict(control=control,seed=seed,per_date_action_counts_preserved=True)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',required=True,choices=['test','prepare','book','score','shuffle'])
    p.add_argument('--case');p.add_argument('--months',type=int,default=8);p.add_argument('--later',action='store_true');p.add_argument('--seed',type=int);a=p.parse_args()
    if a.phase=='test':
        import unittest
        suite=unittest.defaultTestLoader.loadTestsFromNames(['tests.test_straddle_ensemble_search','tests.test_research_memory_guard','tests.test_research_atomic_io'])
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():raise SystemExit(1)
        return
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True);source_manifest()
    if a.phase=='prepare':prepare()
    elif a.phase=='book':book(a.case,a.months,a.later)
    elif a.phase=='score':score(a.case,a.later)
    else:shuffled(a.case,a.seed)


if __name__=='__main__':main()
