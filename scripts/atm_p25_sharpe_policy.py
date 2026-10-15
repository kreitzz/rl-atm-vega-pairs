"""Small-state direct RL policy search on actual training portfolio Sharpe.

Enumerating 3^K assignments (K <= 3) replaces stochastic CEM search. Each policy
is evaluated through sequential holdings, monthly drift and resizing costs.
It is direct policy-search RL, not fitted Q-learning. Market-state partitions
and exposure maps use only completed training months. No source trades change.
"""
import argparse
import gc
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import atm_p25_allocation_q as old
from paper_code.metrics import paired_interval, sharpe

CFG_PATH=Path('config/atm_p25_sharpe_policy.json')
CFG=json.loads(CFG_PATH.read_text())
OUT=Path('artifacts/contracts/atm_p25_sharpe_policy')
REPORT=Path('reports/atm_p25_sharpe_policy')
FEATURES=['relative_sharpe63','vix','correlation63','log_vol_ratio','atm_drawdown63','pair_drawdown63']


def target(ratio,share):
    a=(share/max(float(ratio),1e-8))/((1-share)+share/max(float(ratio),1e-8))
    return np.array([1-a,a,0.])


def prepare(variant,menu):
    data=old.prepare(variant);previous=None
    for m in data['months']:
        neutral=m['targets'][3]
        m['ratio']=neutral[0]/max(neutral[1],1e-12)
        f=dict(zip(old.FEATURES,m['features']))
        f['relative_sharpe63']=f['atm_sharpe63']-f['pair_sharpe63']
        m['state']=np.array([f[k] for k in FEATURES])
        m['targets']=np.stack([target(m['ratio'],s) for s in CFG['menus'][menu]])
        m['before']=np.tile([1.,0.,0.],(3,1)) if previous is None else np.stack([old.drift(w,previous['gross'][1:]) for w in previous['targets']])
        m['paths']=np.stack([[old.month_path(m['gross'],m['cash'],m['cost_rate'],p,a)[0]
                             for a in m['targets']] for p in m['before']])
        m['sums']=m['paths'].sum(axis=2);m['squares']=(m['paths']**2).sum(axis=2)
        previous=m
    return data


def contexts(values,leaves):
    values=np.atleast_2d(values);result=np.zeros(len(values),int);assigned=np.zeros(len(values),bool)
    for i,conditions in enumerate(leaves):
        keep=np.ones(len(values),bool)
        for feature,threshold,above in conditions:
            keep &= values[:,feature]>threshold if above else values[:,feature]<=threshold
        result[keep]=i;assigned |= keep
    assert assigned.all()
    return result


def partition(x,y,kind):
    minleaf=CFG['minimum_leaf_months']
    if kind in ['vix','momentum']:
        j=1 if kind=='vix' else 0;t=float(np.median(x[:,j]));left=x[:,j]<=t
        return [[[j,t,False]],[[j,t,True]]] if min(left.sum(),(~left).sum())>=minleaf else [[]]
    leaves=[dict(idx=np.arange(len(y)),conditions=[])]
    while len(leaves)<int(kind[-1]):
        best=None
        for leaf,record in enumerate(leaves):
            idx=record['idx'];v=y[idx]
            if len(idx)<2*minleaf:continue
            parent=float(((v-v.mean())**2).sum())
            for j in range(x.shape[1]):
                for t in np.unique(np.quantile(x[idx,j],np.arange(.1,1,.1))):
                    left=x[idx,j]<=t
                    if min(left.sum(),(~left).sum())<minleaf:continue
                    a,b=v[left],v[~left]
                    gain=parent-float(((a-a.mean())**2).sum()+((b-b.mean())**2).sum())
                    if gain>1e-9 and (best is None or gain>best[0]):best=(gain,leaf,j,float(t),idx[left],idx[~left])
        if best is None:break
        _,leaf,j,t,a,b=best;conditions=leaves[leaf]['conditions']
        leaves[leaf]=dict(idx=a,conditions=conditions+[[j,t,False]])
        leaves.append(dict(idx=b,conditions=conditions+[[j,t,True]]))
    return [r['conditions'] for r in leaves]


def search(months,states,count):
    maps=np.array(list(itertools.product([1,0,2],repeat=count)),int)
    actions=maps[:,states];previous=np.column_stack([np.ones(len(maps),int),actions[:,:-1]])
    sums=np.stack([m['sums'] for m in months]);squares=np.stack([m['squares'] for m in months])
    columns=np.arange(len(months))[None,:]
    total=sums[columns,previous,actions].sum(axis=1)
    total2=squares[columns,previous,actions].sum(axis=1)
    n=sum(len(m['cash']) for m in months)
    variance=np.maximum((total2-total*total/n)/(n-1),1e-20)
    scores=np.sqrt(252)*(total/n)/np.sqrt(variance)
    objective=scores-CFG['tilt_penalty']*np.mean(actions!=1,axis=1)
    winner=int(np.argmax(objective))
    return maps[winner],float(scores[winner]),float(scores[0]),float(objective[winner]),len(maps)


def fit(data,index,window,kind):
    cutoff=data['months'][index]['cutoff']
    train=[m for m in data['months'][:index] if m['ready'] and m['cutoff']>=cutoff-pd.DateOffset(months=window)]
    if len(train)<CFG['minimum_training_months']:return None
    assert max(m['end'] for m in train)<cutoff
    x=np.stack([m['state'] for m in train])
    y=np.array([(m['paths'][1,2]-m['paths'][1,0]).sum()/max(np.sqrt((m['paths'][1,1]**2).sum()),1e-8) for m in train])
    leaves=partition(x,y,kind);states=contexts(x,leaves)
    choices,score,baseline,objective,n=search(train,states,len(leaves))
    return dict(leaves=leaves,choices=choices.tolist(),training_months=len(train),
                last_training_end=str(max(m['end'] for m in train).date()),cutoff=str(cutoff.date()),
                window=window,partition=kind,training_sharpe=score,balanced_training_sharpe=baseline,
                penalized_objective=objective,policies_enumerated=n,
                occupancy=np.bincount(states,minlength=len(leaves)).tolist())


def case_name(menu,window,kind):return f'{menu}_w{window}_{kind}'


def model_for(data,variant,menu,index,window,kind):
    m=data['months'][index];case=case_name(menu,window,kind)
    path=OUT/'models'/variant/case/(m['month']+'.json')
    if path.exists():return json.loads(path.read_text())
    model=fit(data,index,window,kind)
    if model is not None:old.dump(path,model)
    return model


def replay(data,variant,menu,case,stop,window=None,kind=None,ensemble=False,constant=None,cost_multiplier=1.):
    previous=np.array([1.,0.,0.]);daily=[];decisions=[]
    for index,m in enumerate(data['months']):
        if m['cutoff']>pd.Timestamp(stop):break
        share=.5;models=[];leaf_ids=[]
        if constant is not None:share=constant
        elif m['ready']:
            specifications=[(w,k) for w in CFG['windows'] for k in ['vix','momentum','tree2']] if ensemble else [(window,kind)]
            shares=[]
            for w,k in specifications:
                model=model_for(data,variant,menu,index,w,k)
                if model is None:shares.append(.5);continue
                assert pd.Timestamp(model['last_training_end'])<m['cutoff']
                leaf=int(contexts(m['state'],model['leaves'])[0]);action=model['choices'][leaf]
                shares.append(CFG['menus'][menu][action]);models.append(model);leaf_ids.append(leaf)
            share=float(np.mean(shares))
        allocation=target(m['ratio'],share) if m['ready'] else np.array([1.,0.,0.])
        r,after,cost=old.month_path(m['gross'],m['cash'],m['cost_rate'],previous,allocation,cost_multiplier)
        daily.append(pd.Series(r,index=data['pair'].index[m['indices']]))
        decisions.append(dict(date=m['cutoff'],feature_date=m['feature_date'],atm_risk_share=share,
            pair_weight=allocation[0],atm_weight=allocation[1],cash_weight=allocation[2],
            resizing_cost_fraction=cost,fitted=bool(models),model_count=len(models),
            last_training_end=max([m['last_training_end'] for m in models],default=None),
            minimum_training_months=min([m['training_months'] for m in models],default=0),
            leaves=leaf_ids,training_sharpe_mean=float(np.mean([m['training_sharpe'] for m in models])) if models else None))
        previous=after
    r=pd.concat(daily).rename('excess_return');d=pd.DataFrame(decisions)
    folder=OUT/'replays'/variant/case;folder.mkdir(parents=True,exist_ok=True)
    suffix='validation' if stop==CFG['validation_end'] else 'full'
    pd.DataFrame(dict(excess_return=r,cash_return=data['cash'].reindex(r.index))).to_parquet(folder/(suffix+'.parquet'))
    d.to_csv(folder/(suffix+'_decisions.csv'),index=False)
    return r,d


def check_sources(data,variant):
    paths=data['sources']+[CFG_PATH,Path(__file__),Path('scripts/report_atm_p25_combination.py')]
    record={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    marker=OUT/'source_hashes'/(variant+'.json')
    if marker.exists():assert json.loads(marker.read_text())==record,'Sources changed; use new experiment directory'
    else:old.dump(marker,record)


def validation():
    rows=[]
    for variant in ['tree18','fixed200']:
        for menu in CFG['menus']:
            data=prepare(variant,menu);check_sources(data,variant)
            for window in CFG['windows']:
                for kind in CFG['partitions']:
                    case=case_name(menu,window,kind)
                    r,_=replay(data,variant,menu,case,CFG['validation_end'],window,kind)
                    row=dict(atm_variant=variant,menu=menu,window=window,partition=kind,case=case,validation_sharpe=sharpe(r.iloc[1:]))
                    rows.append(row);pd.DataFrame(rows).to_csv(REPORT/'validation_candidates.csv',index=False)
                    old.progress(phase='validation',**row)
            case=menu+'_ensemble'
            r,_=replay(data,variant,menu,case,CFG['validation_end'],ensemble=True)
            rows.append(dict(atm_variant=variant,menu=menu,window=0,partition='ensemble',case=case,validation_sharpe=sharpe(r.iloc[1:])))
            pd.DataFrame(rows).to_csv(REPORT/'validation_candidates.csv',index=False)
            del data;gc.collect()


def select():
    rows=pd.read_csv(REPORT/'validation_candidates.csv');assert len(rows)==36
    selected={v:g.sort_values(['validation_sharpe','case'],ascending=[False,True]).iloc[0].to_dict() for v,g in rows.groupby('atm_variant')}
    ensembles={v:g.loc[g.partition.eq('ensemble')].sort_values('validation_sharpe',ascending=False).iloc[0].to_dict() for v,g in rows.groupby('atm_variant')}
    old.dump(REPORT/'selection.json',dict(selected=selected,ensemble=ensembles,validation_only=True,
        scope=CFG['scope'],source_hashes={v:json.loads((OUT/'source_hashes'/(v+'.json')).read_text()) for v in selected}))
    old.progress(phase='selection',selected=selected,ensemble=ensembles)


def full_replays():
    selection=json.loads((REPORT/'selection.json').read_text())
    for variant in ['tree18','fixed200']:
        for record in [selection['selected'][variant],selection['ensemble'][variant]]:
            data=prepare(variant,record['menu']);check_sources(data,variant)
            kwargs=dict(window=int(record['window']),kind=record['partition'],ensemble=record['partition']=='ensemble')
            r,d=replay(data,variant,record['menu'],record['case'],CFG['test_end'],**kwargs)
            saved=pd.read_parquet(OUT/'replays'/variant/record['case']/'validation.parquet',use_threads=False)
            np.testing.assert_allclose(r.loc[saved.index],saved.excess_return,atol=1e-14)
            stressed,sd=replay(data,variant,record['menu'],record['case']+'_cost5',CFG['test_end'],cost_multiplier=5.,**kwargs)
            np.testing.assert_allclose(d.atm_risk_share,sd.atm_risk_share)
            del data;gc.collect()
        data=prepare(variant,'balanced')
        for share in [.2,.35,.5,.65,.8]:
            case=f'constant_risk{int(share*100)}'
            r,_=replay(data,variant,'balanced',case,CFG['test_end'],constant=share)
            if share==.5:
                baseline=pd.read_parquet(old.OUT/'replays'/variant/'rolling_risk_balance'/'full.parquet',use_threads=False)
                np.testing.assert_allclose(r,baseline.excess_return,atol=1e-12)
        old.progress(phase='replay_complete',variant=variant)
        del data;gc.collect()


def report():
    selection=json.loads((REPORT/'selection.json').read_text());records=[];annual=[];intervals=[]
    for variant in ['tree18','fixed200']:
        cases=list(dict.fromkeys([selection[x][variant]['case']+suffix for x in ['selected','ensemble'] for suffix in ['','_cost5']]+[f'constant_risk{i}' for i in [20,35,50,65,80]]))
        books={}
        for case in cases:
            folder=OUT/'replays'/variant/case
            book=pd.read_parquet(folder/'full.parquet',use_threads=False);books[case]=book
            d=pd.read_csv(folder/'full_decisions.csv',parse_dates=['date','feature_date','last_training_end'])
            assert d.feature_date.iloc[1:].lt(d.date.iloc[1:]).all()
            assert d.loc[d.fitted,'last_training_end'].lt(d.loc[d.fitted,'date']).all()
            np.testing.assert_allclose(d[['pair_weight','atm_weight','cash_weight']].sum(axis=1),1)
            for period,start,stop in old.blend.PERIODS:
                part=d.loc[d.date.between(start,stop)]
                records.append(dict(atm_variant=variant,case=case,period=period,
                    **old.blend.stats(book.excess_return,book.cash_return,start,stop),
                    mean_atm_capital_pct=100*part.atm_weight.mean(),fitted_months=int(part.fitted.sum()),
                    resizing_cost_bps=float(part.resizing_cost_fraction.sum()*10000)))
            for year in range(2018,2026):annual.append(dict(atm_variant=variant,case=case,year=year,**old.blend.stats(book.excess_return,book.cash_return,f'{year}-01-01',min(f'{year}-12-31',CFG['test_end']))))
        result=pd.DataFrame(records)
        static=result.loc[result.atm_variant.eq(variant)&result.period.eq('validation')&result.case.str.startswith('constant_')].sort_values('annualized_excess_sharpe',ascending=False).iloc[0]['case']
        for family in ['selected','ensemble']:
            case=selection[family][variant]['case']
            for control in ['constant_risk50',static]:
                intervals.append(dict(atm_variant=variant,family=family,control=control,**paired_interval(books[case].excess_return.loc[CFG['test_start']:],books[control].excess_return.loc[CFG['test_start']:],block=63,draws=1000)))
    result=pd.DataFrame(records);result.to_csv(REPORT/'results.csv',index=False)
    pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False)
    pd.DataFrame(intervals).to_csv(REPORT/'sharpe_intervals.csv',index=False)
    old.dump(REPORT/'protocol.json',dict(config=CFG,selection=selection,
        optimizer='Exhaustive enumeration of all 3^K leaf actions, K at most 3. Maximize daily training Sharpe minus 0.05 times fraction of tilted months.',
        transition='Chosen action determines subsequent holdings, drift, next first-session P&L and next resizing cost.',
        evaluation='2018-2021 validation selects procedures. Monthly refits use completed data. Later historical period 2022-Aug2025 has previously been examined.',
        scope=CFG['scope']))
    print(result.loc[result.period.isin(['validation','test']),['atm_variant','case','period','annualized_excess_sharpe','annualized_growth_relative_to_cash_pct']].to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',required=True,choices=['test','validation','select','replay','report'])
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    if args.phase=='test':
        import unittest
        suite=unittest.defaultTestLoader.loadTestsFromNames(['tests.test_atm_p25_sharpe_policy','tests.test_research_memory_guard'])
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():raise SystemExit(1)
    else:dict(validation=validation,select=select,replay=full_replays,report=report)[args.phase]()


if __name__=='__main__':main()
