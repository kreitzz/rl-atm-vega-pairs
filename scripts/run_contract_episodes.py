"""Finite contract episodes: fixed-horizon entry and separate fitted-Q exits.

Preset chronological blocks: forecasts/exits 2010-18, entries 2019-22,
selection 2023, frozen evaluation 2024 and available 2025. Run in xfund.
"""
import json
import hashlib
import pickle
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from scripts.contract_engine import Settings, integer_size
from scripts.contract_q_engine import WAIT,LONG,SHORT,EXIT,evaluate
from scripts.run_contract_q import market_slice,summarize,constant_policy


def regressor():
    return ExtraTreesRegressor(n_estimators=64,max_depth=5,min_samples_leaf=40,
                               random_state=2026,n_jobs=2)


def discounted_step(reward,growth,next_value=0.):
    """Value at decision time; only cash discounting, no preference discount."""
    return (reward+next_value)/growth


class Forecasts:
    def fit(self,bundle,m):
        returns=bundle['stocks'].pivot(index='date',columns='secid',values='return').reindex(columns=m.secids).astype(float)
        sq=np.log1p(returns).pow(2)*252*10000
        lag=pd.concat([sq.rolling(w,min_periods=w).mean() for w in (5,21,63)],axis=1)
        self.lag=lag.replace([np.inf,-np.inf],np.nan).fillna(0)
        future=sq.shift(-1).iloc[::-1].rolling(10,min_periods=10).mean().iloc[::-1]
        train=(lag.index<pd.Timestamp('2019-01-01')) & (lag.index>=pd.Timestamp('2010-04-01'))
        # Purge every outcome whose endpoint crosses the forecast training block.
        endpoints=pd.Series(lag.index,index=lag.index).shift(-10)
        train &= endpoints.lt('2019-01-01').to_numpy() & future.notna().all(axis=1).to_numpy()
        logs=np.log(future.clip(lower=1))
        target=np.column_stack([(logs.iloc[:,0]+logs.iloc[:,1])/2,logs.iloc[:,0]-logs.iloc[:,1]])
        self.variance=make_pipeline(StandardScaler(),Ridge(alpha=100.))
        self.variance.fit(self.lag.loc[train],target[train])
        xx,yy=[],[]
        for t in range(m.n-10):
            if m.dates[t+10]>=pd.Timestamp('2019-01-01') or LONG not in m.legal(t,None):
                continue
            ids=m.candidates[t]
            later=[m.quotes.get((m.dates[t+10],oid)) for oid in ids]
            if any(q is None or not np.isfinite(q['impl_volatility']) or q['impl_volatility']<=0 for q in later):
                continue
            x=m.features(t,None)
            xx.append(np.r_[x[:19],self.lag.loc[m.dates[t]].to_numpy()])
            yy.append([q['impl_volatility']*100-x[i+1] for i,q in enumerate(later)])
        self.iv=make_pipeline(StandardScaler(),Ridge(alpha=100.))
        self.iv.fit(xx,yy)
        self.audit=dict(variance_rows=int(train.sum()),iv_rows=len(xx),last_outcome='2018-12-31',
                        variance_target='10-session log variance: common level and inter-asset difference')
        return self

    def attach(self,m):
        lag=self.lag.reindex(m.dates).to_numpy(float)
        pred=self.variance.predict(lag)
        rv=np.sqrt(np.exp(np.clip(np.column_stack([pred[:,0]+pred[:,1]/2,pred[:,0]-pred[:,1]/2]),0,12)))
        self.cache=(m,lag,rv)
        self.extra_cache={}

    def features(self,m,t,book):
        if not hasattr(self,'cache') or self.cache[0] is not m:
            self.attach(m)
        x=m.features(t,book)
        _,lag,rv=self.cache
        ids=m.candidates[t] if book is None else book.optionids
        key=(t,ids)
        if key in self.extra_cache:
            return np.r_[x,self.extra_cache[key]]
        iv_change=self.iv.predict(np.r_[x[:19],lag[t]][None,:])[0]
        carry=(rv[t]**2-x[1:3]**2)/(2*np.maximum(x[1:3],1))
        cost=0.
        if ids is not None:
            qs=[m.quote(t,oid) for oid in ids]
            qty=integer_size([q['vega']*100 for q in qs],m.cfg.entry_vega_per_leg,m.cfg.max_entry_vega_imbalance)
            if qty is not None:
                # Today's estimated option round-trip cost, not future realized cost.
                cost=sum(n*(100*(q['ask']-q['bid'])+2*m.cfg.option_commission) for n,q in zip(qty,qs))
        extra=np.r_[rv[t],iv_change,carry,carry[0]-carry[1],cost/m.cfg.entry_vega_per_leg,
                    x[1]-x[2],lag[t]]
        self.extra_cache[key]=extra
        return np.r_[x,extra]


def episodes(m,forecasts,horizon):
    """All entry directions; fixed IDs and every possible subsequent exit.

    Future failures raise rather than silently selecting successful paths.
    Entry rejections remain zero-outcome attempts. Complete horizons are purged
    at the block boundary even when an earlier safety exit would be observed.
    """
    rows=[]
    for t in range(m.n-horizon-1):
        if LONG not in m.legal(t,None):
            continue
        x=forecasts.features(m,t,None)
        for action in (LONG,SHORT):
            book,r,info=m.step(t,None,action)
            row=dict(t=t,x=x,action=action,reward=r,growth=1+info['cash_return'],steps=[],status=info['status'])
            j=t+1
            while book is not None:
                age=j-(t+1)
                sx=forecasts.features(m,j,book)
                _,er,ei=m.step(j,book,EXIT)
                nb,hr,hi=m.step(j,book,WAIT)
                row['steps'].append(dict(age=age,x=sx,exit_reward=er,hold_reward=hr,
                    growth=1+hi['cash_return'],terminal=nb is None or age==horizon-1,date=j))
                book=None if age==horizon-1 else nb
                j+=1
            rows.append(row)
        if t and t%750==0:
            print(f'  episodes {m.dates[t].date()}, H={horizon}',flush=True)
    return rows


def episode_value(row,exit_models=None):
    """Realized value under a fixed policy, never the hindsight-best exit."""
    rs=[(row['reward'],row['growth'])]
    for step in row['steps']:
        close=step['terminal']
        if exit_models is not None and not close:
            q=exit_models[step['age']].predict(step['x'][None,:])[0]
            close=q[0]>=q[1]
        rs.append((step['exit_reward'] if close else step['hold_reward'],step['growth']))
        if close:
            break
    value=0.
    for r,g in reversed(rs):
        value=discounted_step(r,g,value)
    return value


def fit_exits(rows,horizon):
    models={}
    for age in reversed(range(horizon)):
        samples=[(row,i,s) for row in rows for i,s in enumerate(row['steps']) if s['age']==age]
        x,y,days=[],[],[]
        continuing=[(row,i,s) for row,i,s in samples if not s['terminal']]
        future_values=iter(models[age+1].predict(np.array([row['steps'][i+1]['x'] for row,i,s in continuing])).max(axis=1)
                           if continuing else [])
        for row,i,s in samples:
            exit_v=discounted_step(s['exit_reward'],s['growth'])
            if s['terminal']:
                hold_v=exit_v
            else:
                future=next(future_values)
                hold_v=discounted_step(s['hold_reward'],s['growth'],future)
            x.append(s['x']); y.append([exit_v,hold_v]); days.append(s['date'])
        counts=Counter(days)
        model=regressor().fit(x,y,sample_weight=[1/counts[d] for d in days])
        model.n_jobs=1
        models[age]=model
        print(f'    fitted exit age {age}, {len(samples)} states',flush=True)
    return models


class EpisodePolicy:
    def __init__(self,forecasts,horizon,entry_models,exit_models=None,threshold=0.):
        self.forecasts,self.horizon,self.entry_models,self.exit_models,self.threshold=forecasts,horizon,entry_models,exit_models,threshold

    def entry_values(self,x):
        return np.array([0.,self.entry_models[LONG].predict(x[None,:])[0],
                         self.entry_models[SHORT].predict(x[None,:])[0]])

    def __call__(self,m,t,book):
        if book is None:
            if LONG not in m.legal(t,None) or t+self.horizon+1>=m.n:
                return WAIT
            q=self.entry_values(self.forecasts.features(m,t,None))
            return int(np.argmax(q)) if q[1:].max()>self.threshold else WAIT
        age=t-book.entry
        if age>=self.horizon-1:
            return EXIT
        if self.exit_models is None:
            return WAIT
        q=self.exit_models[age].predict(self.forecasts.features(m,t,book)[None,:])[0]
        return EXIT if q[0]>=q[1] else WAIT


def fit_entries(rows,forecasts,horizon,exit_models=None):
    models={}
    # Batch predictions once; labels still follow a frozen policy, not an oracle.
    predicted={}
    if exit_models is not None:
        for age,model in exit_models.items():
            steps=[s for row in rows for s in row['steps'] if s['age']==age and not s['terminal']]
            if steps:
                values=model.predict(np.array([s['x'] for s in steps]))
                predicted.update({id(s):v[0]>=v[1] for s,v in zip(steps,values)})
    def label(row):
        rs=[(row['reward'],row['growth'])]
        for s in row['steps']:
            close=s['terminal'] or predicted.get(id(s),False)
            rs.append((s['exit_reward'] if close else s['hold_reward'],s['growth']))
            if close: break
        value=0.
        for r,g in reversed(rs): value=discounted_step(r,g,value)
        return value
    for action in (LONG,SHORT):
        selected=[r for r in rows if r['action']==action]
        model=regressor().fit([r['x'] for r in selected],[label(r) for r in selected])
        model.n_jobs=1
        models[action]=model
    return EpisodePolicy(forecasts,horizon,models,exit_models)


def main():
    out=Path('artifacts/contract_episodes'); out.mkdir(exist_ok=True,parents=True)
    data=Path('data/q_contracts/AMD_NVDA/prepared.pkl')
    bundle=pd.read_pickle(data); cfg=Settings(**bundle['settings'])
    manifest=dict(data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),settings=vars(cfg),
        horizons=[5,10],threshold_dollars=[0,250],exit_training='2010-2018',entry_training='2019-2022',
        validation='2023',test=['2024','2025 available'],cash_ties=True)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    early=market_slice(bundle,'2010-01-01','2019-01-01',cfg)
    entry=market_slice(bundle,'2019-01-01','2023-01-01',cfg)
    validation=market_slice(bundle,'2023-01-01','2024-01-01',cfg)
    forecasts=Forecasts().fit(bundle,early)
    candidates={}; audit={}
    for h in manifest['horizons']:
        print(f'Build H={h} early episodes',flush=True)
        erows=episodes(early,forecasts,h)
        print(f'Fit H={h} exit Q on {len(erows)} episodes',flush=True)
        exits=fit_exits(erows,h)
        print(f'Build H={h} entry episodes',flush=True)
        rows=episodes(entry,forecasts,h)
        audit[str(h)]=dict(exit_episodes=len(erows),entry_episodes=len(rows),
            entry_statuses=dict(Counter(r['status'] for r in rows)),
            mean_fixed_target=float(np.mean([episode_value(r) for r in rows])))
        for kind,ex in [('fixed',None),('q_exit',exits)]:
            fitted=fit_entries(rows,forecasts,h,ex)
            for threshold in manifest['threshold_dollars']:
                name=f'{kind}_h{h}_buffer{threshold}'
                candidates[name]=EpisodePolicy(forecasts,h,fitted.entry_models,ex,threshold)
    validation_rows=[]; selected={}
    for name,policy in candidates.items():
        nav,error=evaluate(validation,policy); nav.to_csv(out/f'validation_{name}.csv')
        result=dict(model=name,**summarize(nav,error,cfg)); validation_rows.append(result)
        print('VALIDATION '+json.dumps(result),flush=True)
    # Independent selectors allow a direct comparison of the two model families.
    for family in ('fixed','q_exit'):
        good=[r for r in validation_rows if r['model'].startswith(family) and r['complete'] and r['excess_sharpe']>0]
        selected[family]=max(good,key=lambda r:r['excess_sharpe'])['model'] if good else 'cash'
    (out/'selection.json').write_text(json.dumps(dict(selected=selected,validation=validation_rows),indent=2))
    # Freeze candidates and selection on disk before reading test outcomes.
    if hasattr(forecasts,'cache'): del forecasts.cache
    with (out/'models.pkl').open('wb') as f: pickle.dump(candidates,f)
    (out/'training_audit.json').write_text(json.dumps(dict(forecasts=forecasts.audit,episodes=audit),indent=2))
    results=[]
    for year in (2024,2025):
        m=market_slice(bundle,f'{year}-01-01',f'{year+1}-01-01',cfg)
        policies=dict(candidates)
        for family,name in selected.items():
            policies[f'selected_{family}']=constant_policy('cash') if name=='cash' else candidates[name]
        policies.update({k:constant_policy(k) for k in ('cash','higher_iv','relative_carry')})
        for name,policy in policies.items():
            nav,error=evaluate(m,policy); nav.to_csv(out/f'{year}_{name}.csv')
            result=dict(year=year,model=name,**summarize(nav,error,cfg)); results.append(result)
            print('TEST '+json.dumps(result),flush=True)
    (out/'summary.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
