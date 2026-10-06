"""Train and evaluate fitted Q on actual contracts, using conda environment xfund.

Download/prepare with mfe_wrds first. Example:
  python -m scripts.run_contract_q --test-years 2024 --iterations 12
Existing IV-proxy Q results are not used for training or validation.
"""
import argparse
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor

from scripts.contract_engine import Settings, UnresolvedPosition
from scripts.contract_q_engine import ContractMarket, FEATURE_NAMES, WAIT, LONG, SHORT, EXIT, evaluate


@dataclass
class Graph:
    x: np.ndarray
    legal: np.ndarray
    date_index: np.ndarray
    source: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    next_state: np.ndarray
    terminal: np.ndarray
    resolved: np.ndarray
    errors: list
    full_hold: list
    dates: pd.DatetimeIndex


def build_graph(m):
    """All reachable single-ticket histories, without using future failures as action masks."""
    xs, masks, dates, edges, errors, full_hold = [], [], [], [], [], []
    def node(t,book):
        i = len(xs)
        xs.append(m.features(t,book))
        masks.append([a in m.legal(t,book) for a in range(4)])
        dates.append(t)
        return i
    for t in range(m.n):
        node(t,None)
    def edge(s,a,r,nxt,t,ok=True):
        edges.append((s,a,r,nxt,t==m.n-1,ok))
    for t in range(m.n-1):
        if t and t%750==0:
            print(f'  transition coverage: {t}/{m.n-1} signal dates',flush=True)
        edge(t,WAIT,0.,t+1,t+1)
        for action in (LONG,SHORT):
            if action not in m.legal(t,None):
                continue
            book,reward,info = m.step(t,None,action)
            total = reward
            if book is None:
                edge(t,action,reward,t+1,t+1)
                full_hold.append(dict(state=t,action=action,reward=reward,end=t+1,status=info['status']))
                continue
            j = t+1
            state = node(j,book)
            edge(t,action,reward,state,j)
            while book is not None and j<m.n-1:
                try:
                    _,er,_ = m.step(j,book,EXIT)
                    edge(state,EXIT,er,j+1,j+1)
                except UnresolvedPosition as exc:
                    edge(state,EXIT,np.nan,j+1,j+1,False)
                    errors.append(dict(date=str(m.dates[j+1].date()),entry=str(m.dates[t].date()),
                                       action='exit',error=str(exc)))
                try:
                    nxt,hr,detail = m.step(j,book,WAIT)
                except UnresolvedPosition as exc:
                    edge(state,WAIT,np.nan,j+1,j+1,False)
                    errors.append(dict(date=str(m.dates[j+1].date()),entry=str(m.dates[t].date()),
                                       action='hold',error=str(exc)))
                    break
                total += hr
                ns = j+1 if nxt is None else node(j+1,nxt)
                edge(state,WAIT,hr,ns,j+1)
                if nxt is None and detail['status']!='terminal_exit':
                    full_hold.append(dict(state=t,action=action,reward=total,end=j+1,status=detail['status']))
                state,book,j = ns,nxt,j+1
    e = np.asarray(edges,float)
    return Graph(np.asarray(xs,np.float32),np.asarray(masks,bool),np.asarray(dates,int),
                 e[:,0].astype(int),e[:,1].astype(int),e[:,2]*10000/m.cfg.capital,
                 e[:,3].astype(int),e[:,4].astype(bool),e[:,5].astype(bool),
                 errors,full_hold,m.dates)


class FittedQ:
    def __init__(self,gamma=.97,downside=0.,iterations=12,min_leaf=128,trees=16):
        self.gamma,self.downside,self.iterations,self.min_leaf,self.trees = gamma,downside,iterations,min_leaf,trees
        self.models = {}

    @classmethod
    def load(cls,path):
        with Path(path).open('rb') as stream:
            saved=pickle.load(stream)
        if tuple(saved['features'])!=FEATURE_NAMES:
            raise ValueError('Saved model feature schema mismatch')
        model=cls(gamma=saved['gamma'],downside=saved['downside'],iterations=len(saved['history']))
        model.models,model.scale,model.history=saved['models'],saved['scale'],saved['history']
        return model

    def predict(self,x):
        return np.column_stack([self.models[a].predict(x) if a in self.models else np.zeros(len(x)) for a in range(4)])

    def fit(self,g):
        valid = g.resolved
        if not valid.any():
            raise ValueError('No resolved contract transitions')
        self.scale = max(float(np.std(g.reward[valid])),.01)
        shaped = g.reward-self.downside*np.minimum(g.reward,0)**2/self.scale
        q = np.zeros((len(g.x),4))
        self.history = []
        for iteration in range(self.iterations):
            best = np.where(g.legal,q,-np.inf).max(axis=1)
            target = shaped + self.gamma*best[g.next_state]*(~g.terminal)
            models = {}
            for action in range(4):
                use = valid & (g.action==action)
                if not use.any():
                    continue
                # Counterfactual positions on a date do not count as independent dates.
                day = g.date_index[g.source[use]]
                counts = np.bincount(day,minlength=len(g.dates))
                weight = 1/counts[day]
                reg = ExtraTreesRegressor(n_estimators=self.trees,max_depth=6,
                    min_samples_leaf=self.min_leaf,random_state=2026,n_jobs=2)
                reg.fit(g.x[g.source[use]],target[use],sample_weight=weight)
                reg.n_jobs=1  # Single-row policy calls do not benefit from thread startup.
                models[action] = reg
            self.models = models
            update = self.predict(g.x)
            residual = float(np.max(abs(update-q)))
            self.history.append(residual)
            q = update
            if (iteration+1)%4==0:
                print(f'  fitted Q sweep {iteration+1}/{self.iterations}, max change {residual:.4f} bps',flush=True)
        return self

    def __call__(self,m,t,book):
        values = self.predict(m.features(t,book)[None,:])[0]
        legal = m.legal(t,book)
        return legal[int(np.argmax(values[list(legal)]))]


def constant_policy(kind):
    def decide(m,t,book):
        if book is not None:
            return WAIT
        if LONG not in m.legal(t,None) or kind=='cash':
            return WAIT
        x = m.features(t,None)
        score = x[2]-x[1] if kind=='higher_iv' else x[3]
        return LONG if score>0 else SHORT if score<0 else WAIT
    return decide


def sharpe(returns):
    a = np.asarray(returns,float)
    sd = np.std(a,ddof=1) if len(a)>1 else 0.
    return float(np.sqrt(252)*a.mean()/sd) if sd>1e-12 else 0.


class FixedHoldSearch:
    """Small tree + exhaustive policy search; same role as CEM, only 3^4 policies.

    Train score uses daily contract P&L on the graph, not overlapping RV-IV labels.
    Any policy hitting an unresolved training edge receives no valid score.
    """
    def fit(self,g):
        labels = pd.DataFrame(g.full_hold)
        target = labels.pivot(index='state',columns='action',values='reward').dropna()
        if len(target)<50 or LONG not in target or SHORT not in target:
            raise ValueError('Too few resolved fixed-hold entry outcomes')
        self.tree = DecisionTreeRegressor(max_leaf_nodes=4,min_samples_leaf=63,random_state=2026)
        self.tree.fit(g.x[target.index],target[LONG]-target[SHORT])
        self.ids = np.unique(self.tree.apply(g.x[target.index]))
        leaves = np.searchsorted(self.ids,self.tree.apply(g.x))
        # A terminal state has no edge and stops the trajectory.
        lookup = {(int(s),int(a)):i for i,(s,a) in enumerate(zip(g.source,g.action))}
        best = None
        for choices in itertools.product((WAIT,LONG,SHORT),repeat=len(self.ids)):
            state, pnl, ok = 0,np.zeros(len(g.dates)),True
            while g.date_index[state]<len(g.dates)-1:
                a = choices[leaves[state]] if g.x[state,19]==0 else WAIT
                if not g.legal[state,a]:
                    a = WAIT
                i = lookup[(state,a)]
                if not g.resolved[i]:
                    ok=False
                    break
                nxt = g.next_state[i]
                pnl[g.date_index[nxt]] += g.reward[i]
                state=nxt
            if ok:
                score=sharpe(pnl[1:])
                if best is None or score>best[0]:
                    best=(score,choices)
        if best is None:
            raise ValueError('No fully resolved training policy')
        self.score,self.policy = best
        return self

    def __call__(self,m,t,book):
        if book is not None:
            return WAIT
        leaf=np.searchsorted(self.ids,self.tree.apply(m.features(t,None)[None,:]))[0]
        a=self.policy[leaf]
        return a if a in m.legal(t,None) else WAIT


def market_slice(bundle,start,end,cfg):
    calendar=bundle['calendar']
    dates=calendar[(calendar>=pd.Timestamp(start))&(calendar<pd.Timestamp(end))]
    quotes=bundle['quotes']
    stocks=bundle['stocks']
    return ContractMarket(quotes[quotes.date.isin(dates)],stocks[stocks.date.isin(dates)],dates,
                          bundle['secids'],cfg,bundle['dividends'],bundle['candidates'])


def summarize(nav,error,cfg):
    result=dict(complete=error is None,error=error,sessions=len(nav),start=str(nav.index.min().date()),
                end=str(nav.index.max().date()),entries=int(nav.status.eq('entered').sum()))
    if error is not None:
        # Partial returns must not masquerade as a completed strategy backtest.
        return result
    r=nav.excess_return.iloc[1:]
    result.update(excess_sharpe=sharpe(r),total_return=float(nav.nav.iloc[-1]/cfg.capital-1),
                  excess_return_initial_capital=float((nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1])/cfg.capital),
                  annualized_excess_vol=float(r.std()*np.sqrt(252)),
                  max_drawdown=float((nav.nav/nav.nav.cummax().clip(lower=cfg.capital)-1).min()),
                  option_cost=float(nav.get('option_cost',pd.Series(dtype=float)).sum()),
                  hedge_cost=float(nav.get('hedge_cost',pd.Series(dtype=float)).sum()))
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,default=Path('data/q_contracts/AMD_NVDA/prepared.pkl'))
    ap.add_argument('--test-years',nargs='+',type=int,default=[2024])
    ap.add_argument('--iterations',type=int,default=12)
    ap.add_argument('--out',type=Path,default=Path('artifacts/contract_q/AMD_NVDA'))
    ap.add_argument('--downside',nargs='+',type=float,default=[0.,.1])
    ap.add_argument('--replay-only',action='store_true',help='Replay saved validation-selected policies; do not refit or reselect')
    args=ap.parse_args()
    if not args.data.exists():
        ap.error('Prepare the actual contract data in mfe_wrds first; see reports/contract_q_plan.md')
    if args.iterations<1 or any(x<0 for x in args.downside):
        ap.error('Iterations must be positive and downside penalties nonnegative')
    # Only the locally created, licensed bundle is accepted by this research CLI.
    bundle=pd.read_pickle(args.data)
    cfg=Settings(**bundle['settings'])
    args.out.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(args.data.read_bytes()).hexdigest()
    manifest=json.loads(args.data.with_suffix('.json').read_text())
    if digest!=manifest['sha256']:
        raise ValueError('Prepared data checksum differs from its manifest')
    provenance=dict(
        prepared_sha256=digest,settings=vars(cfg),test_years=args.test_years,
        iterations=args.iterations,downside_grid=args.downside,
        code_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
            ('scripts/contract_q_engine.py','scripts/run_contract_q.py','scripts/prepare_q_contract_data.py')})
    if args.replay_only:
        original=json.loads((args.out/'run_manifest.json').read_text())
        if original['prepared_sha256']!=digest or original['settings']!=vars(cfg):
            raise ValueError('Replay data/settings differ from the fitted run')
        for year in args.test_years:
            fold=args.out/str(year)
            selected=json.loads((fold/'selected_policy.json').read_text())['model']
            policy=constant_policy('cash') if selected=='cash' else FittedQ.load(fold/f'{selected}.pkl')
            m=market_slice(bundle,f'{year}-01-01',f'{year+1}-01-01',cfg)
            nav,error=evaluate(m,policy)
            nav.to_csv(fold/'replay_selected_q.csv')
            (fold/'replay_manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')
            print(json.dumps(dict(fold=year,selected_model=selected,**summarize(nav,error,cfg))))
        return
    (args.out/'run_manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')
    summary=[]
    for year in args.test_years:
        fold=args.out/str(year)
        fold.mkdir(exist_ok=True)
        train=market_slice(bundle,bundle['calendar'][0],f'{year-1}-01-01',cfg)
        validation=market_slice(bundle,f'{year-1}-01-01',f'{year}-01-01',cfg)
        test=market_slice(bundle,f'{year}-01-01',f'{year+1}-01-01',cfg)
        if train.n<3*252 or validation.n<200:
            raise ValueError('Require at least three training years and one validation year')
        print(f'{year}: build training transitions from {train.n} sessions',flush=True)
        graph=build_graph(train)
        pd.DataFrame(graph.errors).to_csv(fold/'training_unresolved.csv',index=False)
        audit=dict(states=len(graph.x),transitions=len(graph.source),
                   resolved=int(graph.resolved.sum()),unresolved=int((~graph.resolved).sum()),
                   features=list(FEATURE_NAMES),train_start=str(train.dates[0].date()),
                   train_end=str(train.dates[-1].date()),validation_start=str(validation.dates[0].date()),
                   test_start=str(test.dates[0].date()),iterations=args.iterations,downside_grid=args.downside)
        (fold/'training_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        print(f'{year}: {audit["states"]:,} states; {audit["unresolved"]:,} unresolved action outcomes',flush=True)
        candidates=[('cash',constant_policy('cash'))]
        for downside in args.downside:
            print(f'{year}: fitting Q, downside penalty {downside}',flush=True)
            policy=FittedQ(downside=downside,iterations=args.iterations).fit(graph)
            candidates.append((f'q_downside_{downside:g}',policy))
            with (fold/f'q_downside_{downside:g}.pkl').open('wb') as f:
                pickle.dump(dict(models=policy.models,gamma=policy.gamma,downside=policy.downside,
                                 scale=policy.scale,history=policy.history,features=FEATURE_NAMES),f)
        selection=[]
        best=(0.,'cash',candidates[0][1])  # Cash is a genuine validation candidate.
        for name,policy in candidates:
            nav,error=evaluate(validation,policy)
            nav.to_csv(fold/f'validation_{name}.csv')
            row=dict(model=name,**summarize(nav,error,cfg))
            selection.append(row)
            if row['complete'] and row['excess_sharpe']>best[0]:
                best=(row['excess_sharpe'],name,policy)
        (fold/'validation_selection.json').write_text(json.dumps(selection,indent=2)+'\n')
        _,selected_name,selected=best
        (fold/'selected_policy.json').write_text(json.dumps(dict(model=selected_name,validation_sharpe=best[0]),indent=2)+'\n')
        print(f'{year}: validation selected {selected_name}; evaluating frozen policies',flush=True)
        baselines=[('higher_iv',constant_policy('higher_iv')),('relative_carry',constant_policy('relative_carry'))]
        search=FixedHoldSearch().fit(graph)
        baselines.append(('fixed_hold_search',search))
        for name,policy in [('selected_q',selected)]+candidates+baselines:
            nav,error=evaluate(test,policy)
            nav.to_csv(fold/f'test_{name}.csv')
            row=dict(fold=year,model=name,selected_model=selected_name if name=='selected_q' else None,
                     **summarize(nav,error,cfg))
            summary.append(row)
            print(json.dumps(row),flush=True)
        (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    main()
