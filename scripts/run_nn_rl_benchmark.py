"""Matched supervised NN/full-size versus Sharpe policy/sizing benchmark."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from scripts.run_first_pass import fit_policy, net_rewards, payoff_sharpe
from scripts.run_four_pairs import FEATURES, LADDER, entry_cost
from scripts.run_rv_vix_pass import leaf_states
from scripts.run_walk_forward import fold_data
from scripts.explore_etf_pairs import eligible_fold
from scripts.explore_universe_gap import block_difference

NEW_FEATURES = FEATURES + ['rv_adjusted_value', 'spread_reversion']
MODELS = ['rl_base_sized', 'nn_both_fixed', 'rl_both_fixed', 'rl_both_sized']


def add_features(panel):
    p = panel.copy()
    past = p.spread.shift(1).rolling(252, min_periods=126)
    p['rv_adjusted_value'] = (p.trailing_rv_amd-p.iv_amd)-(p.trailing_rv_nvda-p.iv_nvda)
    p['spread_reversion'] = (past.mean()-p.spread)/past.std().replace(0, np.nan)
    return p.replace([np.inf, -np.inf], np.nan).dropna(subset=NEW_FEATURES)


def nn_actions(prediction, cost, multiplier):
    prediction = np.asarray(prediction)
    return np.where(np.abs(prediction) > multiplier*np.asarray(cost), np.sign(prediction), 0.)


def fit_nn(train, validation, test):
    """One hidden layer; training-only scalers, validation-only selection."""
    scaler = StandardScaler().fit(train[NEW_FEATURES])
    xtr = scaler.transform(train[NEW_FEATURES]); xv = scaler.transform(validation[NEW_FEATURES])
    xt = scaler.transform(test[NEW_FEATURES])
    y = train.long_reward.to_numpy(); center = y.mean(); scale = max(y.std(), 1e-8)
    candidates = []; models = {}
    for alpha in [.01, 1.]:
        model = MLPRegressor(hidden_layer_sizes=(16,), activation='tanh', solver='lbfgs',
                             alpha=alpha, max_iter=10000, max_fun=200000, tol=1e-4, random_state=2026)
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(xtr, (y-center)/scale)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in messages)
        vp = model.predict(xv)*scale+center
        models[alpha] = model
        for multiplier in [1., 2., 4.]:
            action = nn_actions(vp, entry_cost(validation, .025), multiplier)
            score = payoff_sharpe(net_rewards(action, validation.long_reward.to_numpy(), entry_cost(validation, .025)))
            candidates.append(dict(model='nn_both_fixed', alpha=alpha, threshold=multiplier,
                                   validation_sharpe=score, converged=converged, iterations=model.n_iter_))
    chosen = max(candidates, key=lambda c:c['validation_sharpe'])
    model = models[chosen['alpha']]
    action = nn_actions(model.predict(xt)*scale+center, entry_cost(test, .025), chosen['threshold'])
    return action, chosen, candidates


def fit_trees(train, validation, test, features, include_fixed):
    candidates=[]; fits=[]
    trcost=entry_cost(train,.025); vacost=entry_cost(validation,.025)
    for leaves in [4,8,16]:
        for minimum in [126,252]:
            tree=DecisionTreeRegressor(max_leaf_nodes=leaves,min_samples_leaf=minimum,random_state=2026)
            tree.fit(train[features],train.long_reward)
            ids=np.unique(tree.apply(train[features])); states=leaf_states(tree,train,features,ids)
            fixed,fs=max([fit_policy(states,train.long_reward.to_numpy(),len(ids),seed,cost=trcost) for seed in range(5)],key=lambda p:p[1])
            sized,ss=max([fit_policy(states,train.long_reward.to_numpy(),len(ids),seed,cost=trcost,action_values=LADDER,initial_policy=fixed) for seed in range(5)],key=lambda p:p[1])
            assert ss >= fs-1e-10
            for kind,policy,score in [('sized',sized,ss)]+([('fixed',fixed,fs)] if include_fixed else []):
                va=policy[leaf_states(tree,validation,features,ids)]
                cfg=dict(model=('rl_both_' if include_fixed else 'rl_base_')+kind,leaves=leaves,min_leaf=minimum,train_sharpe=score,
                         validation_sharpe=payoff_sharpe(net_rewards(va,validation.long_reward.to_numpy(),vacost)))
                candidates.append(cfg); fits.append((cfg,tree,ids,policy))
    selected={}
    for name in dict.fromkeys(c['model'] for c in candidates):
        cfg,tree,ids,policy=max([f for f in fits if f[0]['model']==name],key=lambda f:f[0]['validation_sharpe'])
        selected[name]=(policy[leaf_states(tree,test,features,ids)],cfg)
    return selected,candidates


def run_pair(pair, source, output, resume, refresh_nn=False):
    root=Path(source); out=Path(output)/pair; out.mkdir(exist_ok=True)
    panel=add_features(pd.read_parquet(root/'panels.parquet').xs(pair).sort_index())
    original=pd.read_csv(root/f'{pair}.csv',index_col='date',parse_dates=['date'])
    pieces=[]; configs=[]; candidates=[]
    for year in sorted(original.fold.unique()):
        path=out/f'{year}.csv'; cp=out/f'{year}_configs.json'; ap=out/f'{year}_candidates.json'
        if resume and not refresh_nn and path.exists() and cp.exists() and ap.exists():
            pieces.append(pd.read_csv(path,index_col='date',parse_dates=['date','end']))
            configs.extend(json.loads(cp.read_text())); candidates.extend(json.loads(ap.read_text())); continue
        train,validation,test=fold_data(panel,int(year))
        if not eligible_fold(train,validation,test):
            print(f'Skipping {pair} {year}: insufficient history after feature warmup ({len(train)}/{len(validation)}/{len(test)})',flush=True)
            continue
        assert train.end.max()<validation.index.min() and validation.end.max()<test.index.min()
        expected=original.index[original.fold.eq(year)]
        assert test.index.equals(expected), (pair,year,'test dates changed')
        result=test[['end']].copy(); result['fold']=year
        if refresh_nn and path.exists() and cp.exists() and ap.exists():
            old=pd.read_csv(path,index_col='date',parse_dates=['date','end'])
            if not old.index.equals(test.index):raise ValueError('Refresh dates changed')
            old_cfg=json.loads(cp.read_text())
            selected={c['model']:(old[c['model']+'_action'].to_numpy(),{k:v for k,v in c.items() if k in ['model','leaves','min_leaf','train_sharpe','validation_sharpe']}) for c in old_cfg if c['model']!='nn_both_fixed'}
            cs=[{k:v for k,v in c.items() if k not in ['pair','fold']} for c in json.loads(ap.read_text()) if c['model']!='nn_both_fixed']
        else:
            selected,cs=fit_trees(train,validation,test,FEATURES,False)
            both,bs=fit_trees(train,validation,test,NEW_FEATURES,True); selected.update(both); cs.extend(bs)
        action,cfg,nc=fit_nn(train,validation,test); selected['nn_both_fixed']=(action,cfg); cs.extend(nc)
        fold_configs=[]
        for name,(action,cfg) in selected.items():
            assert np.isin(action,LADDER if name.endswith('sized') else [-1.,0.,1.]).all()
            result[name+'_action']=action
            result[name+'_reward']=net_rewards(action,test.long_reward.to_numpy(),entry_cost(test,.025))
            fold_configs.append(dict(pair=pair,fold=int(year),**cfg,train_entries=len(train),validation_entries=len(validation),test_entries=len(test),
                train_last_outcome=str(train.end.max().date()),validation_first=str(validation.index.min().date()),
                validation_last_outcome=str(validation.end.max().date()),test_first=str(test.index.min().date())))
        all_candidates=[dict(pair=pair,fold=int(year),**c) for c in cs]
        result.to_csv(path); cp.write_text(json.dumps(fold_configs,indent=2)+'\n'); ap.write_text(json.dumps(all_candidates,indent=2)+'\n')
        pieces.append(result); configs.extend(fold_configs); candidates.extend(all_candidates)
        print(f'Completed {pair} {year}',flush=True)
    pred=pd.concat(pieces).sort_index(); pred.to_csv(Path(output)/f'{pair}.csv')
    return pair,configs,candidates


def summarize(output,reports,pairs,configs,candidates):
    scores=[];annual=[];predictions={}
    for pair in pairs:
        pred=pd.read_csv(output/f'{pair}.csv',index_col='date',parse_dates=['date']);predictions[pair]=pred
        for model in MODELS:
            r=pred[model+'_reward'];a=pred[model+'_action']
            scores.append(dict(pair=pair,model=model,entries=len(r),sharpe=payoff_sharpe(r),net_pnl=r.sum(),trade_fraction=a.ne(0).mean(),mean_absolute_size=a.abs().mean()))
            for year,g in r.groupby(r.index.year):annual.append(dict(pair=pair,model=model,year=year,sharpe=payoff_sharpe(g),net_pnl=g.sum()))
    stocks=json.loads(Path('config/universe_gap_pairs.json').read_text())
    etfs=json.loads(Path('config/etf_pairs.json').read_text())['pairs']
    groups={'stocks':[p['pair'] for p in stocks],'etf_stock':[p['pair'] for p in etfs if p['kind']=='ETF_stock' and not p['short_history']]}
    groups['combined']=groups['stocks']+groups['etf_stock']
    groups['combined_without_silver']=[p for p in groups['combined'] if not p.startswith('SIL_')]
    portfolio=[];portfolio_annual=[];contrasts=[]
    for group,names in groups.items():
        if any(n not in predictions for n in names):continue
        common=predictions[names[0]].index
        for pair in names[1:]:common=common.intersection(predictions[pair].index)
        returns={m:pd.concat([predictions[p].loc[common,m+'_reward'] for p in names],axis=1).mean(axis=1) for m in MODELS}
        for model,r in returns.items():
            portfolio.append(dict(group=group,model=model,pairs=len(names),entries=len(r),first=str(r.index.min().date()),last=str(r.index.max().date()),sharpe=payoff_sharpe(r),net_pnl=r.sum()))
            for year,g in r.groupby(r.index.year):portfolio_annual.append(dict(group=group,model=model,year=year,sharpe=payoff_sharpe(g),net_pnl=g.sum()))
        for left,right in [('rl_both_sized','nn_both_fixed'),('rl_both_sized','rl_base_sized'),('rl_both_sized','rl_both_fixed')]:
            ci=block_difference(returns[left].to_numpy(),returns[right].to_numpy(),repeats=2000)
            contrasts.append(dict(group=group,left=left,right=right,sharpe_difference=payoff_sharpe(returns[left])-payoff_sharpe(returns[right]),ci_low=ci[0],ci_high=ci[1]))
    for name,rows in [('scores',scores),('annual',annual),('configs',configs),('candidates',candidates),('portfolio',portfolio),('portfolio_annual',portfolio_annual),('contrasts',contrasts)]:
        pd.DataFrame(rows).to_csv(reports/f'{name}.csv',index=False)
    print(pd.DataFrame(portfolio).to_string(index=False),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=3);ap.add_argument('--resume',action='store_true')
    ap.add_argument('--refresh-nn',action='store_true',help='Refit NN only; preserve saved tree actions for a training-budget audit.')
    ap.add_argument('--pairs',nargs='+');ap.add_argument('--output',type=Path,default=Path('artifacts/nn_rl_benchmark'))
    ap.add_argument('--reports',type=Path,default=Path('reports/nn_rl_benchmark'));args=ap.parse_args()
    args.output.mkdir(exist_ok=True,parents=True);args.reports.mkdir(exist_ok=True,parents=True)
    sources={}
    for root in [Path('artifacts/universe_gap'),Path('artifacts/etf_exploration/run')]:
        for pair in pd.read_parquet(root/'panels.parquet').index.get_level_values('pair').unique():sources[pair]=str(root)
    if args.pairs:sources={p:sources[p] for p in args.pairs}
    settings=dict(features=NEW_FEATURES,models=MODELS,cost_rate=.025,ladder=LADDER.tolist(),nn_hidden=[16],nn_activation='tanh',nn_solver='lbfgs',nn_alpha=[.01,1.],nn_threshold_cost_multipliers=[1,2,4],nn_seed=2026,nn_max_iter=10000,nn_max_fun=200000,nn_tolerance=1e-4,
                  sources={r:hashlib.sha256((Path(r)/'panels.parquet').read_bytes()).hexdigest() for r in sorted(set(sources.values()))},
                  source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    sp=args.output/'settings.json'
    if sp.exists() and json.loads(sp.read_text())!=settings:
        previous=json.loads(sp.read_text())
        mutable={'nn_max_iter','nn_max_fun','nn_tolerance','source_code_sha256'}
        if not args.refresh_nn or {k:v for k,v in previous.items() if k not in mutable}!={k:v for k,v in settings.items() if k not in mutable}:
            raise ValueError('Settings differ: choose another output directory')
        (args.output/'settings_before_nn_refresh.json').write_text(sp.read_text())
    sp.write_text(json.dumps(settings,indent=2)+'\n');(args.reports/'settings.json').write_text(sp.read_text())
    start=time.monotonic(); configs=[];candidates=[]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures=[executor.submit(run_pair,p,r,str(args.output),args.resume,args.refresh_nn) for p,r in sources.items()]
        for future in as_completed(futures):
            pair,cs,ca=future.result();configs.extend(cs);candidates.extend(ca)
            print(f'Finished {pair}; elapsed {(time.monotonic()-start)/60:.1f} minutes',flush=True)
    summarize(args.output,args.reports,list(sources),configs,candidates)


if __name__=='__main__': main()
