"""Broad candidate universe: independent policies condition on past OOS profits."""
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

from scripts.run_first_pass import net_rewards, payoff_sharpe
from scripts.run_four_pairs import entry_cost, LADDER
from scripts.run_rv_vix_pass import fit_tree_policy, leaf_states
from scripts.run_walk_forward import fold_data
from scripts.run_nn_rl_benchmark import NEW_FEATURES, add_features, nn_actions, fit_trees
from scripts.explore_universe_gap import block_difference

MEMORY=['past_net_mean','past_net_sharpe','past_active_fraction','past_observations']
FEATURES=NEW_FEATURES+MEMORY
MODELS=['rl_no_memory','nn_memory','rl_memory_fixed','rl_memory']
ROOT=Path('artifacts/broad_pairs');OUT=ROOT/'memory';REPORTS=Path('reports/broad_pairs')


def pair_panel(definition):
    columns={'iv':'iv_amd','trailing_rv':'trailing_rv_amd','forward_rv':'rv_amd','earnings':'amd_earnings_days_capped'}
    a=pd.read_parquet(ROOT/'assets'/f"{definition['asset_a']}.parquet").rename(columns=columns)
    columns={k:v.replace('amd','nvda') for k,v in columns.items()}
    b=pd.read_parquet(ROOT/'assets'/f"{definition['asset_b']}.parquet").rename(columns=columns)
    cols=['iv_amd','trailing_rv_amd','rv_amd','amd_earnings_days_capped']
    p=a[cols].join(b[[c.replace('amd','nvda') for c in cols]],how='inner').sort_index()
    p['end']=p.index+pd.Timedelta(days=30);p['spread']=p.iv_amd-p.iv_nvda
    p['long_reward']=(p.rv_amd-p.iv_amd)-(p.rv_nvda-p.iv_nvda)
    return add_features(p)


def account(action,panel):
    r=net_rewards(np.asarray(action),panel.long_reward.to_numpy(),entry_cost(panel,.025))
    # A flat entry has known zero payoff even if an unentered trade lacks a label.
    return np.where(np.asarray(action)==0,0.,r)


def history_state(reference,dates,window=252):
    r=reference.dropna(subset=['reward']).sort_values('end')
    result=pd.DataFrame(0.,index=dates,columns=MEMORY)
    if r.empty:return result
    mean=r.reward.rolling(window,min_periods=1).mean();std=r.reward.rolling(window,min_periods=2).std()
    states=pd.DataFrame({'past_net_mean':mean,'past_net_sharpe':np.sqrt(365/30)*mean/std.replace(0,np.nan),
                         'past_active_fraction':r.action.ne(0).rolling(window,min_periods=1).mean(),
                         'past_observations':r.reward.rolling(window,min_periods=1).count()/window}).fillna(0.)
    pos=pd.DatetimeIndex(r.end).searchsorted(dates,side='left')-1
    result.loc[pos>=0]=states.iloc[pos[pos>=0]].to_numpy()
    return result


def reference_predictions(panel):
    pieces=[]
    for year in range(2006,2026,2):
        start=pd.Timestamp(year,1,1);end=pd.Timestamp(year+2,1,1)
        train=panel[(panel.index<start)&(panel.end<start)].dropna(subset=['long_reward'])
        test=panel[(panel.index>=start)&(panel.index<end)]
        if len(train)<504 or test.empty:continue
        tree,ids,policy,_,_=fit_tree_policy(train,NEW_FEATURES,8,126,entry_cost(train,.025),action_values=LADDER)
        a=policy[leaf_states(tree,test,NEW_FEATURES,ids)]
        p=test[['end']].copy();p['action']=a;p['reward']=account(a,test);p['fit_cutoff']=start
        assert train.end.max()<test.index.min()
        pieces.append(p)
    return pd.concat(pieces) if pieces else pd.DataFrame(columns=['end','action','reward','fit_cutoff'])


def fit_neural(train,val,test):
    scaler=StandardScaler().fit(train[FEATURES]);x=scaler.transform(train[FEATURES]);v=scaler.transform(val[FEATURES]);t=scaler.transform(test[FEATURES])
    y=train.long_reward.to_numpy();center=y.mean();scale=max(y.std(),1e-8);models={};candidates=[]
    for alpha in [.01,1.]:
        nn=MLPRegressor(hidden_layer_sizes=(16,),activation='tanh',solver='lbfgs',alpha=alpha,max_iter=10000,max_fun=200000,tol=1e-4,random_state=2026)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always',ConvergenceWarning);nn.fit(x,(y-center)/scale)
        converged=not any(issubclass(w.category,ConvergenceWarning) for w in ws)
        models[alpha]=nn;pred=nn.predict(v)*scale+center
        for threshold in [1.,2.,4.]:
            a=nn_actions(pred,entry_cost(val,.025),threshold)
            candidates.append(dict(model='nn_memory',alpha=alpha,threshold=threshold,validation_sharpe=payoff_sharpe(account(a,val)),converged=converged,iterations=int(nn.n_iter_)))
    chosen=max(candidates,key=lambda r:r['validation_sharpe'])
    pred=models[chosen['alpha']].predict(t)*scale+center
    return nn_actions(pred,entry_cost(test,.025),chosen['threshold']),chosen


def run_pair(definition,years,resume):
    key=definition['pair'];dest=OUT/key;dest.mkdir(exist_ok=True)
    panel=pair_panel(definition);refpath=dest/'reference.parquet'
    if resume and refpath.exists():ref=pd.read_parquet(refpath)
    else:ref=reference_predictions(panel);ref.to_parquet(refpath)
    panel=panel.join(history_state(ref,panel.index));panel.to_parquet(dest/'panel.parquet')
    audit=[]
    for year in years:
        path=dest/f'{year}.csv';cp=dest/f'{year}_configs.json'
        if resume and path.exists() and cp.exists():audit.extend(json.loads(cp.read_text()));continue
        train,val,_=fold_data(panel,year);train=train.dropna(subset=['long_reward']);val=val.dropna(subset=['long_reward'])
        test=panel[(panel.index>=pd.Timestamp(year,1,1))&(panel.index<pd.Timestamp(year+2,1,1))]
        # Eligibility uses only prior data. No minimum future test coverage rule.
        if len(train)<504 or len(val)<252 or test.empty:
            row=dict(pair=key,year=year,model='ineligible',train_rows=len(train),validation_rows=len(val),test_rows=len(test))
            cp.write_text(json.dumps([row],indent=2)+'\n');audit.append(row);continue
        assert train.end.max()<val.index.min() and val.end.max()<test.index.min()
        original,_=fit_trees(train,val,test,NEW_FEATURES,False)
        augmented,_=fit_trees(train,val,test,FEATURES,True)
        nn,ncfg=fit_neural(train,val,test)
        fits={'rl_no_memory':original['rl_base_sized'],'rl_memory':augmented['rl_both_sized'],
              'rl_memory_fixed':augmented['rl_both_fixed'],'nn_memory':(nn,ncfg)}
        p=test[['end','past_net_mean','past_net_sharpe','past_active_fraction','past_observations']].copy()
        selected=[]
        for model,(a,cfg) in fits.items():
            p[model+'_action']=a;p[model+'_reward']=account(a,test)
            cfg={k:v for k,v in cfg.items() if k!='model'}
            row=dict(pair=key,year=year,model=model,**cfg,train_rows=len(train),validation_rows=len(val),test_rows=len(test),
                     train_last_outcome=str(train.end.max().date()),validation_first=str(val.index.min().date()),
                     validation_last_outcome=str(val.end.max().date()),test_first=str(test.index.min().date()))
            selected.append(row)
        p.to_csv(path);cp.write_text(json.dumps(selected,indent=2)+'\n');audit.extend(selected)
    return key,audit


def summarize(screens,definitions,audit):
    rows=[];annual=[];allocations=[];coverage=[];predictions={}
    calendar=pd.DatetimeIndex(pd.read_csv(ROOT/'calendar.csv',parse_dates=['date']).date)
    config=json.loads(Path('config/broad_pair_research.json').read_text());years=config['test_years']
    for definition in definitions:
        key=definition['pair'];pieces=[]
        for year in screens.loc[screens.pair.eq(key),'year']:
            path=OUT/key/f'{year}.csv'
            if path.exists():pieces.append(pd.read_csv(path,index_col=0,parse_dates=[0,'end']))
        if not pieces:continue
        p=pd.concat(pieces).sort_index();predictions[key]=p
        matched=p[[m+'_reward' for m in MODELS]].notna().all(axis=1)
        for m in MODELS:
            r=p.loc[matched,m+'_reward'];a=p.loc[matched,m+'_action']
            rows.append(dict(pair=key,label=definition['label'],industry=definition['industry'],model=m,entries=len(r),missing_outcome_dates=int((~matched).sum()),sharpe=payoff_sharpe(r),net_pnl=r.sum(),trade_fraction=a.ne(0).mean(),mean_size=a.abs().mean()))
            for year,g in r.groupby(r.index.year):annual.append(dict(pair=key,model=m,year=year,sharpe=payoff_sharpe(g),net_pnl=g.sum()))
            for name,mask in [('positive_history',p.past_net_mean.gt(0)&p.past_observations.ge(.5)),('negative_history',p.past_net_mean.lt(0)&p.past_observations.ge(.5)),('short_or_flat_history',p.past_observations.lt(.5)|p.past_net_mean.eq(0))]:
                z=p.loc[mask&matched]
                if len(z):allocations.append(dict(pair=key,model=m,state=name,entries=len(z),trade_fraction=z[m+'_action'].ne(0).mean(),mean_size=z[m+'_action'].abs().mean(),sharpe=payoff_sharpe(z[m+'_reward'])))
    cohorts=[];cohorts_no_metals=[]
    for year in years:
        roster=screens[screens.year.eq(year)]
        dates=calendar[(calendar>=pd.Timestamp(year,1,1))&(calendar<pd.Timestamp(year+2,1,1))&(calendar<='2025-08-29')]
        for group,selected in [('all',roster),('without_metals',roster[~roster.industry.isin(['104','106','109'])&~roster.label.str.startswith('SIL/')])]:
            sums=pd.DataFrame(0.,index=dates,columns=MODELS);missing=pd.Series(False,index=dates);trained=0;missing_count=0
            for key in selected.pair:
                path=OUT/key/f'{year}.csv'
                if not path.exists():continue # Insufficient prior training history means cash, not a reweighted book.
                trained+=1;p=pd.read_csv(path,index_col=0,parse_dates=[0]);p=p.reindex(dates)
                exists=p.end.notna()
                absent_outcome=exists&p[[m+'_reward' for m in MODELS]].isna().any(axis=1)
                missing|=absent_outcome;missing_count+=int(absent_outcome.sum())
                for m in MODELS:sums[m]+=p[m+'_reward'].fillna(0.)/len(selected)
            sums.loc[missing]=np.nan
            sums['fold']=year
            (cohorts if group=='all' else cohorts_no_metals).append(sums)
            coverage.append(dict(group=group,fold=year,candidate_pairs=len(selected),trained_pairs=trained,calendar_dates=len(dates),joint_complete_dates=int((~missing).sum()),unresolved_pair_dates=missing_count))
    portfolios=[];portfolio_annual=[];contrasts=[]
    for group,pieces in [('all',cohorts),('without_metals',cohorts_no_metals)]:
        raw=pd.concat(pieces);raw.to_csv(OUT/f'{group}_portfolio.csv');p=raw.dropna(subset=MODELS)
        for m in MODELS:
            r=p[m];portfolios.append(dict(group=group,model=m,entries=len(r),sharpe=payoff_sharpe(r),net_pnl=r.sum()))
            for year,g in r.groupby(r.index.year):portfolio_annual.append(dict(group=group,model=m,year=year,entries=len(g),sharpe=payoff_sharpe(g),net_pnl=g.sum()))
        for other in ['rl_no_memory','nn_memory','rl_memory_fixed']:
            ci=block_difference(p.rl_memory.to_numpy(),p[other].to_numpy(),repeats=2000)
            contrasts.append(dict(group=group,reference=other,sharpe_change=payoff_sharpe(p.rl_memory)-payoff_sharpe(p[other]),ci_low=ci[0],ci_high=ci[1]))
    for name,data in [('scores',rows),('annual',annual),('allocations',allocations),('coverage',coverage),('configs',audit),('portfolio',portfolios),('portfolio_annual',portfolio_annual),('contrasts',contrasts)]:pd.DataFrame(data).to_csv(REPORTS/f'{name}.csv',index=False)
    print(pd.DataFrame(portfolios).to_string(index=False),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    OUT.mkdir(exist_ok=True);config=json.loads(Path('config/broad_pair_research.json').read_text())
    screens=pd.read_csv(REPORTS/'candidate_screens.csv');screens=screens[screens.year.isin(config['test_years'])]
    definitions=[d for d in json.loads((ROOT/'pairs.json').read_text()) if d['pair'] in set(screens.pair)]
    settings=dict(config=config,features=FEATURES,models=MODELS,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path=OUT/'settings.json'
    if path.exists() and json.loads(path.read_text())!=settings:raise ValueError('Settings changed')
    path.write_text(json.dumps(settings,indent=2)+'\n');(REPORTS/'settings.json').write_text(path.read_text())
    print('Fitting',len(definitions),'unique pairs and',len(screens),'screened pair-folds',flush=True)
    start=time.monotonic();audit=[];done=0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_pair,d,sorted(screens.loc[screens.pair.eq(d['pair']),'year'].tolist()),args.resume) for d in definitions]
        for f in as_completed(futures):
            key,rows=f.result();audit.extend(rows);done+=1
            print(f'{done}/{len(definitions)} pairs completed; {(time.monotonic()-start)/60:.1f} minutes',flush=True)
    summarize(screens,definitions,audit)


if __name__=='__main__':main()
