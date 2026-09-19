"""Four-pair joint Sharpe policy search under one per-entry vega budget."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from scripts.run_first_pass import forward_rv, payoff_sharpe, net_rewards, SHARPE_ANNUALIZATION
from scripts.run_rv_vix_pass import trailing_rv, fit_tree_policy, leaf_states
from scripts.run_walk_forward import fold_data

PAIRS = ['AMD_NVDA', 'XOM_CVX', 'JPM_BAC', 'HD_LOW']
IDS = {'XOM': (104533, 11850), 'CVX': (102968, 14541),
       'JPM': (102936, 47896), 'BAC': (101966, 59408),
       'HD': (105759, 66181), 'LOW': (107045, 61399)}
FEATURES = ['spread', 'amd_earnings_days_capped', 'nvda_earnings_days_capped',
            'trailing_rv_amd', 'trailing_rv_nvda']
LADDER = np.arange(-1, 1.01, .25)


def action_menu():
    """Signed equal-vega pair allocations; total absolute allocation <= 1."""
    return np.array([a for a in itertools.product(range(-4, 5), repeat=4)
                     if sum(abs(x) for x in a) <= 4], dtype=float) / 4


def build_panels(root):
    base = pd.read_csv('data/current_dataset.csv', index_col='date', parse_dates=['date', 'end'])
    base['long_reward'] = (base.rv_amd-base.iv_amd)-(base.rv_nvda-base.iv_nvda)
    panels = {'AMD_NVDA': base}
    om = root/'optionmetrics'
    paths = sorted((om/'atm_iv_compact').rglob('*.parquet')) + sorted((om/'atm_implied_volatility').glob('*.parquet'))
    frames = [pd.read_parquet(p, columns=['secid','date','impl_volatility'],
              filters=[('secid','in',[v[0] for v in IDS.values()]),('days','=',30),('delta','=',50),('cp_flag','=','C')]) for p in paths]
    iv = pd.concat([f for f in frames if len(f)]).drop_duplicates()
    iv['ticker'] = iv.secid.map({v[0]:k for k,v in IDS.items()})
    iv = iv.pivot(index='date',columns='ticker',values='impl_volatility')*100
    returns = []
    for path in sorted((root/'crsp_underlying_daily').glob('*.parquet')):
        d = pd.read_parquet(path,columns=['permno','date','retx'])
        returns.append(d[d.permno.isin([v[1] for v in IDS.values()])])
    r = pd.concat([d for d in returns if len(d)]).drop_duplicates()
    r['ticker'] = r.permno.map({v[1]:k for k,v in IDS.items()})
    r = r.pivot(index='date',columns='ticker',values='retx').sort_index()
    earnings = pd.read_parquet(root/'compustat/underlying_quarterly_earnings.parquet')
    for pair in PAIRS[1:]:
        a,b = pair.split('_')
        ret = r[[a,b]].rename(columns={a:'AMD',b:'NVDA'})
        p = iv[[a,b]].rename(columns={a:'iv_amd',b:'iv_nvda'})
        p = p.where(p>0).dropna().join(forward_rv(ret,p.index),how='inner')
        p = p.join(trailing_rv(ret))
        for ticker,prefix in [(a,'amd'),(b,'nvda')]:
            dates = pd.DatetimeIndex(pd.to_datetime(earnings.loc[earnings.permno.eq(IDS[ticker][1]),'rdq']).dropna().unique()).sort_values()
            pos = dates.searchsorted(p.index,side='left')
            nxt = pd.Series([dates[i] if i<len(dates) else pd.NaT for i in pos],index=p.index)
            p[prefix+'_earnings_days_capped'] = (nxt-pd.Series(p.index,index=p.index)).dt.days.clip(upper=31)
        p['spread'] = p.iv_amd-p.iv_nvda
        p['long_reward'] = (p.rv_amd-p.iv_amd)-(p.rv_nvda-p.iv_nvda)
        panels[pair] = p.dropna().sort_index()
    dates = base.index
    for p in panels.values(): dates = dates.intersection(p.index)
    return {k:p.loc[dates].copy() for k,p in panels.items()}


def action_moments(state, rewards, menu, nstates, cost=1.):
    sums = np.zeros((nstates,len(menu))); squares = sums.copy()
    costs = np.broadcast_to(np.asarray(cost,dtype=float),rewards.shape)
    for s in range(nstates):
        values = rewards[state==s] @ menu.T - costs[state==s] @ np.abs(menu).T
        sums[s] = values.sum(axis=0)
        squares[s] = np.square(values).sum(axis=0)
    return sums,squares


def moment_scores(policies, sums, squares, n):
    rows = np.arange(sums.shape[0])[None,:]
    total = sums[rows,policies].sum(axis=1)
    second = squares[rows,policies].sum(axis=1)
    mean = total/n
    var = np.maximum((second-n*mean**2)/(n-1),0)
    return SHARPE_ANNUALIZATION*np.divide(mean,np.sqrt(var),out=np.zeros_like(mean),where=var>1e-24)


def joint_search(state, rewards, menu, nstates, seed, iterations=80, population=256, cost=1.):
    sums,squares = action_moments(state,rewards,menu,nstates,cost)
    rng = np.random.default_rng(seed)
    probabilities = np.full((nstates,len(menu)),1/len(menu))
    flat = np.flatnonzero((menu==0).all(axis=1))[0]
    best = np.full(nstates,flat); best_score = 0.
    for _ in range(iterations):
        u = rng.random((population,nstates))
        policies = np.column_stack([np.searchsorted(probabilities[s].cumsum(),u[:,s]).clip(max=len(menu)-1) for s in range(nstates)])
        policies[0] = best
        scores = moment_scores(policies,sums,squares,len(rewards))
        winner = scores.argmax()
        if scores[winner]>best_score: best,best_score = policies[winner].copy(),float(scores[winner])
        elite = policies[np.argsort(scores)[-max(2,population//10):]]
        freq = np.stack([np.bincount(elite[:,s],minlength=len(menu))/len(elite) for s in range(nstates)])
        probabilities = .97*(.3*probabilities+.7*freq)+.03/len(menu)
    return menu[best],best_score


def portfolio_reward(actions, rewards, cost=1.):
    return (actions*rewards-cost*np.abs(actions)).sum(axis=1)


def entry_cost(panel, rate):
    return np.ones(len(panel)) if rate is None else rate*(panel.iv_amd+panel.iv_nvda).to_numpy(float)/2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--cost-rate',type=float,help='Pair cost = rate times average entry IV in vol points; default fixed 1 point.')
    parser.add_argument('--resume',action='store_true',help='Reuse local panels and completed folds from this exact experiment.')
    args = parser.parse_args()
    if args.cost_rate is not None and (not np.isfinite(args.cost_rate) or args.cost_rate<0):
        parser.error('--cost-rate must be finite and nonnegative')
    if args.output is None:
        args.output=Path('artifacts/four_pairs' if args.cost_rate is None else f'artifacts/four_pairs_cost_{args.cost_rate:g}')
    args.output.mkdir(parents=True,exist_ok=True)
    settings={'cost_rate':args.cost_rate,'cost_definition':'fixed 1 point' if args.cost_rate is None else 'rate * (entry IV leg A + entry IV leg B) / 2; scaled by absolute allocation'}
    settings_path=args.output/'cost_settings.json'
    if args.resume and settings_path.exists() and json.loads(settings_path.read_text())!=settings:
        parser.error('Cannot resume an experiment with different cost settings')
    if args.resume and args.cost_rate is not None and not settings_path.exists():
        parser.error('Proportional-cost resume requires matching saved settings')
    settings_path.write_text(json.dumps(settings,indent=2)+'\n')
    cache = args.output/'panels.parquet'
    if args.resume and cache.exists():
        packed = pd.read_parquet(cache)
        panels = {k:packed.xs(k,level='pair').sort_index() for k in PAIRS}
    else:
        panels = build_panels(args.source_root)
        pd.concat(panels,names=['pair','date']).to_parquet(cache)
    # One shared tree observes all four states and predicts four forward rewards.
    wide = pd.concat({k:p[FEATURES] for k,p in panels.items()},axis=1)
    wide.columns = ['_'.join(c) for c in wide.columns]
    features = wide.columns.tolist()
    wide['end'] = panels[PAIRS[0]].end
    for k in PAIRS: wide[k+'_gross'] = panels[k].long_reward
    reward_cols = [k+'_gross' for k in PAIRS]
    cost_cols = [k+'_cost' for k in PAIRS]
    for k in PAIRS: wide[k+'_cost']=entry_cost(panels[k],args.cost_rate)
    menu = action_menu(); pieces=[]; configs=[]
    print('Matched rows:',len(wide),'from',wide.index.min(),'to',wide.index.max(),'portfolio actions:',len(menu),flush=True)
    for year in range(2010,2026,2):
        prediction_path=args.output/f'fold_{year}.csv'; config_path=args.output/f'fold_{year}.json'
        if args.resume and prediction_path.exists() and config_path.exists():
            pieces.append(pd.read_csv(prediction_path,index_col='date',parse_dates=['date','end']))
            configs.extend(json.loads(config_path.read_text())); continue
        train,val,test = fold_data(wide,year)
        assert train.end.max()<val.index.min() and val.end.max()<test.index.min()
        out = test[['end']+cost_cols].copy(); out['fold']=year
        fold_configs=[]; independent=[]
        for k in PAIRS:
            p=panels[k]; tr,va,te=p.loc[train.index],p.loc[val.index],p.loc[test.index]
            candidates=[]
            for leaves in [4,8,16]:
                for minimum in [126,252]:
                    tree,ids,policy,score,_=fit_tree_policy(tr,FEATURES,leaves,minimum,entry_cost(tr,args.cost_rate),action_values=LADDER)
                    vs=payoff_sharpe(net_rewards(policy[leaf_states(tree,va,FEATURES,ids)],va.long_reward.to_numpy(),entry_cost(va,args.cost_rate)))
                    candidates.append((vs,tree,ids,policy,leaves,minimum))
            score,tree,ids,policy,leaves,minimum=max(candidates,key=lambda c:c[0])
            action=policy[leaf_states(tree,te,FEATURES,ids)]; independent.append(action)
            out[k+'_solo_action']=action
            out[k+'_solo_reward']=net_rewards(action,te.long_reward.to_numpy(),entry_cost(te,args.cost_rate))
            fold_configs.append(dict(fold=year,model=k,leaves=leaves,min_leaf=minimum,validation_sharpe=score))
        independent=np.column_stack(independent)
        y=test[reward_cols].to_numpy(float)
        test_cost=test[cost_cols].to_numpy(float)
        out['equal_budget_reward']=portfolio_reward(independent/4,y,test_cost)
        # Redistribute unused budget while retaining each independent policy's relative sizes.
        capped=independent/np.maximum(1,np.abs(independent).sum(axis=1))[:,None]
        out['shared_cap_reward']=portfolio_reward(capped,y,test_cost)
        candidates=[]
        for leaves in [4,8,16]:
            for minimum in [126,252]:
                tree=DecisionTreeRegressor(max_leaf_nodes=leaves,min_samples_leaf=minimum,random_state=2026)
                tree.fit(train[features],train[reward_cols])
                ids=np.unique(tree.apply(train[features])); state=np.searchsorted(ids,tree.apply(train[features]))
                fits=[joint_search(state,train[reward_cols].to_numpy(float),menu,len(ids),seed,cost=train[cost_cols].to_numpy(float)) for seed in range(5)]
                policy,training_score=max(fits,key=lambda f:f[1])
                actions=policy[np.searchsorted(ids,tree.apply(val[features]))]
                score=payoff_sharpe(portfolio_reward(actions,val[reward_cols].to_numpy(float),val[cost_cols].to_numpy(float)))
                candidates.append((score,tree,ids,policy,leaves,minimum,training_score))
        score,tree,ids,policy,leaves,minimum,training_score=max(candidates,key=lambda c:c[0])
        actions=policy[np.searchsorted(ids,tree.apply(test[features]))]
        assert (np.abs(actions).sum(axis=1)<=1+1e-12).all()
        out['joint_reward']=portfolio_reward(actions,y,test_cost)
        out['joint_gross_allocation']=np.abs(actions).sum(axis=1)
        for i,k in enumerate(PAIRS):
            out[k+'_joint_action']=actions[:,i]
            out[k+'_joint_reward']=actions[:,i]*y[:,i]-test_cost[:,i]*np.abs(actions[:,i])
        fold_configs.append(dict(fold=year,model='joint',leaves=leaves,min_leaf=minimum,validation_sharpe=score,train_sharpe=training_score))
        out.to_csv(prediction_path); config_path.write_text(json.dumps(fold_configs,indent=2))
        pieces.append(out); configs.extend(fold_configs)
        print(year,'test entries',len(test),'AMD Sharpe',round(payoff_sharpe(out.AMD_NVDA_solo_reward),3),'joint',round(payoff_sharpe(out.joint_reward),3),flush=True)
    result=pd.concat(pieces).sort_index(); result.to_csv(args.output/'predictions.csv')
    models=[k+'_solo' for k in PAIRS]+['equal_budget','shared_cap','joint']
    scores=[]; annual=[]
    for model in models:
        r=result[model+'_reward']
        scores.append(dict(model=model,annualized_payoff_sharpe=payoff_sharpe(r),net_proxy_pnl=float(r.sum()),entries=len(r)))
        for year,g in result.groupby(result.index.year):
            annual.append(dict(year=int(year),model=model,annualized_payoff_sharpe=payoff_sharpe(g[model+'_reward']),net_proxy_pnl=float(g[model+'_reward'].sum())))
    pd.DataFrame(scores).to_csv(args.output/'scores.csv',index=False)
    pd.DataFrame(annual).to_csv(args.output/'annual.csv',index=False)
    pd.DataFrame(configs).to_csv(args.output/'selected_configs.csv',index=False)
    print(pd.DataFrame(scores).to_string(index=False),flush=True)


if __name__=='__main__': main()
