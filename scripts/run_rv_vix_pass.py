"""Shared tree-state policy search with trailing RV and VIX; matched ablations."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text

if __package__:
    from .run_first_pass import load_data, load_return_history, split_data, fit_policy, net_rewards, payoff_sharpe, interval
    from .attach_earnings import attach, load_calendar
else:
    from run_first_pass import load_data, load_return_history, split_data, fit_policy, net_rewards, payoff_sharpe, interval
    from attach_earnings import attach, load_calendar


FEATURE_SETS = {
    'spread': ['spread'],
    'spread_earnings': ['spread', 'amd_earnings_days_capped', 'nvda_earnings_days_capped'],
    'spread_earnings_rv': ['spread', 'amd_earnings_days_capped', 'nvda_earnings_days_capped', 'trailing_rv_amd', 'trailing_rv_nvda'],
    'spread_earnings_rv_vix': ['spread', 'amd_earnings_days_capped', 'nvda_earnings_days_capped', 'trailing_rv_amd', 'trailing_rv_nvda', 'vix_previous_close'],
}


def trailing_rv(returns):
    """Annualized RMS C2C log returns over (entry-30 calendar days, entry]."""
    r = returns.sort_index().astype(float)
    squares = np.log1p(r.where(r > -1)) ** 2
    mean = squares.rolling('30D', min_periods=15, closed='right').mean()
    # Do not silently skip missing returns or use a shorter initial history.
    counts = squares.rolling('30D', closed='right').count()
    expected = pd.Series(1,index=r.index).rolling('30D',closed='right').sum()
    mean = mean.where(counts.eq(expected,axis=0))
    mean.loc[mean.index < mean.index.min()+pd.Timedelta(days=30)] = np.nan
    return (100*np.sqrt(252*mean)).rename(columns={'AMD':'trailing_rv_amd','NVDA':'trailing_rv_nvda'})


def join_vix(panel, vix):
    """Strictly prior session close; never forward-fill from a future date."""
    left = panel.rename_axis('date').reset_index().sort_values('date')
    right = vix.rename(columns={'DATE':'vix_date','CLOSE':'vix_previous_close'})[['vix_date','vix_previous_close']].sort_values('vix_date')
    if right.vix_date.duplicated().any():
        raise ValueError('Duplicate VIX dates')
    merged = pd.merge_asof(left,right,left_on='date',right_on='vix_date',direction='backward',allow_exact_matches=False,tolerance=pd.Timedelta(days=7))
    return merged.set_index('date')


def leaf_states(tree, data, features, leaf_ids):
    leaves = tree.apply(data[features])
    return np.searchsorted(leaf_ids, leaves)


def fit_tree_policy(train, features, max_leaves, min_leaf, cost, action_values=None):
    # Supervised state partition, followed by direct Sharpe policy optimization.
    # The partition is learned only on training rewards; prediction uses state only.
    tree = DecisionTreeRegressor(max_leaf_nodes=max_leaves,min_samples_leaf=min_leaf,random_state=2026)
    tree.fit(train[features],train.long_reward)
    leaf_ids = np.unique(tree.apply(train[features]))
    state = leaf_states(tree,train,features,leaf_ids)
    fits = [fit_policy(state,train.long_reward.to_numpy(),len(leaf_ids),seed,cost=cost) for seed in range(5)]
    policy,score = max(fits,key=lambda x:x[1])
    if action_values is not None:
        fixed_score=score
        fits=[fit_policy(state,train.long_reward.to_numpy(),len(leaf_ids),seed,cost=cost,
                         action_values=action_values,initial_policy=policy) for seed in range(5)]
        policy,score=max(fits,key=lambda x:x[1])
        assert score>=fixed_score-1e-10
    return tree,leaf_ids,policy,score,fits


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--vix',type=Path,default=Path('data/raw/VIX_History.csv'))
    p.add_argument('--output',type=Path,default=Path('artifacts/rv_vix_pass'))
    p.add_argument('--cost',type=float,default=.5)
    args=p.parse_args()
    if args.cost<0:
        p.error('--cost must be nonnegative')
    args.output.mkdir(parents=True,exist_ok=True)
    panel=attach(load_data(args.source_root),load_calendar(args.source_root))
    panel=panel.join(trailing_rv(load_return_history(args.source_root)))
    vix=pd.read_csv(args.vix,parse_dates=['DATE'])
    panel=join_vix(panel,vix)
    for ticker in ['amd','nvda']:
        panel[f'{ticker}_earnings_days_capped']=panel[f'{ticker}_days_to_earnings_ex_post'].clip(upper=31)
    required=FEATURE_SETS['spread_earnings_rv_vix']
    dropped=panel[required].isna().any(axis=1)
    dropped_dates=panel.index[dropped].strftime('%Y-%m-%d').tolist()
    panel=panel.loc[~dropped]
    assert (panel.vix_date<panel.index).all()
    train,val,test=split_data(panel)
    assert train.end.max()<val.index.min() and val.end.max()<test.index.min()
    panel.to_parquet(args.output/'dataset.parquet')
    candidates=[]
    results=[]
    predictions=test.copy()
    selected={}
    for name,features in FEATURE_SETS.items():
        models=[]
        for max_leaves in [4,8,16]:
            for min_leaf in [126,252]:
                tree,leaf_ids,policy,score,fits=fit_tree_policy(train,features,max_leaves,min_leaf,args.cost)
                vs=leaf_states(tree,val,features,leaf_ids)
                vr=net_rewards(policy[vs],val.long_reward.to_numpy(),args.cost)
                candidate={'feature_set':name,'max_leaves':max_leaves,'min_leaf':min_leaf,'actual_leaves':len(leaf_ids),'train_sharpe':score,'validation_sharpe':payoff_sharpe(vr)}
                candidates.append(candidate)
                models.append((candidate,tree,leaf_ids,policy,fits))
        # No test outcome participates in model or feature-set selection.
        cfg,tree,leaf_ids,policy,fits=max(models,key=lambda m:m[0]['validation_sharpe'])
        ts=leaf_states(tree,test,features,leaf_ids)
        actions=policy[ts]
        reward=net_rewards(actions,test.long_reward.to_numpy(),args.cost)
        selected[name]=cfg
        results.append({'feature_set':name,'annualized_sharpe':payoff_sharpe(reward),'sharpe_ci':interval(reward,statistic=payoff_sharpe),'mean_net_reward':float(reward.mean()),'trade_fraction':float(np.mean(actions!=0)), 'seed_test_sharpes':[payoff_sharpe(net_rewards(m[ts],test.long_reward.to_numpy(),args.cost)) for m,_ in fits]})
        predictions[f'{name}_action']=actions
        predictions[f'{name}_reward']=reward
        (args.output/f'{name}_tree.txt').write_text(export_text(tree,feature_names=features,decimals=4)+'\nLeaf action mapping:\n'+str(dict(zip(leaf_ids.tolist(),policy.tolist()))))
        pd.DataFrame({'leaf_id':leaf_ids,'action':policy,'train_count':np.bincount(leaf_states(tree,train,features,leaf_ids),minlength=len(leaf_ids))}).to_csv(args.output/f'{name}_policy.csv',index=False)
    simple=-np.sign(test.spread.to_numpy()).astype(int)
    sr=net_rewards(simple,test.long_reward.to_numpy(),args.cost)
    results.append({'feature_set':'short_higher_iv','annualized_sharpe':payoff_sharpe(sr),'sharpe_ci':interval(sr,statistic=payoff_sharpe),'mean_net_reward':float(sr.mean()),'trade_fraction':float(np.mean(simple!=0))})
    predictions['short_higher_iv_reward']=sr
    predictions.to_csv(args.output/'test_predictions.csv')
    pd.DataFrame(candidates).to_csv(args.output/'candidates.csv',index=False)
    pd.DataFrame(results).to_csv(args.output/'scores.csv',index=False)
    summary={'results':results,'selected':selected,'validation_selected_feature_set':max(selected,key=lambda n:selected[n]['validation_sharpe']),
             'features':FEATURE_SETS,'dropped_dates':dropped_dates,
             'splits':{name:{'first':str(d.index.min().date()),'last':str(d.index.max().date()),'count':len(d)} for name,d in [('train',train),('validation',val),('test',test)]},
             'assumptions':{'cost':args.cost,'sharpe_factor':'sqrt(365/30)','earnings':'actual date known inside 30 days; countdown capped at 31','trailing_rv':'30-calendar-day C2C log-return RMS ending at entry, 252-day annualization','vix':'strictly previous available session close, at most 7 calendar days stale','model':'training-reward regression tree state partition + cross-entropy Sharpe policy search; min leaf 126 or 252'},
             'vix_source':'https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv','vix_sha256':hashlib.sha256(args.vix.read_bytes()).hexdigest()}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
