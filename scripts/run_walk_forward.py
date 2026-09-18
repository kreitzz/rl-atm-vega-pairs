"""Expanding-window evaluation with two-year validation and test blocks."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .run_rv_vix_pass import FEATURE_SETS, fit_tree_policy, leaf_states
    from .run_first_pass import net_rewards, payoff_sharpe, interval
else:
    from run_rv_vix_pass import FEATURE_SETS, fit_tree_policy, leaf_states
    from run_first_pass import net_rewards, payoff_sharpe, interval


def fold_data(panel, test_year):
    validation_start = pd.Timestamp(test_year-2,1,1)
    test_start = pd.Timestamp(test_year,1,1)
    test_end = pd.Timestamp(test_year+2,1,1)
    train = panel[(panel.index < validation_start) & (panel.end < validation_start)]
    validation = panel[(panel.index >= validation_start) & (panel.index < test_start) & (panel.end < test_start)]
    test = panel[(panel.index >= test_start) & (panel.index < test_end)]
    return train,validation,test


def nonoverlap_positions(panel):
    positions=[]
    next_entry=pd.Timestamp.min
    for i,(date,row) in enumerate(panel.iterrows()):
        if date >= next_entry:
            positions.append(i)
            next_entry=row.end
    return np.array(positions,dtype=int)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--panel',type=Path,default=Path('artifacts/rv_vix_pass/dataset.parquet'))
    p.add_argument('--output',type=Path,default=Path('artifacts/walk_forward'))
    p.add_argument('--cost',type=float,default=.5)
    p.add_argument('--features-json',type=Path,help='Optional named feature sets for matched experiments.')
    p.add_argument('--reference-model',default='spread')
    p.add_argument('--comparison-model',default='spread_earnings_rv_vix')
    args=p.parse_args()
    if args.cost<0: p.error('--cost must be nonnegative')
    panel=pd.read_parquet(args.panel).sort_index()
    feature_sets=json.loads(args.features_json.read_text()) if args.features_json else FEATURE_SETS
    if args.reference_model not in feature_sets or args.comparison_model not in feature_sets:
        p.error('Comparison and reference models must be named feature sets')
    columns=sorted({column for features in feature_sets.values() for column in features})
    if panel[columns].isna().any().any():
        p.error('All feature sets must use the same complete panel')
    args.output.mkdir(parents=True,exist_ok=True)
    pieces=[]; fold_scores=[]; configs=[]; candidates=[]; folds=[]
    for year in range(2010,2026,2):
        train,val,test=fold_data(panel,year)
        if test.empty: continue
        assert train.end.max()<val.index.min() and val.end.max()<test.index.min()
        fold=f'{year}-{min(year+1,test.index.max().year)}'
        folds.append({'fold':fold,**{f'{name}_{key}':value for name,d in [('train',train),('validation',val),('test',test)] for key,value in [('first',str(d.index.min().date())),('last',str(d.index.max().date())),('count',len(d)),('last_outcome',str(d.end.max().date()))]}})
        out=test[['end','long_reward','spread']].copy()
        out['fold']=fold
        selected={}
        for name,features in feature_sets.items():
            models=[]
            for max_leaves in [4,8,16]:
                for min_leaf in [126,252]:
                    tree,ids,policy,train_score,_=fit_tree_policy(train,features,max_leaves,min_leaf,args.cost)
                    vr=net_rewards(policy[leaf_states(tree,val,features,ids)],val.long_reward.to_numpy(),args.cost)
                    cfg={'fold':fold,'feature_set':name,'max_leaves':max_leaves,'min_leaf':min_leaf,'actual_leaves':len(ids),'train_sharpe':train_score,'validation_sharpe':payoff_sharpe(vr)}
                    candidates.append(cfg)
                    models.append((cfg,tree,ids,policy))
            cfg,tree,ids,policy=max(models,key=lambda m:m[0]['validation_sharpe'])
            configs.append(cfg)
            selected[name]=cfg
            actions=policy[leaf_states(tree,test,features,ids)]
            out[name+'_action']=actions
            out[name+'_reward']=net_rewards(actions,test.long_reward.to_numpy(),args.cost)
        # This meta-policy selects feature set by preceding validation only.
        winner=max(selected,key=lambda name:selected[name]['validation_sharpe'])
        out['validation_selected_features']=winner
        out['validation_selected_action']=out[winner+'_action']
        out['validation_selected_reward']=out[winner+'_reward']
        out['short_higher_iv_action']=-np.sign(test.spread.to_numpy()).astype(int)
        out['short_higher_iv_reward']=net_rewards(out.short_higher_iv_action,test.long_reward,args.cost)
        for name in list(feature_sets)+['validation_selected','short_higher_iv']:
            fold_scores.append({'fold':fold,'model':name,'annualized_sharpe':payoff_sharpe(out[name+'_reward']),'trade_fraction':float((out[name+'_action']!=0).mean()),'mean_net':float(out[name+'_reward'].mean()),'entries':len(out)})
        pieces.append(out)
        print(f"Completed {fold}: {len(train)} train / {len(val)} validation / {len(test)} test; validation chose {winner}",flush=True)
    combined=pd.concat(pieces).sort_index()
    assert combined.index.is_unique
    positions=nonoverlap_positions(combined)
    scores=[]
    for name in list(feature_sets)+['validation_selected','short_higher_iv']:
        r=combined[name+'_reward'].to_numpy()
        scores.append({'model':name,'annualized_sharpe':payoff_sharpe(r),'sharpe_ci':interval(r,statistic=payoff_sharpe),'trade_fraction':float((combined[name+'_action']!=0).mean()),'mean_net':float(r.mean()),'nonoverlap_sharpe':payoff_sharpe(r[positions])})
    rng=np.random.default_rng(2026)
    differences=[]
    for _ in range(2000):
        idx=((rng.integers(len(combined),size=int(np.ceil(len(combined)/63)))[:,None]+np.arange(63))%len(combined)).ravel()[:len(combined)]
        differences.append(payoff_sharpe(combined[args.comparison_model+'_reward'].to_numpy()[idx])-payoff_sharpe(combined[args.reference_model+'_reward'].to_numpy()[idx]))
    summary={'first':str(combined.index.min().date()),'last':str(combined.index.max().date()),'entries':len(combined),'nonoverlap_entries':len(positions),
             'folds':folds,'scores':scores,'comparison_minus_reference_sharpe_ci':np.quantile(differences,[.025,.975]).tolist(),
             'comparison_model':args.comparison_model,'reference_model':args.reference_model,'feature_sets':feature_sets,
             'source_panel_sha256':hashlib.sha256(args.panel.read_bytes()).hexdigest(),
             'assumptions':{'cost':args.cost,'annualization':'sqrt(365/30)','schedule':'expanding train, prior 2 years validation, next 2 years test','outcome_purging':True,'test_status':'retrospective walk-forward; previously viewed historical data; not pristine holdout','nonoverlap':'greedy entry at first date on/after prior 30-day expiry; one offset diagnostic'}}
    combined.to_csv(args.output/'predictions.csv')
    combined.iloc[positions].to_csv(args.output/'nonoverlap_predictions.csv')
    pd.DataFrame(fold_scores).to_csv(args.output/'fold_scores.csv',index=False)
    pd.DataFrame(configs).to_csv(args.output/'selected_configs.csv',index=False)
    pd.DataFrame(candidates).to_csv(args.output/'candidates.csv',index=False)
    pd.DataFrame(scores).to_csv(args.output/'scores.csv',index=False)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({'entries':len(combined),'nonoverlap_entries':len(positions),'scores':scores,'comparison_minus_reference_sharpe_ci':summary['comparison_minus_reference_sharpe_ci']},indent=2))


if __name__=='__main__': main()
