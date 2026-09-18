"""Paired audit of the earnings-state experiment, without retuning on test data."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

if __package__:
    from .run_first_pass import split_data, model_states, fit_policy, policy_scores, net_rewards, payoff_sharpe
else:
    from run_first_pass import split_data, model_states, fit_policy, policy_scores, net_rewards, payoff_sharpe


def main():
    out = Path('artifacts/earnings_audit')
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet('artifacts/actual_earnings_sharpe_pass/dataset.parquet')
    plain = pd.read_parquet('artifacts/sharpe_pass/dataset.parquet')
    pd.testing.assert_frame_equal(d[plain.columns], plain)
    train,val,test = split_data(d)
    rows = []
    for bins in [5,10,20]:
        edges = np.unique(np.quantile(train.spread,np.arange(1,bins)/bins))
        spread_policy,_ = max([fit_policy(model_states(train,edges),train.long_reward.to_numpy(),len(edges)+1,seed,cost=.5) for seed in range(5)],key=lambda x:x[1])
        for earnings in [False,True]:
            nstates=(len(edges)+1)*(4 if earnings else 1)
            state=model_states(train,edges,earnings)
            policy,score=max([fit_policy(state,train.long_reward.to_numpy(),nstates,seed,cost=.5) for seed in range(5)],key=lambda x:x[1])
            row={'spread_bins':bins,'earnings':earnings,'states':nstates,'min_training_rows_per_state':int(np.bincount(state,minlength=nstates).min())}
            for name,frame in [('train',train),('validation',val),('test',test)]:
                reward=net_rewards(policy[model_states(frame,edges,earnings)],frame.long_reward.to_numpy(),.5)
                row[name+'_sharpe']=payoff_sharpe(reward)
                if earnings:
                    # Explicitly lift spread-only policy into the larger state space.
                    lifted=np.repeat(spread_policy,4)
                    lifted_actions=lifted[model_states(frame,edges,True)]
                    np.testing.assert_array_equal(lifted_actions,spread_policy[model_states(frame,edges)])
            if earnings:
                row['lifted_spread_train_sharpe']=float(policy_scores(np.repeat(spread_policy,4)[None,:],state,train.long_reward.to_numpy(),nstates,cost=.5)[0])
                assert score >= row['lifted_spread_train_sharpe'] - 1e-10
            rows.append(row)
    comparison=pd.DataFrame(rows)
    comparison.to_csv(out/'matched_models.csv',index=False)
    print(comparison.to_string(index=False))
    s=pd.read_csv('artifacts/sharpe_pass/test_predictions.csv',index_col='date',parse_dates=['date'])
    e=pd.read_csv('artifacts/actual_earnings_sharpe_pass/test_predictions.csv',index_col='date',parse_dates=['date'])
    assert s.index.equals(e.index)
    panel=pd.DataFrame({'spread_reward':s.reward,'earnings_reward':e.reward,'spread_action':s.action,'earnings_action':e.action},index=s.index)
    panel['difference']=panel.earnings_reward-panel.spread_reward
    panel['amd_in_30d']=test.amd_days_to_earnings_ex_post<=30
    panel['nvda_in_30d']=test.nvda_days_to_earnings_ex_post<=30
    regime=panel.groupby(['amd_in_30d','nvda_in_30d']).agg(n=('difference','size'),mean_difference=('difference','mean'),total_difference=('difference','sum'),spread_mean=('spread_reward','mean'),earnings_mean=('earnings_reward','mean'))
    regime.to_csv(out/'regime_attribution.csv')
    annual=panel.groupby(panel.index.year).agg(n=('difference','size'),spread_sharpe=('spread_reward',payoff_sharpe),earnings_sharpe=('earnings_reward',payoff_sharpe),mean_difference=('difference','mean'))
    annual.to_csv(out/'annual_comparison.csv')
    rng=np.random.default_rng(2026)
    a=panel.spread_reward.to_numpy();b=panel.earnings_reward.to_numpy()
    diffs=[]
    for _ in range(2000):
        idx=((rng.integers(len(panel),size=int(np.ceil(len(panel)/63)))[:,None]+np.arange(63))%len(panel)).ravel()[:len(panel)]
        diffs.append(payoff_sharpe(b[idx])-payoff_sharpe(a[idx]))
    same_day=(test.amd_days_to_earnings_ex_post==0)|(test.nvda_days_to_earnings_ex_post==0)
    summary={'same_data_rewards':True,'test_action_disagreements':int((s.action!=e.action).sum()),'test_entries':len(panel),
             'sharpe_difference':payoff_sharpe(b)-payoff_sharpe(a),'paired_block_bootstrap_difference_ci':np.quantile(diffs,[.025,.975]).tolist(),
             'same_day_entries':int(same_day.sum()),'exclude_same_day_spread_sharpe':payoff_sharpe(a[~same_day]),'exclude_same_day_earnings_sharpe':payoff_sharpe(b[~same_day])}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(regime.to_string());print(annual.to_string());print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
