"""Matched 200/500/1000/2000-update diagnosis; resume monthly checkpoints."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import signal_allocation_variance as s

OUT=s.ROOT/'convergence'
RISKS=s.RISKS
STEPS=[200,500,1000,2000]


def fitted(data,index):
    periods=data['periods'];current=periods[index];cutoff=current['cutoff']
    train=[i for i,m in enumerate(periods[:index]) if m['end']<cutoff and m['cutoff']>=cutoff-pd.Timedelta(weeks=104)]
    if len(train)<52:return None
    path=OUT/'models'/(str(cutoff.to_period('M'))+'.json')
    if path.exists():return json.loads(path.read_text())
    n=s.N
    states=np.concatenate([np.repeat(np.column_stack([np.tile(periods[i]['market'],(n,1)),periods[i]['before'][:,1]]),n,axis=0) for i in train])
    following=np.concatenate([np.column_stack([np.tile(periods[i+1]['market'],(n,1)),periods[i]['after'][:,1]]) for i in train])
    center=states.mean(axis=0);scale=np.maximum(states.std(axis=0),.1)
    x=s.core.design(states,np.tile(np.arange(n),len(train)*n),center,scale);coef=x.shape[1]
    nx=s.core.design(np.repeat(following,n,axis=0),np.tile(np.arange(n),len(following)),center,scale).reshape(len(following),n,coef)
    projection=x.reshape(len(train),n,n,coef).sum(axis=1).reshape(-1,coef).T/(n*n)
    ref=np.concatenate([periods[i]['paths'][3,3] for i in train]);vol=max(float(ref.std(ddof=1)),1e-8)
    profit=np.concatenate([periods[i]['paths'].sum(axis=-1).ravel()/vol for i in train])
    squares=np.concatenate([(periods[i]['paths']**2).sum(axis=-1).ravel()/vol**2 for i in train])
    penalty=10.*np.eye(coef);penalty[0,0]=.01
    inverse=np.linalg.inv(x.T@x/(n*n)+penalty)
    moments=x.T@np.column_stack([profit,squares])/(n*n)
    reward=moments[:,[0]]-moments[:,[1]]*RISKS[None,:]
    def update(beta):return inverse@(reward+.99*(projection@(nx@beta).max(axis=1)))
    beta=inverse@reward;snapshots={}
    for step in range(2,max(STEPS)+1):
        beta=update(beta)
        if step in STEPS:snapshots[step]=beta.copy()
    assert np.isfinite(beta).all()
    final_predictions=(x@beta).reshape(-1,n,len(RISKS))
    final_adv=final_predictions-final_predictions.mean(axis=1,keepdims=True)
    diagnostics=[]
    for step,b in snapshots.items():
        prediction=(x@b).reshape(-1,n,len(RISKS))
        advantage=prediction-prediction.mean(axis=1,keepdims=True)
        residual=update(b)-b
        for j,risk in enumerate(RISKS):
            greedy=(nx@b)[:,:,j].argmax(axis=1)
            jacobian=.99*inverse@projection@nx[np.arange(len(nx)),greedy]
            diagnostics.append(dict(step=step,risk=risk,
                coefficient_relative_residual=float(np.linalg.norm(residual[:,j])/max(np.linalg.norm(b[:,j]),1e-12)),
                advantage_relative_rmse_vs_2000=float(np.linalg.norm(advantage[:,:,j]-final_adv[:,:,j])/max(np.linalg.norm(final_adv[:,:,j]),1e-12)),
                train_action_disagreement_vs_2000=float((prediction[:,:,j].argmax(axis=1)!=final_predictions[:,:,j].argmax(axis=1)).mean()),
                local_update_spectral_radius=float(np.max(np.abs(np.linalg.eigvals(jacobian)))),
                coefficient_norm=float(np.linalg.norm(b[:,j]))))
    value=dict(cutoff=str(cutoff),last_training_end=str(max(periods[i]['end'] for i in train)),
        center=center.tolist(),scale=scale.tolist(),beta={str(k):v.tolist() for k,v in snapshots.items()},diagnostics=diagnostics)
    assert pd.Timestamp(value['last_training_end'])<cutoff
    s.save(path,value)
    print('Fitted '+str(cutoff.to_period('M')),flush=True)
    return value


def main():
    data=s.load();periods=data['periods'];models=[];model=None;month=None
    for index,m in enumerate(periods):
        key=str(m['cutoff'].to_period('M'))
        if key!=month or model is None:model=fitted(data,index);month=key
        models.append(model)
    cash=np.concatenate([m['cash'] for m in periods]);positions=np.concatenate([m['indices'] for m in periods]);dates=data['pair'].index[positions]
    decision_dates=pd.DatetimeIndex([m['cutoff'] for m in periods]);records=[];choices=[];reference={}
    for step in STEPS:
        for j,risk in enumerate(RISKS):
            previous=3;paths=[];actions=[]
            for m,model in zip(periods,models):
                action=3
                if model:
                    state=np.r_[m['market'],m['before'][previous,1]]
                    score=s.core.design(np.tile(state,(s.N,1)),np.arange(s.N),np.array(model['center']),np.array(model['scale']))@np.asarray(model['beta'][str(step)])[:,j]
                    action=int(score.argmax())
                paths.append(m['paths'][previous,action]);actions.append(action);previous=action
            excess=np.concatenate(paths)
            book=pd.DataFrame(dict(excess_return=excess,cash_return=cash,nav=1e7*np.cumprod(1+excess+cash)),index=dates)
            if step==200:reference[risk]=np.array(actions)
            for period,start,end in s.core.PERIODS:
                mask=(decision_dates>=start)&(decision_dates<=end)
                records.append(dict(step=step,risk=risk,period=period,
                    action_disagreement_vs_200=float((np.asarray(actions)[mask]!=reference[risk][mask]).mean()),
                    mean_straddle_target=float(s.core.SHARES[np.asarray(actions)[mask]].mean()),**s.core.metrics(book,start,end)))
            for date,action in zip(decision_dates,actions):choices.append(dict(step=step,risk=risk,date=date,straddle_weight=s.core.SHARES[action]))
    pd.DataFrame(records).to_csv(s.REPORT/'convergence_results.csv',index=False)
    pd.DataFrame(choices).to_csv(s.REPORT/'convergence_decisions.csv',index=False)
    diagnostics=[]
    for p in sorted((OUT/'models').glob('*.json')):
        v=json.loads(p.read_text());diagnostics.extend(dict(month=p.stem,**r) for r in v['diagnostics'])
    pd.DataFrame(diagnostics).to_csv(s.REPORT/'convergence_diagnostics.csv',index=False)
    baseline=pd.read_csv(s.REPORT/'results.csv')
    for r in records:
        if r['step']!=200:continue
        expected=baseline.loc[(baseline.variance_penalty==r['risk'])&(baseline.period==r['period'])].iloc[0]
        assert abs(r['annualized_excess_sharpe']-expected.annualized_excess_sharpe)<1e-8
        assert abs(r['profit_above_cash']-expected.profit_above_cash)<1e-5
    s.save(s.REPORT/'convergence_verification.json',dict(original_200_step_results_reproduced=True,monthly_models=len(list((OUT/'models').glob('*.json'))),iterations=STEPS,policies=RISKS.tolist(),later_actions_identical_across_budgets=all(r['action_disagreement_vs_200']==0 for r in records if r['period']=='evaluation')))
    print(pd.DataFrame(records).loc[lambda x:x.period.eq('evaluation')].to_string(index=False),flush=True)


if __name__=='__main__':main()
