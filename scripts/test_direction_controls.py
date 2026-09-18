"""Matched-date, matched-size direction controls for each leg and the pair."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .run_first_pass import payoff_sharpe,SHARPE_ANNUALIZATION
else:
    from run_first_pass import payoff_sharpe,SHARPE_ANNUALIZATION


def strata_indices(folds, size):
    frame=pd.DataFrame({'fold':np.asarray(folds),'size':np.asarray(size)})
    return [g.index.to_numpy() for _,g in frame[frame['size']>0].groupby(['fold','size'],sort=True)]


def random_directions(sign, strata, method, rng):
    """Returns signed AMD direction; NVDA takes the opposite direction."""
    result=np.zeros(len(sign),dtype=float)
    for ix in strata:
        values=sign[ix]
        if method=='coin_flip':result[ix]=rng.choice([-1.,1.],size=len(ix))
        elif method=='stratified_shuffle':result[ix]=rng.permutation(values)
        elif method=='stratified_circular_shift':result[ix]=np.roll(values,int(rng.integers(len(ix))))
        else:raise ValueError('Unknown randomization method')
    return result


def leg_payoffs(direction,size,amd_gap,nvda_gap,pair_cost=1.):
    amd_gross=direction*size*amd_gap
    nvda_gross=-direction*size*nvda_gap
    half_cost=.5*pair_cost*size
    return np.column_stack([amd_gross-half_cost,nvda_gross-half_cost,amd_gross+nvda_gross-2*half_cost])


def axis_sharpes(rewards):
    sd=rewards.std(axis=0,ddof=1)
    return SHARPE_ANNUALIZATION*np.divide(rewards.mean(axis=0),sd,out=np.zeros(rewards.shape[1]),where=sd>1e-12)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--predictions',type=Path,default=Path('artifacts/execution_costs/cost_1/predictions.csv'))
    p.add_argument('--panel',type=Path,default=Path('artifacts/short_rv_momentum/dataset.parquet'))
    p.add_argument('--output',type=Path,default=Path('reports/direction_controls'))
    p.add_argument('--artifacts',type=Path,default=Path('artifacts/direction_controls'))
    p.add_argument('--trials',type=int,default=5000)
    p.add_argument('--seed',type=int,default=2026)
    args=p.parse_args()
    if args.trials<100:p.error('Use at least 100 randomization trials')
    pred=pd.read_csv(args.predictions,index_col='date',parse_dates=['date','end'])
    panel=pd.read_parquet(args.panel)
    d=pred.join(panel[['iv_amd','iv_nvda','rv_amd','rv_nvda']],validate='one_to_one')
    assert d[['iv_amd','iv_nvda','rv_amd','rv_nvda']].notna().all().all()
    size=d.rv30cal_action.abs().to_numpy();sign=np.sign(d.rv30cal_action.to_numpy())
    amd_gap=(d.rv_amd-d.iv_amd).to_numpy();nvda_gap=(d.rv_nvda-d.iv_nvda).to_numpy()
    active=size>0
    actual=leg_payoffs(sign,size,amd_gap,nvda_gap)
    np.testing.assert_allclose(actual[:,2],d.rv30cal_reward,atol=1e-10)
    actual_totals=actual.sum(axis=0);actual_sharpes=axis_sharpes(actual)
    names=['AMD','NVDA','PAIR']
    fixed=[];annual=[]
    policies={'policy':sign,'always_long_AMD_short_NVDA':np.ones(len(d)),
              'always_short_AMD_long_NVDA':-np.ones(len(d))}
    for policy,direction in policies.items():
        reward=leg_payoffs(direction,size,amd_gap,nvda_gap)
        for j,leg in enumerate(names):
            if policy=='policy':control='policy'
            elif leg=='PAIR':control=policy
            elif (policy=='always_long_AMD_short_NVDA' and leg=='AMD') or (policy=='always_short_AMD_long_NVDA' and leg=='NVDA'):control='always_long'
            else:control='always_short'
            cost=(1 if leg=='PAIR' else .5)*size.sum()
            fixed.append({'leg':leg,'control':control,'gross_pnl':float(reward[:,j].sum()+cost),'assigned_cost':float(cost),
                          'net_pnl':float(reward[:,j].sum()),'annualized_payoff_sharpe':payoff_sharpe(reward[:,j]),
                          'win_rate_active_entries':float((reward[active,j]>0).mean()),'active_entries':int(active.sum())})
            for year in sorted(d.index.year.unique()):
                mask=d.index.year==year
                annual.append({'year':year,'leg':leg,'control':control,'net_pnl':float(reward[mask,j].sum()),'annualized_payoff_sharpe':payoff_sharpe(reward[mask,j])})
    strata=strata_indices(d.fold,size)
    mixed=sum(len(ix) for ix in strata if len(np.unique(sign[ix]))>1)
    rows=[]
    args.output.mkdir(parents=True,exist_ok=True);args.artifacts.mkdir(parents=True,exist_ok=True)
    # One RNG per control keeps all reported trials reproducible.
    for method in ['coin_flip','stratified_shuffle','stratified_circular_shift']:
        rng=np.random.default_rng(args.seed)
        totals=np.empty((args.trials,3));sharpes=np.empty((args.trials,3))
        annual_random=np.empty((args.trials,len(d.index.year.unique()),3))
        years=sorted(d.index.year.unique())
        year_masks=[d.index.year==year for year in years]
        for i in range(args.trials):
            direction=random_directions(sign,strata,method,rng)
            reward=leg_payoffs(direction,size,amd_gap,nvda_gap)
            totals[i]=reward.sum(axis=0);sharpes[i]=axis_sharpes(reward)
            for yi,mask in enumerate(year_masks):annual_random[i,yi]=reward[mask].sum(axis=0)
        np.savez_compressed(args.artifacts/f'{method}.npz',net_pnl=totals,annualized_sharpe=sharpes,annual_net_pnl=annual_random,years=years)
        for j,leg in enumerate(names):
            pnl_lo,pnl_hi=np.quantile(totals[:,j],[.025,.975]);sr_lo,sr_hi=np.quantile(sharpes[:,j],[.025,.975])
            rows.append({'method':method,'leg':leg,'policy_net_pnl':float(actual_totals[j]),'random_mean_net_pnl':float(totals[:,j].mean()),
                         'random_pnl_p025':pnl_lo,'random_pnl_p975':pnl_hi,
                         'fraction_random_pnl_at_least_policy':float((1+np.sum(totals[:,j]>=actual_totals[j]))/(args.trials+1)),
                         'policy_sharpe':float(actual_sharpes[j]),'random_mean_sharpe':float(sharpes[:,j].mean()),'random_sharpe_p025':sr_lo,'random_sharpe_p975':sr_hi,
                         'fraction_random_sharpe_at_least_policy':float((1+np.sum(sharpes[:,j]>=actual_sharpes[j]))/(args.trials+1))})
        print(f'Completed {method}: {args.trials} trials',flush=True)
    pd.DataFrame(fixed).to_csv(args.output/'fixed_controls.csv',index=False)
    pd.DataFrame(rows).to_csv(args.output/'random_controls.csv',index=False)
    pd.DataFrame(annual).to_csv(args.output/'annual_fixed_controls.csv',index=False)
    summary={'entries':len(d),'active_entries':int(active.sum()),'strata':len(strata),'active_entries_in_mixed_direction_strata':mixed,
             'trials_per_method':args.trials,'seed':args.seed,'pair_cost':1.,'leg_cost_allocation':'50/50 assumption; not measured per-leg execution costs',
             'methods':{'coin_flip':'independent fair directions, preserves dates and sizes but not long/short mix or persistence',
                        'stratified_shuffle':'permute sign within each walk-forward fold and exact size; preserves directional count and vega mix within strata',
                        'stratified_circular_shift':'rotate sign within each fold/size stratum in chronological order; preserves that stratum\'s circular direction runs and exact mix, not all cross-stratum dependence'},
             'scope':'conditional on learned entry selection and size; no retraining; measures direction alignment in this inspected historical sample, not independent evidence of generalizable edge'}
    (args.output/'methodology.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(pd.DataFrame(fixed).to_string(index=False));print(pd.DataFrame(rows).to_string(index=False));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
