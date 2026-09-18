"""Wider execution costs: frozen-policy stress and separately retrained policies."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

if __package__:
    from .run_first_pass import net_rewards,payoff_sharpe
else:
    from run_first_pass import net_rewards,payoff_sharpe


FEATURES={'rv30cal':['spread','amd_earnings_days_capped','nvda_earnings_days_capped','trailing_rv_amd','trailing_rv_nvda']}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--panel',type=Path,default=Path('artifacts/short_rv_momentum/dataset.parquet'))
    p.add_argument('--output',type=Path,default=Path('artifacts/execution_costs'))
    p.add_argument('--report-output',type=Path,default=Path('reports/execution_costs'))
    p.add_argument('--costs',type=float,nargs='+',default=[.5,1.,1.5,2.])
    args=p.parse_args()
    if any(not np.isfinite(c) or c<0 for c in args.costs):p.error('Costs must be finite and nonnegative')
    if .5 not in args.costs:p.error('Include 0.5 as the reference cost')
    args.output.mkdir(parents=True,exist_ok=True)
    args.report_output.mkdir(parents=True,exist_ok=True)
    features_path=args.output/'features.json'
    features_path.write_text(json.dumps(FEATURES,indent=2)+'\n')
    walker=Path(__file__).with_name('run_walk_forward.py')
    runs={}
    for cost in sorted(set(args.costs)):
        destination=args.output/f'cost_{cost:g}'
        log=args.output/f'cost_{cost:g}.log'
        cmd=[sys.executable,str(walker),'--panel',str(args.panel),'--output',str(destination),
             '--features-json',str(features_path),'--reference-model','rv30cal','--comparison-model','rv30cal',
             '--sizing-ladder','0','.25','.5','.75','1','--max-vega-per-dollar','.001','--cost',str(cost)]
        print(f'Training at {cost:g} vol points all-in per full-size pair round trip...',flush=True)
        with log.open('w') as handle:
            subprocess.run(cmd,stdout=handle,stderr=subprocess.STDOUT,check=True)
        summary=json.loads((destination/'summary.json').read_text())
        scores=next(row for row in summary['scores'] if row['model']=='rv30cal')
        predictions=pd.read_csv(destination/'predictions.csv',index_col='date',parse_dates=['date','end'])
        runs[cost]=(scores,predictions)
        print(f'Cost {cost:g}: annualized payoff Sharpe {scores["annualized_sharpe"]:.3f}',flush=True)
    reference=runs[.5][1]
    old_path=Path('artifacts/sizing_ladder/walk_forward/predictions.csv')
    if old_path.exists():
        old=pd.read_csv(old_path,index_col='date',parse_dates=['date'])
        pd.testing.assert_series_equal(reference.rv30cal_action,old.rv30cal_action,check_dtype=False)
        np.testing.assert_allclose(reference.rv30cal_reward,old.rv30cal_reward,atol=1e-12)
    rows=[];annual=[];allocations=[]
    previous_frozen=None
    for cost,(score,d) in runs.items():
        pd.testing.assert_frame_equal(d[['end','long_reward','spread']],reference[['end','long_reward','spread']])
        a=d.rv30cal_action.to_numpy();reward=d.rv30cal_reward.to_numpy()
        frozen=net_rewards(reference.rv30cal_action.to_numpy(),reference.long_reward.to_numpy(),cost)
        np.testing.assert_allclose(reward,net_rewards(a,d.long_reward.to_numpy(),cost),atol=1e-12)
        if previous_frozen is not None:assert np.all(frozen<=previous_frozen+1e-12)
        previous_frozen=frozen
        gross=a*d.long_reward.to_numpy();charges=cost*np.abs(a)
        rng=np.random.default_rng(2026);differences=[]
        for _ in range(2000):
            ix=((rng.integers(len(d),size=int(np.ceil(len(d)/63)))[:,None]+np.arange(63))%len(d)).ravel()[:len(d)]
            differences.append(payoff_sharpe(reward[ix])-payoff_sharpe(frozen[ix]))
        low,high=np.quantile(differences,[.025,.975])
        rows.append({'pair_round_trip_cost':cost,'frozen_policy_sharpe':payoff_sharpe(frozen),'retrained_policy_sharpe':score['annualized_sharpe'],
                     'retrained_sharpe_ci_low':score['sharpe_ci'][0],'retrained_sharpe_ci_high':score['sharpe_ci'][1],
                     'retrained_minus_frozen_ci_low':low,'retrained_minus_frozen_ci_high':high,
                     'trade_fraction':float((a!=0).mean()),'mean_size':float(np.abs(a).mean()),
                     'mean_gross_per_entry':float(gross.mean()),'mean_cost_per_entry':float(charges.mean()),'mean_net_per_entry':float(reward.mean()),
                     'frozen_mean_net_per_entry':float(frozen.mean()),'nonoverlap_retrained_sharpe':score['nonoverlap_sharpe']})
        for size in sorted(np.unique(np.abs(a))):
            allocations.append({'pair_round_trip_cost':cost,'size_fraction':size,'entries':int((np.abs(a)==size).sum())})
        for year,g in d.groupby(d.index.year):
            idx=d.index.year==year
            annual.append({'year':year,'pair_round_trip_cost':cost,'frozen_policy_sharpe':payoff_sharpe(frozen[idx]),
                           'retrained_policy_sharpe':payoff_sharpe(g.rv30cal_reward),
                           'retrained_net_pnl_per_dollar_max_vega_per_leg':float(g.rv30cal_reward.sum()),
                           'frozen_net_pnl_per_dollar_max_vega_per_leg':float(frozen[idx].sum()),
                           'mean_size':float(g.rv30cal_action.abs().mean()),'trade_fraction':float((g.rv30cal_action!=0).mean()),'entries':len(g)})
    result=pd.DataFrame(rows)
    result.to_csv(args.report_output/'comparison.csv',index=False)
    pd.DataFrame(annual).to_csv(args.report_output/'annual.csv',index=False)
    pd.DataFrame(allocations).to_csv(args.report_output/'allocations.csv',index=False)
    print(result.to_string(index=False))


if __name__=='__main__':main()
