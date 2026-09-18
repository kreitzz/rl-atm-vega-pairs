"""Compare matched fixed and ladder policies, including paired uncertainty."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

if __package__:
    from .run_first_pass import payoff_sharpe
else:
    from run_first_pass import payoff_sharpe


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fixed',type=Path,default=Path('artifacts/short_rv_momentum/walk_forward'))
    p.add_argument('--ladder',type=Path,default=Path('artifacts/sizing_ladder/walk_forward'))
    p.add_argument('--output',type=Path,default=Path('reports'))
    args=p.parse_args()
    fixed_summary=json.loads((args.fixed/'summary.json').read_text())
    ladder_summary=json.loads((args.ladder/'summary.json').read_text())
    fixed=pd.read_csv(args.fixed/'predictions.csv',index_col='date',parse_dates=['date','end'])
    ladder=pd.read_csv(args.ladder/'predictions.csv',index_col='date',parse_dates=['date','end'])
    pd.testing.assert_frame_equal(fixed[['end','long_reward','spread']],ladder[['end','long_reward','spread']])
    scale=ladder_summary['assumptions']['max_vega_per_dollar_per_leg_per_entry']
    cost=ladder_summary['assumptions']['cost_per_full_size_pair']
    fixed_scores={r['model']:r for r in fixed_summary['scores']}
    comparisons=[]; sizes=[]; annual=[]
    for row in ladder_summary['scores']:
        name=row['model'];r=ladder[name+'_reward'].to_numpy();baseline=fixed[name+'_reward'].to_numpy()
        a=ladder[name+'_action'].to_numpy()
        rng=np.random.default_rng(2026)
        differences=[]
        for _ in range(2000):
            ix=((rng.integers(len(r),size=int(np.ceil(len(r)/63)))[:,None]+np.arange(63))%len(r)).ravel()[:len(r)]
            differences.append(payoff_sharpe(r[ix])-payoff_sharpe(baseline[ix]))
        lo,hi=np.quantile(differences,[.025,.975])
        # At each entry close, aggregate still-active cohorts including the new one.
        active=np.array([np.abs(a[(ladder.index<=date)&(ladder.end>date)]).sum() for date in ladder.index])*scale
        comparisons.append({'model':name,'fixed_sharpe':fixed_scores[name]['annualized_sharpe'],
                            'ladder_sharpe':row['annualized_sharpe'],'difference':row['annualized_sharpe']-fixed_scores[name]['annualized_sharpe'],
                            'paired_ci_low':lo,'paired_ci_high':hi,'mean_absolute_size':float(np.abs(a).mean()),
                            'trade_fraction':float((a!=0).mean()),'peak_active_vega_per_dollar_per_leg':float(active.max()),
                            'nonoverlap_fixed_sharpe':fixed_scores[name]['nonoverlap_sharpe'],'nonoverlap_ladder_sharpe':row['nonoverlap_sharpe']})
        for size in sorted(np.unique(np.abs(a))):
            chosen=np.abs(a)==size
            perunit=(np.sign(a[chosen])*ladder.long_reward.to_numpy()[chosen]-cost) if size else np.zeros(chosen.sum())
            sizes.append({'model':name,'size_fraction':size,'vega_per_dollar_per_leg':size*scale,'entries':int(chosen.sum()),
                          'mean_net_per_unit_vega_when_traded':float(perunit.mean()) if size else None,
                          'win_rate_when_traded':float((r[chosen]>0).mean()) if size else None})
        for year,g in ladder.groupby(ladder.index.year):
            rr=g[name+'_reward']
            annual.append({'year':year,'model':name,'net_pnl_per_dollar_max_vega_per_leg':float(rr.sum()),
                           'cohort_return_on_reference_capital':float(rr.sum()*scale),'annualized_sharpe':payoff_sharpe(rr),
                           'mean_absolute_size':float(g[name+'_action'].abs().mean()),'entries':len(g)})
    args.output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(comparisons).to_csv(args.output/'sizing_comparison.csv',index=False)
    pd.DataFrame(sizes).to_csv(args.output/'sizing_allocations.csv',index=False)
    pd.DataFrame(annual).to_csv(args.output/'sizing_annual.csv',index=False)
    print(pd.DataFrame(comparisons).to_string(index=False))


if __name__=='__main__':main()
