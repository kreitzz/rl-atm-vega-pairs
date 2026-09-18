"""Annual entry-cohort P&L and payoff Sharpe from walk-forward predictions."""
import argparse
from pathlib import Path
import pandas as pd

if __package__:
    from .run_first_pass import payoff_sharpe
else:
    from run_first_pass import payoff_sharpe


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--predictions',type=Path,default=Path('artifacts/walk_forward/predictions.csv'))
    p.add_argument('--output',type=Path,default=Path('reports/walk_forward_annual.csv'))
    p.add_argument('--models',nargs='+',default=['spread_earnings_rv','spread_earnings_rv_vix'],
                   help='Models to print; the output CSV always includes every model.')
    args=p.parse_args()
    d=pd.read_csv(args.predictions,index_col='date',parse_dates=['date','end'])
    models=[c.removesuffix('_reward') for c in d.columns if c.endswith('_reward') and c!='long_reward']
    rows=[]
    for year,g in d.groupby(d.index.year):
        for model in models:
            reward=g[model+'_reward']
            action=g[model+'_action']
            rows.append({'year':year,'model':model,'entries':len(g),'trades':int((action!=0).sum()),
                         'net_pnl_per_dollar_vega_per_leg':float(reward.sum()),
                         'mean_net_payoff_per_entry':float(reward.mean()),
                         'annualized_payoff_sharpe':payoff_sharpe(reward),
                         'first_entry':str(g.index.min().date()),'last_entry':str(g.index.max().date()),
                         'last_settlement':str(g.end.max().date())})
    result=pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output,index=False)
    focus=result[result.model.isin(args.models)]
    print(focus.pivot(index='year',columns='model',values=['net_pnl_per_dollar_vega_per_leg','annualized_payoff_sharpe']).round(3).to_string())
    print('PERIODS', result.groupby('year').first()[['entries','first_entry','last_entry','last_settlement']].to_string())


if __name__=='__main__':main()
