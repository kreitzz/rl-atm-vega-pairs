"""Direction, initial-vega allocation, and leg-level proxy P&L attribution."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--predictions',type=Path,default=Path('artifacts/execution_costs/cost_1/predictions.csv'))
    p.add_argument('--panel',type=Path,default=Path('artifacts/short_rv_momentum/dataset.parquet'))
    p.add_argument('--output',type=Path,default=Path('reports/reference_positions'))
    args=p.parse_args()
    pred=pd.read_csv(args.predictions,index_col='date',parse_dates=['date','end'])
    panel=pd.read_parquet(args.panel)
    d=pred.join(panel[['iv_amd','iv_nvda','rv_amd','rv_nvda']],how='left',validate='one_to_one')
    assert d[['iv_amd','iv_nvda','rv_amd','rv_nvda']].notna().all().all()
    a=d.rv30cal_action
    d['direction']=np.select([a<0,a>0],['short_AMD_long_NVDA','long_AMD_short_NVDA'],default='flat')
    d['absolute_size']=a.abs()
    d['amd_gross']=a*(d.rv_amd-d.iv_amd)
    d['nvda_gross']=-a*(d.rv_nvda-d.iv_nvda)
    # Infer observed total charges so attribution also respects the saved run.
    d['gross']=d.amd_gross+d.nvda_gross
    d['cost']=d.gross-d.rv30cal_reward
    d['net']=d.rv30cal_reward
    np.testing.assert_allclose(d.gross,a*d.long_reward,atol=1e-10)
    np.testing.assert_allclose(d.cost,a.abs(),atol=1e-10) # reference cost=1.0
    trades=int((a!=0).sum());vega=d.absolute_size.sum()
    rows=[]
    for direction,g in d.groupby('direction',sort=False):
        traded=direction!='flat'
        rows.append({'direction':direction,'entries':len(g),'fraction_all_entries':len(g)/len(d),
                     'fraction_trades':len(g)/trades if traded else 0.,
                     'size_weighted_trade_share':g.absolute_size.sum()/vega,
                     'mean_size':g.absolute_size.mean(),'full_size_equivalent_entries':g.absolute_size.sum(),
                     'amd_gross':g.amd_gross.sum(),'nvda_gross':g.nvda_gross.sum(),'gross':g.gross.sum(),
                     'cost':g.cost.sum(),'net':g.net.sum(),'net_pnl_share':g.net.sum()/d.net.sum(),
                     'win_rate_trades':float((g.net>0).mean()) if traded else None,
                     'mean_net_per_trade':float(g.net.mean()) if traded else None})
    summary=pd.DataFrame(rows)
    year=d.groupby([d.index.year,'direction']).agg(entries=('net','size'),full_size_equivalents=('absolute_size','sum'),amd_gross=('amd_gross','sum'),nvda_gross=('nvda_gross','sum'),cost=('cost','sum'),net=('net','sum')).reset_index(names=['year','direction'])
    allocation=d.groupby(['direction','absolute_size']).agg(entries=('net','size'),amd_gross=('amd_gross','sum'),nvda_gross=('nvda_gross','sum'),cost=('cost','sum'),net=('net','sum')).reset_index()
    legs=pd.DataFrame([{'leg':'AMD','gross':d.amd_gross.sum()},{'leg':'NVDA','gross':d.nvda_gross.sum()},{'leg':'pair_cost','gross':-d.cost.sum()},{'leg':'net','gross':d.net.sum()}])
    args.output.mkdir(parents=True,exist_ok=True)
    summary.to_csv(args.output/'directions.csv',index=False)
    year.to_csv(args.output/'annual_directions.csv',index=False)
    allocation.to_csv(args.output/'sizes.csv',index=False)
    legs.to_csv(args.output/'legs.csv',index=False)
    assert np.isclose(summary.net.sum(),d.net.sum())
    print(summary.to_string(index=False))
    print(legs.to_string(index=False))
    print(year.to_string(index=False))


if __name__=='__main__':main()
