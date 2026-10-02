"""Audit NAV-linked caps and compare results without omitting failed runs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def main():
    root=Path('artifacts/contracts/four_pairs_risk_controlled')
    out=Path('reports/selected_pair_backtest/risk_controlled')
    status=pd.read_csv(out/'status.csv')
    cfg=json.loads((root/'manifest.json').read_text())['settings']
    rows=[];annual=[]
    for row in status[status.status.eq('complete')].itertuples():
        folder=root/str(row.window)/row.pair/row.model
        nav=pd.read_csv(folder/'daily_nav.csv',index_col=0,parse_dates=[0])
        decisions=pd.read_csv(folder/'decisions.csv')
        ratios={'hedge_leverage':nav.gross_hedge_notional/nav.nav,
                'reserve_fraction':nav.capital_reserve/nav.nav,
                'stress_fraction':nav.stress_loss/nav.nav,
                'vega_fraction':nav.gross_vega/nav.nav}
        limits=[cfg['max_hedge_leverage'],cfg['max_reserve_fraction'],cfg['max_stress_loss_fraction'],cfg['max_gross_vega']/cfg['capital']]
        for series,limit in zip(ratios.values(),limits):assert (series<=limit+1e-10).all()
        assert nav.nav.gt(0).all()
        entered=decisions[decisions.status.eq('entered')]
        for field in ['vega_a','vega_b']:
            assert (entered[field]<=cfg['entry_vega_per_leg']*entered.action.abs()*entered.sizing_nav/cfg['capital']+1e-8).all()
        rows.append(dict(window=row.window,pair=row.pair,model=row.model,
                         **{'max_'+k:float(v.max()) for k,v in ratios.items()},
                         worst_daily_return=float(nav.return_.min()),min_nav=float(nav.nav.min()),
                         risk_exit_cohorts=int(nav.risk_exit_cohorts.sum()),
                         blocked_entry_signals=int(decisions.status.eq('risk_limit_entry').sum()),
                         risk_reduction_days=int(nav.pre_trade_risk_breach.sum())))
    pd.DataFrame(rows).to_csv(out/'risk_audit.csv',index=False)
    for model in ['rl_sized','nn_fixed','higher_iv']:
        for window in ['history','2025']:
            path=root/f'{window}_{model}_portfolio.csv'
            if not path.exists():continue
            nav=pd.read_csv(path,index_col=0,parse_dates=[0])
            for year,g in nav.groupby(nav.index.year):
                previous=4*cfg['capital'] if g.index[0]==nav.index[0] else nav.loc[:g.index[0]].nav.iloc[-2]
                annual.append(dict(window=window,model=model,year=int(year),total_return=g.nav.iloc[-1]/previous-1,
                                   annualized_excess_sharpe=np.sqrt(252)*g.excess_return.mean()/g.excess_return.std()))
    pd.DataFrame(annual).to_csv(out/'portfolio_annual.csv',index=False)
    old=pd.read_csv('reports/selected_pair_backtest/half_spread/portfolio.csv',dtype={'window':str})
    new=pd.read_csv(out/'portfolio.csv',dtype={'window':str})
    comparison=old.merge(new,on=['window','model'],how='outer',suffixes=('_fixed','_risk_controlled'))
    comparison.to_csv(out/'portfolio_comparison.csv',index=False)
    print(new.to_string(index=False))
    print(status[(status.model=='rl_sized')][['window','pair','status','total_return','annualized_excess_sharpe','max_drawdown']].to_string(index=False))
    print('All completed runs satisfy end-of-day risk limits and NAV-scaled entry budgets.')


if __name__=='__main__':main()
