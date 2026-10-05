"""Reconstruct delta exposure and the terminal AMD/NVDA drawdown independently."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def main():
    root=Path('artifacts/contracts/four_pairs_half_spread')
    folder=root/'history/AMD_NVDA/rl_sized'
    nav=pd.read_csv(folder/'partial_nav.csv',index_col=0,parse_dates=[0])
    trades=pd.read_csv(folder/'partial_trades.csv',parse_dates=['date'])
    failure=pd.read_csv(root/'status.csv').query('window == "history" and pair == "AMD_NVDA" and model == "rl_sized"').iloc[0]
    terminal=pd.Timestamp(failure['date']); prior=nav.index[-1]
    years=range(nav.index[0].year,terminal.year+1)
    q=pd.concat([pd.read_parquet(Path('artifacts/contracts/four_pairs/cache/AMD_NVDA')/f'{y}.parquet') for y in years])
    q=q.set_index(['date','optionid'])
    stocks=pd.concat([pd.read_parquet(Path('data/option_contracts/underlying')/f'year={y}.parquet',filters=[('secid','in',[101121,108321])]) for y in years])
    stocks['date']=pd.to_datetime(stocks.date);stocks=stocks.set_index(['date','secid'])
    bydate={d:g for d,g in trades.groupby('date')}
    holdings={};max_residual=0.
    for date,row in nav.iterrows():
        for tr in bydate.get(date,pd.DataFrame()).itertuples():
            holdings[tr.optionid]=holdings.get(tr.optionid,0)+tr.quantity
        holdings={oid:qty for oid,qty in holdings.items() if qty}
        delta={101121:0.,108321:0.}
        for oid,qty in holdings.items():
            mark=q.loc[(date,oid)]
            delta[int(mark.secid)]+=qty*100*mark.bs_delta
        for sid in delta:
            residual=abs(delta[sid]+row[f'hedge_{sid}'])
            assert np.isfinite(residual) and residual<1e-7
            max_residual=max(max_residual,residual)
    legs=[]
    for sid,ticker in [(101121,'AMD'),(108321,'NVDA')]:
        option_pnl=vega=0.;contracts=0
        for oid,qty in holdings.items():
            a=q.loc[(prior,oid)]
            if a.secid!=sid:continue
            b=q.loc[(terminal,oid)]
            option_pnl+=qty*100*((b.bid+b.ask-a.bid-a.ask)/2)
            vega+=qty*100*a.bs_vega;contracts+=qty
        previous_spot=stocks.loc[(prior,sid),'close'];spot=stocks.loc[(terminal,sid),'close']
        hedge=nav.loc[prior,f'hedge_{sid}'];hedge_pnl=hedge*(spot-previous_spot)
        legs.append(dict(ticker=ticker,stock_return=spot/previous_spot-1,contracts=int(contracts),
                         prior_vega=vega,hedge_shares=hedge,hedge_notional=abs(hedge*previous_spot),
                         option_mark_pnl=option_pnl,stock_hedge_pnl=hedge_pnl,combined_mark_pnl=option_pnl+hedge_pnl))
    terminal_pnl=failure.nav-nav.nav.iloc[-1]
    market_pnl=sum(x['combined_mark_pnl'] for x in legs)
    summary=dict(prior_date=str(prior.date()),failure_date=str(terminal.date()),initial_capital=1e6,
                 prior_nav=float(nav.nav.iloc[-1]),terminal_nav=float(failure.nav),terminal_pnl=float(terminal_pnl),
                 terminal_pnl_fraction_initial_capital=float(terminal_pnl/1e6),
                 prior_active_cohorts=int(nav.active_cohorts.iloc[-1]),prior_gross_vega=float(nav.gross_vega.iloc[-1]),
                 prior_net_vega=float(nav.net_vega.iloc[-1]),
                 gross_hedge_notional=float(sum(x['hedge_notional'] for x in legs)),
                 option_execution_cost=float(failure.option_execution_cost),commissions=float(trades.commission.sum()),
                 max_delta_residual_shares=float(max_residual),audited_sessions=len(nav),
                 terminal_market_pnl=float(market_pnl),terminal_cost_and_financing_pnl=float(terminal_pnl-market_pnl),legs=legs)
    report=Path('reports/selected_pair_backtest/half_spread')
    (report/'drawdown_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    annual=nav.groupby(nav.index.year).agg(nav_end=('nav','last'),pnl=('pnl','sum'),option_cost=('option_execution_cost','sum'))
    annual.loc[terminal.year,'nav_end']=failure.nav
    annual.loc[terminal.year,'pnl']+=terminal_pnl
    annual.loc[terminal.year,'option_cost']+=failure.option_execution_cost-nav.option_execution_cost.sum()
    annual.to_csv(report/'drawdown_annual.csv')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
