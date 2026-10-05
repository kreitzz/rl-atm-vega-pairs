"""Exact prior-inventory option/hedge attribution on the largest losing days."""
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.black_scholes import call_values
from scripts.diagnose_current_contract_losses import REPORT, POLICIES, read_csv


def main():
    events=pd.read_csv(REPORT/'worst_pair_days.csv',parse_dates=['date'])
    events=events[events.policy.isin(['earlier_filtered_rl','surface_filtered_rl','surface_filtered_nn'])].groupby('policy',sort=False).head(5)
    rows=[]
    for event in events.itertuples():
        root,subdir,*_=POLICIES[event.policy];folder=root/event.pair/subdir
        nav=pd.read_parquet(folder/'daily_nav.parquet');prev=nav.index[nav.index.get_loc(event.date)-1]
        trades=read_csv(folder/'trades.csv',['date'])
        held=trades[trades.date.le(prev)].groupby(['secid','optionid']).quantity.sum();held=held[held.ne(0)]
        event_rows=[]
        for sid,positions in held.groupby(level='secid'):
            frames=[]
            for year in sorted({prev.year,event.date.year}):
                p=Path('data/contract_history_shared')/str(year)/f'{int(sid)}.parquet'
                frames.append(pd.read_parquet(p,filters=[('date','in',[prev,event.date]),('optionid','in',positions.index.get_level_values('optionid').tolist())]))
            q=pd.concat(frames,ignore_index=True).set_index(['date','optionid'])
            reconstructed_hedge=0.
            for (_,oid),qty in positions.items():
                a=q.loc[(prev,oid)];b=q.loc[(event.date,oid)]
                ratio=nav.loc[event.date,f'split_factor_{int(sid)}']
                old_mid=(a.bid+a.ask)/2;new_mid=(b.bid+b.ask)/2
                spot_move=ratio*b.close-a.close
                option_pnl=qty*100*(ratio*new_mid-old_mid)
                hedge_pnl=-qty*100*a.delta*spot_move
                reconstructed_hedge-=qty*100*a.delta
                constant_iv_price=float(call_values(b.close-b.known_dividend_pv,b.strike,b.dte/365,b.bs_rate,a.bs_iv)[0])
                constant_iv=qty*100*(ratio*constant_iv_price-old_mid-a.delta*spot_move)
                remarking=qty*100*ratio*(new_mid-constant_iv_price)
                np.testing.assert_allclose(constant_iv+remarking,option_pnl+hedge_pnl,atol=1e-7,rtol=0)
                event_rows.append(dict(policy=event.policy,pair=event.pair,label=event.label,date=event.date,secid=int(sid),optionid=int(oid),
                    prior_quantity=qty,stock_return=ratio*b.close/a.close-1,prior_iv=a.bs_iv*100,current_iv=b.bs_iv*100,
                    prior_bid=a.bid,prior_ask=a.ask,current_bid=b.bid,current_ask=b.ask,current_dte=b.dte,
                    prior_relative_width=(a.ask-a.bid)/old_mid,current_relative_width=(b.ask-b.bid)/new_mid,
                    option_mid_pnl=option_pnl,stock_hedge_gross_pnl=hedge_pnl,gross_hedged_pnl=option_pnl+hedge_pnl,
                    spot_time_rates_dividends_at_prior_iv=constant_iv,repricing_residual=remarking,
                    valid_iv_both=bool(np.isfinite([a.bs_iv,b.bs_iv]).all())))
            np.testing.assert_allclose(reconstructed_hedge,nav.loc[prev,f'hedge_{int(sid)}'],atol=1e-7,rtol=0)
        np.testing.assert_allclose(sum(r['gross_hedged_pnl'] for r in event_rows),event.option_mid_pnl+event.stock_hedge_gross_pnl,atol=1e-6,rtol=0)
        rows.extend(event_rows)
    frame=pd.DataFrame(rows);frame.to_csv(REPORT/'worst_day_contracts.csv',index=False)
    grouped=frame.groupby(['policy','pair','label','date','secid']).agg(stock_return=('stock_return','first'),
        option_mid_pnl=('option_mid_pnl','sum'),stock_hedge_gross_pnl=('stock_hedge_gross_pnl','sum'),
        gross_hedged_pnl=('gross_hedged_pnl','sum'),spot_time_rates_dividends_at_prior_iv=('spot_time_rates_dividends_at_prior_iv','sum'),
        repricing_residual=('repricing_residual','sum'),valid_iv_both=('valid_iv_both','all')).reset_index()
    grouped.to_csv(REPORT/'worst_day_legs.csv',index=False)
    print(grouped[grouped.date.eq('2022-01-24')].round(3).to_string(index=False))


if __name__=='__main__':main()
