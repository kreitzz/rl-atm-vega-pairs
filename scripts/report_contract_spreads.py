"""Observed executed-quote spreads in volatility points, without changing fills."""
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.black_scholes import invert_calls

ROOT=Path('artifacts/contracts/four_pairs')
TICKERS={101121:'AMD',108321:'NVDA',104533:'XOM',102968:'CVX',102936:'JPM',101966:'BAC',105759:'HD',107045:'LOW'}


def main():
    rows=[]
    for window in ['2025','history']:
        for pair in ['AMD_NVDA','XOM_CVX','JPM_BAC','HD_LOW']:
            folder=ROOT/window/pair/'rl_sized'
            path=folder/'trades.csv'
            partial=False
            if not path.exists():path=folder/'partial_trades.csv';partial=True
            if not path.exists():continue
            trades=pd.read_csv(path,parse_dates=['date'])
            columns=['date','optionid','secid','bid','ask','bs_vega','close','known_dividend_pv','strike','dte','bs_rate']
            q=pd.concat([pd.read_parquet(ROOT/'cache'/pair/f'{y}.parquet',columns=columns)
                         for y in sorted(trades.date.dt.year.unique())],ignore_index=True)
            t=trades.merge(q,on=['date','optionid','secid'],how='left',validate='many_to_one')
            assert t.bid.notna().all() and t.ask.notna().all()
            # Match the explicitly disclosed envelope on crossed held quotes.
            t['raw_crossed']=t.bid.gt(t.ask)
            bid=t[['bid','ask']].min(axis=1);ask=t[['bid','ask']].max(axis=1)
            effective=t.close-t.known_dividend_pv
            biv=invert_calls(effective,t.strike,t.dte/365,t.bs_rate,bid)[0]
            aiv=invert_calls(effective,t.strike,t.dte/365,t.bs_rate,ask)[0]
            t['exact_width_vol_points']=(aiv-biv)*100
            t['full_width_vol_points']=(ask-bid)/t.bs_vega.where(t.bs_vega.gt(0))
            t['commission_vol_points']=.65/(100*t.bs_vega.where(t.bs_vega.gt(0)))
            t['quantity_abs']=t.quantity.abs()
            for (secid,reason),g in t.groupby(['secid','reason']):
                valid=g.bs_vega.gt(0)&np.isfinite(g.bs_vega)
                v=g.loc[valid]
                weights=v.quantity_abs*100*v.bs_vega
                width=(v.ask-v.bid).abs()
                rows.append(dict(window=window,pair=pair,ticker=TICKERS[secid],reason=reason,partial_run=partial,
                    fills=len(g),exact_iv_fits=int(g.exact_width_vol_points.notna().sum()),
                    median_full_spread_vol=float(g.full_width_vol_points.median()),
                    p90_full_spread_vol=float(g.full_width_vol_points.quantile(.9)),
                    median_exact_iv_spread_vol=float(g.exact_width_vol_points.median()),
                    vega_weighted_full_spread_vol=float((v.quantity_abs*100*width).sum()/weights.sum()) if weights.sum()>0 else np.nan,
                    median_commission_vol_per_fill=float(g.commission_vol_points.median()),
                    spread_cost_dollars=float((g.quantity_abs*100*(g.ask-g.bid).abs()/2).sum()),
                    commission_dollars=float(g.commission.sum()),crossed_quotes=int(g.raw_crossed.sum())))
    result=pd.DataFrame(rows)
    out=Path('reports/selected_pair_backtest');out.mkdir(exist_ok=True)
    result.to_csv(out/'execution_spreads.csv',index=False)
    print(result[result.reason.eq('entry')].to_string(index=False))


if __name__=='__main__':main()
