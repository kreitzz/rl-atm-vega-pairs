"""Describe matched ATM call/put marks; parity residuals are not arbitrage claims."""
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    rows=[]
    for p in sorted(Path('data/call_put/history').glob('*/*_selection.parquet')):
        put=pd.read_parquet(p);callpath=Path('data/contract_history_shared')/p.parent.name/p.name
        if put.empty or not callpath.exists():continue
        call=pd.read_parquet(callpath)
        keys=['date','secid','expiry','strike'];cols=keys+['bid','ask','bs_iv','bs_vega','bs_rate','known_dividend_pv','close']
        x=call[cols].merge(put[cols],on=keys,suffixes=('_c','_p'),validate='one_to_one')
        carry=x.close_c-x.known_dividend_pv_c-x.strike*np.exp(-x.bs_rate_c*(x.expiry-x.date).dt.days/365)
        residual=(x.bid_c+x.ask_c-x.bid_p-x.ask_p)/2-carry
        scale=(x.bs_vega_c+x.bs_vega_p)/2
        for sid,g in x.groupby('secid'):
            rv=(residual.loc[g.index]/scale.loc[g.index]).replace([np.inf,-np.inf],np.nan).dropna()
            iv=(g.bs_iv_p-g.bs_iv_c)*100
            rows.append(dict(year=int(p.parent.name),secid=int(sid),matched_dates=len(g),
                median_put_minus_call_iv=iv.median(),median_abs_parity_vol_points=rv.abs().median(),
                p95_abs_parity_vol_points=rv.abs().quantile(.95),
                parity_outside_quote_interval=int(((g.bid_c-g.ask_p>carry.loc[g.index])|(g.ask_c-g.bid_p<carry.loc[g.index])).sum())))
    dest=Path('reports/option_flex');dest.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(dest/'matched_mark_quality.csv',index=False)
    print(f'Checked {sum(r["matched_dates"] for r in rows):,} matched signal-date call/put marks')


if __name__=='__main__':main()
