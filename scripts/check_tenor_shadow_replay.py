"""Reconcile new real-contract fixed-capital labels with isolated full-engine trades."""
from dataclasses import replace
import json
import numpy as np
import pandas as pd
from scripts.run_tenor_comparison import TenorMarket,OUTPUT,REPORT,ROOT,config,definitions,mark_book
from scripts.select_atm_contracts import load_calendar
from scripts.contract_engine import simulate
from scripts.contract_learning_labels import LADDER

def main():
    d=next(d for d in definitions() if d['label']=='AMD/NVDA')
    cal=load_calendar(ROOT);cal=cal[(cal>='2010-01-01')&(cal<='2025-08-29')];market=TenorMarket(cal);checks=[]
    for tenor in [14,45]:
        folder=OUTPUT/d['pair']/f'{tenor}d';panel=pd.read_parquet(folder/'states.parquet')
        with np.load(folder/'labels.npz',allow_pickle=False) as z:
            daily=z['daily'];ends=pd.DatetimeIndex(z['ends']);vegas=z['vegas'];statuses=z['statuses']
        rows=np.flatnonzero((panel.index>='2018-01-01')&(statuses=='complete')&(vegas[:,0]>0)&(vegas[:,-1]>0))
        assert len(rows)>0
        q,s,selection,divs=market.for_tenor(d['secids'],tenor)
        cfg=replace(config(tenor),risk_controls=False) # Standalone labels intentionally omit portfolio entry controls.
        for i in rows[:2]:
            start=panel.index[i];segment=cal[(cal>=start)&(cal<=ends[i])];n=len(segment)-1
            for j,action in enumerate(LADDER):
                nav,_,_=simulate(q,s,pd.Series([action],index=[start]),segment,d['secids'],cfg,divs,selection)
                mark_book(nav,market.rates,cfg.capital)
                expected=(nav.pnl-nav.cash_benchmark.diff().fillna(0)).iloc[1:].to_numpy()/cfg.entry_vega_per_leg
                np.testing.assert_allclose(daily[i,j,:n],expected,atol=1e-8,rtol=0)
                np.testing.assert_allclose(daily[i,j,n:],0,atol=1e-8,rtol=0)
                checks.append(dict(tenor=tenor,signal=str(start.date()),action=float(action),max_error=float(np.max(abs(daily[i,j,:n]-expected)))))
    (REPORT/'real_data_reconciliation.json').write_text(json.dumps(dict(pair=d['label'],convention='fixed capital, no portfolio risk rejection',checks=checks),indent=2)+'\n')
    print(f'{len(checks)} real-data action/entry trajectories matched through expiry')

if __name__=='__main__':main()
