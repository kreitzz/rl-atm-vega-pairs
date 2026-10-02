"""Reconcile expanded real-contract shadow paths against the existing engine."""
import argparse
from dataclasses import replace
import json
import numpy as np
import pandas as pd
from scripts.run_curve_trading import CurveMarket,config,ROOT,REPORT,BASE,definitions
from scripts.curve_trading.policy import Replay,ATM
from scripts.curve_trading.config import ACTIONS
from scripts.curve_trading.unit_paths import UnitPaths,pair_labels
from scripts.option_flex.contract_learning_labels import ShadowLabels
from scripts.option_flex.batched_contract_labels import build_labels
from scripts.option_flex.contract_engine import simulate
from scripts.run_aligned_contract_learning import mark_book
from scripts.select_atm_contracts import load_calendar

def main():
    p=argparse.ArgumentParser();p.add_argument('--calls-only',action='store_true');args=p.parse_args()
    cal=load_calendar(ROOT);cal=cal[(cal>='2010-01-01')&(cal<='2025-08-29')]
    market=CurveMarket(cal);cfg=config();records=[]
    defs=definitions();d=next(d for d in defs if 'AMD' in d['label'] and 'NVDA' in d['label'])
    q,s,sel,div=market.curve_pair(d['secids'])
    for cp in (['C'] if args.calls_only else ['C','P']):
        for tenor in [14,30,45]:
            for ba,bb in [(25,75),(75,25),('atm','atm')]:
                chosen=pd.concat([sel[sel.secid.eq(sid)&sel.tenor.eq(tenor)&sel.cp_flag.eq(cp)&sel.bucket.eq(str(b))]
                                  for sid,b in zip(d['secids'],[ba,bb])],ignore_index=True)
                dates=chosen.groupby('date').secid.nunique();dates=dates[(dates==2)&(dates.index>='2018-01-01')&(dates.index<'2025-01-01')].index
                if len(dates)<4:raise ValueError(f'Not enough real-data checks for {(tenor,cp,ba,bb)}')
                dates=dates[np.linspace(0,len(dates)-1,4,dtype=int)]
                selected=chosen[chosen.date.isin(dates)]
                quotes=q[q.optionid.isin(selected.optionid)].copy()
                units=[UnitPaths(quotes[quotes.secid.eq(sid)],s[s.secid.eq(sid)],selected[selected.secid.eq(sid)],cal,dates,div[div.secid.eq(sid)],cfg) for sid in d['secids']]
                actual,ends,vega,status=pair_labels(*units,(tenor,cp,ba,bb),cfg)
                old=ShadowLabels(quotes,s,selected,cal,d['secids'],cfg,div)
                expected,oldends,oldvega,oldstatus=build_labels(old,cal.get_indexer(dates),batch_size=4)
                error=abs(actual-expected[:,[0,1,2,3,5,6,7,8]])
                np.testing.assert_allclose(actual,expected[:,[0,1,2,3,5,6,7,8]],atol=1e-8,equal_nan=True)
                np.testing.assert_allclose(vega,oldvega[:,[0,1,2,3,5,6,7,8]])
                for j,date in enumerate(dates):
                    if status[j]!='complete':continue
                    for direction,index in [(-1.,0),(1.,7)]:
                        subcal=cal[(cal>=date)&(cal<=cal[ends[j]])]
                        nav,_,dec=simulate(quotes,s,pd.Series([direction],index=[date]),subcal,d['secids'],replace(cfg,risk_controls=False),div,selected)
                        mark_book(nav,market.rates,cfg.capital)
                        excess=(nav.nav.diff()-nav.cash_benchmark.diff()).iloc[1:]/cfg.entry_vega_per_leg
                        np.testing.assert_allclose(actual[j,index,:len(excess)],excess,atol=1e-8)
                row=dict(tenor=tenor,cp=cp,bucket_a=ba,bucket_b=bb,cohorts=len(dates),complete=int((status=='complete').sum()),
                         max_label_error=float(np.nanmax(error)) if np.isfinite(error).any() else None)
                records.append(row);print(row,flush=True)
    controls=[]
    if not args.calls_only:
        ranked=sorted(defs,key=lambda d:-json.loads((BASE/d['pair']/'30d/status.json').read_text())['entries'])[:3]
        for d in ranked:
            q,s,sel,div=market.curve_pair(d['secids'])
            original=pd.read_csv(BASE/d['pair']/'30d/actions.csv',index_col=0,parse_dates=[0]).rl_sized
            codes=pd.Series([ACTIONS.index((ATM,float(a))) if a else 0 for a in original],index=original.index)
            nav,t,_=Replay(q,s,sel,d['secids'],div,cfg).run(codes,cal[cal>='2018-01-01'])
            mark_book(nav,market.rates,cfg.capital)
            expected=pd.read_parquet(BASE/d['pair']/'30d/daily_nav.parquet')
            np.testing.assert_allclose(nav.nav,expected.nav,atol=1e-5,rtol=0)
            row=dict(pair=d['pair'],label=d['label'],max_nav_error=float(abs(nav.nav-expected.nav).max()))
            controls.append(row);print(row,flush=True)
    REPORT.mkdir(parents=True,exist_ok=True)
    (REPORT/('call_reconciliation.json' if args.calls_only else 'real_data_reconciliation.json')).write_text(json.dumps(dict(checks=records,frozen_controls=controls),indent=2)+'\n')

if __name__=='__main__':main()
