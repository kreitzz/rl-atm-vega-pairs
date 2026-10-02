"""Reconcile long-duration call/put shadow labels and full hedge replay."""
from dataclasses import replace
import json
import numpy as np
import pandas as pd
from scripts.run_curve90 import CurveMarket,config,ROOT,OUTPUT,definitions
from scripts.curve90.unit_paths import UnitPaths,pair_labels
from scripts.option_flex.contract_learning_labels import ShadowLabels
from scripts.option_flex.batched_contract_labels import build_labels
from scripts.option_flex.contract_engine import simulate
from scripts.run_aligned_contract_learning import mark_book
from scripts.select_atm_contracts import load_calendar

def main():
    cal=load_calendar(ROOT);cal=cal[(cal>='2010-01-01')&(cal<='2025-08-29')]
    market=CurveMarket(cal);cfg=config();d=next(d for d in definitions() if d['label']=='AMD/NVDA')
    q,s,sel,div=market.curve_pair(d['secids']);records=[]
    for cp in ['C','P']:
        for ba,bb in [('atm','atm'),(25,75),(75,25)]:
            chosen=pd.concat([sel[sel.secid.eq(sid)&sel.cp_flag.eq(cp)&sel.bucket.eq(str(b))] for sid,b in zip(d['secids'],[ba,bb])])
            dates=chosen.groupby('date').secid.nunique();dates=dates[(dates==2)&(dates.index>='2018-01-01')&(dates.index<'2025-01-01')].index
            assert len(dates)>=4,'Insufficient reconciliation coverage'
            dates=dates[np.linspace(0,len(dates)-1,4,dtype=int)];selected=chosen[chosen.date.isin(dates)]
            quotes=q[q.optionid.isin(selected.optionid)]
            units=[UnitPaths(quotes[quotes.secid.eq(sid)],s[s.secid.eq(sid)],selected[selected.secid.eq(sid)],cal,dates,div[div.secid.eq(sid)],cfg) for sid in d['secids']]
            actual,ends,vegas,status=pair_labels(*units,(90,cp,ba,bb),cfg)
            expected,_,v,_=build_labels(ShadowLabels(quotes,s,selected,cal,d['secids'],cfg,div),cal.get_indexer(dates),batch_size=4)
            expected=expected[:,[0,1,2,3,5,6,7,8]]
            np.testing.assert_allclose(actual,expected,atol=1e-8,equal_nan=True)
            np.testing.assert_allclose(vegas,v[:,[0,1,2,3,5,6,7,8]])
            for j,date in enumerate(dates):
                if status[j]!='complete':continue
                for direction,k in [(-1.,0),(1.,7)]:
                    sub=cal[(cal>=date)&(cal<=cal[ends[j]])]
                    nav,_,_=simulate(quotes,s,pd.Series([direction],index=[date]),sub,d['secids'],replace(cfg,risk_controls=False),div,selected)
                    mark_book(nav,market.rates,cfg.capital)
                    e=(nav.nav.diff()-nav.cash_benchmark.diff()).iloc[1:]/cfg.entry_vega_per_leg
                    np.testing.assert_allclose(actual[j,k,:len(e)],e,atol=1e-8)
            records.append(dict(cp=cp,bucket_a=ba,bucket_b=bb,complete=int((status=='complete').sum()),max_error=float(np.nanmax(abs(actual-expected)))))
    assert sum(r['complete'] for r in records)>=12
    (OUTPUT/'reconciliation.json').write_text(json.dumps(records,indent=2)+'\n');print('90D real-data reconciliation passed',flush=True)

if __name__=='__main__':main()
