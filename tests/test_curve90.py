import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
from scripts.curve90.config import TENORS,COMBOS,ACTIONS,RUNOFF,config
from scripts.curve90.unit_paths import UnitPaths,pair_labels
from scripts.option_flex.contract_learning_labels import ShadowLabels
from scripts.option_flex.batched_contract_labels import build_labels
from scripts.option_flex.contract_engine import simulate
from scripts.run_aligned_contract_learning import mark_book

class Curve90Tests(unittest.TestCase):
    def test_long_expiry_matches_existing_engine(self):
        cal=pd.bdate_range('2020-01-02','2020-04-24');cfg=replace(config(),risk_controls=False)
        for cp in ['C','P']:
            quotes=[];stocks=[]
            for date in cal:
                for sid,expiry in [(1,pd.Timestamp('2020-03-31')),(2,pd.Timestamp('2020-04-03'))]:
                    stocks.append(dict(date=date,secid=sid,close=102. if sid==1 else 100.,cfadj=1.,rate=.04))
                    if date<expiry:quotes.append(dict(date=date,secid=sid,optionid=sid,expiry=expiry,cp_flag=cp,
                        strike=100.,bid=4.9,ask=5.1,delta=.5 if cp=='C' else -.5,vega=.1,multiplier=100.,cfadj=1.,
                        standard=True,open_interest=1000,bs_iv=.4,bs_rate=.04,known_dividend_pv=0.))
            q=pd.DataFrame(quotes);s=pd.DataFrame(stocks)
            sel=q[q.date.eq(cal[0])][['date','secid','optionid','cp_flag']].copy();sel['tenor']=90;sel['bucket']=np.where(sel.secid.eq(1),'25','75')
            div=pd.DataFrame(columns=['secid','ex_date','declare_date','amount'])
            units=[UnitPaths(q[q.secid.eq(sid)],s[s.secid.eq(sid)],sel[sel.secid.eq(sid)],cal,cal[:1],div,cfg) for sid in [1,2]]
            actual,ends,vegas,status=pair_labels(*units,(90,cp,25,75),cfg)
            expected,_,_,_=build_labels(ShadowLabels(q,s,sel,cal,[1,2],cfg,div),np.array([0]),batch_size=1)
            np.testing.assert_allclose(actual,expected[:,[0,1,2,3,5,6,7,8]],atol=1e-8)
            self.assertEqual(status[0],'complete');self.assertGreater(ends[0],57)
            for action,k in [(-1.,0),(1.,7)]:
                nav,_,_=simulate(q,s,pd.Series([action],index=cal[:1]),cal,[1,2],cfg,div,sel)
                mark_book(nav,pd.Series(.04,index=cal),cfg.capital)
                e=(nav.nav.diff()-nav.cash_benchmark.diff()).iloc[1:ends[0]+1]/cfg.entry_vega_per_leg
                np.testing.assert_allclose(actual[0,k,:len(e)],e,atol=1e-8)

    def test_only_90_day_choices(self):
        self.assertEqual(TENORS,[90]);self.assertEqual(len(COMBOS),72);self.assertEqual(len(ACTIONS),577)
        self.assertGreater(RUNOFF,config().max_dte)

if __name__=='__main__':unittest.main()
