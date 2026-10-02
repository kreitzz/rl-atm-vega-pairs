import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
from scripts.prepare_tenor_comparison import config, TENORS
from scripts.select_atm_contracts import select_candidates
from scripts.contract_learning_labels import ShadowLabels, LADDER
from scripts.batched_contract_labels import build_labels
from scripts.contract_engine import simulate
from scripts.run_aligned_contract_learning import mark_book
from tests.test_expiry_holding import fixture

class TenorTests(unittest.TestCase):
    def test_only_tenor_selection_settings_change(self):
        from dataclasses import asdict
        base=asdict(config(30))
        for tenor in TENORS:
            actual=asdict(config(tenor))
            for k in base:
                if k not in ['target_dte','min_dte','max_dte']:self.assertEqual(actual[k],base[k])

    def test_selection_uses_next_session_tenor(self):
        q,_,_,_,_=fixture();r=q.iloc[0].to_dict()
        signal=pd.Timestamp('2025-01-03');entry=pd.Timestamp('2025-01-06')
        r.update(date=signal,secid=1,cp_flag='C',standard=True,multiplier=100,bid=4.,ask=4.1,
                 delta=.5,vega=.1,strike=100.,open_interest=500,cfadj=1.)
        quotes=pd.DataFrame([dict(r,optionid=i,expiry=entry+pd.Timedelta(days=dte)) for i,dte in enumerate([14,30,45,65])])
        stock=pd.DataFrame([dict(secid=1,date=signal,entry_date=entry,close=100.)])
        for tenor in TENORS:
            chosen=select_candidates(quotes,stock,config(tenor))
            self.assertEqual(len(chosen),1);self.assertEqual(chosen.entry_dte.iloc[0],tenor)

    def test_common_runoff_and_completed_outcome_cutoffs(self):
        from unittest.mock import patch
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from scripts.run_tenor_comparison import train_tenor, COMMON_RUNOFF
        dates=pd.bdate_range('2010-01-01','2018-02-28')
        panel=pd.DataFrame({'x':0.,'eligible':True},index=dates)
        for side in ['long','short']:
            for days in [91,365]:
                panel[f'profit_{side}_{days}d']=1.;panel[f'count_{side}_{days}d']=100
        seen=[]
        def validate(a,q,s,selection,divs,secids,cal,cfg,rates,prepared):
            self.assertLessEqual(a.index.max(),cal[-1]-pd.Timedelta(days=COMMON_RUNOFF))
            seen.append(cfg.target_dte)
            return True,1.,None
        for tenor in TENORS:
            cfg=config(tenor)
            with TemporaryDirectory() as tmp, patch('scripts.run_tenor_comparison.PreparedMarket'), \
                 patch('scripts.run_tenor_comparison.fit_rl',return_value=(None,{})), \
                 patch('scripts.run_tenor_comparison.predict',side_effect=lambda model,x,neural:np.ones(len(x))), \
                 patch('scripts.run_tenor_comparison.validate_policy',side_effect=validate):
                a,f=train_tenor({'screen_years':[2018],'secids':[1,2],'label':'test'},panel,
                    np.zeros((len(dates),9,cfg.max_dte+2)),dates+pd.Timedelta(days=tenor),dates,None,None,None,None,cfg,
                    pd.Series(0.,index=dates),pd.Timestamp('2018-01-01'),dates[-1],Path(tmp),
                    feature_columns=['x'],memory_filters={'rl_sized':'memory_both'},recent_refit=True)
            self.assertTrue((pd.to_datetime(f.last_training_outcome)<pd.to_datetime(f.validation_start)).all())
            self.assertTrue((pd.to_datetime(f.last_recent_outcome)<pd.to_datetime(f.fit)).all())
        self.assertEqual(set(seen),set(TENORS))

    def test_short_and_long_labels_match_full_expiry_engine_all_sizes(self):
        for tenor,dte in [(14,14),(45,55)]:
            q0,s0,_,_,cfg=fixture()
            cfg=replace(cfg,min_dte=TENORS[tenor][0],max_dte=TENORS[tenor][1],target_dte=tenor,
                entry_vega_per_leg=80,option_penalty_vol_points=.1,stock_cost_bps=1,stock_borrow_rate=.003)
            dates=pd.bdate_range('2025-01-03','2025-03-10');expiry=dates[1]+pd.Timedelta(days=dte)
            q0=q0.groupby('secid').head(1);s0=s0.groupby('secid').head(1)
            q=pd.concat([q0.assign(date=d,expiry=expiry,bs_rate=.04) for d in dates if d<expiry],ignore_index=True)
            s=pd.concat([s0.assign(date=d,rate=.04) for d in dates],ignore_index=True)
            selection=q[['date','secid','optionid']];div=pd.DataFrame(columns=['secid','ex_date','declare_date','amount'])
            daily,ends,_,status=build_labels(ShadowLabels(q,s,selection,dates,[1,2],cfg,div),[0])
            self.assertEqual(status[0],'complete')
            self.assertEqual(daily.shape[-1],cfg.max_dte+2)
            if tenor==45:self.assertGreater(ends[0],32)
            for j,action in enumerate(LADDER):
                nav,_,_=simulate(q,s,pd.Series([action],index=dates[:1]),dates,[1,2],cfg,div,selection)
                mark_book(nav,pd.Series(.04,index=dates),cfg.capital)
                expected=(nav.pnl-nav.cash_benchmark.diff().fillna(0)).iloc[1:ends[0]+1]/cfg.entry_vega_per_leg
                np.testing.assert_allclose(daily[0,j,:len(expected)],expected,rtol=0,atol=1e-8)
                np.testing.assert_allclose(daily[0,j,len(expected):],0,rtol=0,atol=1e-8)

if __name__=='__main__':unittest.main()
