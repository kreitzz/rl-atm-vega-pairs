import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import replace
import numpy as np
import pandas as pd
from scripts.batched_contract_labels import build_labels
from scripts.contract_learning_labels import ShadowLabels, LADDER, profitability_memory
from scripts.contract_engine import simulate
from scripts.run_aligned_contract_learning import mark_book
from scripts.run_aligned_contract_learning import train_pair
from tests.test_expiry_holding import fixture


class ExpiryLabelTests(unittest.TestCase):
    def compare(self, split=False, invalid=False):
        q,s,a,d,c=fixture()
        c=replace(c,entry_vega_per_leg=80, max_gross_vega=1000,
            option_penalty_vol_points=.1,stock_cost_bps=1,stock_borrow_rate=.003,
            support_integer_splits=True)
        s['rate']=.04;q['bs_rate']=.04
        s.loc[s.secid.eq(1),'close']+=np.tile(np.arange(len(d))*.1,1)
        if split:
            mask=q.date.ge(d[5])&q.secid.eq(1)
            q.loc[mask,['bid','ask','strike','vega']]/=2;q.loc[mask,'cfadj']=2
            mask=s.date.ge(d[5])&s.secid.eq(1)
            s.loc[mask,'close']/=2;s.loc[mask,'cfadj']=2
        if invalid:q.loc[q.date.eq(d[3]),['delta','vega','bs_iv']]=np.nan
        div=pd.DataFrame([dict(secid=1,declare_date=d[0],ex_date=d[6],amount=.1)])
        selected=q[['date','secid','optionid']]
        maker=ShadowLabels(q,s,selected,d,[1,2],c,div)
        daily,ends,vegas,status=build_labels(maker,np.array([0,1]),batch_size=2)
        for row,i in enumerate([0,1]):
            self.assertEqual(status[row],'complete')
            self.assertEqual(d[ends[row]],pd.Timestamp('2025-02-07'))
            for j,action in enumerate(LADDER):
                nav,t,_=simulate(q,s,pd.Series([action],index=d[i:i+1]),d[i:], [1,2],c,div,selected)
                rates=pd.Series(.04,index=nav.index);mark_book(nav,rates,c.capital)
                excess=nav.pnl-nav.cash_benchmark.diff().fillna(0)
                expected=excess.iloc[1:ends[row]-i+1].to_numpy()/c.entry_vega_per_leg
                np.testing.assert_allclose(daily[row,j,:len(expected)],expected,atol=1e-8,rtol=0)
                np.testing.assert_allclose(daily[row,j,len(expected):],0,atol=1e-12)

    def test_daily_cost_and_expiry_cashflows_match_engine(self):self.compare()
    def test_split_and_dividends_match_engine(self):self.compare(split=True)
    def test_causal_hedge_fallback_matches_engine(self):self.compare(invalid=True)

    def test_45_day_contract_is_not_truncated_at_old_32_session_width(self):
        q,s,_,_,c=fixture()
        dates=pd.bdate_range('2025-01-03','2025-02-21')
        q0=q.groupby('secid').head(1);s0=s.groupby('secid').head(1)
        q=pd.concat([q0.assign(date=date,expiry=pd.Timestamp('2025-02-20')) for date in dates if date<pd.Timestamp('2025-02-20')])
        s=pd.concat([s0.assign(date=date) for date in dates])
        selected=q[['date','secid','optionid']]
        div=pd.DataFrame(columns=['secid','ex_date','declare_date','amount'])
        maker=ShadowLabels(q,s,selected,dates,[1,2],c,div)
        daily,ends,_,status=build_labels(maker,[0])
        self.assertEqual(status[0],'complete');self.assertGreater(ends[0],32)
        nav,_,_=simulate(q,s,pd.Series([1.],index=dates[:1]),dates,[1,2],c,div,selected)
        np.testing.assert_allclose(daily[0,-1,:ends[0]],nav.pnl.iloc[1:ends[0]+1]/c.entry_vega_per_leg,atol=1e-8)

    def test_memory_waits_for_last_leg_expiry(self):
        dates=pd.bdate_range('2025-01-02','2025-02-10')
        ends=pd.DatetimeIndex([pd.Timestamp('2025-02-07')]*len(dates))
        mem=profitability_memory(dates,ends,np.ones((len(dates),2)),np.ones((len(dates),2),bool))
        self.assertEqual(mem.loc[:'2025-02-07','count_long_91d'].sum(),0)
        self.assertGreater(mem.loc['2025-02-10','count_long_91d'],0)

    def test_blocked_validation_directions_skip_fitting_and_stay_flat(self):
        dates=pd.bdate_range('2010-01-01','2018-01-31')
        panel=pd.DataFrame({'x':0.,'eligible':True},index=dates)
        for side in ['long','short']:
            for days in [91,365]:
                panel[f'profit_{side}_{days}d']=0.;panel[f'count_{side}_{days}d']=0.
        c=fixture()[-1]
        with TemporaryDirectory() as temp, patch('scripts.run_aligned_contract_learning.PreparedMarket'), \
                patch('scripts.run_aligned_contract_learning.fit_rl',side_effect=AssertionError('unnecessary fit')), \
                patch('scripts.run_aligned_contract_learning.fit_nn',side_effect=AssertionError('unnecessary fit')):
            a,f=train_pair({'screen_years':[2018],'secids':[1,2],'label':'test'},panel,
                np.zeros((len(dates),9,47)),dates+pd.Timedelta(days=30),dates,None,None,None,None,
                c,pd.Series(0.,index=dates),pd.Timestamp('2018-01-01'),dates[-1],Path(temp),
                feature_columns=['x'],memory_filters={'rl_sized':'memory_both','nn_fixed':'memory_year'},recent_refit=True)
        self.assertTrue(a.eq(0).all().all())
        self.assertEqual(set(f.status),{'flat_no_validation_opportunities'})
