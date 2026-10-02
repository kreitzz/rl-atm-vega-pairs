import unittest
from unittest.mock import patch
from dataclasses import replace

import numpy as np
import pandas as pd

from scripts.run_aligned_contract_learning import contract_states, completed_training_mask, validate_policy, screen_dates
from scripts.contract_engine import PreparedMarket,simulate
from tests.test_contract_engine import fixture


class AlignedLearningTests(unittest.TestCase):
    def test_prepared_market_reuses_only_data_not_portfolio_state(self):
        q,s,a,d,c=fixture();selection=q[['date','secid','optionid']]
        prepared=PreparedMarket(q,s,selection)
        for action in [1.,-1.,0.,1.]:
            signals=a*action
            expected=simulate(q,s,signals,d,[1,2],c,selections=selection)
            observed=simulate(q,s,signals,d,[1,2],c,selections=selection,prepared=prepared)
            for left,right in zip(expected,observed):pd.testing.assert_frame_equal(left,right)
        with self.assertRaisesRegex(ValueError,'different input frames'):
            simulate(q.copy(),s,a,d,[1,2],c,selections=selection,prepared=prepared)

    def test_states_use_contract_iv_and_do_not_change_with_future_marks(self):
        dates=pd.bdate_range('2018-01-01',periods=400)
        t=np.arange(len(dates));ret=.01*np.sin(t)
        assets=pd.DataFrame({'ret':ret,'trailing_rv':20.,'earnings':31.,'iv':999.},index=dates)
        selections=[];stocks=[]
        for sid in [1,2]:
            selections.append(pd.DataFrame({'date':dates,'secid':sid,'bs_iv':.2+sid*.01+.01*np.sin(t/sid),
                                             'bs_vega':.2,'expiry':dates+pd.Timedelta(days=30),'strike':100.}))
            stocks.append(pd.DataFrame({'date':dates,'secid':sid,'close':100.}))
        selection=pd.concat(selections);stock=pd.concat(stocks)
        definition={'group':'broad','secids':[1,2],'source_pair':'s1p1__s2p2'}
        cfg=fixture()[-1]
        with patch('scripts.run_aligned_contract_learning.pd.read_parquet',return_value=assets):
            before,_=contract_states(definition,selection,stock,dates,cfg)
            selection.loc[selection.date>=dates[350],'bs_iv']=9.
            after,_=contract_states(definition,selection,stock,dates,cfg)
        pd.testing.assert_frame_equal(before.loc[:dates[349]],after.loc[:dates[349]])
        self.assertTrue(before.iv_amd.between(19,23).all())
        self.assertAlmostEqual(before.commission_vol_roundtrip.iloc[0],4*cfg.option_commission/20)

    def test_training_purges_validation_boundary_and_unknown_outcomes(self):
        panel=pd.DataFrame(index=pd.to_datetime(['2018-01-01','2019-12-01','2019-12-02','2019-12-03']))
        ends=pd.to_datetime(['2018-02-01','2020-01-01','2019-12-31','2019-12-31'])
        daily=np.zeros((4,9,32));daily[3]=np.nan
        result=completed_training_mask(panel,ends,daily,pd.Timestamp('2020-01-01'))
        np.testing.assert_array_equal(result,[True,False,True,False])

    def test_validation_uses_full_engine_and_rejects_unresolved_book(self):
        q,s,a,d,c=fixture();c=replace(c,option_spread_fraction=0)
        selection=q[q.date.eq(d[0])][['date','secid','optionid']]
        dividends=pd.DataFrame(columns=['secid','ex_date','declare_date','amount'])
        q=q[~(q.date.eq(d[2])&q.optionid.eq(1))]
        enabled,score,error=validate_policy(a,q,s,selection,dividends,[1,2],d,c,pd.Series(0.,index=d))
        self.assertFalse(enabled);self.assertIsNone(score);self.assertIn('missing held option',error)

    def test_historical_roster_does_not_expand_to_future_fold(self):
        dates=pd.to_datetime(['2017-12-29','2018-01-02','2019-12-31','2020-01-02'])
        np.testing.assert_array_equal(screen_dates({'screen_years':[2018]},dates),[False,True,True,False])

    def test_quarterly_discovery_does_not_enable_other_quarters(self):
        dates=pd.to_datetime(['2018-03-30','2018-04-02','2018-06-29','2018-07-02'])
        np.testing.assert_array_equal(screen_dates({'screen_quarters':['2018Q2']},dates),[False,True,True,False])


if __name__=='__main__':unittest.main()
