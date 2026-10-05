import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from scripts.run_recent_contract_refit import recent_training_mask, train, NEW_FEATURES, EXTRA


class RecentRefitTest(unittest.TestCase):
    def test_insufficient_recent_history_stays_flat(self):
        calendar=pd.bdate_range('2019-12-01','2020-01-31')
        panel=pd.DataFrame(1.,index=calendar,columns=list(dict.fromkeys(NEW_FEATURES+EXTRA)))
        panel['eligible']=True
        actions=pd.DataFrame(0.,index=calendar[calendar>='2020-01-01'],columns=['rl_sized','nn_fixed'])
        fits=pd.DataFrame([dict(fit='2020-01-01',model='rl_sized',status='enabled',validation_sharpe=.5)])
        result,logs=train(panel,np.zeros((len(panel),9,32)),calendar+pd.Timedelta(days=30),calendar,actions,fits)
        self.assertTrue(result.eq(0).all().all())
        self.assertEqual(logs.status.iloc[0],'insufficient_recent_history')

    def test_only_completed_past_outcomes_in_five_year_window(self):
        dates=pd.to_datetime(['2014-12-31','2015-01-01','2019-12-01','2019-12-02','2019-12-03','2020-01-01'])
        ends=pd.to_datetime(['2015-01-02','2015-02-01','2019-12-31','2020-01-01','2019-12-31','2020-01-02'])
        daily=np.zeros((6,9,32));daily[4]=np.nan
        np.testing.assert_array_equal(recent_training_mask(dates,ends,daily,pd.Timestamp('2020-01-01')),
                                      [False,True,True,False,False,False])

    def test_refit_keeps_original_month_gate_and_beta(self):
        calendar=pd.bdate_range('2016-01-01','2020-03-31')
        panel=pd.DataFrame(1.,index=calendar,columns=list(dict.fromkeys(NEW_FEATURES+EXTRA)))
        panel['eligible']=True
        panel.loc['2020-01-02','eligible']=False
        ends=calendar+pd.Timedelta(days=30);daily=np.zeros((len(panel),9,32))
        actions=pd.DataFrame(0.,index=calendar[calendar>='2020-01-01'],columns=['rl_sized','nn_fixed'])
        fits=pd.DataFrame([dict(fit='2020-01-01',model='rl_sized',status='enabled',validation_sharpe=.5),
                           dict(fit='2020-02-01',model='rl_sized',status='flat_validation_gate',validation_sharpe=-.5)])
        with (patch('scripts.run_recent_contract_refit.fit_rl',return_value=(object(),{})) as fit,
              patch('scripts.run_recent_contract_refit.predict',side_effect=lambda model,x,neural:np.full(len(x),.5))):
            result,logs=train(panel,daily,ends,calendar,actions,fits)
        self.assertEqual(fit.call_count,1)
        self.assertEqual(result.loc['2020-01-02','rl_sized'],0)
        self.assertEqual(result.loc['2020-01-03','rl_sized'],.5)
        self.assertTrue(result.loc['2020-02-01':].eq(0).all().all())
        self.assertTrue(result.nn_fixed.eq(0).all())
        self.assertLess(pd.Timestamp(logs.last_outcome.iloc[0]),pd.Timestamp('2020-01-01'))


if __name__=='__main__':unittest.main()
