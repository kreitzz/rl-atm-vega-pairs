import unittest
import numpy as np
import pandas as pd
from scripts.build_broad_pairs import forward_series
from scripts.run_first_pass import forward_rv
from scripts.run_broad_memory import history_state, account


class BroadMemoryTests(unittest.TestCase):
    def test_fast_forward_rv_matches_existing_definition_with_missing_returns(self):
        dates=pd.bdate_range('2020-01-01',periods=90)
        ret=pd.Series(np.sin(np.arange(90))*.01,index=dates);ret.iloc[40]=np.nan
        expected=forward_rv(pd.DataFrame({'AMD':ret,'NVDA':ret}),dates).rv_amd
        actual=forward_series(ret)
        np.testing.assert_allclose(actual.loc[expected.index],expected,atol=1e-12)
        self.assertEqual(set(actual.dropna().index),set(expected.index))

    def test_profit_state_waits_until_after_outcome_date(self):
        dates=pd.date_range('2020-01-01',periods=8)
        ref=pd.DataFrame({'end':[dates[2],dates[4]],'reward':[2.,-4.],'action':[1.,-.5]},index=dates[:2])
        actual=history_state(ref,dates,window=2)
        self.assertEqual(actual.loc[dates[2],'past_observations'],0.)
        self.assertEqual(actual.loc[dates[3],'past_net_mean'],2.)
        self.assertEqual(actual.loc[dates[4],'past_net_mean'],2.)
        self.assertEqual(actual.loc[dates[5],'past_net_mean'],-1.)
        ref.loc[dates[1],'reward']=1e9
        changed=history_state(ref,dates,window=2)
        pd.testing.assert_frame_equal(actual.loc[:dates[4]],changed.loc[:dates[4]])

    def test_missing_trade_outcome_is_not_assumed_zero(self):
        p=pd.DataFrame({'long_reward':[np.nan,np.nan,3.],'iv_amd':[40.,40.,40.],'iv_nvda':[40.,40.,40.]})
        actual=account([0,1,.5],p)
        self.assertEqual(actual[0],0.)
        self.assertTrue(np.isnan(actual[1]))
        self.assertEqual(actual[2],1.)


if __name__=='__main__':unittest.main()
