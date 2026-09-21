import unittest
import numpy as np
import pandas as pd
from scripts.run_nn_rl_benchmark import add_features, nn_actions


class BenchmarkTests(unittest.TestCase):
    def test_nn_is_full_size_or_flat_and_requires_cost_coverage(self):
        prediction=np.array([-3.,-2.,-.5,0.,.5,2.,3.])
        actual=nn_actions(prediction,np.ones(7),2.)
        np.testing.assert_array_equal(actual,[-1,0,0,0,0,0,1])
        np.testing.assert_array_equal(nn_actions([2.,2.],[1.,3.],1.),[1,0])

    def test_feature_history_does_not_use_future_or_today_in_mean(self):
        dates=pd.date_range('2020-01-01',periods=300)
        p=pd.DataFrame({'spread':np.arange(300.),'iv_amd':40.,'iv_nvda':20.,
                        'trailing_rv_amd':30.,'trailing_rv_nvda':15.,
                        'amd_earnings_days_capped':31.,'nvda_earnings_days_capped':31.},index=dates)
        expected=add_features(p)
        self.assertAlmostEqual(expected.iloc[0].spread_reversion,(np.arange(126.).mean()-126)/np.arange(126.).std(ddof=1))
        self.assertEqual(expected.iloc[0].rv_adjusted_value,-5.)
        p.loc[dates[201]:,'spread']=1e9
        actual=add_features(p)
        pd.testing.assert_series_equal(expected.loc[:dates[200],'spread_reversion'],actual.loc[:dates[200],'spread_reversion'])


if __name__=='__main__':unittest.main()
