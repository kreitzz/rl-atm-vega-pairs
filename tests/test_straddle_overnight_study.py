import unittest
import numpy as np
import pandas as pd
from scripts.straddle_overnight_study import market_features,specifications


class OvernightFeatures(unittest.TestCase):
    def data(self,n=180):
        dates=pd.bdate_range('2020-01-01',periods=n)
        x=pd.Series(np.sin(np.arange(n)*.7)*.01,index=dates);r=2*x
        iv=pd.Series(.4+np.sin(np.arange(n)*.3)*.03,index=dates)
        vix=pd.Series(20+np.sin(np.arange(n)*.5),index=dates)
        proxy=pd.DataFrame(dict(return_sum=100*x+r,return_count=101),index=dates)
        return r,iv,vix,proxy

    def test_market_factor_leaves_own_name_out(self):
        f=market_features(*self.data())
        self.assertAlmostEqual(f.spot_market_beta126.iloc[-1],2.,places=10)
        self.assertAlmostEqual(f.spot_market_corr126.iloc[-1],1.,places=10)

    def test_future_observations_do_not_change_past_features(self):
        data=self.data();full=market_features(*data)
        past=market_features(*(x.iloc[:140] for x in data))
        pd.testing.assert_frame_equal(full.iloc[:140],past)

    def test_declared_matrix_is_unique_and_fixed(self):
        specs=specifications();self.assertEqual(len(specs),12)
        self.assertEqual(len(set(s['name'] for s in specs)),12)
        self.assertEqual({s['months'] for s in specs},{18,36})
