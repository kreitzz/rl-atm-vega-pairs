import unittest
import numpy as np
import pandas as pd
from scripts.build_spread_history import spread_history_features


class SpreadHistoryTests(unittest.TestCase):
    def test_baseline_excludes_current_and_future(self):
        s=pd.Series(np.arange(100,dtype=float),index=pd.bdate_range('2020-01-01',periods=100))
        x=spread_history_features(s)
        self.assertAlmostEqual(x.iloc[63].spread_mean_63,np.arange(63).mean())
        self.assertAlmostEqual(x.iloc[63].spread_z_63,(63-np.arange(63).mean())/np.arange(63).std(ddof=1))
        self.assertEqual(x.iloc[63].spread_change_21,21.)
        s.iloc[64:]=1000
        pd.testing.assert_series_equal(spread_history_features(s).iloc[63],x.iloc[63])
        s.iloc[63]=500
        self.assertEqual(spread_history_features(s).iloc[63].spread_mean_63,x.iloc[63].spread_mean_63)

    def test_warmup_and_constant_spread_are_explicit(self):
        s=pd.Series(np.ones(70))
        x=spread_history_features(s)
        self.assertTrue(x.spread_z_63.iloc[:63].isna().all())
        self.assertEqual(x.spread_z_63.iloc[63],0.)
        s.iloc[63]=2.
        self.assertEqual(spread_history_features(s).spread_z_63.iloc[63],10.)


if __name__=='__main__':unittest.main()
