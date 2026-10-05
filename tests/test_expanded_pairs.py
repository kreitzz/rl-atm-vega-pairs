import unittest
import numpy as np
import pandas as pd
from scripts.discover_expanded_pairs import pair_statistics


class ExpandedPairTests(unittest.TestCase):
    def test_formation_excludes_current_and_future_returns(self):
        dates=pd.bdate_range('2020-01-01',periods=150)
        b=np.random.default_rng(1).normal(0,.01,150)
        r=pd.DataFrame({'a':.9*b,'b':b},index=dates)
        beta,corr=pair_statistics(r,dates[130])
        self.assertAlmostEqual(beta.loc['a','b'],.9)
        self.assertAlmostEqual(corr.loc['a','b'],1.)
        r.loc[dates[130]:,'a']=99
        after,after_corr=pair_statistics(r,dates[130])
        pd.testing.assert_frame_equal(beta,after);pd.testing.assert_frame_equal(corr,after_corr)

    def test_requires_full_window_and_does_not_fill_missing_returns(self):
        dates=pd.bdate_range('2020-01-01',periods=150)
        r=pd.DataFrame(np.random.default_rng(2).normal(size=(150,2)),index=dates,columns=['a','b'])
        self.assertTrue(pair_statistics(r,dates[100])[0].empty)
        r.loc[dates[120],'a']=np.nan
        beta,_=pair_statistics(r,dates[130])
        self.assertEqual(list(beta.columns),['b'])


if __name__=='__main__':unittest.main()
