import unittest
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scripts.run_contract_tail_moments import trailing_moments


class TailMomentsTest(unittest.TestCase):
    def test_unbiased_sample_moments_match_scipy(self):
        dates=pd.bdate_range('2020-01-01',periods=90)
        values=np.random.default_rng(12).standard_t(6,size=90)*.01
        result=trailing_moments(pd.Series(values,index=dates),dates)
        self.assertEqual(result.available.iloc[61],0)
        self.assertEqual(result.available.iloc[62],1)
        self.assertAlmostEqual(result['skew'].iloc[-1],skew(values[-63:],bias=False))
        self.assertAlmostEqual(result['kurt'].iloc[-1],kurtosis(values[-63:],bias=False,fisher=True))

    def test_future_return_changes_do_not_change_past_states(self):
        dates=pd.bdate_range('2020-01-01',periods=100)
        returns=pd.Series(np.random.default_rng(7).normal(size=100),index=dates)
        original=trailing_moments(returns,dates)
        returns.iloc[80:]=1000
        pd.testing.assert_frame_equal(original.iloc[:80],trailing_moments(returns,dates).iloc[:80])

    def test_missing_sessions_and_constant_history_are_flagged(self):
        dates=pd.bdate_range('2020-01-01',periods=150)
        returns=pd.Series(np.random.default_rng(1).normal(size=150),index=dates).drop(dates[70])
        result=trailing_moments(returns,dates)
        self.assertTrue(result.available.iloc[70:133].eq(0).all())
        self.assertEqual(result.available.iloc[133],1)
        constant=trailing_moments(pd.Series(.01,index=dates),dates)
        self.assertTrue(constant.eq(0).all().all())


if __name__=='__main__':unittest.main()
