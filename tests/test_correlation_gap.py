import unittest
import numpy as np
import pandas as pd
from scripts.diagnose_correlation_gap import rolling_dependence


class DependenceTests(unittest.TestCase):
    def test_beta_and_correlation_are_distinct(self):
        b=pd.Series(np.random.default_rng(7).normal(size=80))
        result=rolling_dependence(2*b,b,20).dropna()
        np.testing.assert_allclose(result.correlation,1,atol=1e-12)
        np.testing.assert_allclose(result.beta,2,atol=1e-12)

    def test_future_changes_do_not_change_past_features(self):
        rng=np.random.default_rng(4)
        a=pd.Series(rng.normal(size=100));b=pd.Series(rng.normal(size=100))
        before=rolling_dependence(a,b,20)
        a.iloc[70:]=100;b.iloc[70:]=-100
        after=rolling_dependence(a,b,20)
        pd.testing.assert_frame_equal(before.iloc[:70],after.iloc[:70])
        self.assertTrue(before.iloc[:19].isna().all().all())


if __name__=='__main__':unittest.main()
