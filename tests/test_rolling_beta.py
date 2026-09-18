import unittest
import numpy as np
import pandas as pd
from scripts.build_rolling_beta import rolling_beta


class RollingBetaTests(unittest.TestCase):
    def test_known_slope_with_intercept_and_no_future_leak(self):
        x=np.tile([-.02,.01,.03,-.01],40)
        d=pd.DataFrame({'NVDA':x,'AMD':2*x+.003},index=pd.bdate_range('2020-01-01',periods=len(x)))
        result=rolling_beta(d,63)
        self.assertTrue(result.iloc[:62].isna().all())
        np.testing.assert_allclose(result.iloc[62:],2.,atol=1e-12)
        d.iloc[101:,d.columns.get_loc('AMD')]=.9
        self.assertAlmostEqual(rolling_beta(d,63).iloc[100],2.)

    def test_missing_return_invalidates_window(self):
        x=np.arange(100,dtype=float)/1000
        d=pd.DataFrame({'AMD':x*1.5,'NVDA':x})
        d.loc[50,'AMD']=np.nan
        self.assertTrue(np.isnan(rolling_beta(d,63).iloc[80]))

    def test_zero_denominator_is_missing(self):
        d=pd.DataFrame({'AMD':np.arange(100)/1000,'NVDA':np.zeros(100)})
        self.assertTrue(rolling_beta(d,63).isna().all())


if __name__=='__main__':unittest.main()
