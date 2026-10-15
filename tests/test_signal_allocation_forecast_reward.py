import unittest
import numpy as np
import pandas as pd
from scripts.signal_allocation_forecast_reward import covariance_forecast


class ForecastTests(unittest.TestCase):
    def test_future_return_cannot_change_covariance(self):
        dates=pd.bdate_range('2020-01-01',periods=100)
        a=pd.Series(np.sin(np.arange(100))*.001,index=dates);b=pd.Series(np.cos(np.arange(100))*.002,index=dates)
        old=covariance_forecast(a,b,dates[0],dates[70],63)
        a.iloc[70:]=100;b.iloc[70:]=-100
        np.testing.assert_array_equal(old,covariance_forecast(a,b,dates[0],dates[70],63))

    def test_covariance_is_positive_semidefinite(self):
        dates=pd.bdate_range('2020-01-01',periods=100)
        a=pd.Series(np.sin(np.arange(100))*.001,index=dates);b=pd.Series(np.cos(np.arange(100))*.002,index=dates)
        matrix=covariance_forecast(a,b,dates[0],dates[-1],126)
        self.assertTrue((np.linalg.eigvalsh(matrix)>=0).all())
        np.testing.assert_allclose(matrix,matrix.T,atol=1e-15)


if __name__=='__main__':unittest.main()
