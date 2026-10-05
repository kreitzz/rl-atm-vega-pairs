import unittest
import numpy as np
import pandas as pd
from scripts.run_atm65_employee_sentiment import add_sentiment, ADDED


class EmployeeFeatureTests(unittest.TestCase):
    def test_orientation_missingness_and_original_state(self):
        dates = pd.date_range('2020-01-01', periods=3)
        panel = pd.DataFrame({'spread': [2., 3., 4.], 'profit_long_91d': [1., 2., 3.]}, index=dates)
        a = pd.Series([4., np.nan, 2.], index=dates)
        b = pd.Series([3., 4., np.nan], index=dates)
        got = add_sentiment(panel, a, b)
        pd.testing.assert_frame_equal(got[panel.columns], panel)
        np.testing.assert_array_equal(got[ADDED[0]], [1., 0., 0.])
        np.testing.assert_array_equal(got[ADDED[1]], [1., 0., 0.])
        reverse = add_sentiment(panel, b, a)
        np.testing.assert_array_equal(reverse[ADDED[0]], -got[ADDED[0]])


if __name__ == '__main__': unittest.main()
