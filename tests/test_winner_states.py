import unittest
import numpy as np
import pandas as pd
from scripts.diagnose_winners import matured_history


class WinnerStateTests(unittest.TestCase):
    def test_outcome_unavailable_until_after_maturity(self):
        dates = pd.date_range('2020-01-01', periods=10)
        panel = pd.DataFrame({'end': dates+pd.Timedelta(days=2), 'long_reward': np.arange(10.)}, index=dates)
        result = matured_history(panel, window=2, minimum=2)
        self.assertTrue(result.loc['2020-01-04'].isna().all())
        self.assertEqual(result.loc['2020-01-05','mean'], .5)
        self.assertEqual(result.loc['2020-01-06','mean'], 1.5)

    def test_unmatured_outcomes_cannot_change_current_state(self):
        dates = pd.date_range('2020-01-01', periods=20)
        panel = pd.DataFrame({'end': dates+pd.Timedelta(days=3), 'long_reward': np.arange(20.)}, index=dates)
        expected = matured_history(panel, window=4, minimum=2)
        panel.loc[panel.end >= dates[10], 'long_reward'] = 1e9
        actual = matured_history(panel, window=4, minimum=2)
        pd.testing.assert_frame_equal(expected.loc[:dates[10]], actual.loc[:dates[10]])


if __name__ == '__main__': unittest.main()
