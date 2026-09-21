import unittest
import pandas as pd
from scripts.explore_etf_pairs import add_countdown, eligible_fold


class ETFPairTests(unittest.TestCase):
    def test_etf_has_no_corporate_earnings_event(self):
        p=pd.DataFrame(index=pd.date_range('2020-01-01',periods=5))
        add_countdown(p,[], 'amd',True)
        self.assertTrue(p.amd_earnings_days_capped.eq(31).all())

    def test_stock_countdown_and_cap(self):
        dates=pd.date_range('2018-01-01',periods=24,freq='90D')
        p=pd.DataFrame(index=[dates[10]-pd.Timedelta(days=2),dates[10],dates[10]+pd.Timedelta(days=1)])
        add_countdown(p,dates,'nvda',False)
        self.assertEqual(p.nvda_earnings_days_capped.tolist(),[2,0,31])

    def test_minimum_training_and_validation_history(self):
        self.assertTrue(eligible_fold(range(504),range(252),range(126)))
        self.assertFalse(eligible_fold(range(503),range(252),range(126)))
        self.assertFalse(eligible_fold(range(504),range(251),range(126)))
        self.assertFalse(eligible_fold(range(504),range(252),range(125)))


if __name__=='__main__':unittest.main()
