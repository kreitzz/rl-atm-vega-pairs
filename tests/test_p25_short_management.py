import unittest
import pandas as pd

from scripts.run_p25_short_management import training_episode


class ShortHistoryTest(unittest.TestCase):
    def test_six_month_fit_is_possible_and_purges_unfinished_outcomes(self):
        cal=pd.bdate_range('2018-01-02','2019-12-31')
        cut=pd.Timestamp('2018-07-01')
        rows=pd.DataFrame({'date':pd.bdate_range('2018-01-15',periods=16),'known_contract_expiry':pd.Timestamp('2018-03-01')})
        bad=pd.DataFrame({'date':[pd.Timestamp('2018-04-01'),pd.Timestamp('2017-12-29')],
                          'known_contract_expiry':[pd.Timestamp('2018-07-02'),pd.Timestamp('2018-02-01')]})
        use,history,ready,minimum=training_episode(pd.concat([rows,bad],ignore_index=True),cal,cut,6)
        self.assertTrue(ready)
        self.assertEqual(len(use),16)
        self.assertEqual(minimum,100)
        self.assertTrue((history<cut).all())
        self.assertTrue((use.date>=cut-pd.DateOffset(months=6)).all())
        self.assertTrue((use.known_contract_expiry<cut).all())

    def test_one_year_requires_full_history_and_sparse_cases_hold(self):
        cal=pd.bdate_range('2018-01-02','2019-12-31')
        rows=pd.DataFrame({'date':pd.bdate_range('2018-01-15',periods=16),'known_contract_expiry':pd.Timestamp('2018-03-01')})
        self.assertFalse(training_episode(rows,cal,pd.Timestamp('2018-07-01'),12)[2])
        self.assertTrue(training_episode(rows,cal,pd.Timestamp('2019-01-01'),12)[2])
        self.assertFalse(training_episode(rows.iloc[:15],cal,pd.Timestamp('2018-07-01'),6)[2])


if __name__=='__main__':unittest.main()
