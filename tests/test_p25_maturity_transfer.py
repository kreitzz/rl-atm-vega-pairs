import unittest
import numpy as np
import pandas as pd
from scripts.p25_maturity_transfer import remap, common_entries, period_stats


class MaturityTest(unittest.TestCase):
    def test_same_cutoff_and_decisions_with_missing_contracts_explicit(self):
        rows=pd.DataFrame({'date':pd.to_datetime(['2025-05-11','2025-05-12']),
            'pair':['1__2']*2,'secid_a':[1]*2,'secid_b':[2]*2,'direction':[1,-1],'base_level':[2,4]})
        picked=common_entries(rows,'2025-08-29')
        self.assertEqual(len(picked),1)
        choices={(1,'P25'):pd.DataFrame({'date':picked.date,'optionid':[99]}),
                 (2,'P25'):pd.DataFrame({'date':pd.to_datetime([]),'optionid':[]})}
        result=remap(picked,choices,'P25')
        pd.testing.assert_frame_equal(result[picked.columns],picked)
        self.assertEqual(result.optionid_a.iloc[0],99)
        self.assertTrue(np.isnan(result.optionid_b.iloc[0]))

    def test_later_calendar_window_keeps_first_day_and_pnl_from_open_inventory(self):
        idx=pd.to_datetime(['2020-12-31','2021-01-04','2021-01-05'])
        nav=pd.DataFrame({'excess_return':[0,.01,.02],'return_':[0,.01,.02],
            'cash_benchmark':[100,101,102],'pnl':[0,11,21],'active_cohorts':[1,1,1]},index=idx)
        decisions=pd.DataFrame({'status':['entered'],'entry_date':pd.to_datetime(['2020-12-31']),'pair':['1__2']})
        result=period_stats(nav,decisions,'2021-01-01','2021-12-31')
        self.assertEqual(result['sessions'],2)
        self.assertEqual(result['entries'],0)
        self.assertEqual(result['net_excess_pnl'],30)
        self.assertEqual(result['active_sessions'],2)


if __name__=='__main__':
    unittest.main()
