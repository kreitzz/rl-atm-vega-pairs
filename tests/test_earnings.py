import unittest
import numpy as np
import pandas as pd
from scripts.attach_earnings import attach
from scripts.run_first_pass import model_states


class EarningsTests(unittest.TestCase):
    def test_future_reports_do_not_change_estimated_countdown(self):
        dates = pd.to_datetime(['2019-01-01','2019-04-02','2019-07-02','2019-10-01','2020-01-01'])
        c = pd.DataFrame({'ticker':['AMD']*5+['NVDA']*5,'earnings_date':list(dates)*2})
        p = pd.DataFrame({'end':pd.to_datetime(['2019-11-30','2019-10-31'])},index=pd.to_datetime(['2019-10-31','2019-10-01']))
        a = attach(p,c)
        c.loc[c.earnings_date > p.index.max(),'earnings_date'] += pd.Timedelta(days=20)
        b = attach(p,c)
        for ticker in ['amd','nvda']:
            np.testing.assert_array_equal(a[f'{ticker}_estimated_days_until_earnings'],b[f'{ticker}_estimated_days_until_earnings'])
            self.assertEqual(a.loc['2019-10-31',f'{ticker}_estimated_days_until_earnings'],61.)
            self.assertEqual(a.loc['2019-10-01',f'{ticker}_last_earnings_date'],pd.Timestamp('2019-07-02'))
            self.assertTrue(a.loc['2019-10-01',f'{ticker}_same_day_earnings_time_unknown'])

    def test_joint_state_uses_both_countdowns(self):
        p = pd.DataFrame({'spread':[-1,-1,-1,-1,1],
                          'amd_estimated_days_until_earnings':[40,40,30,30,40],
                          'nvda_estimated_days_until_earnings':[40,30,40,30,40]})
        np.testing.assert_array_equal(model_states(p,np.array([0.]),True,'estimated'),[0,1,2,3,4])
        p['amd_days_to_earnings_ex_post'] = p.amd_estimated_days_until_earnings
        p['nvda_days_to_earnings_ex_post'] = p.nvda_estimated_days_until_earnings
        np.testing.assert_array_equal(model_states(p,np.array([0.]),True,'actual'),[0,1,2,3,4])


if __name__ == '__main__':
    unittest.main()
