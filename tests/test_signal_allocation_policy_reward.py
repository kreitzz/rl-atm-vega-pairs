import unittest
import numpy as np
import pandas as pd
from scripts.signal_allocation_policy_reward import choose_reference


class PolicyReferenceTests(unittest.TestCase):
    def data(self):
        dates=pd.date_range('2018-01-01',periods=55,freq='W-MON');periods=[]
        for d in dates:
            paths=np.empty((7,7,5))
            for a in range(7):paths[:,a,:]=np.array([.001,-.001,.002,-.002,.001])*(1+a*.1)+a*.0001
            periods.append(dict(cutoff=d,end=d+pd.Timedelta(days=4),paths=paths))
        return dict(periods=periods)

    def test_reference_is_selected_from_completed_training_only(self):
        data=self.data();original=choose_reference(data,52)
        self.assertEqual(original['action'],6)
        self.assertLess(original['last'],data['periods'][52]['cutoff'])
        for m in data['periods'][52:]:m['paths'][:,0,:]=1000.
        changed=choose_reference(data,52)
        self.assertEqual(original['action'],changed['action'])
        np.testing.assert_array_equal(original['scores'],changed['scores'])

    def test_minimum_history_is_required(self):self.assertIsNone(choose_reference(self.data(),40))


if __name__=='__main__':unittest.main()
