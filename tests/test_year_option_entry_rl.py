import unittest

import numpy as np
import pandas as pd

from scripts.run_year_option_entry_rl import completed_entries,entry_level,opening_target


class EntryQTest(unittest.TestCase):
    def episode(self,first='2020-01-01'):
        return dict(date=pd.Timestamp(first),dates=pd.bdate_range(first,periods=15),state_known=np.ones(15,bool))

    def test_can_train_first_week_without_waiting_for_expiry(self):
        ep = self.episode()
        available = completed_entries([ep],ep['dates'][6])
        self.assertEqual(len(available),1)
        self.assertEqual(available[0][1],5)

    def test_entry_label_ending_on_or_after_cutoff_is_purged(self):
        ep = self.episode()
        self.assertEqual(completed_entries([ep],ep['dates'][5]),[])
        self.assertEqual(completed_entries([ep],ep['dates'][4]),[])

    def test_unobserved_first_week_waits_for_next_observed_decision(self):
        ep = self.episode()
        ep['state_known'][5] = False
        self.assertEqual(completed_entries([ep],ep['dates'][10]),[])
        self.assertEqual(completed_entries([ep],ep['dates'][11])[0][1],10)

    def test_first_block_fees_are_not_subtracted_twice(self):
        self.assertAlmostEqual(opening_target(-.3,1.,0.,.995),.695)

    def test_zero_action_and_cold_start_have_explicit_values(self):
        class Model:
            def __init__(self,value):
                self.value = value
            def predict(self,x):
                return np.full(len(x),self.value)
        state = np.zeros(8)
        self.assertEqual(entry_level(None,state),2)
        self.assertEqual(entry_level(Model(-1.),state),0)
        self.assertEqual(entry_level(Model(0.),state),0)
        self.assertEqual(entry_level(Model(.1),state),2)


if __name__=='__main__':
    unittest.main()
