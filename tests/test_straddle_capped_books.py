import unittest
import numpy as np
from scripts import straddle_capped_books as study


class CappedBookTests(unittest.TestCase):
    def test_ladder_ends_at_full_size_and_target_is_not_75_percent(self):
        self.assertEqual(study.CFG['actions'],[2,3,4])
        paths=np.arange(15.).reshape(3,5)
        record,actual=study.full_size_target(dict(finite_label=True,full_profit=paths[1].sum()),paths)
        self.assertEqual(record['full_profit'],paths[2].sum())
        np.testing.assert_array_equal(actual,paths)

    def test_missing_unit_does_not_acquire_a_reward(self):
        record,paths=study.full_size_target(dict(finite_label=False,full_profit=np.nan),np.zeros((3,36)))
        self.assertTrue(np.isnan(record['full_profit']))


if __name__=='__main__':unittest.main()
