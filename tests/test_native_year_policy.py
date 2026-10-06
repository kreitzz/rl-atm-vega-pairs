import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import native_year_policy as p


class NativeYearPolicyTest(unittest.TestCase):
    def test_training_requires_completed_native_outcome(self):
        cutoff = pd.Timestamp('2018-01-01')
        frame = pd.DataFrame(dict(date=pd.to_datetime(['2014-01-01']*4),
              end=pd.to_datetime(['2017-12-29', '2018-01-01', '2017-12-29', '2017-12-29']),
              signal_eligible=[True, True, False, True], finite_label=[True, True, True, False]))
        self.assertEqual(p.training_rows(frame, cutoff).index.tolist(), [0])

    def test_decode_preserves_direction_and_declared_size(self):
        direction, size = p.decode([0, 1, 3, 4, 5, 6, 8])
        np.testing.assert_array_equal(direction, [0, -1, -1, -1, 1, 1, 1])
        np.testing.assert_array_equal(size, [0, 4, 2, 1, 1, 2, 4])

    def test_enumeration_chooses_sharpe_and_uses_flat(self):
        rng = np.random.default_rng(2026)
        book = np.zeros((1000, 18))
        book[:, 8] = 1 + rng.normal(0, 1, 1000)
        book[:, 1] = -1 + rng.normal(0, 1, 1000)
        book[:, 9+8] = -1 + rng.normal(0, 1, 1000)
        book[:, 9+1] = -1 + rng.normal(0, 1, 1000)
        choices, details = p.select_policy(book, np.array([0, 1, 8]), 2)
        np.testing.assert_array_equal(choices, [8, 0])
        self.assertEqual(details['assignments_evaluated'], 9)

    def test_equal_sharpe_tie_does_not_invent_size_advantage(self):
        book = np.zeros((1000, 9))
        r = np.sin(np.arange(1000)) + .1
        book[:, 8], book[:, 6], book[:, 5] = r, .5*r, .25*r
        choices, _ = p.select_policy(book, np.array([0, 5, 6, 8]), 1)
        np.testing.assert_array_equal(choices, [5])

    def test_memory_only_credits_completed_cohorts(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels = np.zeros((3, 9, 2))
            labels[0, 8] = 1.
            labels[1, 8] = 10000.
            np.save(Path(tmp)/'a.npy', labels)
            frame = pd.DataFrame(dict(pair='a', date=pd.to_datetime(['2014-01-01', '2015-01-01', '2016-01-01']),
               end=pd.to_datetime(['2015-01-01', '2017-01-01', '2017-01-01']), signal_eligible=True,
               finite_label=True, label_status='complete', label_row=[0, 1, 2]))
            result = p.native_memory(frame, Path(tmp))
            self.assertEqual(result.memory_count.iloc[1], 0.)
            self.assertEqual(result.memory_count.iloc[2], 1.)
            self.assertEqual(result.memory_long.iloc[2], 2.)

    def test_flat_encoding_preserves_every_row_and_zero_exposure(self):
        frame = pd.DataFrame(dict(direction=[0, -1, 1], learned_level=[0, 2, 4]))
        result = p.executable_rows(frame)
        self.assertEqual(len(result), len(frame))
        np.testing.assert_array_equal(result.direction*result.learned_level, frame.direction*frame.learned_level)
        np.testing.assert_array_equal(result.native_requested_direction, frame.direction)
        self.assertEqual(result.direction.iloc[0], 1)
        with self.assertRaises(ValueError):
            p.executable_rows(pd.DataFrame(dict(direction=[0], learned_level=[2])))

    def test_pair_names_follow_actual_regression_orientation(self):
        # The older human label is ARKK/TSLA, while the canonical IDs put
        # TSLA first. A display reversal would misstate every trade direction.
        from scripts.run_native_year_entry import roster
        row = roster().set_index('pair').loc['143439__205711']
        self.assertEqual(row.label, 'TSLA/ARKK')
        self.assertEqual((row.secid_a, row.secid_b), (143439, 205711))

    def test_size_only_rewards_follow_forecast_without_reversing_trade(self):
        rewards = np.zeros((9, 2))
        rewards[[1, 3, 4]] = [[-100, 7], [-50, 4], [-25, 2]]
        rewards[[5, 6, 8]] = [[25, 2], [50, 4], [100, 7]]
        short = p.forecast_size_rewards(rewards, -1)
        np.testing.assert_array_equal(short[5], [-25, 2])
        np.testing.assert_array_equal(short[6], [-50, 4])
        np.testing.assert_array_equal(short[8], [-100, 7])
        np.testing.assert_array_equal(p.forecast_size_rewards(rewards, 1)[8], [100, 7])
        self.assertFalse(p.forecast_size_rewards(rewards, 0).any())
        self.assertNotIn(0, p.FORECAST_MENUS['forecast_rl_sized_no_flat'])


if __name__ == '__main__':
    unittest.main()
