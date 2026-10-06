import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.run_tail_put25_legacy_rl import score_all, training_selection
from scripts.tail_put25_pooled import FEATURES
from scripts.tail_put25_policy import CODES
from scripts.tail_put25_rule_sizing import SIZE_CODES


class LegacyPooledSizingTest(unittest.TestCase):
    def test_training_purges_unfinished_and_nonprofitable_experience(self):
        meta = pd.DataFrame(dict(date=pd.to_datetime(['2017-11-01'] * 4),
            end=pd.to_datetime(['2017-12-15', '2018-01-01', '2017-12-15', '2017-12-15']),
            finite_label=True, count_91d=21, count_365d=[63, 63, 62, 63],
            profit_91d=[1., 1., 1., -1.], profit_365d=1.))
        np.testing.assert_array_equal(training_selection(meta, '2018-01-01'), [True, False, False, False])

    def test_all_observed_states_scored_but_only_reference_trades_sized(self):
        dates = pd.bdate_range('2018-01-02', periods=4)
        panel = pd.DataFrame(0., index=dates, columns=FEATURES)
        panel['observed'] = True
        reference = pd.Series([CODES[-1], 0, CODES[1], 0], index=dates)
        calendar = dates.append(pd.DatetimeIndex(['2018-06-01']))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / '2018-01-01.joblib').touch()
            with patch('scripts.run_tail_put25_legacy_rl.joblib.load', return_value=None), \
                 patch('scripts.run_tail_put25_legacy_rl.predict', return_value=np.array([SIZE_CODES[2], SIZE_CODES[4], SIZE_CODES[3], SIZE_CODES[4]])):
                actions, allocation, scored, fallback = score_all(panel, reference, calendar, root)
        self.assertEqual(scored, 4); self.assertEqual(fallback, 0)
        np.testing.assert_array_equal(actions, [CODES[6], 0, CODES[2], 0])
        np.testing.assert_array_equal(allocation, [.5, 1., .75, 1.])

    def test_global_cold_start_preserves_reference_instead_of_vetoing_pairs(self):
        dates = pd.bdate_range('2018-01-02', periods=2)
        panel = pd.DataFrame(0., index=dates, columns=FEATURES); panel['observed'] = True
        reference = pd.Series([CODES[-1], CODES[1]], index=dates)
        calendar = dates.append(pd.DatetimeIndex(['2018-06-01']))
        with tempfile.TemporaryDirectory() as temp:
            actions, _, scored, fallback = score_all(panel, reference, calendar, Path(temp))
        pd.testing.assert_series_equal(actions, reference.rename('code'))
        self.assertEqual(scored, 0); self.assertEqual(fallback, 2)


if __name__ == '__main__': unittest.main()
