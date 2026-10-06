import unittest

import numpy as np

from scripts.tail_put25_rule_sizing import CODES, SIZE_CODES, translate, sizing_labels, states, columns
from scripts.curve_trading.config import ACTIONS
from tests.test_tail_put25_atm_features import TailATMFeaturesTest


class RuleSizingTest(unittest.TestCase):
    def test_full_allocation_exactly_recovers_rule_and_never_invents_trade(self):
        rule = np.array([0, CODES[1], CODES[-1]] * 4)
        np.testing.assert_array_equal(translate(np.full(12, SIZE_CODES[-1]), rule), rule)
        for alias in SIZE_CODES:
            result = translate(np.full(12, alias), rule)
            a = np.array([ACTIONS[int(c)][1] if c else 0 for c in result])
            b = np.array([ACTIONS[int(c)][1] if c else 0 for c in rule])
            self.assertTrue(((a == 0) | (np.sign(a) == np.sign(b))).all())
            self.assertTrue((abs(a) <= abs(b)).all())
        np.testing.assert_array_equal(translate(np.full(12, SIZE_CODES[0]), rule), np.zeros(12))

    def test_training_labels_keep_actual_directional_integer_sizing_outcomes(self):
        daily = np.broadcast_to(np.arange(9)[None, :, None], (3, 9, 4)).copy()
        output = sizing_labels(daily, [0, CODES[1], CODES[-1]])
        np.testing.assert_array_equal(output[0], np.zeros((5, 4)))
        np.testing.assert_array_equal(output[1, :, 0], [0, 4, 3, 2, 1])
        np.testing.assert_array_equal(output[2, :, 0], [0, 5, 6, 7, 8])

    def test_optional_reversion_warmup_does_not_veto_rule_signal(self):
        dates, panel, memory, a, b = TailATMFeaturesTest().fixture()
        result = states(a, b, panel, dates, memory, .65)
        self.assertEqual(len(columns(result)), 49)
        self.assertTrue(result.feature_available.iloc[62])
        self.assertEqual(result.spread_reversion_available.iloc[62], 0)
        self.assertEqual(result.spread_reversion.iloc[62], 0)
        self.assertTrue(np.isfinite(result.loc[result.feature_available, columns(result)]).all().all())


if __name__ == '__main__':
    unittest.main()
