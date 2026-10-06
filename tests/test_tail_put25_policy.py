import unittest
import numpy as np
import pandas as pd
from scripts import tail_put_policy as put10, tail_put25_policy as put25
from scripts.curve_trading.config import COMBOS, ACTIONS


class Put25PolicyTest(unittest.TestCase):
    def test_same_learning_kernel_and_sizes_only_contract_ids_change(self):
        rng = np.random.default_rng(10)
        n = 520; x = pd.DataFrame(rng.normal(size=(n, 2)), columns=put10.FEATURES)
        daily = rng.normal(size=(n, 9, 3)); daily[:, 0] = 0
        positions = np.arange(n); ends = positions + 3
        a, detail_a = put10.fit_tail(x, daily, positions, ends)
        b, detail_b = put25.fit_tail(x, daily, positions, ends)
        self.assertEqual(detail_a, detail_b)
        np.testing.assert_array_equal(a[0].tree_.feature, b[0].tree_.feature)
        np.testing.assert_array_equal(a[0].tree_.threshold, b[0].tree_.threshold)
        size_a = [ACTIONS[int(c)][1] if c else 0 for c in a[2]]
        size_b = [ACTIONS[int(c)][1] if c else 0 for c in b[2]]
        self.assertEqual(size_a, size_b)
        self.assertTrue(all(COMBOS[ACTIONS[c][0]] == (30, 'P', 25, 25) for c in put25.CODES if c))
        self.assertEqual(put25.FEATURES, put10.FEATURES)

    def test_put25_rule_uses_only_put25_direction_codes(self):
        dates = pd.bdate_range('2018-01-01', periods=3)
        state = pd.DataFrame({'eligible': True, 'feature_available': True,
                              'wing_iv_gap_pct': [.1, .5, .9]}, index=dates)
        result = put25.simple_rule(dates, state, np.ones((3, 2), bool), np.ones(3, bool))
        np.testing.assert_array_equal(result, [put25.CODES[-1], 0, put25.CODES[1]])


if __name__ == '__main__':
    unittest.main()
