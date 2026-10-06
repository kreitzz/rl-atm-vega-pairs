import unittest
import numpy as np
import pandas as pd
from scripts.tail_put_policy import states, fit_tail, permitted, simple_rule, CODES, ALIASES, FEATURES
from scripts.curve_trading.policy import fit


class TailPutPolicyTest(unittest.TestCase):
    def test_ranks_are_causal_and_missing_contracts_are_not_filled(self):
        cal = pd.bdate_range('2018-01-01', periods=320)
        a = pd.DataFrame({'date': cal, 'bs_iv': np.linspace(.3, .5, len(cal))})
        b = pd.DataFrame({'date': cal, 'bs_iv': .25})
        panel = pd.DataFrame({'eligible': True}, index=cal)
        original = states(a, b, panel, cal)
        changed = a.copy(); changed.loc[250:, 'bs_iv'] = 3.
        pd.testing.assert_frame_equal(original.iloc[:250], states(changed, b, panel, cal).iloc[:250])
        missing = states(a.drop(index=200), b, panel, cal)
        self.assertFalse(missing.feature_available.iloc[200])
        self.assertTrue(np.isnan(missing.wing_iv_gap.iloc[200]))
        sparse_panel = panel.drop(index=cal[120:200])
        sparse = states(a, b, sparse_panel, cal)
        expected = original.wing_iv_gap.reindex(sparse_panel.index).reindex(cal).rolling(252, min_periods=63).rank(pct=True)
        pd.testing.assert_series_equal(sparse.wing_iv_gap_pct, expected.reindex(sparse_panel.index), check_names=False)

    def test_action_mapping_preserves_reference_optimizer(self):
        rng = np.random.default_rng(42)
        n = 520; x = pd.DataFrame(rng.normal(size=(n, 2)), columns=FEATURES)
        daily = rng.normal(scale=.1, size=(n, 9, 4)); daily[:, 0] = 0
        daily[:, -1, 0] += x.iloc[:, 0].to_numpy() * .3
        positions = np.arange(n); ends = positions + 4
        original, details = fit(x, daily, positions, ends, ALIASES)
        mapped, new_details = fit_tail(x, daily, positions, ends)
        self.assertEqual(details, new_details)
        np.testing.assert_array_equal(original[0].tree_.feature, mapped[0].tree_.feature)
        lookup = dict(zip(ALIASES, CODES))
        np.testing.assert_array_equal(mapped[2], [lookup[int(c)] for c in original[2]])

    def test_directional_memory_gates_and_missing_state_block_orders(self):
        dates = pd.bdate_range('2018-01-01', periods=5)
        state = pd.DataFrame({'eligible': [True, True, False, True, True],
            'feature_available': [True, True, True, False, True],
            'wing_iv_gap_pct': [.1, .9, .1, .1, .5]}, index=dates)
        gates = np.ones((5, 2), bool); gates[1, 0] = False
        result = simple_rule(dates, state, gates, np.ones(5, bool))
        np.testing.assert_array_equal(result, [CODES[-1], 0, 0, 0, 0])
        with self.assertRaises(ValueError):
            permitted([1]*5, dates, state, gates, np.ones(5, bool))


if __name__ == '__main__':
    unittest.main()
