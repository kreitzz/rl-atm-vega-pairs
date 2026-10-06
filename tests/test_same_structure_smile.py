import unittest
import numpy as np
import pandas as pd

from scripts import same_structure_smile as p


class SharedStructureTest(unittest.TestCase):
    def rows(self):
        rows = pd.DataFrame(dict(rule_direction=[1, -1], iv_gap_relative=[-.2, .3],
            iv_rank=[.1, .9], vrp_gap_relative=[-.5, .6], vrp_gap_relative_available=[1, 1],
            dte_gap=[2., -3.], dte_gap_available=[1, 1], observed=True, eligible=True,
            context=True, target_long=[8., 3.], target_short=[-5., 9.], iv_mean=.3))
        for col in ['rv_mean', 'beta', 'correlation', 'dte_mean', 'commission_vol_points', 'vix']:
            rows[col] = 0.
            rows[col+'_available'] = 0.
        for days in [91, 365]:
            for side in ['long', 'short']:
                rows[f'profit_{side}_{days}d'] = 1.
                rows[f'count_{side}_{days}d'] = 100.
                rows[f'profit_{side}_{days}d_available'] = 1.
                rows[f'count_{side}_{days}d_available'] = 1.
        return rows

    def test_identical_inputs_across_wings_get_identical_state_and_gates(self):
        panels = [p.oriented_states(self.rows().assign(wing=wing)) for wing in ['ATM', 'P25', 'P75']]
        for panel in panels:
            self.assertEqual(panel.signal_eligible.tolist(), [True, True])
            self.assertEqual(panel.oriented_iv_gap.tolist(), [-.2, -.3])
            self.assertEqual(panel.oriented_dte_gap.tolist(), [2., 3.])
            self.assertEqual(panel.target.tolist(), [8., 9.])
        pd.testing.assert_frame_equal(panels[0][p.FEATURES], panels[1][p.FEATURES])
        pd.testing.assert_frame_equal(panels[1][p.FEATURES], panels[2][p.FEATURES])

    def test_gate_uses_proposed_direction_and_ignores_missing_optional_features(self):
        rows = self.rows()
        rows.loc[0, 'profit_short_91d'] = -1.
        rows.loc[1, 'profit_short_91d'] = -1.
        result = p.oriented_states(rows)
        self.assertEqual(result.signal_eligible.tolist(), [True, False])
        self.assertTrue(result.vix_available.eq(0).all())
        self.assertTrue(np.isfinite(result[p.FEATURES]).all().all())

    def test_labels_match_vega_sizing_for_both_directions_and_cash(self):
        labels = np.broadcast_to(np.arange(9)[None, :, None], (2, 9, 3)).copy()
        paths = p.directional_paths(labels, np.array([1, -1]))
        np.testing.assert_array_equal(paths[:, :, 0], [[0, 5, 6, 7, 8], [0, 4, 3, 2, 1]])
        np.testing.assert_array_equal(paths[:, 0], 0)


if __name__ == '__main__':
    unittest.main()
