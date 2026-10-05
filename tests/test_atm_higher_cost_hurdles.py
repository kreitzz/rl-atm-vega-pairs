"""Higher artificial training costs must not alter actual execution fees."""
from dataclasses import replace
import unittest
import numpy as np
import pandas as pd
from tests.test_atm_cost_hurdles import CostHurdleTests
from scripts.run_atm_cost_hurdles import make_units, prepare_hurdle, reward_and_memory
from scripts.curve_trading.policy import ATM, Replay


class HigherHurdleTests(unittest.TestCase):
    def test_costs_above_original_match_fresh_paths(self):
        q, s, selection, divs, cal, cfg, panel, units = CostHurdleTests().inputs()
        original = reward_and_memory(units, cfg)[0]
        for fee in [1., 1.5]:
            training_cfg = replace(cfg, option_commission=fee)
            state, labels, _, _, _ = prepare_hurdle(panel, units, units, fee)
            fresh = make_units(panel, q, s, selection, divs, cal, training_cfg)
            expected = reward_and_memory(fresh, training_cfg)[0]
            np.testing.assert_allclose(labels[ATM], expected, atol=1e-7)
            self.assertTrue((expected.sum(axis=2) < original.sum(axis=2)).all())
            np.testing.assert_allclose(state.commission_vol_entry, .13 * fee / .65)
            replay = Replay(q, s, selection, [1, 2], divs, replace(cfg, option_commission=.03))
            _, trades, _ = replay.run(pd.Series([1 + ATM * 8 + 7], index=cal[:1]), cal)
            entry = trades[trades.reason.eq('entry')]
            self.assertEqual(len(entry), 2)
            np.testing.assert_allclose(entry.commission, abs(entry.quantity) * .03)
        self.assertEqual(units[0].cfg.option_commission, .65)
