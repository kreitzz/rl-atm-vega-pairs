import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
from tests.test_expiry_holding import fixture
from scripts.run_atm_cost_hurdles import make_units, prepare_hurdle, reward_and_memory
from scripts.curve_trading.policy import ATM, Replay


class CostHurdleTests(unittest.TestCase):
    def inputs(self):
        q, s, _, cal, cfg = fixture()
        cfg = replace(cfg, entry_vega_per_leg=80, max_gross_vega=1000, option_penalty_vol_points=.1)
        selection = q[['date', 'secid', 'optionid', 'cp_flag']].copy()
        selection['tenor'] = 30; selection['bucket'] = 'atm'
        divs = pd.DataFrame(columns=['secid', 'ex_date', 'declare_date', 'amount'])
        panel = pd.DataFrame({'commission_a': .065, 'commission_b': .065,
                              'commission_vol_entry': .13, 'signal': [1., 2.]}, index=cal[:2])
        units = make_units(panel, q, s, selection, divs, cal, cfg)
        panel = panel.join(reward_and_memory(units, cfg)[-1])
        return q, s, selection, divs, cal, cfg, panel, units

    def test_hurdles_reprice_training_only_and_match_fresh_paths(self):
        q, s, selection, divs, cal, cfg, panel, units = self.inputs()
        for fee in [.03, .15, .30, .65]:
            state, labels, _, _, _ = prepare_hurdle(panel, units, units, fee)
            expected = reward_and_memory(make_units(panel, q, s, selection, divs, cal, replace(cfg, option_commission=fee)), replace(cfg, option_commission=fee))[0]
            np.testing.assert_allclose(labels[ATM], expected, atol=1e-7)
            np.testing.assert_allclose(state.commission_vol_entry, .13 * fee / .65)
            pd.testing.assert_series_equal(state.signal, panel.signal)
        self.assertEqual(units[0].cfg.option_commission, .65)

    def test_execution_cost_stays_three_cents(self):
        q, s, selection, divs, cal, cfg, panel, units = self.inputs()
        prepare_hurdle(panel, units, units, .30)
        replay = Replay(q, s, selection, [1, 2], divs, replace(cfg, option_commission=.03))
        actions = pd.Series([1 + ATM * 8 + 7], index=cal[:1])
        _, trades, _ = replay.run(actions, cal)
        entry = trades[trades.reason.eq('entry')]
        self.assertEqual(len(entry), 2)
        np.testing.assert_allclose(entry.commission, abs(entry.quantity) * .03)


if __name__ == '__main__': unittest.main()
