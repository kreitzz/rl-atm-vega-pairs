import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
from tests.test_expiry_holding import fixture
from scripts.retrain_atm_commission import fast_memory, cheaper_units, rebuild
from scripts.contract_learning_labels import profitability_memory
from scripts.curve_trading.unit_paths import UnitPaths, pair_labels


class AtmCommissionTrainingTests(unittest.TestCase):
    def test_empty_state_history_is_valid_cash_input(self):
        q, s, _, cal, cfg = fixture()
        cfg = replace(cfg, option_commission=.03, option_penalty_vol_points=.1)
        selection = q[['date', 'secid', 'optionid', 'cp_flag']].copy()
        selection['tenor'] = 30; selection['bucket'] = 'atm'
        columns = ['commission_a', 'commission_b', 'commission_vol_entry']
        columns += [f'{prefix}_{side}_{days}d' for days in [91, 365]
                    for side in ['short', 'long'] for prefix in ['profit', 'count']]
        panel = pd.DataFrame(columns=columns, index=pd.DatetimeIndex([]), dtype=float)
        divs = pd.DataFrame(columns=['secid', 'ex_date', 'declare_date', 'amount'])
        changed, labels, _, _, _, checks = rebuild(panel, q, s, selection, divs, cal, cfg)
        self.assertTrue(changed.empty)
        self.assertEqual(labels.daily.shape[0], 0)
        self.assertEqual(checks['original_memory_max_error'], 0.)

    def test_memory_matches_reference_and_ignores_unfinished_outcomes(self):
        dates = pd.bdate_range('2018-01-01', periods=350)
        ends = dates + pd.to_timedelta(np.arange(350) % 30 + 2, unit='D')
        rewards = np.column_stack([np.sin(np.arange(350)), np.cos(np.arange(350))])
        entered = np.ones_like(rewards, dtype=bool); entered[::13] = False; rewards[::17] = np.nan
        actual = fast_memory(dates, ends, rewards, entered)
        pd.testing.assert_frame_equal(actual, profitability_memory(dates, ends, rewards, entered), check_dtype=False, atol=1e-12)
        changed = rewards.copy(); changed[ends >= dates[220]] = 1e9
        pd.testing.assert_frame_equal(fast_memory(dates, ends, changed, entered).iloc[:221], actual.iloc[:221], atol=1e-12)

    def test_unit_repricing_matches_complete_rebuild(self):
        q, s, _, cal, old = fixture()
        old = replace(old, entry_vega_per_leg=80, max_gross_vega=1000, borrowing_spread=.01,
                      option_penalty_vol_points=.1)
        new = replace(old, option_commission=.03)
        q['bs_rate'] = .04; s['rate'] = .04
        selection = q[['date', 'secid', 'optionid', 'cp_flag']].copy()
        selection['tenor'] = 30; selection['bucket'] = 'atm'
        divs = pd.DataFrame(columns=['secid', 'ex_date', 'declare_date', 'amount'])
        def units(cfg):
            return [UnitPaths(q[q.secid.eq(sid)], s[s.secid.eq(sid)], selection[selection.secid.eq(sid)], cal, cal[:2], divs, cfg) for sid in [1, 2]]
        original = units(old)
        cheaper = [cheaper_units(u, new) for u in original]
        expected = pair_labels(*units(new), (30, 'C', 'atm', 'atm'), new)
        actual = pair_labels(*cheaper, (30, 'C', 'atm', 'atm'), new)
        for a, b in zip(actual[:3], expected[:3]): np.testing.assert_allclose(a, b, atol=1e-9)
        before = pair_labels(*original, (30, 'C', 'atm', 'atm'), old)[0].sum(axis=2)
        self.assertTrue((actual[0].sum(axis=2) > before).all())


if __name__ == '__main__': unittest.main()
