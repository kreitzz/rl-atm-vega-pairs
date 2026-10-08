"""Focused checks for observation before next-session management orders."""
import os
for _name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'ARROW_NUM_THREADS']:
    os.environ[_name] = '1'

from copy import deepcopy
import inspect
import unittest
from unittest.mock import patch

import pandas as pd

from scripts import uniform_smile_engine
from scripts.p25_rl.managed_ledger import (instrumentation_info,
    simulate_with_management_observer, _instrument_function)
from scripts.uniform_smile_risk import book_risk
from tests.test_p25_rl_ledger import tiny_inputs


class ManagedLedgerTests(unittest.TestCase):
    def test_guarded_structure_and_hold_observer_preserve_original_ledger(self):
        _, info = _instrument_function(inspect.getsource(uniform_smile_engine.simulate))
        self.assertEqual(info, instrumentation_info())
        self.assertTrue(info['accounting_ast_unchanged'])
        seen = []
        args = tiny_inputs()
        hold = lambda date, held: 1.
        with patch.object(uniform_smile_engine, 'book_risk', book_risk):
            expected = uniform_smile_engine.simulate(*args, management_policy=hold)

        def observer(date, day, positions, pairs, global_risk):
            seen.append((date, deepcopy(day), deepcopy(positions)))

        observed = simulate_with_management_observer(
            *args, management_policy=hold, session_observer=observer)
        without_observer = simulate_with_management_observer(*args, management_policy=hold)
        for actual in [observed, without_observer]:
            for want, got in zip(expected[:3], actual[:3]):
                pd.testing.assert_frame_equal(want, got, check_exact=True)
            self.assertEqual(expected[3], actual[3])
        self.assertEqual([date for date, _, _ in seen], list(args[2]))
        for date, day, positions in seen:
            self.assertEqual(day, observed[0].loc[date].to_dict() | {'date': date})
            for position in positions:
                self.assertIn('entry_iv', position)
                self.assertIn('initial_quantity', position)
                self.assertIn('management_fraction', position)
                self.assertIn('entry_date', position)
                self.assertIn('quote', position)
                self.assertEqual(position['quote']['date'], date)
                self.assertEqual(position['spot'], 100.)

    def test_close_instruction_fills_only_next_observed_session(self):
        args = tiny_inputs()
        dates = args[2]
        instructions = {}

        def observer(date, day, positions, pairs, global_risk):
            instructions[date] = 0. if date == dates[1] else 1.

        def management(date, held):
            # The dictionary lookup also proves the observer ran before this
            # original per-cohort callback at that completed close.
            return instructions[date]

        nav, trades, decisions, audit = simulate_with_management_observer(
            *args, management_policy=management, session_observer=observer)
        exits = trades.loc[trades.reason.eq('learned_policy_exit')]
        self.assertEqual(len(exits), 2)
        self.assertTrue(exits.date.eq(dates[2]).all())
        self.assertTrue(exits.exit_signal_date.eq(dates[1]).all())
        self.assertTrue(trades.loc[trades.reason.eq('entry')].date.eq(dates[1]).all())
        self.assertEqual(nav.loc[dates[1], 'active_cohorts'], 1)
        self.assertEqual(nav.loc[dates[2], 'active_cohorts'], 0)
        self.assertEqual(decisions.iloc[0].status, 'entered')
        self.assertLess(audit['max_ledger_error'], 1e-5)

    def test_snapshot_mutation_does_not_change_hold_ledger_or_management(self):
        args = tiny_inputs()
        hold = lambda date, held: 1.
        expected = simulate_with_management_observer(*args, management_policy=hold)

        def observer(date, day, positions, pairs, global_risk):
            day['nav'] = -1.
            for position in positions:
                position['quantity'] = 0
                position['initial_quantity'] = 0
                position['management_fraction'] = 0.
                position['quote']['bid'] = -1.
                position['spot'] = -1.
            positions.clear()
            for metrics in pairs.values():
                metrics['target_hedge'].clear()
            pairs.clear()
            global_risk['target_hedge'].clear()

        actual = simulate_with_management_observer(
            *args, management_policy=hold, session_observer=observer)
        for want, got in zip(expected[:3], actual[:3]):
            pd.testing.assert_frame_equal(want, got, check_exact=True)
        self.assertEqual(expected[3], actual[3])


if __name__ == '__main__':
    unittest.main()
