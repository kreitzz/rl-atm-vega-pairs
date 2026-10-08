import os
for _name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_name] = '1'

from copy import deepcopy
import inspect
import unittest
from unittest.mock import patch

import pandas as pd

from scripts import uniform_smile_engine
from scripts.option_flex.contract_engine import MarketRow, Settings
from scripts.p25_rl.ledger import (instrumentation_info, simulate_with_observer,
                                   _instrument_function)
from scripts.uniform_smile_risk import book_risk


class TinyMarket:
    """Five sessions, one executable pair and deterministic expiry runoff."""
    def __init__(self, calendar):
        self.calendar = calendar
        self.divs = pd.DataFrame(columns=['secid', 'ex_date', 'amount'])

    def stock(self, date, secid):
        return 100., 1.

    def rate(self, date):
        return .001

    def dividend_pv(self, date, secid, expiry, rate):
        return 0.

    def quote(self, date, optionid):
        return MarketRow(secid=1 if optionid == 11 else 2, optionid=optionid,
                         date=date, strike=100., expiry=self.calendar[-1],
                         cp_flag='P', cfadj=1., multiplier=100., standard=True,
                         bid=1., ask=1.02, delta=-.5, vega=.2,
                         bs_iv=.2, bs_rate=.001, known_dividend_pv=0.,
                         bs_status='fitted', open_interest=1000)


def tiny_inputs():
    calendar = pd.bdate_range('2020-01-06', periods=5)
    signals = pd.DataFrame([dict(date=calendar[0], pair='1_2', wing='P25',
                                 secid_a=1, secid_b=2, optionid_a=11,
                                 optionid_b=22, base_level=4, direction=1)])
    cfg = Settings(capital=10_000_000, max_gross_vega=200_000,
                   hold_to_expiry=True, risk_controls=True, min_dte=1,
                   max_dte=10, max_moneyness_distance=1.,
                   option_commission=.03, option_penalty_vol_points=.1,
                   max_stress_loss_fraction=.9)
    return TinyMarket(calendar), signals, calendar, cfg, 1_000_000


class LedgerObserverTests(unittest.TestCase):
    def test_structural_identity_and_provenance_are_detached(self):
        source = inspect.getsource(uniform_smile_engine.simulate)
        _, info = _instrument_function(source)
        self.assertTrue(info['accounting_ast_unchanged'])
        self.assertEqual(info, instrumentation_info())
        original_args = list(inspect.signature(uniform_smile_engine.simulate).parameters)
        observed_args = list(inspect.signature(simulate_with_observer).parameters)
        self.assertEqual(observed_args, original_args + ['session_observer'])
        info['accounting_ast_unchanged'] = False
        self.assertTrue(instrumentation_info()['accounting_ast_unchanged'])

    def test_callback_can_mutate_snapshots_without_changing_ledger(self):
        seen = []
        args = tiny_inputs()
        with patch.object(uniform_smile_engine, 'book_risk', book_risk):
            expected = uniform_smile_engine.simulate(*args)

        def observe(date, daily, positions, pairs, global_metrics):
            seen.append((date, deepcopy(daily), deepcopy(positions)))
            daily['nav'] = -1.
            if positions:
                positions[0]['quantity'] = 0
                positions.clear()
            for metrics in pairs.values():
                metrics['target_hedge'].clear()
            global_metrics['target_hedge'].clear()
            global_metrics['gross_vega'] = -1.

        actual = simulate_with_observer(*args, session_observer=observe)
        for want, got in zip(expected[:3], actual[:3]):
            pd.testing.assert_frame_equal(want, got, check_exact=True)
        self.assertEqual(expected[3], actual[3])
        self.assertEqual([row[0] for row in seen], list(args[2]))
        self.assertFalse(seen[0][2])
        self.assertTrue(seen[1][2])
        self.assertFalse(seen[-1][2])
        for date, daily, _ in seen:
            self.assertEqual(daily['nav'], actual[0].loc[date, 'nav'])

    def test_completed_close_can_control_next_session_only(self):
        args = tiny_inputs()
        decisions = {}

        def observer(date, daily, positions, pairs, global_metrics):
            decisions[date] = 0

        def policy(row, context):
            return decisions[pd.Timestamp(row['date'])]

        daily, trades, fills, audit = simulate_with_observer(
            *args, policy=policy, session_observer=observer)
        self.assertTrue(trades.empty)
        self.assertEqual(fills.iloc[0].signal_date, args[2][0])
        self.assertEqual(fills.iloc[0].entry_date, args[2][1])
        self.assertEqual(fills.iloc[0].level, 0)
        self.assertTrue(daily.active_cohorts.eq(0).all())
        self.assertLess(audit['max_ledger_error'], 1e-5)


if __name__ == '__main__':
    unittest.main()
