import unittest
import numpy as np
import pandas as pd

from tests.test_contract_engine import fixture
from scripts.option_flex.contract_engine import Settings, simulate as original
from scripts.option_flex.pooled_tail_engine import simulate
from scripts.run_selected_pair_backtest import audit


class PooledAccountingTest(unittest.TestCase):
    def setup_book(self):
        q, s, a, dates, _ = fixture()
        q['expiry'] = dates[-1]
        q['bs_iv'] = .4
        q['bs_rate'] = 0.
        q['known_dividend_pv'] = 0.
        cfg = Settings(min_dte=1, hold_to_expiry=True, option_spread_fraction=0.,
            option_penalty_vol_points=.1, option_commission=.03, entry_vega_per_leg=10,
            max_gross_vega=100, stock_cost_bps=1., stock_borrow_rate=0.)
        return q, s, a, dates, cfg

    def test_ordinary_book_exactly_matches_frozen_engine(self):
        q, s, a, dates, cfg = self.setup_book()
        before = original(q, s, a, dates, [1, 2], cfg)
        after = simulate(q, s, a, dates, [1, 2], cfg)
        pd.testing.assert_frame_equal(before[0], after[0].drop(columns=['causal_held_mark_fallbacks']))
        pd.testing.assert_frame_equal(before[1], after[1])
        pd.testing.assert_frame_equal(before[2], after[2])

    def test_announced_event_closes_both_and_charges_exit_costs(self):
        q, s, a, dates, cfg = self.setup_book()
        a = pd.Series([1., 1.], index=dates[:2])
        q.loc[q.secid.eq(1) & q.date.ge(dates[3]), 'standard'] = False
        event = dict(secid=1, announced=str(dates[0]), effective=str(dates[3]))
        nav, trades, decisions = simulate(q, s, a, dates, [1, 2], cfg, corporate_events=[event])
        exits = trades[trades.reason.eq('announced_corporate_action_exit')]
        self.assertEqual(set(exits.secid), {1, 2})
        self.assertTrue(exits.date.eq(dates[2]).all())
        self.assertAlmostEqual(exits.commission.sum(), .06)
        self.assertTrue(exits.execution_penalty.gt(0).all())
        self.assertIn('announced_corporate_action_no_entry', decisions.status.to_list())
        audit(nav, trades, s, cfg, [1, 2])

    def test_future_announcement_cannot_trigger_exit(self):
        q, s, a, dates, cfg = self.setup_book()
        event = dict(secid=1, announced=str(dates[-1] + pd.Timedelta(days=1)), effective=str(dates[3]))
        _, trades, _ = simulate(q, s, a, dates, [1, 2], cfg, corporate_events=[event])
        self.assertFalse(trades.reason.eq('announced_corporate_action_exit').any())

    def test_missing_held_quote_uses_causal_iv_and_ledger_reconciles(self):
        q, s, a, dates, cfg = self.setup_book()
        q = q[~(q.date.eq(dates[2]) & q.optionid.eq(1))]
        nav, trades, _ = simulate(q, s, a, dates, [1, 2], cfg)
        self.assertEqual(nav.causal_held_mark_fallbacks.sum(), 1)
        self.assertEqual(nav.active_cohorts.iloc[-1], 0)
        audit(nav, trades, s, cfg, [1, 2])

    def test_missing_entry_quote_is_never_synthesized(self):
        q, s, a, dates, cfg = self.setup_book()
        q = q[~(q.date.eq(dates[1]) & q.optionid.eq(1))]
        nav, trades, decisions = simulate(q, s, a, dates, [1, 2], cfg)
        self.assertTrue(trades.empty)
        self.assertEqual(nav.causal_held_mark_fallbacks.sum(), 0)
        self.assertEqual(decisions.status.iloc[0], 'entry_quote_unavailable')


if __name__ == '__main__':
    unittest.main()
