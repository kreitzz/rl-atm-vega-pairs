import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from tests.test_joint_smile_engine import JointEngineTest, Market
from scripts.curve_trading.config import config
from scripts.uniform_smile_engine import simulate
from scripts.run_p25_long_management import held_rule


class ManagementTest(unittest.TestCase):
    def setup_book(self):
        fixture = JointEngineTest()
        q, s, _, cal, divs = fixture.inputs()
        return Market(q, s, divs), fixture.signals(cal, 1, 4), cal, replace(config(), option_commission=.03)

    def run_book(self, market, rows, cal, cfg, callback=None):
        return simulate(market, rows, cal, cfg, cfg.capital, management_policy=callback)

    def reconcile(self, nav, trades, cfg):
        flow = (-trades.quantity*100*trades.price-trades.commission-trades.execution_penalty).groupby(trades.date).sum().reindex(nav.index, fill_value=0)
        expected = flow+nav.option_value.diff().fillna(nav.option_value)+nav.hedge_gross_pnl-nav.hedge_cost+nav.interest-nav.stock_borrow_cost+nav.dividend_cash
        np.testing.assert_allclose(expected, nav.pnl, atol=1e-7, rtol=0)
        self.assertTrue(trades.groupby(['cohort', 'optionid']).quantity.sum().eq(0).all())
        self.assertAlmostEqual(nav.option_value.iloc[-1], 0.)
        self.assertAlmostEqual(nav.stock_value.iloc[-1], 0.)

    def test_hold_callback_preserves_original_book(self):
        market, rows, cal, cfg = self.setup_book()
        old = self.run_book(market, rows, cal, cfg)
        new = self.run_book(market, rows, cal, cfg, lambda date, held: 1.)
        for actual, expected in zip(new[:3], old[:3]):
            pd.testing.assert_frame_equal(actual, expected)
        self.reconcile(new[0], new[1], cfg)

    def test_close_instructions_fill_next_session_and_pay_both_legs(self):
        market, rows, cal, cfg = self.setup_book()
        seen = []
        def callback(date, held):
            seen.append(date)
            return 0.
        nav, trades, _, audit = self.run_book(market, rows, cal, cfg, callback)
        entry = trades.loc[trades.reason.eq('entry')]
        exits = trades.loc[trades.reason.eq('learned_policy_exit')]
        self.assertEqual(seen, [cal[1]])
        self.assertEqual(len(exits), 2)
        self.assertTrue(exits.date.eq(cal[2]).all())
        self.assertTrue(exits.exit_signal_date.eq(cal[1]).all())
        self.assertEqual(exits.quantity.tolist(), (-entry.quantity).tolist())
        np.testing.assert_allclose(exits.commission, exits.quantity.abs()*.03)
        self.assertGreater(nav.option_execution_cost.loc[cal[2]], exits.commission.sum())
        self.assertLess(audit['max_ledger_error'], 1e-7)
        self.reconcile(nav, trades, cfg)

    def test_half_trim_is_integer_balanced_and_cannot_repeat(self):
        market, rows, cal, cfg = self.setup_book()
        nav, trades, _, _ = self.run_book(market, rows, cal, cfg, lambda date, held: .5)
        entry = trades.loc[trades.reason.eq('entry')].set_index('optionid')
        trims = trades.loc[trades.reason.eq('learned_policy_trim')].set_index('optionid')
        self.assertEqual(len(trims), 2)
        self.assertTrue(trims.date.eq(cal[2]).all())
        remain = (entry.quantity+trims.quantity).abs()
        np.testing.assert_array_equal(remain, np.floor(entry.quantity.abs()*.5))
        self.assertGreaterEqual(remain.min(), 1)
        self.assertLessEqual((remain/entry.quantity.abs()).max()-(remain/entry.quantity.abs()).min(), .1)
        self.reconcile(nav, trades, cfg)

    def test_missing_raw_quote_blocks_both_exit_fills_until_retry(self):
        market, rows, cal, cfg = self.setup_book()
        del market.q[cal[2], 1]
        nav, trades, _, audit = self.run_book(market, rows, cal, cfg, lambda date, held: 0.)
        exits = trades.loc[trades.reason.eq('learned_policy_exit')]
        self.assertEqual(len(exits), 2)
        self.assertTrue(exits.date.eq(cal[3]).all())
        self.assertEqual(audit['management_log'][0]['status'], 'no_executable_exit_quote')
        self.assertGreater(nav.option_value.loc[cal[2]], -100000.)
        self.reconcile(nav, trades, cfg)

    def test_single_contract_leg_cannot_be_zeroed_by_half_trim(self):
        market, rows, cal, cfg = self.setup_book()
        v = market.quote(cal[1], 2).vega*100
        cfg = replace(cfg, entry_vega_per_leg=v)
        nav, trades, decisions, audit = self.run_book(market, rows, cal, cfg, lambda date, held: .5)
        self.assertEqual(decisions.status.iloc[0], 'entered')
        self.assertEqual(trades.loc[trades.reason.eq('entry'), 'quantity'].abs().min(), 1)
        self.assertFalse(trades.reason.eq('learned_policy_trim').any())
        self.assertTrue(any(r['status']=='integer_trim_unavailable' for r in audit['management_log']))
        self.reconcile(nav, trades, cfg)

    def test_future_realized_variance_cannot_change_current_action(self):
        date = pd.Timestamp('2020-01-02')
        dates = pd.bdate_range(date, periods=2)
        rv = {sid: pd.DataFrame({'rv20':[.09, .16], 'rv60':[.09, .16]}, index=dates) for sid in [1, 2]}
        held = [dict(secid=1, quantity=2, entry_iv=.4, quote=dict(bs_iv=.4, delta=-.25)), dict(secid=2, quantity=-1, entry_iv=.3, quote=dict(bs_iv=.3, delta=-.25))]
        first = held_rule(date, held, rv, 'carry_trim')
        for state in rv.values(): state.loc[dates[1]] = 1000.
        self.assertEqual(first, held_rule(date, held, rv, 'carry_trim'))

    def test_locked_counts_reproduce_fills_and_reject_invalid_counts(self):
        market, rows, cal, cfg = self.setup_book()
        old = self.run_book(market, rows, cal, cfg)
        entry = old[1].loc[old[1].reason.eq('entry')]
        cohort = entry.cohort.iloc[0]
        counts = {cohort: tuple(entry.quantity.abs())}
        new = simulate(market, rows, cal, cfg, cfg.capital, locked_entry_counts=counts)
        for actual, expected in zip(new[:3], old[:3]): pd.testing.assert_frame_equal(actual, expected)
        with self.assertRaises(ValueError):
            simulate(market, rows, cal, cfg, cfg.capital, locked_entry_counts={cohort:(0,1)})
        with self.assertRaises(ValueError):
            simulate(market, rows, cal, cfg, cfg.capital, locked_entry_counts={cohort:(100,1)})


if __name__ == '__main__': unittest.main()
