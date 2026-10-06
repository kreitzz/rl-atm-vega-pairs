import unittest

import numpy as np
import pandas as pd

from scripts.uniform_smile_engine import simulate as original
from scripts.year_prefix_engine import simulate
from tests.test_p25_management import ManagementTest


class PrefixEngineTest(unittest.TestCase):
    def test_default_engine_reproduces_all_original_ledger_fields(self):
        market,rows,cal,cfg = ManagementTest().setup_book()
        expected = original(market,rows,cal,cfg,cfg.capital)
        actual = simulate(market,rows,cal,cfg,cfg.capital)
        for a,b in zip(actual[:3],expected[:3]):
            pd.testing.assert_frame_equal(a,b)

    def test_marked_prefix_equals_full_replay_prefix_without_forced_exit(self):
        market,rows,cal,cfg = ManagementTest().setup_book()
        complete = original(market,rows,cal,cfg,cfg.capital)
        partial = cal[:3]
        cutoff = partial[-1]
        old_quote = market.quote
        def past_quote(date,oid):
            self.assertLessEqual(date,cutoff)
            return old_quote(date,oid)
        market.quote = past_quote
        nav,trades,decisions,audit = simulate(market,rows,partial,cfg,cfg.capital,
                                            require_runoff=False,settlement_calendar=cal)
        pd.testing.assert_frame_equal(nav,complete[0].loc[:cutoff])
        pd.testing.assert_frame_equal(trades,complete[1].loc[complete[1].date<=cutoff])
        self.assertEqual(decisions.status.iloc[0],'entered')
        self.assertGreater(nav.active_cohorts.iloc[-1],0)
        self.assertFalse(trades.reason.eq('scheduled_exit').any())
        flow = (-trades.quantity*100*trades.price-trades.commission-trades.execution_penalty).groupby(trades.date).sum().reindex(nav.index,fill_value=0)
        expected = flow+nav.option_value.diff().fillna(nav.option_value)+nav.hedge_gross_pnl-nav.hedge_cost+nav.interest-nav.stock_borrow_cost+nav.dividend_cash
        np.testing.assert_allclose(expected,nav.pnl,atol=1e-7)
        self.assertLess(audit['max_ledger_error'],1e-7)

    def test_partial_mode_requires_a_full_settlement_calendar(self):
        market,rows,cal,cfg = ManagementTest().setup_book()
        with self.assertRaises(ValueError):
            simulate(market,rows,cal[:3],cfg,cfg.capital,require_runoff=False)

    def test_resumed_chunks_reproduce_cash_and_pending_management_fills(self):
        market,rows,cal,cfg=ManagementTest().setup_book()
        for callback in [None,lambda date,held:.5,lambda date,held:0.]:
            whole=simulate(market,rows,cal,cfg,cfg.capital,management_policy=callback)
            first=simulate(market,rows,cal,cfg,cfg.capital,management_policy=callback,through_date=cal[1])
            second=simulate(market,rows,cal,cfg,cfg.capital,management_policy=callback,
                            resume_state=first[3]['resume_state'])
            for k in range(3):
                actual=pd.concat([first[k],second[k]])
                if k:actual=actual.reset_index(drop=True)
                pd.testing.assert_frame_equal(actual,whole[k],check_dtype=False)


if __name__=='__main__':
    unittest.main()
