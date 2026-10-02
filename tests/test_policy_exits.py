import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
from tests.test_expiry_holding import fixture
from scripts.option_flex.contract_engine import simulate as original
from scripts.option_flex.exit_engine import simulate
from scripts.run_selected_pair_backtest import audit

class PolicyExitTests(unittest.TestCase):
    def test_disabled_exits_reproduce_checkpoint_engine_exactly(self):
        q,s,a,d,c=fixture()
        for left,right in zip(original(q,s,a,d,[1,2],c),simulate(q,s,a,d,[1,2],c)):
            pd.testing.assert_frame_equal(left,right)

    def test_reversal_closes_both_legs_next_session_and_charges_roundtrip(self):
        q,s,a,d,c=fixture();c=replace(c,option_commission=.03,option_penalty_vol_points=.1,stock_cost_bps=1)
        exits=pd.Series([-1.],index=d[3:4])
        nav,t,_=simulate(q,s,a,d,[1,2],c,exit_signals=exits,exit_rule='reverse')
        closed=t[t.reason.eq('signal_reversal_exit')]
        self.assertEqual(set(closed.secid),{1,2});self.assertTrue(closed.date.eq(d[4]).all())
        self.assertTrue(closed.exit_signal_date.eq(d[3]).all())
        self.assertAlmostEqual(t.commission.sum(),.12)
        self.assertAlmostEqual(t.execution_penalty.sum(),4.)
        self.assertEqual(nav.loc[d[4],'option_value'],0)
        self.assertEqual(nav.loc[d[4],'stock_value'],0)
        audit(nav,t,s,c,[1,2])

    def test_flat_exit_is_optional_and_missing_signal_does_not_close(self):
        q,s,a,d,c=fixture();exits=pd.Series([0.],index=d[3:4])
        _,t,_=simulate(q,s,a,d,[1,2],c,exit_signals=exits,exit_rule='reverse')
        self.assertFalse(t.reason.eq('signal_flat_exit').any())
        nav,t,_=simulate(q,s,a,d,[1,2],c,exit_signals=exits,exit_rule='flat_or_reverse')
        self.assertTrue(t.loc[t.reason.eq('signal_flat_exit'),'date'].eq(d[4]).all())
        self.assertGreater(nav.loc[d[3],'active_cohorts'],0)

if __name__=='__main__':unittest.main()
