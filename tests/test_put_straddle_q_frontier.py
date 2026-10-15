import copy
import unittest

import numpy as np
import pandas as pd

from scripts import put_straddle_q_frontier as q


def market(straddle, pair=None, cost=0.):
    n=len(straddle)
    return dict(gross=np.column_stack([1+np.array(pair if pair is not None else np.zeros(n)),
        1+np.array(straddle),np.ones(n)]),cash=np.zeros(n),costs=np.full((n,2),cost),
        cash_fractions=np.ones((n,2)),elapsed=np.full(n,1/365))


class FrontierQTests(unittest.TestCase):
    def test_first_session_uses_previous_holdings(self):
        z=q.week_path(market([.10,.10]),q.TARGETS[0][None],3,0.,None)
        np.testing.assert_allclose(z['daily'][:,0],[0.,.10],atol=1e-12)

    def test_budget_crossing_waits_until_next_close_and_can_overshoot(self):
        z=q.week_path(market([-.06,-.04,.10]),q.TARGETS[3][None],3,0.,.05)
        np.testing.assert_allclose(z['daily'][:,0],[-.06,-.04,0.],atol=1e-12)
        self.assertTrue(z['stopped'][0])
        self.assertAlmostEqual(z['drawdown'][0],1-.94*.96)
        self.assertTrue(z['hard_crossings'][0,0])
        self.assertFalse(z['hard_crossings'][1:,0].any())

    def test_soft_budget_reduces_straddles_next_close(self):
        z=q.week_path(market([-.041,-.001,.50]),q.TARGETS[7][None],7,0.,.05)
        np.testing.assert_allclose(z['daily_weights'][1,0],q.TARGETS[1],atol=1e-12)
        self.assertAlmostEqual(z['daily'][2,0],0.)
        self.assertFalse(z['stopped'][0])

    def test_cash_inside_source_books_offsets_synthetic_cash_short(self):
        m=market([0.]);m['gross'][:]=1.001;m['cash'][:]=.001
        z=q.week_path(m,q.TARGETS[7][None],7,0.,None)
        self.assertAlmostEqual(z['daily'][0,0],0.,places=12)

    def test_actual_negative_pooled_cash_pays_borrowing_spread(self):
        m=market([0.]);m['cash_fractions'][:]=0.
        z=q.week_path(m,q.TARGETS[7][None],7,0.,None)
        self.assertAlmostEqual(z['daily'][0,0],-.01/365,places=12)

    def test_execution_cost_does_not_create_a_pre_fee_highwater_mark(self):
        z=q.week_path(market([.01],cost=.001),q.TARGETS[0][None],3,0.,None)
        self.assertAlmostEqual(z['drawdown'][0],.001)
        self.assertAlmostEqual(z['nav'][0,0],.999)

    def toy_data(self):
        dates=pd.date_range('2018-01-01',periods=55,freq='W-MON')
        data=dict(weeks=[dict(cutoff=d,end=d+pd.Timedelta(days=4)) for d in dates])
        trans=dict(row=np.tile([0,1,q.A],(55,1)),following=np.tile([0,1,0],(55,1)),
            profit=np.tile([1.,0.,10.],(55,1)),variance=np.zeros((55,3)),
            stop_penalty=np.zeros((55,3)),done=np.tile([True,False,True],(55,1)))
        return data,trans

    def test_future_week_rewards_cannot_change_fit(self):
        data,trans=self.toy_data()
        original=q.fit(data,trans,52,104,.05,.97,'unlimited')
        changed=copy.deepcopy(trans);changed['profit'][52:]=1e12
        self.assertEqual(original,q.fit(data,changed,52,104,.05,.97,'unlimited'))
        self.assertEqual(original['training_weeks'],52)
        self.assertLess(pd.Timestamp(original['last_training_end']),data['weeks'][52]['cutoff'])

    def test_continuation_changes_action_in_a_sequential_problem(self):
        data,trans=self.toy_data()
        myopic=np.array(q.fit(data,trans,52,104,0.,0.,'unlimited')['q'])
        sequential=np.array(q.fit(data,trans,52,104,0.,.97,'unlimited')['q'])
        self.assertGreater(myopic[0,0],myopic[0,1])
        self.assertGreater(sequential[0,1],sequential[0,0])

    def test_pending_budget_state_only_allows_cash(self):
        legal=q.legal_actions(.05)
        np.testing.assert_array_equal(np.flatnonzero(legal[-1]),[0])
        self.assertTrue(q.legal_actions(None).all())
        self.assertGreaterEqual(q.drawdown_index(.81*.05,.05),4)


if __name__=='__main__':unittest.main()
