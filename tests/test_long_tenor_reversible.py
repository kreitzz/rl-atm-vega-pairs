from dataclasses import replace
import unittest

import numpy as np
import pandas as pd

from tests import test_p25_management as fixtures
from scripts.uniform_smile_engine import simulate as frozen
from scripts.reversible_management_engine import simulate
from scripts.run_long_tenor_reversible import block_reward,callback,choose_actions,enough_history,inputs,risk_coefficient,transitions
from scripts.run_long_tenor_fitted_q import causal_transitions


class ReversibleTest(unittest.TestCase):
    def book(self):
        return fixtures.ManagementTest().setup_book()

    def reconcile(self,nav,trades,cfg):
        fixtures.ManagementTest().reconcile(nav,trades,cfg)

    def trim_then_restore(self,date,held,cal):
        if date==cal[1]:
            return .5
        if date==cal[2]:
            return 1.
        return min(p['management_fraction'] for p in held)

    def test_fork_hold_matches_every_original_ledger_field(self):
        market,rows,cal,cfg = self.book()
        expected = frozen(market,rows,cal,cfg,cfg.capital)
        actual = simulate(market,rows,cal,cfg,cfg.capital,management_policy=lambda d,h:1.,reversible_management=True)
        for a,b in zip(actual[:3],expected[:3]):
            pd.testing.assert_frame_equal(a,b)

    def test_trim_and_restore_execute_next_day_pay_costs_and_restore_original_counts(self):
        market,rows,cal,cfg = self.book()
        nav,trades,_,audit = simulate(market,rows,cal,cfg,cfg.capital,
            management_policy=lambda d,h:self.trim_then_restore(d,h,cal),reversible_management=True)
        entry = trades.loc[trades.reason.eq('entry')].set_index('optionid')
        trim = trades.loc[trades.reason.eq('learned_policy_trim')].set_index('optionid')
        restore = trades.loc[trades.reason.eq('learned_policy_restore')].set_index('optionid')
        self.assertEqual(len(restore),2)
        self.assertTrue(restore.date.eq(cal[3]).all())
        self.assertTrue(restore.exit_signal_date.eq(cal[2]).all())
        np.testing.assert_array_equal(entry.quantity+trim.quantity+restore.quantity,entry.quantity)
        np.testing.assert_allclose(restore.commission,restore.quantity.abs()*cfg.option_commission)
        self.assertGreater(nav.option_execution_cost.loc[cal[3]],restore.commission.sum())
        self.assertLess(audit['max_ledger_error'],1e-7)
        self.reconcile(nav,trades,cfg)

    def test_irreversible_mode_cannot_restore(self):
        market,rows,cal,cfg = self.book()
        nav,trades,_,_ = simulate(market,rows,cal,cfg,cfg.capital,
            management_policy=lambda d,h:self.trim_then_restore(d,h,cal),reversible_management=False)
        self.assertFalse(trades.reason.eq('learned_policy_restore').any())
        self.reconcile(nav,trades,cfg)

    def test_missing_quote_blocks_atomic_restoration(self):
        market,rows,cal,cfg = self.book()
        del market.q[cal[3],1]
        nav,trades,_,audit = simulate(market,rows,cal,cfg,cfg.capital,
            management_policy=lambda d,h:self.trim_then_restore(d,h,cal),reversible_management=True)
        self.assertFalse(trades.reason.eq('learned_policy_restore').any())
        self.assertTrue(any(r['status']=='no_executable_exit_quote' for r in audit['management_log']))
        self.reconcile(nav,trades,cfg)

    def test_restore_risk_breach_rejects_both_legs_without_partial_fill(self):
        market,rows,cal,cfg = self.book()
        old = frozen(market,rows,cal,cfg,cfg.capital)
        cfg = replace(cfg,max_gross_vega=old[0].gross_vega.loc[cal[1]]*1.2)
        for oid in [1,2]:
            market.q[cal[3],oid]['vega'] *= 2
        nav,trades,_,audit = simulate(market,rows,cal,cfg,cfg.capital,
            management_policy=lambda d,h:self.trim_then_restore(d,h,cal),reversible_management=True)
        self.assertFalse(trades.reason.eq('learned_policy_restore').any())
        self.assertTrue(any(r['status']=='restore_risk_limit' for r in audit['management_log']))
        self.reconcile(nav,trades,cfg)

    def test_vega_mismatch_rejects_atomic_restoration(self):
        market,rows,cal,cfg = self.book()
        market.q[cal[3],1]['vega'] *= 2
        nav,trades,_,audit = simulate(market,rows,cal,cfg,cfg.capital,
            management_policy=lambda d,h:self.trim_then_restore(d,h,cal),reversible_management=True)
        self.assertFalse(trades.reason.eq('learned_policy_restore').any())
        self.assertTrue(any(r['status']=='restore_vega_mismatch' for r in audit['management_log']))
        self.reconcile(nav,trades,cfg)

    def test_closed_cohort_cannot_be_reopened(self):
        market,rows,cal,cfg = self.book()
        nav,trades,_,_ = simulate(market,rows,cal,cfg,cfg.capital,
            management_policy=lambda d,h:0. if d==cal[1] else 1.,reversible_management=True)
        self.assertFalse(trades.reason.eq('learned_policy_restore').any())
        self.assertEqual(trades.reason.eq('entry').sum(),2)
        self.reconcile(nav,trades,cfg)

    def episode(self):
        n = 12
        return dict(cohort='a',dates=pd.bdate_range('2020-01-01',periods=n),
            states=np.zeros((n,13)),state_known=np.ones(n,bool),
            can_exit=np.ones(n,bool),restore_valid=np.ones(n,bool),split_seen=np.zeros(n,bool),
            hedge=np.zeros((n,2)),hprev=np.zeros((n,2)),spot=np.ones((n,2))*100,
            gross_training=np.ones(n),exit_cost_training=np.ones(n)*4,entry_vega=1.)

    def test_shadow_restoration_pays_positive_cost_and_keeps_prior_exposure_until_fill(self):
        reward,retained = block_reward(self.episode(),5,10,.5,1.)
        self.assertEqual(reward,3.5)
        self.assertEqual(retained,1.)

    def test_failed_shadow_restore_keeps_inventory(self):
        ep = self.episode()
        ep['restore_valid'][6] = False
        reward,retained = block_reward(ep,5,10,.5,1.)
        self.assertEqual(reward,2.5)
        self.assertEqual(retained,.5)

    def test_future_payoffs_cannot_change_completed_prefix(self):
        ep = self.episode()
        cutoff = ep['dates'][11]
        first = causal_transitions(transitions(ep,True),cutoff)
        ep['gross_training'][11] = 1000
        ep['restore_valid'][11] = False
        second = causal_transitions(transitions(ep,True),cutoff)
        self.assertEqual([r['reward'] for r in first],[r['reward'] for r in second])

    def test_inventory_exposures_scale_but_market_and_book_state_do_not(self):
        x = inputs(np.ones((1,13)),np.array([.5]),np.array([1.]))
        np.testing.assert_array_equal(x[0,7:12],np.full(5,.5))
        self.assertEqual(x[0,6],1.)
        self.assertEqual(x[0,12],1.)

    def test_idle_day_does_not_automatically_restore_a_trimmed_position(self):
        cal = pd.bdate_range('2020-01-01',periods=12)
        policy = callback({cal[0]:None},{},cal,None,True)
        held = [dict(management_fraction=.5,entry_date=cal[0])]*2
        self.assertEqual(policy(cal[2],held),.5)

    def test_restore_action_is_available_only_in_reversible_mode(self):
        class Tree:
            def predict(self,x):
                return x[:,-1]
        class Model:
            estimators_ = [Tree(),Tree()]
        state,fraction = np.zeros((1,13)),np.array([.5])
        self.assertEqual(choose_actions(Model(),state,fraction,True)[0],1.)
        self.assertEqual(choose_actions(Model(),state,fraction,False)[0],.5)

    def test_past_calibrated_risk_utility_can_prefer_half_for_lower_edge(self):
        coefficient = risk_coefficient(np.array([2.,-1.]),np.ones(2))
        self.assertAlmostEqual(coefficient,.2)
        rewards = np.array([1.,-.8])
        utility = lambda size:np.mean(size*rewards-.5*coefficient*(size*rewards)**2)
        self.assertGreater(utility(.5),utility(1.))
        self.assertGreater(utility(.5),utility(0.))

    def test_negative_reference_mean_cannot_reward_variance(self):
        self.assertEqual(risk_coefficient(np.array([-2.,1.]),np.ones(2)),0.)

    def test_action_augmentation_cannot_advance_training_activation(self):
        for reversible,rows_per_step in [(False,5),(True,6)]:
            self.assertFalse(enough_history(dict(cohorts=25,transitions=120*rows_per_step),reversible))
            self.assertTrue(enough_history(dict(cohorts=16,transitions=128*rows_per_step),reversible))
            self.assertFalse(enough_history(dict(cohorts=15,transitions=128*rows_per_step),reversible))


if __name__=='__main__':
    unittest.main()
