import unittest

import numpy as np
import pandas as pd

from scripts.run_year_option_action_critics import bellman_target,choose_actions,features,fit_q,group_transitions
from scripts import run_long_tenor_reversible as previous


class ActionCriticTest(unittest.TestCase):
    def records(self,values):
        records = []
        for index,(state_value,payoffs) in enumerate(values):
            state = np.zeros(13)
            state[0] = state_value
            for action,reward in zip(previous.ACTIONS,payoffs):
                records.append(dict(cohort=f'c{index}',start=pd.Timestamp('2020-01-01'),
                    end=pd.Timestamp('2020-01-08'),state=state,fraction=1.,action=float(action),
                    reward=reward,next_state=state.copy(),next_fraction=action,terminal=True,discount=.98))
        return records

    def test_grouping_matches_actions_by_identity_not_input_order(self):
        records = self.records([(0.,[-1.,1.,3.])])
        batches = group_transitions(records[::-1])
        self.assertEqual([r['reward'] for r in batches[0]],[-1.,1.,3.])

    def test_incomplete_or_duplicated_counterfactuals_are_rejected(self):
        records = self.records([(0.,[-1.,1.,3.])])
        with self.assertRaises(AssertionError):
            group_transitions(records[:-1])
        with self.assertRaises(AssertionError):
            group_transitions(records+[records[0]])

    def test_explicit_outputs_distinguish_actions_without_any_state_split(self):
        records = self.records([(0.,[-1.,1.,3.])]*64)
        model,_ = fit_q(records,dict(iterations=2,discount=.98,risk=False))
        np.testing.assert_allclose(model.predict(features(np.zeros((1,13)),np.ones(1))),[[-1.,1.,3.]])
        self.assertEqual(choose_actions(model,np.zeros((1,13)),np.ones(1))[0],1.)

    def test_risk_reward_produces_half_size_in_lower_edge_state(self):
        values = [(0.,[0.,r/2,r]) for r in [2.,-1.]*64]
        values += [(1.,[0.,r/2,r]) for r in [1.,-.8]*64]
        model,coefficient = fit_q(self.records(values),dict(iterations=2,discount=.995,risk=True))
        self.assertGreater(coefficient,0.)
        states = np.zeros((2,13))
        states[1,0] = 1.
        np.testing.assert_array_equal(choose_actions(model,states,np.ones(2)),[1.,.5])

    def test_closed_inventory_never_restores_and_equal_values_hold(self):
        class Tree:
            def predict(self,x):
                return np.zeros((len(x),3))
        class Model:
            estimators_ = [Tree(),Tree()]
        np.testing.assert_array_equal(choose_actions(Model(),np.zeros((3,13)),np.array([0.,.5,1.])),[0.,.5,1.])

    def test_terminal_actions_have_no_continuation_value(self):
        target = bellman_target(np.array([[1.,2.,3.]]),np.full((1,3),100.),
                                np.array([[True,False,False]]),np.array([[.98]]))
        np.testing.assert_allclose(target,[[1.,100.,101.]])

    def test_greedy_action_is_not_suppressed_by_ensemble_penalty(self):
        class Tree:
            def __init__(self,half):
                self.half = half
            def predict(self,x):
                return np.tile([0.,self.half,0.],(len(x),1))
        class Model:
            estimators_ = [Tree(3.),Tree(-1.)]
        states,fraction = np.zeros((1,13)),np.ones(1)
        self.assertEqual(choose_actions(Model(),states,fraction,penalty=1.)[0],1.)
        self.assertEqual(choose_actions(Model(),states,fraction,penalty=0.)[0],.5)

    def test_full_year_reward_cannot_reach_first_week_in_sixteen_updates(self):
        utility = np.zeros(52)
        utility[-1] = 1.
        terminal = np.arange(52)==51
        value = np.zeros(52)
        for step in range(64):
            value = bellman_target(utility,np.r_[value[1:],0.],terminal,.995)
            if step==15:
                self.assertEqual(value[0],0.)
        self.assertAlmostEqual(value[0],.995**51)

    def test_future_actions_and_rewards_do_not_enter_completed_history(self):
        records = self.records([(0.,[-1.,1.,3.])]*64)
        future = [dict(r,start=pd.Timestamp('2020-02-01'),end=pd.Timestamp('2020-02-08'),reward=1e6) for r in records]
        first = previous.old.causal_transitions(records+future,pd.Timestamp('2020-01-15'))
        second = previous.old.causal_transitions(records,pd.Timestamp('2020-01-15'))
        self.assertEqual(len(first),len(second))
        np.testing.assert_array_equal([r['reward'] for r in first],[r['reward'] for r in second])


if __name__=='__main__':
    unittest.main()
