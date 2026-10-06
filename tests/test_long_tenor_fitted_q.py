import unittest

import numpy as np
import pandas as pd

from scripts.run_long_tenor_fitted_q import block_reward, callback, causal_transitions, choose_actions, episode_transitions


class FittedQTest(unittest.TestCase):
    def episode(self):
        n = 12
        return dict(cohort='a', dates=pd.bdate_range('2020-01-01',periods=n),
            state_known=np.ones(n,bool),states=np.zeros((n,6)),
            can_exit=np.ones(n,bool),split_seen=np.zeros(n,bool),
            gross_training=np.ones(n),entry_vega=1.,
            hedge=np.zeros((n,2)),hprev=np.zeros((n,2)),spot=np.ones((n,2))*100,
            exit_cost_training=np.ones(n)*4)

    def test_close_keeps_next_close_pnl_and_pays_exit_once(self):
        reward, retained = block_reward(self.episode(),5,10,1.,0.)
        self.assertEqual(reward,-1.)
        self.assertEqual(retained,0.)

    def test_trim_keeps_old_exposure_until_fill_then_half(self):
        reward, retained = block_reward(self.episode(),5,10,1.,.5)
        self.assertEqual(reward,2.)
        self.assertEqual(retained,.5)

    def test_no_quote_does_not_create_an_exit(self):
        ep = self.episode()
        ep['can_exit'][6] = False
        reward, retained = block_reward(ep,5,10,1.,0.)
        self.assertEqual(reward,5.)
        self.assertEqual(retained,1.)

    def test_weekly_prefix_can_train_before_expiry_without_using_future_step(self):
        records = episode_transitions(self.episode())
        cutoff = self.episode()['dates'][11]
        past = causal_transitions(records,cutoff)
        self.assertTrue(past)
        self.assertTrue(all(r['end']<cutoff for r in past))
        self.assertTrue(all(not r['terminal'] or r['next_fraction']==0 for r in past))

    def test_action_mask_never_reexpands_trimmed_inventory(self):
        class Model:
            def predict(self,x):
                return x[:,-1]
        selected = choose_actions(Model(),np.zeros((2,6)),np.array([.5,1.]))
        np.testing.assert_array_equal(selected,[.5,1.])

    def test_equal_values_choose_hold(self):
        class Model:
            def predict(self,x):
                return np.zeros(len(x))
        selected = choose_actions(Model(),np.zeros((2,6)),np.array([.5,1.]))
        np.testing.assert_array_equal(selected,[.5,1.])

    def test_future_reward_does_not_change_completed_prefix(self):
        ep = self.episode()
        cutoff = ep['dates'][11]
        first = causal_transitions(episode_transitions(ep),cutoff)
        ep['gross_training'][11] = 10000
        second = causal_transitions(episode_transitions(ep),cutoff)
        self.assertEqual([r['reward'] for r in first],[r['reward'] for r in second])

    def test_disagreement_penalty_uses_paired_advantages_and_preserves_hold(self):
        class Tree:
            def __init__(self, close):
                self.close = close
            def predict(self,x):
                return np.where(x[:,-1]==0,self.close,0.)
        class Model:
            estimators_ = [Tree(3.),Tree(-1.)]
            def predict(self,x):
                return np.mean([tree.predict(x) for tree in self.estimators_],axis=0)
        state, fraction = np.zeros((1,6)),np.ones(1)
        self.assertEqual(choose_actions(Model(),state,fraction)[0],0.)
        self.assertEqual(choose_actions(Model(),state,fraction,conservative=True)[0],1.)

    def test_conservative_action_mask_preserves_half_inventory(self):
        class Tree:
            def predict(self,x):
                return x[:,-1]
        class Model:
            estimators_ = [Tree(),Tree()]
            def predict(self,x):
                return x[:,-1]
        self.assertEqual(choose_actions(Model(),np.zeros((1,6)),np.array([.5]),conservative=True)[0],.5)

    def test_unpaired_natural_runoff_is_not_a_forced_management_exit(self):
        policy = callback({}, {}, pd.bdate_range('2020-01-01',periods=2))
        self.assertEqual(policy(pd.Timestamp('2020-01-01'),[{'quantity':1}]),1.)


if __name__=='__main__':
    unittest.main()
