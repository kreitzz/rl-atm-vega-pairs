import unittest
from itertools import product
import numpy as np
from scripts.run_first_pass import net_rewards,policy_scores,payoff_sharpe,fit_policy


class SizingTests(unittest.TestCase):
    def test_cost_scales_with_vega_and_fixed_actions_unchanged(self):
        actions=np.array([-1.,-.5,0.,.25,1.])
        actual=net_rewards(actions,np.full(5,2.),.5)
        np.testing.assert_allclose(actual,[-2.5,-1.25,0,.375,1.5])
        fixed=np.array([-1,0,1])
        np.testing.assert_array_equal(net_rewards(fixed,2.,.5),fixed*2.-.5*(fixed!=0))

    def test_fractional_aggregate_sharpe_matches_explicit_rewards(self):
        rng=np.random.default_rng(42)
        state=np.tile([0,1],300)
        reward=rng.normal(np.where(state==0,2.,-3.),np.where(state==0,1.,8.))
        levels=np.arange(-1.,1.01,.25)
        policies=np.array(list(product(levels,repeat=2)))
        scores=policy_scores(policies,state,reward,2,cost=.5)
        direct=np.array([payoff_sharpe(net_rewards(p[state],reward,.5)) for p in policies])
        np.testing.assert_allclose(scores,direct,atol=1e-12)
        initial=np.array([1.,-1.])
        policy,score=fit_policy(state,reward,2,seed=0,cost=.5,action_values=levels,initial_policy=initial)
        self.assertAlmostEqual(score,direct.max())
        self.assertGreaterEqual(score,payoff_sharpe(net_rewards(initial[state],reward,.5)))
        self.assertTrue(np.any((np.abs(policy)>0)&(np.abs(policy)<1)))

    def test_uniform_leverage_does_not_change_sharpe(self):
        rewards=np.array([1.,-2.,3.,0.,5.])
        self.assertAlmostEqual(payoff_sharpe(rewards),payoff_sharpe(rewards*.001))


if __name__=='__main__':unittest.main()
