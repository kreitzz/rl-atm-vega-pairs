import unittest
import numpy as np
from scripts.run_four_pairs import action_menu, action_moments, moment_scores, portfolio_reward, joint_search, entry_cost
from scripts.run_first_pass import payoff_sharpe, policy_scores
from scripts.run_walk_forward import fold_data
import pandas as pd


class FourPairTests(unittest.TestCase):
    def test_entry_iv_cost_examples(self):
        panel=pd.DataFrame({'iv_amd':[40.,20.,40.],'iv_nvda':[40.,20.,20.]})
        np.testing.assert_allclose(entry_cost(panel,.025),[1.,.5,.75])

    def test_variable_cost_moments(self):
        rng=np.random.default_rng(87)
        y=rng.normal(size=(150,4))*3
        costs=rng.uniform(.2,2,size=y.shape)
        state=np.arange(len(y))%3; menu=action_menu()
        policies=rng.integers(len(menu),size=(20,3))
        sums,squares=action_moments(state,y,menu,3,costs)
        expected=[payoff_sharpe(portfolio_reward(menu[p[state]],y,costs)) for p in policies]
        np.testing.assert_allclose(moment_scores(policies,sums,squares,len(y)),expected,atol=1e-12)
        allocations=menu[policies,0]
        expected=[payoff_sharpe(a[state]*y[:,0]-np.abs(a[state])*costs[:,0]) for a in allocations]
        np.testing.assert_allclose(policy_scores(allocations,state,y[:,0],3,costs[:,0]),expected,atol=1e-12)
        np.testing.assert_allclose(policy_scores(allocations,state,y[:,0],3,1.),policy_scores(allocations,state,y[:,0],3,np.ones(len(y))),atol=1e-12)

    def test_budget_and_cost(self):
        menu=action_menu()
        self.assertTrue((np.abs(menu).sum(axis=1)<=1).all())
        self.assertTrue(((menu==0).all(axis=1)).any())
        self.assertTrue(((menu==[.25,-.25,.25,-.25]).all(axis=1)).any())
        a=np.array([[.5,-.25,0,0],[0,0,0,0]])
        r=np.array([[3,2,7,1],[9,9,9,9]])
        np.testing.assert_allclose(portfolio_reward(a,r),[.25,0])

    def test_joint_moments_include_cross_pair_covariance(self):
        rng=np.random.default_rng(5)
        y=rng.normal(size=(120,4)); y[:,1]=2*y[:,0]+.1*y[:,1]
        state=np.arange(120)%3; menu=action_menu()
        policies=rng.integers(len(menu),size=(15,3))
        sums,squares=action_moments(state,y,menu,3)
        actual=moment_scores(policies,sums,squares,len(y))
        expected=[payoff_sharpe(portfolio_reward(menu[p[state]],y)) for p in policies]
        np.testing.assert_allclose(actual,expected,atol=1e-12)

    def test_search_can_choose_flat_when_cost_dominates(self):
        y=np.zeros((100,4)); state=np.arange(100)%2
        a,score=joint_search(state,y,action_menu(),2,4,iterations=5,population=32)
        np.testing.assert_array_equal(a,np.zeros((2,4)))
        self.assertEqual(score,0)

    def test_fold_purges_future_outcomes(self):
        dates=pd.date_range('2005-01-01','2011-12-31',freq='B')
        d=pd.DataFrame({'end':dates+pd.Timedelta(days=30)},index=dates)
        tr,va,te=fold_data(d,2010)
        self.assertLess(tr.end.max(),va.index.min())
        self.assertLess(va.end.max(),te.index.min())


if __name__=='__main__': unittest.main()
