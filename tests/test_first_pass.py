import unittest
import numpy as np
import pandas as pd
from scripts.run_first_pass import forward_rv, fit_policy, policy_scores, payoff_sharpe, net_rewards, split_data


class FirstPassTests(unittest.TestCase):
    def test_sharpe_annualizes_the_30_day_horizon(self):
        rewards = np.array([0., 1., 2., 3.])
        expected = rewards.mean() / rewards.std(ddof=1) * np.sqrt(365 / 30)
        self.assertAlmostEqual(payoff_sharpe(rewards), expected)
        self.assertEqual(payoff_sharpe(np.zeros(4)), 0.)

    def test_forward_window_excludes_entry_and_later_returns(self):
        dates = pd.bdate_range('2020-01-01', '2020-03-01')
        r = pd.DataFrame(.01, index=dates, columns=['AMD', 'NVDA'])
        entry = pd.Timestamp('2020-01-06')
        r.loc[entry] = .9
        r.loc[r.index > entry + pd.Timedelta(days=30)] = .8
        rv = forward_rv(r, [entry])
        self.assertAlmostEqual(rv.loc[entry, 'rv_amd'], np.sqrt(252)*np.log1p(.01)*100)
        self.assertTrue(forward_rv(r, [pd.Timestamp('2020-02-20')]).empty)

    def test_split_purges_unsettled_rewards(self):
        dates = pd.to_datetime(['2016-12-01','2016-12-15','2017-01-03','2020-12-15','2021-01-04'])
        d = pd.DataFrame({'end':dates+pd.Timedelta(days=30)}, index=dates)
        train,val,test = split_data(d)
        self.assertEqual(len(train), 1)
        self.assertEqual(len(val), 1)
        self.assertEqual(len(test), 1)

    def test_cost_can_make_flat_optimal(self):
        state = np.tile([0,1,2],100)
        reward = np.where(state == 0, .2, np.where(state == 1, 2., -3.))
        policy, score = fit_policy(state,reward,3,seed=1,cost=.5,iterations=30)
        np.testing.assert_array_equal(policy,[0,1,-1])
        self.assertGreater(score,0)
        np.testing.assert_array_equal(net_rewards(np.array([0,1,-1]), np.ones(3),.5),[0,.5,-1.5])

    def test_aggregate_sharpe_matches_direct_and_exhaustive_optimum(self):
        from itertools import product
        rng = np.random.default_rng(7)
        state = np.tile([0,1,2],100)
        reward = rng.normal(size=len(state)) + np.choose(state,[.1,2.,-3.])
        policies = np.array(list(product([-1,0,1],repeat=3)))
        scores = policy_scores(policies,state,reward,3,cost=.5)
        direct = [payoff_sharpe(net_rewards(p[state],reward,.5)) for p in policies]
        np.testing.assert_allclose(scores,direct,atol=1e-12)
        _,score = fit_policy(state,reward,3,seed=1,cost=.5,iterations=30)
        self.assertAlmostEqual(score,max(direct))


if __name__ == '__main__':
    unittest.main()
