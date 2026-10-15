import unittest
import numpy as np
from scripts import signal_allocation_objectives as q


class ObjectiveRewardsTests(unittest.TestCase):
    def setUp(self):
        self.reference=np.array([.003,-.002,.004,.001,-.001,.002,.001,.003])
        self.old=self.reference[-2:]

    def parts(self,candidate):
        r=self.reference
        return q.reward_parts(np.asarray(candidate)[None,None,:],np.zeros(2),self.old,float(r.sum()),float(r@r),len(r))[0,0]

    def test_replacement_equals_direct_realized_sharpe_change(self):
        candidate=np.array([-.001,.005]);replaced=np.r_[self.reference[:-2],candidate]
        sh=lambda r:np.sqrt(252)*r.mean()/r.std(ddof=1)
        expected=(sh(replaced)-sh(self.reference))*len(self.reference)/252
        self.assertAlmostEqual(self.parts(candidate)[2],expected,places=12)

    def test_local_reward_matches_infinitesimal_replacement(self):
        delta=np.array([.001,-.002]);epsilon=1e-6
        parts=self.parts(self.old+epsilon*delta)
        self.assertAlmostEqual(parts[1]/epsilon,parts[2]/epsilon,places=5)

    def test_profit_reward_is_dollars_above_cash(self):
        parts=self.parts(np.array([.01,0.]))
        self.assertAlmostEqual(parts[0],100000/250000,places=12)

    def test_reference_replacement_has_zero_sharpe_reward(self):
        self.assertAlmostEqual(self.parts(self.old)[1],0.)
        self.assertAlmostEqual(self.parts(self.old)[2],0.)

    def test_two_formulations_share_one_profit_endpoint(self):
        rows=q.objectives();self.assertEqual(len(rows),9)
        self.assertEqual(len({r['key'] for r in rows}),9)
        self.assertEqual(sum(r['weight']==1 for r in rows),1)


if __name__=='__main__':unittest.main()
