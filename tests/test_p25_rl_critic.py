"""Action effects must survive even when states have identical inputs."""
import os
for _key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS','ARROW_NUM_THREADS']:
    os.environ[_key]='1'

import unittest
import numpy as np
from scripts.p25_rl.critic import ActionForest


class CriticTests(unittest.TestCase):
    def test_independent_action_values_and_aligned_trees(self):
        actions=np.repeat(np.arange(3),60)
        states=np.zeros((len(actions),2))
        x=np.column_stack([states,np.eye(3)[actions]])
        y=np.array([2.,-5.,7.])[actions]
        critic=ActionForest(3).fit(x,y,sample_weight=np.ones(len(y)))
        probe=np.column_stack([np.zeros((3,2)),np.eye(3)])
        np.testing.assert_allclose(critic.predict(probe),[2.,-5.,7.])
        np.testing.assert_allclose(np.mean([tree.predict(probe) for tree in critic.estimators_],axis=0),critic.predict(probe))
        self.assertEqual(critic.action_counts_,{0:60,1:60,2:60})
        self.assertEqual(len(critic.feature_importances_),5)
        np.testing.assert_array_equal(critic.feature_importances_[-3:],np.zeros(3))

    def test_absent_action_is_supported_only_when_observed(self):
        actions=np.repeat([0,2],60)
        x=np.column_stack([np.zeros((120,2)),np.eye(4)[actions]])
        critic=ActionForest(4).fit(x,np.where(actions==0,1.,3.))
        self.assertEqual(critic.present_actions_,[0,2])
        np.testing.assert_allclose(critic.predict(x),np.where(actions==0,1.,3.))
        unseen=np.column_stack([np.zeros((1,2)),np.eye(4)[[3]]])
        with self.assertRaises(ValueError):critic.predict(unseen)
        with self.assertRaises(ValueError):critic.estimators_[0].predict(unseen)

    def test_rejects_invalid_action_encoding(self):
        critic=ActionForest(3)
        x=np.array([[0.,0.,.5,.5,0.]])
        with self.assertRaises(ValueError):critic.fit(x,np.array([1.]))


if __name__=='__main__':unittest.main()
