import unittest
import numpy as np
from scripts import signal_capital_allocation as q


class SignalAllocationTests(unittest.TestCase):
    def test_targets_sum_to_one_and_include_both_endpoints(self):
        np.testing.assert_allclose(q.TARGETS.sum(axis=1),1)
        np.testing.assert_array_equal(q.TARGETS[0],[1,0,0])
        np.testing.assert_array_equal(q.TARGETS[-1],[0,1,0])

    def test_pure_source_endpoints_are_unchanged_even_with_large_losses(self):
        gross=np.array([[1.01,.8,1.],[1.02,.9,1.],[.99,1.6,1.]])
        for action in [0,q.N-1]:
            path,after,charge=q.engine.month_path(gross,np.zeros(3),np.array([.05,.05]),q.TARGETS[action],q.TARGETS[action])
            np.testing.assert_allclose(path,gross[:,action//(q.N-1)]-1,atol=1e-14)
            self.assertAlmostEqual(charge,0.)

    def test_change_executes_after_first_return(self):
        gross=np.array([[1.01,1.2,1.],[1.02,1.3,1.]])
        path,_,_=q.engine.month_path(gross,np.zeros(2),np.zeros(2),q.TARGETS[0],q.TARGETS[-1])
        np.testing.assert_allclose(path,[.01,.3],atol=1e-14)


if __name__=='__main__':unittest.main()
