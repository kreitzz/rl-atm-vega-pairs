import unittest
import numpy as np
from scripts.test_direction_controls import leg_payoffs,random_directions,strata_indices


class DirectionControlTests(unittest.TestCase):
    def test_leg_cost_split_reconciles_pair(self):
        direction=np.array([1.,-1.,0.]);size=np.array([.5,1.,0.])
        a=np.array([2.,3.,4.]);n=np.array([1.,-1.,2.])
        rewards=leg_payoffs(direction,size,a,n)
        np.testing.assert_allclose(rewards[:,0]+rewards[:,1],rewards[:,2])
        np.testing.assert_allclose(rewards[:,2],direction*size*(a-n)-size)
        self.assertEqual(rewards[2].sum(),0.)

    def test_stratified_controls_preserve_signed_vega_mix_and_flats(self):
        signs=np.array([1.,-1.,1.,-1.,1.,0.])
        size=np.array([.5,.5,1.,1.,1.,0.])
        strata=strata_indices(['a']*6,size)
        for method in ['stratified_shuffle','stratified_circular_shift']:
            r=random_directions(signs,strata,method,np.random.default_rng(3))
            for ix in strata:np.testing.assert_array_equal(np.sort(r[ix]),np.sort(signs[ix]))
            self.assertEqual(r[-1],0.)
            self.assertAlmostEqual(np.sum(r*size),np.sum(signs*size))

    def test_randomization_is_reproducible(self):
        signs=np.array([1.,-1.,1.,-1.]);strata=[np.arange(4)]
        for method in ['coin_flip','stratified_shuffle','stratified_circular_shift']:
            x=random_directions(signs,strata,method,np.random.default_rng(8))
            y=random_directions(signs,strata,method,np.random.default_rng(8))
            np.testing.assert_array_equal(x,y)
            self.assertTrue(np.isin(x,[-1.,1.]).all())


if __name__=='__main__':unittest.main()
