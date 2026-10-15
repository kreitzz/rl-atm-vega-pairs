import unittest
import numpy as np
import pandas as pd
from scripts.straddle_policy_refinement import moments,action_map,fit_model,FEATURES


class RefinementTests(unittest.TestCase):
    def test_uniform_weights_reproduce_standard_daily_statistics(self):
        x=np.random.default_rng(5).normal(size=(100,9))
        a,b=moments(x);c,d=moments(x,np.ones(100))
        np.testing.assert_allclose(a,c,atol=1e-14);np.testing.assert_allclose(b,d,atol=1e-14)

    def test_ordered_action_constraint(self):
        x=np.random.default_rng(7).normal(size=(150,9))
        x[:,0]+=.2;x[:,3]+=.1;x[:,8]+=.4
        choices,_,_=action_map(x,np.array([50,50,50]),dict(ordered=True))
        self.assertTrue((np.diff(choices)>=0).all());self.assertTrue(np.isin(choices,[2,3,4]).all())

    def test_tied_scaled_paths_prefer_half(self):
        r=np.random.default_rng(8).normal(.1,1,size=200)
        choices,_,_=action_map(np.column_stack([r,1.5*r,2*r]),np.array([50]),{})
        np.testing.assert_array_equal(choices,[2])

    def test_future_labels_cannot_change_recency_fit(self):
        rng=np.random.default_rng(4);count=160
        cal=pd.bdate_range('2018-01-01','2020-12-31')
        meta=pd.DataFrame(rng.normal(size=(count,len(FEATURES['performance']))),columns=FEATURES['performance'])
        meta['date']=cal[:count];meta['end']=cal[:count]+pd.Timedelta(days=30)
        meta['entry_position']=np.arange(count)+1;meta['finite_label']=True
        meta.loc[count-2:,'end']=pd.to_datetime(['2020-01-01','2020-02-01'])
        paths=rng.normal(.1,1,size=(count,3,10))
        spec=dict(features='performance',months=48,half_life_months=12,ordered=True)
        before=fit_model(meta,paths,'2020-01-01',spec,cal);paths[-2:]*=1e12
        self.assertEqual(before,fit_model(meta,paths,'2020-01-01',spec,cal))
        self.assertEqual(before['training_samples'],158)


if __name__=='__main__':unittest.main()
