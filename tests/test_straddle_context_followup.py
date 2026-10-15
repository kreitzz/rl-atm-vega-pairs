import unittest

import numpy as np
import pandas as pd

from scripts.straddle_context_followup import (CFG, experience_features,
    dynamics_features, window_moments, predict, fit_model, configure, FEATURES)


class StraddleContextTests(unittest.TestCase):
    def labels(self):
        return pd.DataFrame(dict(end=pd.to_datetime(['2020-01-01','2020-01-02','2020-01-03']),
                                 full_profit=[450.,-900.,450000.]))

    def test_completion_date_is_strictly_before_decision(self):
        result=window_moments(pd.to_datetime(['2020-01-01','2020-01-02','2020-01-03']),self.labels(),365)
        np.testing.assert_array_equal(result[:,0],[0,1,2])
        np.testing.assert_array_equal(result[:,1],[0,1,-1])

    def test_future_outcome_does_not_change_memory(self):
        dates=pd.to_datetime(['2020-01-02','2020-01-03'])
        labels=self.labels()
        a=experience_features(dates,labels,labels)
        labels.loc[2,'full_profit']=-1e12
        b=experience_features(dates,labels,labels)
        pd.testing.assert_frame_equal(a,b)

    def test_missing_name_history_shrinks_to_pool(self):
        labels=self.labels();dates=pd.to_datetime(['2020-01-03'])
        f=experience_features(dates,labels.iloc[:0],labels)
        self.assertAlmostEqual(f.name_profit365_shrunk.iloc[0],f.pool_profit365_score.iloc[0])
        self.assertEqual(f.name_history_count.iloc[0],0)
        self.assertTrue(np.isfinite(f).all().all())

    def test_dynamics_are_prefix_invariant(self):
        dates=pd.bdate_range('2019-01-01',periods=200)
        r=pd.Series(np.sin(np.arange(200))*.01,index=dates)
        iv=pd.Series(.3+np.cos(np.arange(200)/7)*.03,index=dates)
        vix=iv*60
        full=dynamics_features(r,iv,vix)
        prefix=dynamics_features(r.iloc[:150],iv.iloc[:150],vix.iloc[:150])
        pd.testing.assert_frame_equal(full.iloc[:150],prefix)

    def test_prediction_clips_outliers_and_stays_in_ladder(self):
        model=dict(choices=[2,3,4],features=['x'],center=[0],scale=[1],clip=5,
                   beta=[0,1],cuts=[-1,1])
        np.testing.assert_array_equal(predict(model,pd.DataFrame(dict(x=[-1e9,0,1e9]))),[2,3,4])
        self.assertEqual(len(CFG['specifications']),10)
        self.assertEqual(len(set(s['name'] for s in CFG['specifications'])),10)

    def test_fit_cannot_use_outcomes_finishing_at_or_after_cutoff(self):
        configure()
        generator=np.random.default_rng(17)
        calendar=pd.bdate_range('2018-01-01','2020-12-31')
        count=160; dates=calendar[:count]
        meta=pd.DataFrame(generator.normal(size=(count,len(FEATURES['memory']))),columns=FEATURES['memory'])
        meta['date']=dates;meta['end']=dates+pd.Timedelta(days=30)
        meta['entry_position']=np.arange(count)+1;meta['finite_label']=True
        meta.loc[count-2:,'end']=pd.to_datetime(['2020-01-01','2020-02-01'])
        paths=generator.normal(0.2,1,size=(count,3,10))
        spec=CFG['specifications'][0]
        before=fit_model(meta,paths,'2020-01-01',spec,calendar)
        paths[count-2:]*=1e12
        after=fit_model(meta,paths,'2020-01-01',spec,calendar)
        self.assertEqual(before,after)
        self.assertEqual(before['training_samples'],count-2)


if __name__=='__main__':unittest.main()
