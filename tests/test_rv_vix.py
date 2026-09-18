import unittest
import numpy as np
import pandas as pd
from scripts.run_rv_vix_pass import trailing_rv, join_vix, fit_tree_policy, leaf_states


class RVVixTests(unittest.TestCase):
    def test_trailing_rv_excludes_future_and_old_returns(self):
        dates=pd.bdate_range('2020-01-01','2020-04-01')
        r=pd.DataFrame(.01,index=dates,columns=['AMD','NVDA'])
        entry=pd.Timestamp('2020-02-10')
        r.loc[r.index<=entry-pd.Timedelta(days=30)] = .8
        r.loc[r.index>entry]=.9
        actual=trailing_rv(r).loc[entry,'trailing_rv_amd']
        self.assertAlmostEqual(actual,100*np.sqrt(252)*np.log1p(.01))
        r.loc['2020-02-05','AMD']=np.nan
        self.assertTrue(np.isnan(trailing_rv(r).loc[entry,'trailing_rv_amd']))

    def test_vix_uses_strictly_prior_close(self):
        panel=pd.DataFrame(index=pd.to_datetime(['2020-01-06','2020-02-01']))
        vix=pd.DataFrame({'DATE':pd.to_datetime(['2020-01-03','2020-01-06']),'CLOSE':[12.,99.]})
        got=join_vix(panel,vix)
        self.assertEqual(got.loc['2020-01-06','vix_previous_close'],12.)
        self.assertTrue(np.isnan(got.loc['2020-02-01','vix_previous_close']))

    def test_shared_tree_leaves_obey_minimum_and_map_states(self):
        x=np.linspace(-1,1,300)
        train=pd.DataFrame({'spread':x,'long_reward':3*x+np.sin(x*10)})
        tree,ids,policy,score,_=fit_tree_policy(train,['spread'],8,50,.5)
        state=leaf_states(tree,train,['spread'],ids)
        self.assertGreaterEqual(np.bincount(state).min(),50)
        self.assertEqual(len(policy),len(ids))
        self.assertGreaterEqual(score,0.)


if __name__=='__main__':
    unittest.main()
