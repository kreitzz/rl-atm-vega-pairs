import unittest
import numpy as np
import pandas as pd
from scripts.discover_contract_signals import training_mask, predict, summarize

class SignalDiscoveryTests(unittest.TestCase):
    def test_purges_unfinished_and_boundary_outcomes(self):
        d=pd.DataFrame({'date':pd.to_datetime(['2019-01-01','2019-12-20','2019-12-01','2020-01-02']),
                        'end':pd.to_datetime(['2019-02-01','2020-01-15','2020-01-01','2020-02-01'])})
        self.assertEqual(training_mask(d,pd.Timestamp('2020-01-01')).tolist(),[True,False,False,False])

    def test_predictions_do_not_depend_on_evaluation_labels_or_other_test_rows(self):
        rng=np.random.default_rng(4)
        tr=pd.DataFrame({'x':rng.normal(size=100),'pair':['a']*50+['b']*50})
        tr['long_net']=2*tr.x-.2; tr['short_net']=-2*tr.x-.2
        te=tr.iloc[:5].copy()
        p=predict(tr,te,['x'])
        te['long_net']=1e20; te['short_net']=-1e20
        np.testing.assert_array_equal(p,predict(tr,te,['x']))
        te.loc[te.index[-1],'x']=1e20
        np.testing.assert_allclose(p[:-1],predict(tr,te,['x'])[:-1],rtol=0,atol=1e-14)

    def test_flat_has_zero_payoff(self):
        g=pd.DataFrame({'action':[0,1,-1],'long_net':[99,2,99],'short_net':[99,99,-1],
            'pred_long':[0,1,-1],'pred_short':[0,-1,1],'directional_edge':[0,2,1],'pair':['a']*3})
        s=summarize(g)
        self.assertEqual(s['trades'],2)
        self.assertAlmostEqual(s['mean_net_per_opportunity'],1/3)
        self.assertAlmostEqual(s['mean_net_per_trade'],.5)

if __name__=='__main__': unittest.main()
