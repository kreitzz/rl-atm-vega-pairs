import unittest
import numpy as np
import pandas as pd
from scripts.forecast_contract_variance import variance_target,event_features,forecast

class VarianceForecastTests(unittest.TestCase):
    def test_holding_target_excludes_entry_and_requires_complete_path(self):
        self.assertAlmostEqual(variance_target([999.,999.,.01,.04,999.],1,3),6.3)
        self.assertTrue(np.isnan(variance_target([1.,2.,np.nan],1,2)))
        self.assertTrue(np.isnan(variance_target([1.,2.],1,1)))

    def test_uncapped_events_respect_contract_horizon(self):
        dates=pd.to_datetime(['2020-01-02','2020-02-10'])
        count,days=event_features(dates,pd.Timestamp('2020-01-03'),pd.Timestamp('2020-01-06'),pd.Timestamp('2020-02-14'))
        self.assertEqual((count,days),(1,38))
        self.assertEqual(event_features(dates,pd.Timestamp('2020-01-03'),pd.Timestamp('2020-01-06'),pd.Timestamp('2020-01-31'))[0],0)

    def test_heldout_labels_do_not_affect_forecast(self):
        x=np.linspace(-3,1,100)
        tr=pd.DataFrame({'x':x,'target':np.exp(x),'secid':[1]*50+[2]*50})
        te=tr.iloc[:4].copy()
        expected=forecast(tr,te,['x'])
        te['target']=1e9
        np.testing.assert_array_equal(expected,forecast(tr,te,['x']))
        self.assertTrue((expected>0).all())

if __name__=='__main__':unittest.main()
