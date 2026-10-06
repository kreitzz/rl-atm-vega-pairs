import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scripts.p25_risk_sizing import risk_features, inventory_context, inventory_level


class RiskSizingTest(unittest.TestCase):
    def test_rv_is_oriented_to_long_and_short_for_both_directions(self):
        rows=pd.DataFrame(dict(direction=[1,-1],correlation=[.5,.5],beta=[1.,1.],
            correlation_available=[1,1],beta_available=[1,1],iv_mean=[.3,.3]))
        x=risk_features(rows,[.2,.2],[.4,.4],[.3,.3],[.2,.2])
        np.testing.assert_allclose(x.long_rv_iv,[1.5,.5])
        np.testing.assert_allclose(x.short_rv_iv,[.5,1.5])
        np.testing.assert_allclose(x.log_vol_mismatch,np.log(2))

    def test_missing_optional_data_does_not_remove_a_signal(self):
        rows=pd.DataFrame(dict(direction=[1],correlation=[0.],beta=[0.],
            correlation_available=[0],beta_available=[0],iv_mean=[.3]))
        x=risk_features(rows,[.2],[.4],[np.nan],[.2])
        self.assertEqual(len(x),1)
        self.assertEqual(x.long_rv_iv_available.iloc[0],0)
        self.assertEqual(x.short_rv_iv.iloc[0],.5)
        self.assertEqual(x.log_vol_mismatch_available.iloc[0],0)
        self.assertTrue(np.isfinite(x[['long_rv_iv','short_rv_iv','log_vol_mismatch']]).all().all())

    def test_future_returns_cannot_change_past_rv_input(self):
        index=pd.date_range('2020-01-01',periods=30)
        returns=pd.Series(.01,index=index)
        rv=(returns.pow(2).rolling(20,min_periods=10).mean()*252)**.5
        changed=returns.copy();changed.iloc[21:]=1.
        other=(changed.pow(2).rolling(20,min_periods=10).mean()*252)**.5
        pd.testing.assert_series_equal(rv.iloc[:21],other.iloc[:21])

    def test_inventory_counts_both_names_and_uses_prior_funded_budget(self):
        cfg=SimpleNamespace(entry_vega_per_leg=450.,capital=1_000_000.)
        row=dict(secid_a=1,secid_b=2)
        context=inventory_context(row,{1:900.,2:450.},2_000_000.,cfg)
        self.assertEqual(context['name_inventory_units'],1.)
        self.assertEqual(inventory_level(4,context),3)
        self.assertEqual(inventory_level(4,dict(name_inventory_units=0.)),4)
        self.assertEqual(inventory_level(4,dict(name_inventory_units=10.)),2)
        self.assertEqual(inventory_level(2,dict(name_inventory_units=10.)),2)


if __name__=='__main__':
    unittest.main()
