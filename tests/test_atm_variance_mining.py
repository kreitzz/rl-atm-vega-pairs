import unittest
import numpy as np
import pandas as pd
from scripts.mine_atm_variance_signals import holding_variance,rolling_factor_residual,forecast,price_features,deduplicate_legs,factor_residuals


class MiningTests(unittest.TestCase):
    def test_target_excludes_entry_move_and_rejects_missing_path(self):
        result=holding_variance([999.,999.,.01,.04,np.nan],[0,1,2],[3,3,4])
        self.assertAlmostEqual(result[0],6.3)
        self.assertAlmostEqual(result[1],10.08)
        self.assertTrue(np.isnan(result[2]))

    def test_factor_coefficients_ignore_current_and_future_targets(self):
        index=pd.bdate_range('2020-01-01',periods=190)
        rng=np.random.default_rng(123);x=pd.DataFrame(rng.normal(size=(190,3)),index=index)
        y=pd.Series(x.to_numpy()@np.array([.4,-.2,.1]),index=index)
        a=rolling_factor_residual(y,x)
        changed=y.copy();changed.iloc[150:]+=1000
        b=rolling_factor_residual(changed,x)
        pd.testing.assert_series_equal(a.iloc[:150],b.iloc[:150])
        self.assertAlmostEqual(b.iloc[150]-a.iloc[150],1000.)
        self.assertTrue(a.iloc[:126].isna().all())

    def test_forecast_never_uses_test_targets(self):
        n=180;x=np.linspace(-1,1,n)
        train=pd.DataFrame(dict(feature=x,target=.04*np.exp(x),iv=.2,secid=np.repeat([1,2],90)))
        test=train.iloc[:10].copy();a=forecast(train,test,['feature'])
        test.target=1e8
        np.testing.assert_array_equal(a,forecast(train,test,['feature']))
        self.assertTrue(np.isfinite(a).all() and (a>0).all())

    def test_daily_bar_features_are_causal_and_split_adjusted(self):
        calendar=pd.bdate_range('2020-01-01',periods=100)
        p=100*np.exp(np.arange(100)*.001)
        data=pd.DataFrame(dict(date=calendar,close=p,open=p*.999,high=p*1.002,low=p*.997,cfadj=1.))
        a,_=price_features(data,calendar)
        split=data.copy();split.loc[60:,['close','open','high','low']]/=2;split.loc[60:,'cfadj']=2.
        b,_=price_features(split,calendar);pd.testing.assert_frame_equal(a,b)
        data.loc[70:,['close','open','high','low']]*=20
        c,_=price_features(data,calendar);pd.testing.assert_frame_equal(a.iloc[:70],c.iloc[:70])

    def test_duplicate_assets_do_not_gain_training_weight(self):
        row=dict(date=pd.Timestamp('2020-01-01'),secid=1,optionid=10,iv=.2,target=.03,long_net=.4,short_net=-.7)
        data=pd.DataFrame([row,row]);self.assertEqual(len(deduplicate_legs(data)),1)
        data.loc[1,'long_net']=99
        with self.assertRaises(AssertionError):deduplicate_legs(data)

    def test_peer_factor_membership_does_not_use_future_outcome_labels(self):
        calendar=pd.bdate_range('2020-01-01',periods=150);rng=np.random.default_rng(7)
        data=pd.DataFrame([dict(date=date,secid=sid,forecast_vrp=rng.normal(),dte=30.,industry='10',target=1.)
            for sid in range(25) for date in calendar])
        left=factor_residuals(data,calendar)
        data.loc[data.secid.eq(0),'target']=np.nan
        right=factor_residuals(data,calendar)
        columns=['date','secid','factor_vrp_residual','sector_factor_available']
        pd.testing.assert_frame_equal(left[columns],right[columns])


if __name__=='__main__':unittest.main()
