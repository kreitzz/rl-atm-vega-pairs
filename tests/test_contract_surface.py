import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from scripts.build_contract_surface import candidates, reduce_points, surface_values


class ContractSurfaceTest(unittest.TestCase):
    def test_absent_surface_and_empty_trading_history_have_numeric_states(self):
        from scripts.run_contract_surface_features import augment, FEATURES, VALUES
        absent=pd.DataFrame(columns=VALUES,index=pd.DatetimeIndex([]))
        with patch('scripts.run_contract_surface_features.surface_asset',return_value=absent):
            for dates in [pd.DatetimeIndex([]),pd.date_range('2020-01-01',periods=2)]:
                result=augment({'secids':[1,2]},pd.DataFrame(index=dates),dates)
                self.assertTrue(np.isfinite(result[FEATURES]).all().all())
                self.assertTrue(result[FEATURES].eq(0).all().all())

    def fixture(self):
        date=pd.Timestamp('2020-01-02');middle=date+pd.Timedelta(days=30)
        selected=pd.DataFrame([dict(date=date,secid=1,expiry=middle,strike=100.,bs_iv=.30)])
        stocks=pd.DataFrame([dict(date=date,secid=1,close=100.)])
        rows=[]
        for i,(dte,strike,delta) in enumerate([(30,90,.75),(30,110,.25),(15,100,.5),(45,100,.5),(31,89,.75),(15,105,.4)]):
            rows.append(dict(date=date,secid=1,optionid=i+1,exdate=date+pd.Timedelta(days=dte),strike_price=strike*1000,
                delta=delta,vega=10.,cp_flag='C',best_bid=3.,best_offer=3.2,ss_flag='0',contract_size=100,open_interest=150))
        return pd.DataFrame(rows),selected,stocks

    def test_selection_stays_on_expiry_and_batch_reduction_is_exact(self):
        raw,selected,stocks=self.fixture()
        whole=candidates(raw,selected,stocks)
        parts=reduce_points(pd.concat([candidates(raw.iloc[:2],selected,stocks),candidates(raw.iloc[2:],selected,stocks)]))
        pd.testing.assert_frame_equal(whole,parts)
        self.assertEqual(whole.set_index('slot').optionid.to_dict(),dict(back=4,front=3,lower=1,upper=2))
        raw.loc[0,'best_offer']=10.
        self.assertNotIn('lower',candidates(raw,selected,stocks).slot.tolist())

    def test_actual_maturity_slope_and_curvature(self):
        raw,selected,stocks=self.fixture();points=candidates(raw,selected,stocks)
        points['bs_iv']=points.slot.map(dict(lower=.36,upper=.32,front=.28,back=.34))
        points['bs_delta']=points.delta
        result=surface_values(selected,points).iloc[0]
        self.assertAlmostEqual(result.downside_skew,6.)
        self.assertAlmostEqual(result.smile_curvature,4.)
        self.assertAlmostEqual(result.term_slope,6.)
        self.assertAlmostEqual(result.term_bend,-1.)
        points.loc[points.slot.eq('back'),'expiry']=selected.expiry.iloc[0]
        self.assertTrue(np.isnan(surface_values(selected,points).term_slope.iloc[0]))

    def test_missing_wing_not_fabricated_and_future_rows_cannot_change_today(self):
        raw,selected,stocks=self.fixture();first=candidates(raw,selected,stocks)
        future=raw.copy();future['date']+=pd.Timedelta(days=1);future['exdate']+=pd.Timedelta(days=1)
        pd.testing.assert_frame_equal(first,candidates(pd.concat([raw,future]),selected,stocks))
        points=first[first.slot.ne('upper')].copy();points['bs_iv']=.3;points['bs_delta']=points.delta
        result=surface_values(selected,points)
        self.assertTrue(result.smile_curvature.isna().all())
        self.assertTrue(result.downside_skew.notna().all())


if __name__=='__main__':unittest.main()
