import unittest
import numpy as np
import pandas as pd

from scripts.explore_opposite_wing import wing_features, parity_matches


class OppositeWingTest(unittest.TestCase):
    def asset(self,level):
        dates=pd.bdate_range('2015-01-01',periods=350)
        frame=pd.DataFrame(dict(bs_iv=level+np.arange(350)*.0001,dte=30,bs_delta=-.75,bs_vega=.1,
            bid=4.,ask=4.1,strike=100.,expiry=dates+pd.Timedelta(days=30)),index=dates)
        return dict(frames={('P',75):frame},premium_z={('P',75):pd.Series(0.,index=dates)},rv=pd.Series(.2,index=dates))

    def test_future_selection_cannot_change_earlier_features(self):
        a,b=self.asset(.3),self.asset(.35)
        x=wing_features(a,b,'P',75)
        a['frames']['P',75].loc[a['frames']['P',75].index[300:],'bs_iv']=2.
        y=wing_features(a,b,'P',75)
        pd.testing.assert_frame_equal(x.iloc[:300],y.iloc[:300])

    def test_parity_comparison_requires_both_strikes_and_expiries(self):
        dates=pd.bdate_range('2018-01-02',periods=3)
        put=pd.DataFrame(dict(pair='a',date=dates,strike_a=100.,strike_b=50.,expiry_a=pd.Timestamp('2018-02-01'),
            expiry_b=pd.Timestamp('2018-02-01'),resolved=True,eligible=True,legacy=True))
        call=put.copy(); call.loc[1,'strike_b']=51.; call.loc[2,'expiry_a']=pd.Timestamp('2018-02-02')
        matches=parity_matches(put,call)
        np.testing.assert_array_equal(matches.same_strikes_expiries,[True,False,False])


if __name__=='__main__':unittest.main()
