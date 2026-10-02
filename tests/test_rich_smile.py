import unittest
import numpy as np
import pandas as pd
from scripts.build_rich_smile import bracket_candidates,reduce_brackets,interpolate,TARGETS,VALUES
from scripts.run_rich_smile import augment_smile

class SmileTests(unittest.TestCase):
    def quotes(self,deltas):
        return pd.DataFrame([dict(date=pd.Timestamp('2020-01-02'),secid=1,optionid=i,bs_delta=d,
            bs_iv=.4+.1*(d-.5),bs_status='ok',relative_width=.05) for i,d in enumerate(deltas)])
    def selected(self):
        return pd.DataFrame([dict(date=pd.Timestamp('2020-01-02'),secid=1,bs_iv=.4,expiry=pd.Timestamp('2020-02-01'))])
    def test_exact_points_and_normalized_smile(self):
        x=interpolate(bracket_candidates(self.quotes(TARGETS)),self.selected()).iloc[0]
        self.assertAlmostEqual(x.iv90,.44);self.assertAlmostEqual(x.down10,.1)
        self.assertAlmostEqual(x.rr25,-.125);self.assertAlmostEqual(x.bf25,0)
    def test_no_extrapolation_or_wide_gap_interpolation(self):
        x=interpolate(bracket_candidates(self.quotes([.2,.3,.7,.8])),self.selected()).iloc[0]
        self.assertTrue(np.isnan(x.iv10));self.assertTrue(np.isnan(x.iv90));self.assertTrue(np.isnan(x.iv50))
        self.assertAlmostEqual(x.iv25,.375);self.assertAlmostEqual(x.iv75,.425)
    def test_chunk_invariance(self):
        q=self.quotes([.08,.12,.2,.3,.45,.55,.7,.8,.88,.92])
        a=bracket_candidates(q)
        b=reduce_brackets(pd.concat([bracket_candidates(q.iloc[:4]),bracket_candidates(q.iloc[4:])]))
        pd.testing.assert_frame_equal(a,b)
    def test_changes_are_causal_calendar_lags_and_missing_is_flagged(self):
        cal=pd.bdate_range('2020-01-01',periods=50);base=pd.DataFrame({'x':1.},index=cal)
        frame=pd.DataFrame({v:np.arange(50,dtype=float) for v in VALUES},index=cal)
        frame.loc[cal[30],'down10']=np.nan
        a,levels,changes=augment_smile(base,[frame,frame*.5],cal)
        b,_,_=augment_smile(base.iloc[:40],[frame.iloc[:40],frame.iloc[:40]*.5],cal[:40])
        pd.testing.assert_frame_equal(a.iloc[:40],b)
        self.assertEqual(a.loc[cal[29],'smile_down10_change5_a'],5.)
        self.assertEqual(a.loc[cal[35],'smile_down10_change5_a_available'],0.)
        self.assertEqual(a.loc[cal[35],'smile_down10_change5_a'],0.)
        self.assertEqual(a.loc[cal[29],'smile_down10_change5_difference'],2.5)
        self.assertEqual(len(levels),48);self.assertEqual(len(changes),96)

if __name__=='__main__':unittest.main()
