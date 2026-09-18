import unittest
import numpy as np
import pandas as pd
from scripts.build_short_rv_momentum import short_rv, momentum_12_1, feature_sets


class ShortRVMomentumTests(unittest.TestCase):
    def test_short_rv_uses_only_requested_past_returns(self):
        d=pd.DataFrame({'AMD':[.8]*10+[.01]*5+[.9]*5})
        rv=short_rv(d,5)
        self.assertAlmostEqual(rv.iloc[14,0],100*np.sqrt(252)*np.log1p(.01))
        self.assertTrue(rv.iloc[:4,0].isna().all())
        d.iloc[12,0]=np.nan
        self.assertTrue(np.isnan(short_rv(d,5).iloc[14,0]))

    def test_momentum_excludes_recent_month_and_future(self):
        d=pd.DataFrame({'AMD':np.full(300,.001)})
        d.iloc[231:]=.5
        actual=momentum_12_1(d)
        self.assertAlmostEqual(actual.iloc[251,0],100*((1.001)**231-1),places=9)
        self.assertTrue(actual.iloc[:251,0].isna().all())
        d.iloc[250:]=-.9
        self.assertEqual(momentum_12_1(d).iloc[251,0],actual.iloc[251,0])
        d.iloc[100,0]=np.nan
        self.assertTrue(np.isnan(momentum_12_1(d).iloc[251,0]))

    def test_matched_variants_keep_earnings_and_both_legs(self):
        fs=feature_sets()
        self.assertEqual(len(fs),12)
        for name,columns in fs.items():
            self.assertIn('amd_earnings_days_capped',columns)
            self.assertIn('nvda_earnings_days_capped',columns)
            if 'momentum' in name:
                self.assertIn('momentum_12_1_amd',columns)
                self.assertIn('momentum_12_1_nvda',columns)


if __name__=='__main__': unittest.main()
