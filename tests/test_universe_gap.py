import unittest
import numpy as np
import pandas as pd
from scripts.explore_universe_gap import correct_earnings, block_difference


class UniverseGapTests(unittest.TestCase):
    def test_earnings_overrides_replace_late_dates_only(self):
        d=pd.DataFrame({'permno':[48486,48486,48486,1],
                        'datadate':pd.to_datetime(['2007-06-30','2007-09-30','2007-12-31','2007-06-30']),
                        'rdq':pd.to_datetime(['2008-03-31','2008-03-31','2008-03-31','2007-07-01'])})
        corrected=correct_earnings(d)
        expected=pd.to_datetime(['2007-07-24','2007-10-10','2008-01-24','2007-07-01'])
        self.assertEqual(corrected.rdq.tolist(),expected.tolist())
        self.assertEqual(d.rdq.iloc[0],pd.Timestamp('2008-03-31'))

    def test_paired_block_difference_identity(self):
        r=np.random.default_rng(7).normal(size=300)
        np.testing.assert_array_equal(block_difference(r,r,repeats=50),[0,0])


if __name__=='__main__':unittest.main()
