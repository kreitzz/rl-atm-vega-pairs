import unittest
import pandas as pd
from scripts.run_walk_forward import fold_data, nonoverlap_positions


class WalkForwardTests(unittest.TestCase):
    def test_fold_purges_unsettled_outcomes(self):
        dates=pd.to_datetime(['2007-12-01','2007-12-15','2008-01-02','2009-12-15','2010-01-04','2011-12-30','2012-01-03'])
        panel=pd.DataFrame({'end':dates+pd.Timedelta(days=30)},index=dates)
        train,val,test=fold_data(panel,2010)
        self.assertEqual(list(train.index),[dates[0]])
        self.assertEqual(list(val.index),[dates[2]])
        self.assertEqual(list(test.index),[dates[4],dates[5]])
        self.assertLess(train.end.max(),val.index.min())
        self.assertLess(val.end.max(),test.index.min())

    def test_nonoverlap_uses_expiry_not_row_count(self):
        dates=pd.to_datetime(['2020-01-02','2020-01-31','2020-02-03','2020-03-03','2020-03-04'])
        panel=pd.DataFrame({'end':dates+pd.Timedelta(days=30)},index=dates)
        self.assertEqual(nonoverlap_positions(panel).tolist(),[0,2,4])


if __name__=='__main__': unittest.main()
