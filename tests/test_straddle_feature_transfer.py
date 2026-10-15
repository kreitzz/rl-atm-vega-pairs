import unittest
import numpy as np
import pandas as pd
from scripts import straddle_feature_transfer as study


class TransferTests(unittest.TestCase):
    def test_memory_requires_completion_before_signal_and_respects_window(self):
        labels=pd.DataFrame(dict(finite_label=[True]*3,end=pd.to_datetime(['2020-01-01','2020-01-03','2020-06-01']),
                                 full_profit=[450.,-900.,1e12]))
        dates=pd.to_datetime(['2020-01-01','2020-01-02','2020-01-03','2020-01-04','2020-05-01'])
        actual=study.profit_memory(dates,labels)
        np.testing.assert_allclose(actual.profit_91d,[np.nan,1.,1.,-.5,np.nan],equal_nan=True)
        np.testing.assert_array_equal(actual.count_91d,[0,1,1,2,0])
        self.assertAlmostEqual(actual.profit_365d.iloc[-1],-.5)

    def test_price_zscore_uses_only_previous_history(self):
        series=pd.Series(np.arange(300,dtype=float))
        original=study.past_z(series)
        series.iloc[250:]=1e12
        pd.testing.assert_series_equal(original.iloc[:250],study.past_z(series).iloc[:250])
        self.assertGreater(original.iloc[252],1.)

    def test_feature_groups_retain_control_and_every_missingness_flag(self):
        study.configure()
        for fields in study.core.FEATURES.values():
            self.assertEqual(fields[:3],study.CONTROL)
            self.assertEqual(len(fields),len(set(fields)))
        for field in study.EXTRA:self.assertIn(field+'_available',study.core.FEATURES['all'])
        self.assertEqual(study.CFG['actions'],[2,3,4])


if __name__=='__main__':unittest.main()
