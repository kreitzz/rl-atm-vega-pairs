import unittest
import numpy as np
import pandas as pd
from scripts.straddle_performance_followup import benchmark_features,contract_features,CFG


class PerformanceContextTests(unittest.TestCase):
    def test_benchmark_features_do_not_use_future_marks(self):
        dates=pd.bdate_range('2019-01-01',periods=180)
        r=np.sin(np.arange(180))*.001
        nav=pd.DataFrame(dict(excess_return=r,nav=1e7*np.cumprod(1+r),
            cash_benchmark=1e7,capital_reserve=1e6),index=dates)
        pd.testing.assert_frame_equal(benchmark_features(nav).iloc[:130],benchmark_features(nav.iloc[:130]))
        nav.iloc[130:,nav.columns.get_loc('excess_return')]=100
        pd.testing.assert_frame_equal(benchmark_features(nav).iloc[:130],benchmark_features(nav.iloc[:130]))

    def test_rounding_uses_total_straddle_vega_and_half_budget(self):
        # A300-dollar-vega straddle fits full size but no whole half-size contract.
        f=contract_features(pd.Series([.06/300,.06/100,0.]))
        self.assertEqual(f.half_vega_utilization.iloc[0],0)
        self.assertAlmostEqual(f.full_vega_utilization.iloc[0],300/450)
        self.assertAlmostEqual(f.half_vega_utilization.iloc[1],200/225)
        self.assertEqual(f.signal_vega_available.iloc[2],0)
        self.assertTrue(np.isfinite(f).all().all())
        self.assertEqual(len(CFG['specifications']),4)


if __name__=='__main__':unittest.main()
