"""Safeguards for chronological opportunity-filter analysis."""
import unittest
import numpy as np
import pandas as pd
from scripts.analyze_p25_opportunity_filter import (
    paired_sharpe_interval, prediction_diagnostics, pair_diagnostics)


class OpportunityAnalysisTests(unittest.TestCase):
    def test_paired_resampling_preserves_identical_strategies(self):
        dates = pd.bdate_range('2022-01-03', periods=150)
        returns = pd.Series(np.sin(np.arange(150))*.001+.0001, index=dates)
        result = paired_sharpe_interval(returns, returns.copy(), block=63, replications=20)
        self.assertEqual((result['point'], result['lower95'], result['upper95']), (0., 0., 0.))

    def test_same_length_different_dates_is_not_paired(self):
        a = pd.Series([.01, -.01, .005], index=pd.bdate_range('2022-01-03', periods=3))
        b = pd.Series(a.to_numpy(), index=pd.bdate_range('2022-01-04', periods=3))
        with self.assertRaisesRegex(ValueError, 'identical date indexes'):
            paired_sharpe_interval(a, b, block=1, replications=20)

    def test_unresolved_and_future_labels_excluded_not_zeroed(self):
        data = pd.DataFrame(dict(case='ridge', date=pd.to_datetime(['2025-06-01']*3),
            pair=['a', 'b', 'c'], prediction=[1., 2., 3.], target=[2., np.nan, 1e9],
            selected=[True, True, False], end=pd.to_datetime(['2025-07-01', None, '2025-09-01'])))
        bins, coverage = prediction_diagnostics(data)
        row = bins.loc[bins.period.eq('evaluation') & bins.group.eq('all')].iloc[0]
        self.assertEqual(row.observations, 1)
        self.assertEqual(row.mean_target, 2.)
        counts = coverage.loc[coverage.period.eq('evaluation')].iloc[0]
        self.assertEqual(counts.candidates, 3)
        self.assertEqual(counts.matured_outcomes, 1)

    def test_pair_attribution_must_reconcile_shared_financing(self):
        dates = pd.to_datetime(['2022-01-03', '2022-01-04'])
        nav = pd.DataFrame(dict(nav=[100., 103.], cash_benchmark=[100., 100.]), index=dates)
        daily = pd.DataFrame(dict(pair_a=[0., 2.]), index=dates)
        financing = pd.Series([0., 1.], index=dates)
        rows, concentration = pair_diagnostics('ridge', daily, financing, nav)
        self.assertEqual(next(r for r in rows if r['period']=='evaluation')['net_before_shared_financing'], 2.)
        with self.assertRaises(AssertionError):
            pair_diagnostics('ridge', daily, financing*0, nav)


if __name__ == '__main__':
    unittest.main()
