import unittest
import numpy as np
import pandas as pd
from scripts.diagnose_p25_maturity import (entry_greeks, pair_metadata, maturity_band,
    intended_entry_dates, saved_path_metrics, frozen_leaf_ids)
from scripts.option_flex.black_scholes import option_values


class MaturityDiagnosticTest(unittest.TestCase):
    def test_put_gamma_and_theta_match_price_derivatives(self):
        s, k, days, r, sigma, pv = 100., 95., 30., .04, .3, .5
        greek = entry_greeks(s, k, days, r, sigma, pv)
        def price(spot, dte):
            return float(option_values(spot-pv, k, dte/365, r, sigma, 'P')[0])
        h = .01
        gamma = (price(s+h, days)-2*price(s, days)+price(s-h, days))/h**2
        theta = (price(s, days-.001)-price(s, days+.001))/.002
        self.assertAlmostEqual(greek['gamma'], gamma, places=6)
        self.assertAlmostEqual(greek['theta_day'], theta, places=7)

    def test_calendar_gap_is_not_pair_holding_duration(self):
        entry = pd.Timestamp('2022-01-03')
        legs = pd.DataFrame(dict(cohort=['2021-12-31|A__B|P25']*2,
            role=['long', 'short'], entry_date=[entry]*2,
            exit_date=[entry+pd.Timedelta(days=24), entry+pd.Timedelta(days=38)],
            entry_dte=[25, 39], holding_days=[24, 38], initial_vega=[450., 440.],
            gamma_1pct_per_vega=[.1, .2], theta_day_per_vega=[-.2, -.1]))
        row = pair_metadata(legs).iloc[0]
        self.assertEqual(row.expiry_gap_days, 14)
        self.assertEqual(row.both_legs_open_days, 24)
        self.assertEqual(row.cohort_holding_days, 38)
        self.assertEqual(row.one_leg_tail_days, 14)
        self.assertAlmostEqual(row.gamma_1pct_gross_per_mean_vega, (45+88)/445)
        self.assertAlmostEqual(row.theta_day_net_per_mean_vega, (-90+44)/445)

    def test_band_boundaries_do_not_hide_short_or_long_durations(self):
        bands = maturity_band(pd.Series([14, 21, 27, 28, 34, 35, 45, 46]))
        self.assertEqual(bands.tolist(), ['<21', '21-27', '21-27', '28-34', '28-34', '35-45', '35-45', '46+'])

    def test_signal_to_entry_uses_next_session_not_calendar_day(self):
        signals = pd.Series(pd.to_datetime(['2022-01-14', '2022-01-18']))
        calendar = pd.to_datetime(['2022-01-14', '2022-01-18', '2022-01-19'])
        self.assertEqual(intended_entry_dates(signals, calendar).tolist(),
                         list(pd.to_datetime(['2022-01-18', '2022-01-19'])))

    def test_path_risk_retains_real_zero_days_excludes_padding(self):
        result = saved_path_metrics(np.array([-2., 0., 3., -1., 0., 0.]), 4)
        self.assertAlmostEqual(result['daily_sd'], np.std([-2., 0., 3., -1.], ddof=1))
        self.assertEqual(result['path_max_drawdown'], 2.)
        self.assertEqual(result['realized_quadratic_exposure'], 14.)
        with self.assertRaises(ValueError):
            saved_path_metrics(np.array([-2., 0., 3., -1., 1.]), 4)

    def test_saved_tree_traversal_preserves_float32_split_boundary(self):
        from types import SimpleNamespace
        tree = SimpleNamespace(children_left=np.array([1, -1, -1]),
            children_right=np.array([2, -1, -1]), feature=np.array([0, -2, -2]),
            threshold=np.array([.1000000015, -2., -2.]))
        model = dict(tree=SimpleNamespace(tree_=tree), features=['x'])
        self.assertEqual(frozen_leaf_ids(model, pd.DataFrame({'x': [.1000000016, .2]})).tolist(), [1, 2])


if __name__ == '__main__':
    unittest.main()
