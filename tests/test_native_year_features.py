import unittest

import numpy as np
import pandas as pd

from scripts import native_year_features as f


class NativeYearFeaturesTest(unittest.TestCase):
    def test_factor_reset_does_not_create_shares(self):
        rows = pd.DataFrame(dict(secid=1, date=pd.bdate_range('2019-05-30', periods=3),
             close=[50.31, 49.66, 50.], cfadj=[2., 2.04, 2.04],
             **{'return': [0., 49.66/50.31-1, 50./49.66-1]}))
        clean, notes = f.reconcile_adjustments(rows)
        np.testing.assert_allclose(clean.cfadj, [2., 2., 2.])
        np.testing.assert_allclose(clean.source_cfadj, rows.cfadj)
        self.assertEqual(len(notes), 1)

    def test_true_split_preserved_and_future_adjustments_do_not_change_past(self):
        rows = pd.DataFrame(dict(secid=1, date=pd.bdate_range('2020-01-01', periods=4),
             close=[100., 50., 51., 25.5], cfadj=[1., 2., 2.04, 4.08],
             **{'return': [0., 0., .02, 0.]}))
        clean, notes = f.reconcile_adjustments(rows)
        np.testing.assert_allclose(clean.cfadj, [1., 2., 2., 4.])
        prefix, _ = f.reconcile_adjustments(rows.iloc[:3])
        pd.testing.assert_series_equal(clean.cfadj.iloc[:3], prefix.cfadj)
        self.assertEqual(len(notes), 1)

    def test_premature_reverse_split_follows_already_published_occ_notice(self):
        rows = pd.DataFrame(dict(secid=111450, date=pd.bdate_range('2022-07-25', periods=4),
            close=[.326, .3685, .36, 3.75], cfadj=[1., .1, 1., .1],
            **{'return': [0., -.886963, -.023066, .041667]}))
        clean, _ = f.reconcile_adjustments(rows)
        np.testing.assert_allclose(clean.cfadj, [1., 1., 1., .1])
        prefix, _ = f.reconcile_adjustments(rows.iloc[:3])
        pd.testing.assert_series_equal(clean.cfadj.iloc[:3], prefix.cfadj)

    def test_forward_variance_excludes_signal_return(self):
        dates = pd.bdate_range('2010-01-01', periods=600)
        r = np.linspace(.001, .02, len(dates))
        close = pd.Series(100 * np.exp(np.cumsum(r)), index=dates)
        frame, returns = f.variance_inputs(close, dates)
        self.assertAlmostEqual(frame.future_variance.iloc[260], 252 * returns.iloc[261:513].pow(2).mean())
        self.assertEqual(frame.label_end.iloc[260], dates[512])

    def test_label_at_cutoff_is_not_completed(self):
        cutoff = pd.Timestamp('2018-01-01')
        rows = pd.DataFrame(dict(date=[pd.Timestamp('2016-01-01')]*3,
                                 label_end=[cutoff-pd.Timedelta(days=1), cutoff, cutoff+pd.Timedelta(days=1)],
                                 rv252=[.04]*3, future_variance=[.05]*3))
        for col in f.FORECAST_FEATURES:
            rows[col] = 1.
        self.assertEqual(f.completed_forecasts(rows, cutoff).index.tolist(), [0])

    def test_forecast_ignores_uncompleted_targets(self):
        dates = pd.date_range('2013-01-01', periods=180, freq='MS')
        rows = pd.DataFrame(dict(date=dates, label_end=dates+pd.DateOffset(years=1),
                                 secid=1, rv252=.04, future_variance=.06))
        for col in f.FORECAST_FEATURES:
            rows[col] = np.linspace(.5, 1., len(rows))
        forecast, _ = f.forecast_rows(rows)
        changed = rows.copy()
        cutoff = pd.Timestamp('2024-01-01')
        changed.loc[changed.label_end.ge(cutoff), 'future_variance'] = 10000.
        other, _ = f.forecast_rows(changed)
        pd.testing.assert_series_equal(forecast.loc[forecast.date.lt('2024-04-01'), 'forecast_variance'],
                                       other.loc[other.date.lt('2024-04-01'), 'forecast_variance'])

    def test_name_z_excludes_current_premium_and_future_quotes(self):
        dates = pd.date_range('2013-01-01', periods=16, freq='MS')
        rows = pd.DataFrame(dict(date=dates, secid=1, bs_iv=.3, rv252=.04,
                                 forecast_variance=np.linspace(.03, .08, 16)))
        result = f.name_premiums(rows)
        past = result.carry_forecast.iloc[:12]
        self.assertAlmostEqual(result.carry_z.iloc[12], (result.carry_forecast.iloc[12]-past.mean())/past.std())
        changed = rows.copy()
        changed.loc[13:, 'bs_iv'] = 2.
        pd.testing.assert_series_equal(result.carry_z.iloc[:13], f.name_premiums(changed).carry_z.iloc[:13])

    def test_long_cheapest_short_richest_and_no_missing_trade(self):
        np.testing.assert_array_equal(f.native_direction([.1, -.1, 0, np.nan]), [1, -1, 0, 0])

    def test_dividend_prefilter_preserves_full_fit(self):
        from scripts.run_native_year_entry import relevant_dividends, old
        date = pd.Timestamp('2020-01-02')
        quotes = pd.DataFrame(dict(date=[date], secid=[1], expiry=[pd.Timestamp('2021-01-03')],
                   strike=[100.], cp_flag=['P'], bid=[9.9], ask=[10.1], standard=[True], multiplier=[100.]))
        stocks = pd.DataFrame(dict(date=[date], secid=[1], close=[100.]))
        zero = pd.DataFrame(dict(date=[date], days=[365], rate=[1.]))
        divs = pd.DataFrame(dict(secid=[1]*4, amount=[1.]*4, cancel_flag='0', distr_type='1', currency='USD',
                   adj_factor=1, approx_flag='0', liquid_flag='0',
                   ex_date=pd.to_datetime(['2020-07-01', '2019-07-01', '2022-07-01', '2020-08-01']),
                   declare_date=pd.to_datetime(['2019-12-01', '2019-01-01', '2019-12-01', '2020-08-01'])))
        pruned = relevant_dividends(quotes, divs)
        self.assertEqual(len(pruned), 1)
        pd.testing.assert_frame_equal(old.fit_quotes(quotes, stocks, zero, divs),
                                       old.fit_quotes(quotes, stocks, zero, pruned))

    def test_native_iv_history_uses_only_previous_native_quotes(self):
        frame = pd.DataFrame(dict(pair='a', date=pd.date_range('2013-01-01', periods=10, freq='MS'),
                                  optionid_a=1, optionid_b=2, iv_mean=.4, iv_gap_relative=np.arange(10)*.01))
        first = f.native_iv_history(frame)
        changed = frame.copy()
        changed.loc[7:, 'iv_gap_relative'] = 1000.
        second = f.native_iv_history(changed)
        pd.testing.assert_series_equal(first.native_iv_gap_z.iloc[:7], second.native_iv_gap_z.iloc[:7])
        self.assertEqual(first.iv_history_direction.iloc[6], -1)


if __name__ == '__main__':
    unittest.main()
