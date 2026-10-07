import unittest

import numpy as np
import pandas as pd

from scripts.p25_earnings_features import leg_features, orient_legs


class EarningsStateTests(unittest.TestCase):
    def test_prior_only_ignores_changed_future_reports(self):
        past = pd.to_datetime(['2022-01-20','2022-04-20','2022-07-20','2022-10-20'])
        decision = pd.to_datetime(['2022-11-01'])
        a = leg_features(decision,[30],past.append(pd.DatetimeIndex(['2023-01-10'])))
        b = leg_features(decision,[30],past.append(pd.DatetimeIndex(['2023-02-28'])))
        pd.testing.assert_frame_equal(a,b)

    def test_known_calendar_includes_expiry_boundary(self):
        events = pd.to_datetime(['2023-01-20'])
        result = leg_features(pd.to_datetime(['2023-01-01','2023-01-01']),[19,18],events,mode='historical_calendar')
        self.assertEqual(result.earnings_days.tolist(),[19.,19.])
        self.assertEqual(result.earnings_before_expiry.tolist(),[1.,0.])

    def test_unknown_date_is_flagged_and_does_not_drop_row(self):
        result = leg_features(pd.to_datetime(['2023-01-01']),[30],[])
        self.assertEqual(len(result),1)
        self.assertTrue(np.isfinite(result).all().all())
        self.assertEqual(result.earnings_days_available.iloc[0],0)
        self.assertEqual(result.earnings_before_expiry_available.iloc[0],0)

    def test_etf_has_no_issuer_event(self):
        result = leg_features(pd.to_datetime(['2023-01-01']),[365],[],etf=True)
        self.assertEqual(result.earnings_days_available.iloc[0],1)
        self.assertEqual(result.earnings_before_expiry.iloc[0],0)

    def test_orientation_keeps_long_and_short_distinct(self):
        dates = pd.to_datetime(['2023-01-01','2023-01-01'])
        a = leg_features(dates,[30,30],pd.to_datetime(['2023-01-10']),mode='historical_calendar')
        b = leg_features(dates,[30,30],pd.to_datetime(['2023-02-10']),mode='historical_calendar')
        result = orient_legs(a,b,[1,-1])
        self.assertEqual(result.long_earnings_days.tolist(),[9.,40.])
        self.assertEqual(result.short_earnings_days.tolist(),[40.,9.])
        self.assertEqual(result.long_earnings_before_expiry.tolist(),[1.,0.])

    def test_same_day_release_is_not_prior_information(self):
        events = pd.to_datetime(['2022-01-01','2022-04-01','2022-07-01','2022-10-01'])
        date = pd.to_datetime(['2022-10-01'])
        with_same_day = leg_features(date,[30],events)
        without = leg_features(date,[30],events[:-1])
        pd.testing.assert_frame_equal(with_same_day,without)


if __name__ == '__main__':
    unittest.main()
