from decimal import Decimal
import unittest
import numpy as np
import pandas as pd

from scripts.test_wrds_alternative_pairs import (build_review_states, monthly_date,
    numeric_ids, rolling_events, text_counts)
from scripts.test_wrds_pair_signals import match_states


class AlternativeTimingTests(unittest.TestCase):
    def test_decimal_identifiers_match_numpy_ids_after_normalization(self):
        states=pd.DataFrame({'rcid':[Decimal('123')],
                            'available':pd.to_datetime(['2020-01-01']),'feature':[2.]})
        states.rcid=pd.Series(numeric_ids(states.rcid)).astype(np.int64)
        got=match_states(pd.to_datetime(['2020-01-02']),numeric_ids([np.int64(123)]),
                         states,'rcid',['feature'],7)
        self.assertEqual(got.feature.iloc[0],2.)

    def test_monthly_features_wait_for_month_end_and_lag(self):
        result=monthly_date(pd.Series(pd.to_datetime(['2020-01-01','2020-02-01'])),15)
        self.assertEqual(result.iloc[0],pd.Timestamp('2020-02-17'))
        self.assertEqual(result.iloc[1],pd.Timestamp('2020-03-16'))

    def test_future_events_and_post_coverage_zeros_are_excluded(self):
        events=pd.DataFrame({'id':['A','A'],'available':pd.to_datetime(['2020-01-02','2020-01-15']),
                             'amount':[2.,999.]})
        dates=pd.to_datetime(['2020-01-03','2020-01-10','2020-01-20'])
        got=rolling_events(events,['A'],dates,'id',['amount'],30,pd.Timestamp('2020-01-18'))
        self.assertEqual(got.amount.iloc[0],2.)
        self.assertEqual(got.amount.iloc[1],2.)
        self.assertTrue(np.isnan(got.amount.iloc[2]))

    def review_fixture(self,count=10):
        months=pd.date_range('2020-01-01',periods=7,freq='MS')
        return pd.DataFrame({'rcid':123,'month':months,'overall_n':count,'overall_sum':count*3.,
          'leadership_n':count,'leadership_sum':count*3.,'outlook_n':count,'outlook_sum':0.})

    def test_future_reviews_do_not_change_historical_scores(self):
        raw=self.review_fixture();original=build_review_states(raw)
        raw.loc[6,'overall_sum']=50.;raw.loc[6,'outlook_sum']=10.
        changed=build_review_states(raw)
        pd.testing.assert_frame_equal(original.iloc[:6],changed.iloc[:6])
        self.assertNotEqual(original.review_rating_change.iloc[6],changed.review_rating_change.iloc[6])

    def test_sparse_review_counts_stay_missing(self):
        states=build_review_states(self.review_fixture(count=5))
        self.assertTrue(states[['review_rating_change','review_outlook_change','review_leadership']].isna().all().all())

    def test_fixed_lexicon_counts(self):
        self.assertEqual(text_counts('Strong growth; risk may decline.'),(5,2,2))


if __name__=='__main__':unittest.main()
