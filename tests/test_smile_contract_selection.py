import unittest
import pandas as pd
from scripts.smile_contract_selection import select, reduce_choices


class SelectionTest(unittest.TestCase):
    def quotes(self):
        return pd.DataFrame(dict(date=pd.Timestamp('2020-01-02'), secid=1,
            expiry=pd.to_datetime(['2020-02-02','2020-02-02','2020-02-07','2020-02-07']),
            optionid=[1,2,3,4], bs_delta=[-.25,-.75,-.25,-.75],
            bs_status='ok', bs_vega=.1, standard=True, multiplier=100,
            cp_flag='P', bid=5., ask=5.1, open_interest=[100,0,100,10],
            close=100., strike=[95,105,95,105]))

    def following(self):
        return pd.Series(pd.Timestamp('2020-01-03'), index=[pd.Timestamp('2020-01-02')])

    def test_each_bucket_can_use_a_different_expiry_when_quotes_require_it(self):
        selected = select(self.quotes(), self.following(), 'P').set_index('bucket')
        self.assertEqual(selected.at['25','optionid'], 1)
        self.assertEqual(selected.at['75','optionid'], 4)
        self.assertEqual(selected.at['75','open_interest'], 10)

    def test_expiry_closeness_precedes_delta_closeness_and_future_oi_is_unused(self):
        q = self.quotes()
        q.loc[0,'bs_delta'] = -.28
        self.assertEqual(select(q, self.following(), 'P').set_index('bucket').at['25','optionid'], 1)
        # Best-per-chunk reduction must equal selection over the entire batch.
        chunks = pd.concat([select(q.iloc[:2], self.following(), 'P'), select(q.iloc[2:], self.following(), 'P')])
        pd.testing.assert_frame_equal(reduce_choices(chunks), select(q, self.following(), 'P'))

    def test_entry_dte_is_measured_at_next_session_and_crossed_quotes_rejected(self):
        q = self.quotes()
        q.loc[0,'expiry'] = pd.Timestamp('2020-01-23')  # Signal21D, entry20D.
        q.loc[1,'open_interest'] = 10
        q.loc[1,'ask'] = 4.
        result = select(q, self.following(), 'P')
        self.assertEqual(set(result.optionid), {3,4})


if __name__ == '__main__':
    unittest.main()
