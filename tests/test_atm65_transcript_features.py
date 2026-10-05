import unittest
import numpy as np
import pandas as pd
from scripts.atm65_transcript_features import add_transcripts, FIELDS
from scripts.test_wrds_pair_signals import match_states


class TranscriptFeatureTests(unittest.TestCase):
    def test_orientation_individual_missing_flags_and_unchanged_state(self):
        dates=pd.date_range('2020-01-01',periods=2)
        panel=pd.DataFrame({'spread':[1.,2.]},index=dates)
        first=pd.DataFrame({FIELDS[0]:[4.,5.],FIELDS[1]:[np.nan,3.],FIELDS[2]:[2.,2.]},index=dates)
        second=pd.DataFrame({FIELDS[0]:[3.,1.],FIELDS[1]:[1.,2.],FIELDS[2]:[2.,np.nan]},index=dates)
        got=add_transcripts(panel,first,second);reverse=add_transcripts(panel,second,first)
        pd.testing.assert_frame_equal(got[panel.columns],panel)
        for field in FIELDS:
            gap='transcript_'+field+'_gap';flag='transcript_'+field+'_available'
            np.testing.assert_array_equal(got[gap],-reverse[gap])
            np.testing.assert_array_equal(got[flag],reverse[flag])
        np.testing.assert_array_equal(got['transcript_call_qa_uncertainty_available'],[0.,1.])
        np.testing.assert_array_equal(got['transcript_call_analyst_negative_available'],[1.,0.])

    def test_future_transcripts_do_not_change_earlier_signal_and_stale_data_expire(self):
        dates=pd.to_datetime(['2020-01-01','2020-01-03','2020-08-01'])
        states=pd.DataFrame({'companyid':[123,123],'available':pd.to_datetime(['2020-01-02','2020-08-02']),
                             **{field:[1.,999.] for field in FIELDS}})
        got=match_states(dates,np.array([123]*3),states,'companyid',FIELDS,180)
        self.assertTrue(got.iloc[0].isna().all())
        self.assertTrue(got.iloc[1].eq(1.).all())
        self.assertTrue(got.iloc[2].isna().all())
        states.loc[1,FIELDS]=0.
        changed=match_states(dates,np.array([123]*3),states,'companyid',FIELDS,180)
        pd.testing.assert_frame_equal(got,changed)


if __name__=='__main__': unittest.main()
