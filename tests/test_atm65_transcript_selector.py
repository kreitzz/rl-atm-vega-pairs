import unittest
import pandas as pd
from scripts.run_atm65_transcript_selector import choose_actions


class SelectorTests(unittest.TestCase):
    def test_past_scores_ties_and_disabled_proposals(self):
        dates=pd.to_datetime(['2020-01-02','2020-02-03','2020-03-02'])
        old=pd.Series([1,1,1],index=dates);new=pd.Series([2,2,2],index=dates)
        fits=pd.DataFrame({'fit':['2020-01-01','2020-02-01','2020-03-01'],
            'status':['enabled']*3,'validation_sharpe':[.5,.5,.5],
            'validation_end':['2019-12-01','2020-01-01','2020-02-01']})
        candidate=fits.rename(columns={'status':'rl_transcripts_status','validation_sharpe':'rl_transcripts_validation_sharpe'}).copy()
        candidate.rl_transcripts_validation_sharpe=[.6,.5,999.]
        candidate.loc[2,'rl_transcripts_status']='flat_validation_gate'
        got,_=choose_actions(old,new,fits,candidate,'rl',False)
        self.assertEqual(got.tolist(),[2,1,1])
        future=candidate.copy();future.loc[2,'rl_transcripts_validation_sharpe']=-999.
        changed,_=choose_actions(old,new,fits,future,'rl',False)
        pd.testing.assert_series_equal(got,changed)

    def test_future_validation_date_is_rejected(self):
        dates=pd.to_datetime(['2020-01-02']);a=pd.Series([1],index=dates)
        fits=pd.DataFrame({'fit':['2020-01-01'],'status':['enabled'],'validation_sharpe':[.5],
                           'validation_end':['2020-01-02']})
        new=fits.rename(columns={'status':'rl_transcripts_status','validation_sharpe':'rl_transcripts_validation_sharpe'})
        with self.assertRaises(AssertionError):choose_actions(a,a,fits,new,'rl',False)


if __name__=='__main__':unittest.main()
