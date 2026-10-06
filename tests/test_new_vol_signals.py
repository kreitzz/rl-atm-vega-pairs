import unittest
import numpy as np
import pandas as pd
from scripts.screen_new_vol_signals import causal_global, daily_features, dated_events, event_panel, borrow_features
from scripts.test_wrds_pair_signals import match_states


class SignalTimingTest(unittest.TestCase):
    def test_close_data_cannot_enter_same_day(self):
        dates=pd.to_datetime(['2024-01-05','2024-01-08','2024-01-09'])
        raw=pd.DataFrame(dict(secid=[1,1],date=['2024-01-05']*2,cp_flag=['C','P'],
                              volume=[10,30],open_interest=[100,100]))
        state=daily_features(raw,dates)
        matched=match_states(dates,[1,1,1],state,'secid',['put_call_volume','volume_oi'],7)
        self.assertTrue(matched.iloc[0].isna().all())
        self.assertAlmostEqual(matched.iloc[1].volume_oi,.2)
        self.assertAlmostEqual(matched.iloc[1].put_call_volume,np.log(31/11))

    def test_event_waits_for_latest_record_and_expires(self):
        raw=pd.DataFrame(dict(company_fkey=['123'],file_date=['2024-01-03'],
                              file_accepted=['20240105180000']))
        events=dated_events(raw,['file_date','file_accepted'])
        dates=pd.to_datetime(['2024-01-05','2024-01-08','2024-04-08'])
        panel=event_panel(events,['0000000123'],dates,'event')
        np.testing.assert_allclose(panel.event,[0,np.log(2),0])

    def test_global_lag_and_staleness(self):
        series=pd.Series([10.],index=pd.to_datetime(['2024-01-05']))
        dates=pd.to_datetime(['2024-01-05','2024-01-08','2024-01-16'])
        values=causal_global(series,dates)
        self.assertTrue(np.isnan(values.iloc[0]))
        self.assertEqual(values.iloc[1],10.)
        self.assertTrue(np.isnan(values.iloc[2]))

    def test_borrow_sentinel_does_not_become_level_or_change(self):
        dates=pd.bdate_range('2024-01-02',periods=23)
        raw=pd.DataFrame(dict(secid=1,date=dates,borrowrate=.01))
        raw.loc[0,'borrowrate']=-99.99
        clean=borrow_features(raw)
        self.assertTrue(np.isnan(clean.borrow_level.iloc[0]))
        self.assertTrue(np.isnan(clean.borrow_change.iloc[21]))
        self.assertEqual(clean.borrow_change.iloc[22],0.)


if __name__=='__main__':unittest.main()
