import unittest
import numpy as np
import pandas as pd
from scripts.atm65_volume_oi_features import turnover,add_volume_oi
from scripts.test_wrds_pair_signals import match_states
from scripts import run_atm65_volume_oi as runner


class VolumeOITest(unittest.TestCase):
    def test_aggregate_ratio_lag_and_leg_order(self):
        dates=pd.to_datetime(['2024-01-05','2024-01-08'])
        raw=pd.DataFrame(dict(secid=[1,1,2,2],date=['2024-01-05']*4,
            cp_flag=['C','P','C','P'],volume=[10,30,10,10],open_interest=[100,300,100,100]))
        state=turnover(raw)
        legs=[match_states(dates,[sid,sid],state,'secid',['volume_oi'],7).volume_oi for sid in [1,2]]
        panel=pd.DataFrame({'original':[5.,6.]},index=dates)
        result=add_volume_oi(panel,*legs)
        np.testing.assert_allclose(result.volume_oi_gap,[0.,0.],atol=1e-15)
        np.testing.assert_array_equal(result.volume_oi_available,[0,1])
        pd.testing.assert_frame_equal(result[panel.columns],panel)
        reversed_result=add_volume_oi(panel,legs[0],legs[1]*2)
        self.assertAlmostEqual(reversed_result.volume_oi_gap.iloc[1],-.1)

    def test_missing_side_or_invalid_oi_is_unavailable(self):
        raw=pd.DataFrame(dict(secid=[1,2,2],date=['2024-01-05']*3,
            cp_flag=['C','C','P'],volume=[10,10,20],open_interest=[100,0,0]))
        self.assertTrue(turnover(raw).volume_oi.isna().all())

    def test_rl_only_and_separate_output(self):
        runner.configure()
        self.assertEqual(runner.MODELS,['rl_volume_oi'])
        self.assertNotEqual(runner.OUT,runner.EMPLOYEE)
        self.assertEqual(runner.ADDED,['volume_oi_gap','volume_oi_available'])


if __name__=='__main__':unittest.main()
