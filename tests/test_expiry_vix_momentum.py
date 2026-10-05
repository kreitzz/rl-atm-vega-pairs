import unittest
import numpy as np
import pandas as pd
from scripts.build_short_rv_momentum import momentum_12_1
from scripts.run_expiry_vix_momentum import add_states


class ExpiryMacroStatesTests(unittest.TestCase):
    def test_prior_vix_and_skipped_month_ignore_future_changes(self):
        dates=pd.bdate_range('2017-01-02',periods=400)
        returns=pd.Series(.001,index=dates)
        original=momentum_12_1(returns)
        changed=returns.copy();changed.iloc[330:]=.5
        later=momentum_12_1(changed)
        pd.testing.assert_series_equal(original.iloc[:351],later.iloc[:351])
        self.assertAlmostEqual(original.iloc[300],100*((1.001)**231-1))
        panel=pd.DataFrame({'eligible':True},index=dates[300:350])
        vix=pd.DataFrame({'DATE':dates,'CLOSE':np.arange(len(dates),dtype=float)})
        before=add_states(panel,vix,[original,original])
        vix.loc[vix.DATE>=dates[325],'CLOSE']=999
        after=add_states(panel,vix,[original,later])
        pd.testing.assert_frame_equal(before.loc[:dates[325]],after.loc[:dates[325]])
        self.assertEqual(before.vix_previous_close.iloc[0],299)
        self.assertTrue(before.index.equals(panel.index))
        pd.testing.assert_frame_equal(before[panel.columns],panel)

    def test_missing_momentum_is_flagged_without_dropping_dates(self):
        dates=pd.bdate_range('2020-01-01',periods=5)
        panel=pd.DataFrame({'eligible':True},index=dates[1:])
        vix=pd.DataFrame({'DATE':dates,'CLOSE':20.})
        momentum=pd.Series(np.nan,index=dates)
        result=add_states(panel,vix,[momentum,momentum])
        self.assertTrue(result.index.equals(panel.index))
        pd.testing.assert_frame_equal(result[panel.columns],panel)
        self.assertTrue(result.momentum_12_1_a.eq(0).all())
        self.assertTrue(result.momentum_12_1_a_available.eq(0).all())


if __name__=='__main__':unittest.main()
