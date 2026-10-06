import unittest

import numpy as np
import pandas as pd

from scripts.run_year_option_sharpe_policy import observed_prefixes,select_policy,state


class SharpePolicyTest(unittest.TestCase):
    def test_open_contract_can_contribute_only_its_observed_prefix(self):
        ep = dict(date=pd.Timestamp('2020-01-01'),dates=pd.bdate_range('2020-01-01',periods=260))
        cutoff = ep['dates'][30]
        observed = observed_prefixes([ep],cutoff,48)
        self.assertEqual(len(observed),1)
        self.assertEqual(observed[0][1].tolist(),list(range(30)))
        self.assertTrue((ep['dates'][observed[0][1]]<cutoff).all())

    def test_insufficient_observed_holding_days_are_not_training_labels(self):
        ep = dict(date=pd.Timestamp('2020-01-01'),dates=pd.bdate_range('2020-01-01',periods=260))
        self.assertEqual(observed_prefixes([ep],ep['dates'][20],48),[])

    def test_constant_leverage_tie_uses_smaller_budget(self):
        book = np.array([[1.],[-.5],[2.],[.1]])
        choice,_ = select_policy(book,np.array([[2.],[3.],[4.]]))
        np.testing.assert_array_equal(choice,[2])

    def test_policy_optimizes_portfolio_sharpe_rather_than_total_pnl(self):
        book = np.array([[1.,10.],[1.,-9.],[1.,10.],[1.,-9.]])
        grid = np.array([[2.,2.],[4.,2.],[2.,4.],[4.,4.]])
        choice,_ = select_policy(book,grid)
        np.testing.assert_array_equal(choice,[4,2])

    def test_pair_encoding_uses_known_identity_and_keeps_native_features(self):
        ep = dict(entry_features=np.arange(8),pair='AMD/NVDA')
        encoded = state(ep,['AMD/NVDA','UAL/AAL'],True)
        np.testing.assert_array_equal(encoded,np.r_[np.arange(8),1.,0.])


if __name__=='__main__':
    unittest.main()
