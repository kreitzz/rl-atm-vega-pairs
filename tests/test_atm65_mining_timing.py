import unittest
import numpy as np
import pandas as pd
from scripts.run_atm65_signal_mining import fit_masks

class MiningTimingTests(unittest.TestCase):
    def test_entry_before_cutoff_does_not_make_its_future_outcome_available(self):
        index=pd.to_datetime(['2018-12-01','2018-12-20','2019-12-01','2019-12-20'])
        ends=pd.to_datetime(['2018-12-31','2019-01-01','2019-12-31','2020-01-01'])
        proposal,recent=fit_masks(index,ends,np.ones(4,bool),pd.Timestamp('2020-01-01'))
        np.testing.assert_array_equal(proposal,[True,False,False,False])
        np.testing.assert_array_equal(recent,[True,True,True,False])

if __name__=='__main__':unittest.main()
