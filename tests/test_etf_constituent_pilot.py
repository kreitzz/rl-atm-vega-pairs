import unittest
import numpy as np
import pandas as pd
from scripts.run_etf_constituent_pilot import definitions, eligibility


class ETFConstituentTest(unittest.TestCase):
    def test_beta_ablation_preserves_correlation_and_missing_gate(self):
        panel=pd.DataFrame({'beta':[.84,.85,1.15,1.16,2.,1.,np.nan],
                            'correlation':[.5,.5,.5,.5,.49,np.nan,.9]})
        self.assertEqual(eligibility(panel,'tight').tolist(),[False,True,True,False,False,False,False])
        self.assertEqual(eligibility(panel,'no_beta_cutoff').tolist(),[True,True,True,True,False,False,False])

    def test_frozen_roster_precedes_evaluation(self):
        roster=definitions()
        self.assertEqual(len(roster),7)
        self.assertEqual(len({d['pair'] for d in roster}),7)
        for d in roster:
            self.assertLess(pd.Timestamp(d['holdings_snapshot']),pd.Timestamp('2019-01-01'))
            self.assertEqual(min(d['screen_years']),2019)
            self.assertTrue(d['label'].startswith(('OIH/','SIL/')))
