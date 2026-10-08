"""Matched myopic controls change discounting without changing the game."""
import os
for _key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS','ARROW_NUM_THREADS']:
    os.environ[_key]='1'

import unittest
from scripts.p25_rl import action_q,management,temporal


class TemporalControlTests(unittest.TestCase):
    def test_controls_change_only_gamma_from_unshaped_q(self):
        parents={'entry_myopic_capacity':action_q.SPECS['action_q_capacity'],
                 'manage_myopic_capacity':management.SPECS['manage_q']}
        for case,parent in parents.items():
            with self.subTest(case=case):
                spec=temporal.SPECS[case]
                self.assertEqual(spec,dict(parent,gamma=0.))
                self.assertEqual(spec['penalty'],0.)
                self.assertNotEqual(spec['gamma'],parent['gamma'])
        self.assertEqual(temporal.CANDIDATES,[])
        self.assertEqual(temporal.KINDS,{'entry_myopic_capacity':'entry',
                                       'manage_myopic_capacity':'management'})

    def test_outputs_are_isolated_from_parent_experiments(self):
        self.assertNotEqual(temporal.OUT,action_q.OUT)
        self.assertNotEqual(temporal.OUT,management.OUT)
        self.assertNotEqual(temporal.REPORT,action_q.REPORT)
        self.assertNotEqual(temporal.REPORT,management.REPORT)
        self.assertIn('management_log.csv',temporal.FILES)


if __name__=='__main__':unittest.main()
