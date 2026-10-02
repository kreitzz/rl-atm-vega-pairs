import unittest
from scripts.continue_atm_nn_cost_hurdles import common_roster


class CommonRosterTests(unittest.TestCase):
    def test_one_unresolved_window_excludes_pair_from_all_comparisons(self):
        defs=[{'pair':'a'}, {'pair':'b'}, {'pair':'c'}]; hurdles=[.03,.65,1.5]
        statuses={(m,d['pair'],f,p):{'status':'complete','cash_only':True}
                  for m in ['nn_fixed','rl_sized'] for d in defs for f in hurdles for p in ['full','recent']}
        statuses[('nn_fixed','b',.65,'recent')]={'status':'incomplete'}
        statuses[('rl_sized','c',.03,'full')]={'status':'incomplete'}
        self.assertEqual(common_roster(defs,statuses,hurdles),[defs[0]])
