import unittest
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from scripts.run_atm65_ablation import (SHORT, LONG, confidence_codes, confidence_thresholds,
                                       full_size, tree_baseline, read_actions)

class AblationTests(unittest.TestCase):
    def test_empty_saved_actions_keep_a_datetime_index(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'actions.csv';path.write_text(',code\n')
            result=read_actions(path)
            self.assertIsInstance(result.index,pd.DatetimeIndex)
            self.assertEqual(len(result),0)

    def test_sizing_preserves_directions_and_can_choose_every_rung(self):
        values=np.array([[v,-1.] for v in [1.,3.,5.,7.]]+[[-1.,v] for v in [1.,3.,5.,7.]]+[[0.,-1.]])
        codes=confidence_codes(values,np.array([2.,4.,6.]))
        np.testing.assert_array_equal(codes,[SHORT+3,SHORT+2,SHORT+1,SHORT,SHORT+4,SHORT+5,SHORT+6,LONG,0])
        np.testing.assert_array_equal(full_size(codes),[SHORT]*4+[LONG]*4+[0])

    def test_confidence_thresholds_exclude_forbidden_training_rows(self):
        prediction=np.array([[1.,-1.],[2.,-1.],[3.,-1.],[4.,-1.],[1e9,-1.],[-2.,-1.]])
        allowed=np.ones_like(prediction,dtype=bool);allowed[4,0]=False
        np.testing.assert_allclose(confidence_thresholds(prediction,allowed),[1.75,2.5,3.25])

    def test_supervised_tree_uses_shared_leaves_but_selects_positive_expected_payoff(self):
        x=pd.DataFrame({'state':[-1.,-1.,1.,1.]})
        tree=DecisionTreeRegressor(max_leaf_nodes=2,random_state=2026).fit(x,[-1.,-1.,1.,1.])
        leaves=np.unique(tree.apply(x));labels=np.zeros((4,9,2))
        labels[:2,1,0]=5.;labels[:2,-1,0]=-3.
        labels[2:,1,0]=-1.;labels[2:,-1,0]=-2.
        result=tree_baseline((tree,leaves,np.array([LONG,LONG])),x,labels)
        self.assertIs(result[0],tree)
        np.testing.assert_array_equal(result[1],leaves)
        np.testing.assert_array_equal(result[2],[SHORT,0])

if __name__=='__main__':unittest.main()
