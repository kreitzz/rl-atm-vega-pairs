import unittest
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from scripts.joint_smile_policy import optimize,predict
from scripts.tail_put25_pooled import completed_mask


class JointPolicyTest(unittest.TestCase):
    def test_joint_objective_learns_offsetting_daily_risk(self):
        rng=np.random.default_rng(18);shock=rng.normal(0,2,252)
        book=np.zeros((252,10))
        for size in range(1,5):
            book[:,size]=(shock+.2)*size/4
            book[:,5+size]=(-shock+.2+rng.normal(0,.05,252))*size/4
        levels,details=optimize(book,[4,0],['P25','P75'])
        self.assertTrue((levels>0).all())
        self.assertGreater(details['training_shadow_daily_sharpe'],details['frozen_sizes_shadow_daily_sharpe'])

    def test_predictions_use_actual_inventory_not_reference_inventory(self):
        tree=DecisionTreeRegressor(max_leaf_nodes=2).fit([[0.],[0.1],[.8],[1.]],[4,4,0,0])
        leaves=np.unique(tree.apply([[0.],[1.]]))
        model={'groups':{('P75',3):(tree,leaves,0)},'levels':np.array([4,0])}
        row={'wing':'P75','base_level':3,'pair_inventory_vega':0.}
        self.assertEqual(predict(model,row,['pair_inventory_vega'],{'pair_inventory_vega':1.}),0)
        self.assertEqual(predict(model,row,['pair_inventory_vega'],{'pair_inventory_vega':0.}),4)
        row['wing']='ATM'
        self.assertEqual(predict(model,row,['pair_inventory_vega'],{}),3)

    def test_training_requires_completed_outcomes_strictly_before_fit(self):
        dates=pd.to_datetime(['2020-01-01','2020-01-01','2014-01-01','2020-02-01'])
        ends=pd.to_datetime(['2020-01-31','2020-02-01','2014-02-01','2020-02-02'])
        self.assertEqual(completed_mask(dates,ends,'2020-02-01').tolist(),[True,False,False,False])
