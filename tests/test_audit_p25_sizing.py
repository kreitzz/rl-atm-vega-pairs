"""Focused safeguards for the exploratory sizing comparison."""
import unittest
import numpy as np
import pandas as pd
from scripts.audit_p25_sizing import train_simple_rules,tercile_levels
from scripts.diagnose_p25_pca import training_window


class SizingAuditTests(unittest.TestCase):
    def test_completed_window_ignores_unavailable_outcomes(self):
        dates=pd.to_datetime(['2021-06-01','2021-07-01','2021-12-01','2021-12-02','2022-01-01'])
        data=pd.DataFrame(dict(date=dates,end=pd.to_datetime(['2021-07-01','2021-08-01','2021-12-31','2022-01-01','2022-02-01']),
            signal_eligible=True,finite_label=True,rank_strength=[.1,.2,.3,.4,.5],target=[1.,2.,3.,4.,5.],flag=1.))
        columns=['rank_strength','flag']
        initial=train_simple_rules(training_window(data,'2022-01-01',months=6),columns)
        changed=data.copy()
        changed.loc[[0,3,4],['target','rank_strength']]=1e9
        other=train_simple_rules(training_window(changed,'2022-01-01',months=6),columns)
        np.testing.assert_array_equal(initial['rank_edges'],other['rank_edges'])
        np.testing.assert_array_equal(initial['ridge'].coef_,other['ridge'].coef_)
        np.testing.assert_array_equal(initial['prediction_edges'],other['prediction_edges'])

    def test_standardization_and_constant_flags(self):
        rows=pd.DataFrame(dict(rank_strength=[.1,.2,.3,.4,.5,.6],vol=[1.,4.,2.,6.,3.,8.],flag=1.,target=[2.,1.,5.,3.,9.,4.]))
        columns=['rank_strength','vol','flag']
        fitted=train_simple_rules(rows,columns)
        scaled=rows.copy();scaled['vol']=scaled.vol*100+8
        other=train_simple_rules(scaled,columns)
        np.testing.assert_allclose(fitted['ridge'].coef_,other['ridge'].coef_,atol=1e-12)
        self.assertEqual(fitted['ridge'].coef_[-1],0.)
        self.assertEqual(fitted['sd'][-1],1.)

    def test_action_bounds_and_predeclared_ties(self):
        np.testing.assert_array_equal(tercile_levels([-100,.3,.5,.7,100],[.3,.7]),[2,3,3,4,4])
        np.testing.assert_array_equal(tercile_levels([0,.5,1],[.5,.5]),[2,4,4])


if __name__=='__main__':unittest.main()
