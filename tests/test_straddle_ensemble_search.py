import unittest
import numpy as np
import pandas as pd
from scripts.straddle_ensemble_search import combine,expert_context,quantize,CFG


class EnsembleTests(unittest.TestCase):
    def test_disagreement_averages_to_native75(self):
        a=np.array([2,4,2,4,3]);b=np.array([4,2,2,4,4])
        np.testing.assert_array_equal(combine('mean_ab',a,b,a,None),[3,3,2,4,3])
        np.testing.assert_array_equal(quantize([2.5,3.5]),[3,3])

    def test_every_declared_policy_stays_in_original_ladder(self):
        a=np.array([2,3,4]);b=np.array([4,2,3]);c=np.array([3,4,2])
        context=pd.DataFrame({f'{v}_{e}_{n}':[.1,.2,.3] for v in ['sharpe','vol'] for e in ['a','b','c'] for n in [63,126]})
        for name in CFG['cases']:self.assertTrue(np.isin(combine(name,a,b,c,context),[2,3,4]).all(),name)

    def test_performance_weighting_uses_no_future_marks(self):
        dates=pd.bdate_range('2018-01-01',periods=180)
        nav=pd.DataFrame(dict(excess_return=np.sin(np.arange(180))*.001),index=dates)
        full=expert_context(dates,dict(a=nav,b=nav,c=nav))
        past=expert_context(dates[:130],dict(a=nav.iloc[:130],b=nav.iloc[:130],c=nav.iloc[:130]))
        pd.testing.assert_frame_equal(full.iloc[:130],past)


if __name__=='__main__':unittest.main()
