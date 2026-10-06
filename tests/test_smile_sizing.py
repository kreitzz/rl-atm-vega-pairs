import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from scripts.smile_sizing_policy import FEATURES,features,translate,directional_labels,training_mask,fit_sizes
from scripts.curve_trading.config import COMBOS,ACTIONS
from scripts.run_smile_sizing import scoped_inputs


class SmileSizingTest(unittest.TestCase):
    def test_dividends_cannot_leak_from_other_repaired_issuer(self):
        divs=pd.DataFrame(dict(secid=[105003,127149,105003],amount=[.1,.2,.3]))
        result=(pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),divs)
        with patch('scripts.run_smile_sizing.inherited_inputs',return_value=result):
            _,_,_,fcx=scoped_inputs([], [105003])
            _,_,_,teck=scoped_inputs([], [127149])
        self.assertEqual(set(fcx.secid),{105003});self.assertAlmostEqual(fcx.amount.sum(),.4)
        self.assertEqual(set(teck.secid),{127149});self.assertAlmostEqual(teck.amount.sum(),.2)

    def test_sizing_preserves_both_types_buckets_and_direction(self):
        for cp,bucket in [('P',75),('C',25)]:
            first=1+COMBOS.index((30,cp,bucket,bucket))*8
            rule=np.array([0,first,first+7])
            np.testing.assert_array_equal(translate([4]*3,rule,cp,bucket),rule)
            for level in range(5):
                output=translate([level]*3,rule,cp,bucket)
                for old,new in zip(rule,output):
                    if not new:continue
                    combo,size=ACTIONS[new];self.assertEqual(COMBOS[combo],(30,cp,bucket,bucket))
                    self.assertEqual(np.sign(size),np.sign(ACTIONS[old][1]));self.assertEqual(abs(size),level/4)

    def test_exact_directional_size_paths_and_flat(self):
        daily=np.broadcast_to(np.arange(8)[None,:,None],(3,8,2)).copy()
        result=directional_labels(daily,[.9,.1,.5])
        np.testing.assert_array_equal(result[:,:,0],[[0,3,2,1,0],[0,4,5,6,7],[0,0,0,0,0]])

    def test_missing_optional_signals_keep_observed_state(self):
        p=pd.DataFrame(dict(observed=[True],iv_mean=[.3],wing_iv_gap=[.1],wing_iv_rank=[.1],
            vol_factor_residual_gap=[np.nan],risk_reversal_gap=[np.nan],right_steepness_gap=[np.nan],
            beta=[np.nan],correlation=[np.nan],vix=[np.nan],profit_91d=[0.],count_91d=[0.],profit_365d=[0.],count_365d=[0.]))
        x=features(p);self.assertTrue(np.isfinite(x[FEATURES]).all().all())
        self.assertEqual(x.factor_cheapness_available.iloc[0],0.)
        self.assertAlmostEqual(x.relative_iv_cheapness.iloc[0],-1/3)

    def test_incomplete_and_equal_cutoff_outcomes_are_purged(self):
        m=pd.DataFrame(dict(date=pd.to_datetime(['2017-11-01']*3),end=pd.to_datetime(['2017-12-29','2018-01-01','2018-01-02']),finite_label=True))
        np.testing.assert_array_equal(training_mask(m,'2018-01-01'),[True,False,False])

    def test_floor_keeps_full_size_when_uniform_edge_is_positive(self):
        rng=np.random.default_rng(1);base=rng.normal(.1,.2,600)
        book=base[:,None]*np.array([0,.25,.5,.75,1.])[None]
        policy,info=fit_sizes(book,None,np.array([0]),[2,3,4])
        self.assertIn(int(policy[2][0]),[2,3,4])
        self.assertGreaterEqual(info['training_shadow_daily_sharpe'],info['baseline_shadow_daily_sharpe']-1e-12)


if __name__=='__main__':unittest.main()
