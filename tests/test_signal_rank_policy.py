import unittest
import numpy as np
import pandas as pd
from scripts.run_signal_rank_policy import rank_panel, eligibility, feature_columns, VARIANTS, ACTIONS, FLAT
from scripts.option_flex_ablation import eligibility as prior_gate, VARIANTS as PRIOR

class SignalRankTests(unittest.TestCase):
    def panel(self):
        n=320
        p=pd.DataFrame(index=pd.bdate_range('2010-01-01',periods=n))
        p['spread']=np.arange(n,dtype=float)
        p['spread_reversion']=-p.spread
        p['surface_term_slope_difference']=p.spread
        p['surface_term_slope_difference_available']=1.
        p['eligible']=True; p['CC_available']=True
        for side in ['long','short']:
            for days in [91,365]:
                p[f'profit_{side}_{days}d']=-1.
                p[f'count_{side}_{days}d']=100
        return p

    def test_prefix_invariance_and_missing_warmup(self):
        p=self.panel(); a=rank_panel(p); b=rank_panel(p.iloc[:200])
        pd.testing.assert_frame_equal(a.iloc[:200],b)
        self.assertEqual(a.rank_spread.iloc[62],1.)
        self.assertEqual(a.rank_spread.iloc[61],.5)
        self.assertEqual(a.rank_spread_available.iloc[61],0.)
        self.assertAlmostEqual(a.rank_spread_reversion.iloc[-1],1/252)
        p.loc[p.index[-1],'surface_term_slope_difference_available']=0
        c=rank_panel(p)
        self.assertEqual(c.rank_surface_term_slope_difference_available.iloc[-1],0.)
        self.assertEqual(c.rank_surface_term_slope_difference.iloc[-1],.5)

    def test_gate_removal_preserves_eligibility_availability_and_action_set(self):
        p=self.panel(); p.loc[p.index[0],'eligible']=False; p.loc[p.index[1],'CC_available']=False
        spec=VARIANTS['rank_no_memory_gate']; g=eligibility(p,spec)
        long=ACTIONS.index(('CC',1.))
        self.assertFalse(g[long].iloc[0]); self.assertFalse(g[long].iloc[1]); self.assertTrue(g[long].iloc[2])
        self.assertTrue(g[FLAT].all())
        for i,(combo,size) in enumerate(ACTIONS):
            if combo!='CC': self.assertFalse(g[i].any())
        pd.testing.assert_frame_equal(eligibility(p,VARIANTS['rank_with_memory_gate']),prior_gate(p,PRIOR['cc_matched_rows']))

    def test_search_unchanged_against_prior_calls_control(self):
        from scripts.run_signal_rank_policy import fit, predict
        from scripts.option_flex_ablation import fit as old_fit, predict as old_predict
        rng=np.random.default_rng(3); n=252
        x=pd.DataFrame(rng.normal(size=(n,3)))
        labels=rng.normal(size=(n,33,4)); labels[:,FLAT]=0
        positions=np.arange(n); ends=positions+4
        a, da=fit(x,labels,positions,ends,VARIANTS['rank_no_memory_gate'])
        b, db=old_fit(x,labels,positions,ends,PRIOR['cc_matched_rows'])
        np.testing.assert_array_equal(predict(a,x),old_predict(b,x))
        self.assertEqual(da,db)

    def test_rank_features_have_no_profitability_or_outcomes(self):
        p=rank_panel(self.panel())
        for name in ['rank_with_memory_gate','rank_no_memory_gate','rank_term_no_memory_gate']:
            cols=feature_columns(p,VARIANTS[name])
            self.assertTrue(all(c.startswith('rank_') for c in cols))
            self.assertTrue(np.isfinite(p[cols]).all().all())

if __name__=='__main__': unittest.main()
