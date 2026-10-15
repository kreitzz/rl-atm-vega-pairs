"""Causality, sparse fallback, and exact sizing-menu checks."""
import unittest
import numpy as np
import pandas as pd
from scripts import straddle_entry_refinement as study


class EntryRefinementTests(unittest.TestCase):
    def test_future_completed_rewards_cannot_change_fit(self):
        cal = pd.bdate_range('2018-01-01', periods=180)
        dates = cal[np.arange(80)]
        meta = pd.DataFrame(dict(date=dates, end=dates+pd.offsets.BDay(25),
            finite_label=True, entry_position=np.arange(80), residual_memory=np.linspace(-.4,-.1,80),
            iv=.4, vix=20.))
        paths = np.random.default_rng(13).normal(0, 1, (80, 3, 36))
        spec = dict(name='synthetic', features='core', kind='tree', leaves=2, months=36)
        cutoff=cal[95]
        original=study.fit_model(meta, paths, cutoff, spec, cal)
        paths[meta.end.ge(cutoff)]=1e9
        self.assertEqual(original, study.fit_model(meta, paths, cutoff, spec, cal))
        self.assertEqual(original['training_samples'],70)
        self.assertLess(pd.Timestamp(original['last_completed']),cutoff)

    def test_sparse_and_missing_feature_predictions_keep_proposal(self):
        rows=pd.DataFrame(dict(residual_memory=[np.nan,-.2],iv=[.4,.4],vix=[20.,20.]))
        model=dict(choices=[2], features=study.FEATURES['core'])
        np.testing.assert_array_equal(study.predict(model,rows),[2,2])
        model.update(choices=[4,3],tree=[[[0,-.1,False]],[[0,-.1,True]]])
        np.testing.assert_array_equal(study.predict(model,rows),[2,4])

    def test_path_target_is_scale_invariant_and_safe_at_zero(self):
        paths=np.random.default_rng(3).normal(size=(4,3,36));paths[0]=0
        a=study.risk_adjusted_target(paths)
        np.testing.assert_allclose(a,study.risk_adjusted_target(paths*7))
        self.assertEqual(a[0],0)
        self.assertTrue(np.isfinite(a).all())

    def test_exact_action_search_handles_identical_action_sharpes(self):
        y=np.random.default_rng(3).normal(.1,1,100)
        book=np.column_stack([.5*y,.75*y,y])
        actions,score=study.map_actions(book,np.array([40]))
        np.testing.assert_array_equal(actions,[2])
        self.assertAlmostEqual(score,study.sharpe(y))


if __name__=='__main__':unittest.main()
