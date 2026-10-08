import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k]='1'
import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scripts.p25_direct_policy import make_policy, size_codes, ShadowScorer, shadow_scorer, fit_direct
from scripts.p25_representation import state_book
TEST_TMP = Path(__file__).resolve().parents[1]/'artifacts/contracts/p25_direct_policy_tests'
TEST_TMP.mkdir(parents=True, exist_ok=True)


class DirectPolicyTests(unittest.TestCase):
    def test_calendar_direction_and_action_specific_paths(self):
        cal = pd.bdate_range('2020-01-01', periods=10)
        rows = pd.DataFrame(dict(pair=['a','a','b'], position=[0,2,1], label_row=[0,1,0],
                                 direction=[1,-1,1], end=[cal[4],cal[6],cal[5]]))
        rng = np.random.default_rng(31)
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            folder = Path(tmp)
            for pair in ['a','b']:
                (folder/pair).mkdir()
                np.save(folder/pair/'P25_labels.npy', rng.normal(size=(2,9,4)))
            scorer = shadow_scorer(rows, cal, folder)
            book = state_book(rows, np.arange(3), 3, cal, folder)
            choices = np.array([[2,3,4],[4,2,3],[3,3,3]])
            expected = np.column_stack([book[:, np.arange(3)*5+c].sum(axis=1) for c in choices])
            np.testing.assert_allclose(scorer.daily(choices), expected)
            np.testing.assert_allclose(scorer.score(choices), np.sqrt(252)*expected.mean(axis=0)/expected.std(axis=0,ddof=1))

    def test_train_only_basis_additivity_and_serialization(self):
        rows = pd.DataFrame({'x': np.linspace(-2,2,30), 'constant': np.ones(30)})
        for kind in ['linear','gam']:
            model = make_policy(rows, ['x','constant'], kind)
            model.coefficients = np.linspace(-1,1,len(model.coefficients))
            before = model.mains['x'].knots.copy()
            evaluation = pd.DataFrame({'x':[-100.,np.nan,100.], 'constant':[4.,1.,6.]})
            np.testing.assert_allclose(model.contributions(evaluation).sum(axis=1), model.predict(evaluation))
            np.testing.assert_array_equal(model.mains['x'].knots,before)
            np.testing.assert_array_equal(model.contributions(evaluation)['constant'],0.)
            with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
                p=Path(tmp)/'model.joblib';joblib.dump(model,p)
                np.testing.assert_array_equal(joblib.load(p).size_codes(evaluation),model.size_codes(evaluation))

    def test_exact_budget_reproducibility_and_fullsize_anchor(self):
        rows=pd.DataFrame({'x':np.linspace(-1,1,12)})
        scorer=ShadowScorer(csr_matrix(np.random.default_rng(14).normal(.04,1,(40,36))),12)
        results=[]
        for _ in range(2):
            model,info=fit_direct(make_policy(rows,['x'],'linear'),rows,scorer,iterations=5,population=20)
            self.assertEqual(info['evaluations'],100)
            self.assertGreaterEqual(info['training_shadow_daily_sharpe'],info['baseline_shadow_daily_sharpe'])
            self.assertTrue(np.all(np.diff([r['best_score'] for r in info['trace']])>=0))
            results.append(model.coefficients)
        np.testing.assert_array_equal(*results)

    def test_boundaries_and_invalid_actions(self):
        np.testing.assert_array_equal(size_codes([-1,-.5,0,.5,1]),[2,3,3,4,4])
        with self.assertRaises(ValueError):size_codes([np.nan])
        with self.assertRaises(ValueError):ShadowScorer(csr_matrix(np.ones((3,6))),2).score([[0,4]])


if __name__=='__main__':unittest.main()
