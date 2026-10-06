import unittest
import numpy as np
import pandas as pd

from scripts.run_tail_put25_residual_rl import context_mask, moments, score_choices, fit_allocations
from scripts.tail_put25_rule_sizing import SIZE_CODES


class ResidualAllocationTest(unittest.TestCase):
    def test_context_uses_original_history_then_exact_evaluation_calendar(self):
        dates = pd.to_datetime(['2017-12-01','2017-12-04','2018-01-02','2018-01-03'])
        pre = dates[[0, 2, 3]]; post = dates[[3]]
        np.testing.assert_array_equal(context_mask(dates, pre, post), [True, False, False, True])

    def test_robust_full_size_is_zero_relative_reward(self):
        rng = np.random.default_rng(10); book = rng.normal(.01, .1, size=(1000, 10))
        dates = pd.bdate_range('2013-01-02', periods=1000)
        stats = moments(book, dates, True)
        self.assertGreaterEqual(len(stats), 2)
        self.assertAlmostEqual(float(score_choices([[4,4]], stats, True)[0]), 0., places=12)

    def test_floor_policy_never_returns_flat_and_includes_baseline(self):
        rng = np.random.default_rng(12); n = 800
        book = rng.normal(.01, .05, size=(n, 10)); dates = pd.bdate_range('2013-01-02', periods=n)
        policy, detail = fit_allocations(book, dates, None, np.array([1, 2]))
        self.assertTrue(np.isin(policy[2], SIZE_CODES[2:]).all())
        self.assertGreaterEqual(detail['training_shadow_daily_sharpe'], detail['baseline_shadow_daily_sharpe']-1e-12)

    def test_robust_reward_penalizes_fold_instability(self):
        # One candidate beats full size by a stable amount; another changes sign.
        cov = np.eye(5)
        a = np.array([0., 0., .2, .15, .1]); b = np.array([0., 0., 0., .15, .1])
        score = score_choices([[2],[3],[4]], [(a,cov),(b,cov)], True)
        self.assertGreater(score[1], score[0]); self.assertGreater(score[1], score[2])


if __name__ == '__main__': unittest.main()
