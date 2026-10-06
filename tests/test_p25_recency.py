import unittest
import numpy as np
import pandas as pd
from scripts.p25_recency import completed_months, recency_weights, weighted_moments, fit_recency
from scripts.p25_action_windows import fit_menu


class RecencyTest(unittest.TestCase):
    def test_equal_weights_reproduce_unweighted_mean_and_covariance(self):
        book = np.random.default_rng(3).normal(size=(300, 15))
        mean, cov = weighted_moments(book, np.ones(len(book)))[0]
        np.testing.assert_allclose(mean, book.mean(axis=0), atol=1e-14)
        np.testing.assert_allclose(cov, np.cov(book, rowvar=False), atol=1e-14)
        self.assertGreaterEqual(np.linalg.eigvalsh(cov).min(), -1e-12)

    def test_weights_are_causal_and_decay_by_declared_half_life(self):
        cutoff = pd.Timestamp('2023-01-01')
        dates = [cutoff-pd.Timedelta(days=365.25/2), cutoff-pd.Timedelta(days=365.25)]
        np.testing.assert_allclose(recency_weights(dates, cutoff, 6), [.5, .25])
        with self.assertRaises(ValueError): recency_weights([cutoff], cutoff, 6)
        with self.assertRaises(ValueError): recency_weights([cutoff+pd.Timedelta(days=1)], cutoff, 6)

    def test_month_cutoff_excludes_unfinished_and_out_of_window_cohorts(self):
        dates = pd.to_datetime(['2022-06-30','2022-07-01','2022-12-01','2023-01-01'])
        ends = pd.to_datetime(['2022-08-01','2022-08-01','2023-01-01','2023-02-01'])
        np.testing.assert_array_equal(completed_months(dates, ends, '2023-01-01', 6), [False,True,False,False])

    def test_uniform_optimizer_identity_and_higher_floor(self):
        book = np.random.default_rng(4).normal(.02, 1, (300, 15))
        dates = pd.bdate_range('2020-01-01', periods=300); cutoff = dates[-1]+pd.Timedelta(days=1)
        old, a = fit_menu(book, dates, [2,3,4])
        new, b = fit_recency(book, dates, [2,3,4], cutoff)
        np.testing.assert_array_equal(old,new); self.assertEqual(a,b)
        levels, log = fit_recency(book, dates, [3,4], cutoff, 6)
        self.assertTrue(np.isin(levels,[3,4]).all())
        self.assertGreater(log['effective_daily_observations'],1)


if __name__ == '__main__': unittest.main()
