"""Leakage boundaries and geometry of the exploratory PCA diagnostic."""
import unittest
import numpy as np
import pandas as pd
from scripts.diagnose_p25_pca import training_window, standardized_pca, subspace_overlap


class PcaDiagnosticTest(unittest.TestCase):
    def test_future_and_unfinished_rows_cannot_change_fit(self):
        dates = pd.to_datetime(['2020-07-01', '2021-06-01', '2021-12-01', '2022-01-01', '2020-06-30'])
        ends = pd.to_datetime(['2020-08-01', '2021-07-01', '2022-01-01', '2022-02-01', '2020-08-01'])
        rows = pd.DataFrame(dict(date=dates, end=ends, signal_eligible=True, finite_label=True,
                                 a=[1., 2., 1e9, 1e10, 1e11], b=[2., 1., -1e9, -1e10, -1e11]))
        train = training_window(rows, '2022-01-01')
        self.assertEqual(list(train.index), [0, 1])
        first = standardized_pca(train[['a', 'b']])
        rows.loc[2:, ['a', 'b']] = -999.
        second = standardized_pca(training_window(rows, '2022-01-01')[['a', 'b']])
        np.testing.assert_allclose(first['weights'], second['weights'])

    def test_standardization_removes_units_and_handles_constant_flags(self):
        x = pd.DataFrame(np.random.default_rng(4).normal(size=(60, 3)), columns=list('abc'))
        x['constant'] = 1.
        first = standardized_pca(x)
        shifted = x.copy()
        shifted['a'] = shifted.a * 10000 + 300
        second = standardized_pca(shifted)
        np.testing.assert_allclose(first['ratio'], second['ratio'], atol=1e-12)
        np.testing.assert_allclose(first['weights'], second['weights'], atol=1e-12)
        self.assertFalse(first['active'][-1])
        np.testing.assert_array_equal(first['weights'][:, -1], 0.)

    def test_eigenvalues_and_loadings_reconstruct_correlation(self):
        x = pd.DataFrame(np.random.default_rng(8).normal(size=(100, 5)))
        result = standardized_pca(x)
        self.assertAlmostEqual(result['ratio'].sum(), 1.)
        np.testing.assert_allclose(result['weights'] @ result['weights'].T, np.eye(5), atol=1e-12)
        np.testing.assert_allclose(result['loadings'].T @ result['loadings'], x.corr(), atol=1e-12)

    def test_stability_ignores_sign_flips_and_component_rotations(self):
        weights = np.eye(5)
        rotated = weights.copy()
        rotated[:2] = np.array([[.6, -.8], [.8, .6]]) @ weights[:2]
        rotated[2] *= -1
        self.assertAlmostEqual(subspace_overlap(weights, rotated), 1.)


if __name__ == '__main__':
    unittest.main()
