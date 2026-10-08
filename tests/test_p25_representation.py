"""Leakage, information-preservation and allocation-search checks."""
import unittest
import numpy as np
import pandas as pd
from scripts.diagnose_p25_pca import CORE, training_window
from scripts.p25_representation import fit_representation, transform, fit_menu


class RepresentationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(31)
        self.inputs = CORE + ['economic4', 'economic4_available']
        self.rows = pd.DataFrame(rng.normal(size=(40, 4)), columns=self.inputs[:-1])
        self.rows['economic4_available'] = 1.

    def test_all_components_reconstruct_training_and_unseen_inputs(self):
        rep = fit_representation(self.rows, self.inputs, 'pca_all')
        unseen = self.rows.iloc[:5].copy()
        unseen[CORE] += 10.
        for frame in [self.rows, unseen]:
            scores = transform(frame, rep)[:, :4]
            reconstructed = (scores @ rep['components']) * rep['scale'] + rep['mean']
            np.testing.assert_allclose(reconstructed, frame[rep['columns']], atol=1e-12)

    def test_unfinished_and_future_rows_cannot_change_pca(self):
        frame = self.rows.copy()
        frame['date'] = pd.Timestamp('2021-07-01')
        frame['end'] = pd.Timestamp('2021-08-01')
        frame['signal_eligible'] = frame['finite_label'] = True
        frame.loc[0, 'end'] = pd.Timestamp('2022-01-02')
        frame.loc[1, 'date'] = pd.Timestamp('2022-01-01')
        before = fit_representation(training_window(frame, '2022-01-01'), self.inputs, 'pca3')
        frame.loc[[0, 1], CORE] = 1e9
        after = fit_representation(training_window(frame, '2022-01-01'), self.inputs, 'pca3')
        np.testing.assert_array_equal(before['components'], after['components'])
        np.testing.assert_array_equal(before['mean'], after['mean'])

    def test_constant_inputs_and_flags(self):
        frame = self.rows.copy()
        frame['economic4'] = 0.
        rep = fit_representation(frame, self.inputs, 'pca80')
        result = transform(frame, rep)
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(result[:, -1], 1.)
        self.assertGreaterEqual(rep['variance_retained'], .8)
        core = fit_representation(frame, self.inputs, 'core3')
        np.testing.assert_array_equal(transform(frame, core)[:, :3], frame[CORE])

    def test_search_finds_known_best_action(self):
        # With one leaf and a proportional return path, action3 has strictly
        # greater Sharpe through an added constant drift.
        rng = np.random.default_rng(7)
        book = np.zeros((300, 5))
        noise = rng.normal(size=300)
        book[:, 2] = noise + .01
        book[:, 3] = noise + .1
        book[:, 4] = noise + .02
        choices, details = fit_menu(book)
        np.testing.assert_array_equal(choices, [3])
        self.assertGreater(details['training_shadow_daily_sharpe'], details['baseline_shadow_daily_sharpe'])


if __name__ == '__main__':
    unittest.main()
