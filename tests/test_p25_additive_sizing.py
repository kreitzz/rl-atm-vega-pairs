"""Structural guarantees for smooth additive scores and their matched size head."""
import io
import unittest

import joblib
import numpy as np
import pandas as pd

from scripts.p25_additive_sizing import fit_model, tercile_size_codes


FEATURES = ["rank_strength", "correlation", "vix", "profit_365d", "count_365d", "iv_mean"]


def fixture(n=96):
    rng = np.random.default_rng(2026)
    x = rng.uniform(-1, 1, size=(n, len(FEATURES)))
    rows = pd.DataFrame(x, columns=FEATURES)
    rows["target"] = 2+x[:, 0]**2+.5*x[:, 1]+x[:, 0]*x[:, 2]-.4*x[:, 3]*x[:, 4]
    return rows


class AdditiveSizingTests(unittest.TestCase):
    def test_main_model_is_additive_and_contributions_reconcile(self):
        rows = fixture()
        model = fit_model(rows, FEATURES, kind="gam")
        anchor = rows.iloc[[0]].copy()
        a, b, both = anchor.copy(), anchor.copy(), anchor.copy()
        a["rank_strength"] = .8
        b["vix"] = -.7
        both["rank_strength"], both["vix"] = .8, -.7
        np.testing.assert_allclose(model.predict(both)-model.predict(a)-model.predict(b)+model.predict(anchor), 0, atol=1e-12)
        np.testing.assert_allclose(model.contributions(rows).sum(axis=1), model.predict(rows), atol=1e-12)
        self.assertTrue(all(term["columns"] <= 4 for term in model.metadata["groups"]))

    def test_interactions_are_orthogonal_to_their_mains_and_bivariate(self):
        rows = fixture()
        model = fit_model(rows, FEATURES, kind="gam_interactions")
        bases = model.term_bases(rows)
        for name, term in model.interactions.items():
            local = np.column_stack([np.ones(len(rows)), bases[term.first], bases[term.second]])
            np.testing.assert_allclose(local.T@bases[name]/len(rows), 0, atol=1e-10)
            self.assertLessEqual(bases[name].shape[1], 16)
            changed = rows.copy()
            unrelated = next(f for f in FEATURES if f not in [term.first, term.second])
            changed[unrelated] = 1e6
            np.testing.assert_array_equal(bases[name], model.term_bases(changed)[name])
        np.testing.assert_allclose(model.contributions(rows).sum(axis=1), model.predict(rows), atol=1e-12)

    def test_each_active_term_has_unit_group_rms(self):
        model = fit_model(fixture(), FEATURES, kind="gam_interactions")
        for basis in model.term_bases(fixture()).values():
            np.testing.assert_allclose(basis.mean(axis=0), 0, atol=1e-12)
            self.assertAlmostEqual(float(np.sqrt(np.mean(np.sum(basis**2, axis=1)))), 1.)

    def test_constant_and_missing_inputs_remain_zero_contributions(self):
        rows = fixture()
        rows["rank_strength"] = 1.
        rows["iv_mean"] = np.nan
        for kind in ["linear", "gam", "gam_interactions"]:
            model = fit_model(rows, FEATURES, kind=kind)
            changed = rows.copy()
            changed["rank_strength"], changed["iv_mean"] = 9e9, 7e8
            contributions = model.contributions(changed)
            np.testing.assert_array_equal(contributions.rank_strength, 0.)
            np.testing.assert_array_equal(contributions.iv_mean, 0.)
            self.assertTrue(np.isfinite(model.predict(changed)).all())

    def test_serialized_model_preserves_predictions_terms_and_sizes(self):
        rows = fixture()
        model = fit_model(rows, FEATURES, kind="gam_interactions")
        stream = io.BytesIO()
        joblib.dump(model, stream)
        stream.seek(0)
        recovered = joblib.load(stream)
        np.testing.assert_array_equal(model.predict(rows), recovered.predict(rows))
        np.testing.assert_array_equal(model.contributions(rows), recovered.contributions(rows))
        np.testing.assert_array_equal(model.sizes(rows), recovered.sizes(rows))

    def test_basis_and_size_edges_are_fitted_only_on_training_rows(self):
        rows = fixture()
        model = fit_model(rows, FEATURES, kind="gam")
        original_edges = model.training_score_edges.copy()
        original_predictions = model.predict(rows)
        future = rows.iloc[:3].copy()
        future.loc[:, FEATURES] = 1e6
        future["target"] = 1e20
        self.assertTrue(np.isfinite(model.predict(future)).all())
        for feature in FEATURES:
            np.testing.assert_array_equal(model.mains[feature].knots,
                                          np.unique(np.quantile(rows[feature], [0, .5, 1])))
            self.assertEqual(model.feature_ranges[feature], [rows[feature].min(), rows[feature].max()])
        np.testing.assert_array_equal(original_edges, model.training_score_edges)
        np.testing.assert_array_equal(original_predictions, model.predict(rows))

    def test_tercile_head_boundaries_and_validation(self):
        np.testing.assert_array_equal(tercile_size_codes([-1, 0, .1, 1, 2], [0, 1]), [2, 3, 3, 4, 4])
        np.testing.assert_array_equal(tercile_size_codes([0, 1], [0, 0]), [4, 4])
        rows = fixture()
        self.assertTrue(set(fit_model(rows, FEATURES, "linear").sizes(rows)).issubset({.5, .75, 1.}))
        with self.assertRaises(ValueError):
            fit_model(rows, ["target", *FEATURES])
        rows.loc[0, "target"] = np.nan
        with self.assertRaises(ValueError):
            fit_model(rows, FEATURES)


if __name__ == "__main__":
    unittest.main()
