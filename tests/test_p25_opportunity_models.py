"""Causality and action safeguards for contextual opportunity selection."""
import unittest

import numpy as np
import pandas as pd

from scripts.p25_opportunity_models import (MonthlyRule, best_per_pair_date,
    completed_window, fit_month, make_pipeline, tune_hurdle)


def fixture():
    dates = pd.bdate_range("2019-01-01", "2021-02-01")
    x = np.sin(np.arange(len(dates)) * .19)
    return pd.DataFrame(dict(date=dates, end=dates + pd.Timedelta(days=20),
        pair="A/B", direction=1, x=x, flag=1., missing=np.nan,
        target=.3 + .8 * x))


class OpportunityModelsTests(unittest.TestCase):
    def test_completed_window_strict_boundary(self):
        frame = pd.DataFrame(dict(date=pd.to_datetime(["2019-06-30", "2020-11-01", "2020-12-01"]),
            end=pd.to_datetime(["2019-07-05", "2020-12-31", "2021-01-01"]), target=[1., 2., 3.]))
        self.assertEqual(completed_window(frame, "2021-01-01").index.tolist(), [1])

    def test_future_and_unfinished_labels_cannot_change_fit_or_selection(self):
        rows = fixture()
        cutoff = pd.Timestamp("2021-01-01")
        changed = rows.copy()
        unavailable = (changed.end >= cutoff) | (changed.date >= cutoff)
        changed.loc[unavailable, ["target", "x"]] = 1e12
        first = fit_month(rows, ["x", "flag", "missing"], cutoff)
        other = fit_month(changed, ["x", "flag", "missing"], cutoff)
        self.assertEqual(first.metadata, other.metadata)
        np.testing.assert_allclose(first.predict_scores(rows), other.predict_scores(rows))
        self.assertTrue(first.model is not None)
        # Live prediction uses predictors alone, regardless of outcomes present.
        probe = rows.tail(10).copy()
        expected = first.select(probe).selected.to_numpy()
        probe["target"] = 1e20
        probe["end"] = pd.NaT
        np.testing.assert_array_equal(expected, first.select(probe).selected)

    def test_inner_folds_only_use_completed_training(self):
        model = fit_month(fixture(), ["x"], "2021-01-01")
        for fold in model.metadata["folds"]:
            self.assertLess(pd.Timestamp(fold["latest_training_outcome"]), pd.Timestamp(fold["start"]))
        self.assertLess(pd.Timestamp(model.metadata["latest_training_outcome"]), pd.Timestamp("2021-01-01"))

    def test_incomplete_label_flags_excluded_from_training_but_never_selection(self):
        rows = fixture()
        rows["finite_label"] = True
        rows.loc[rows.index % 10 == 0, "finite_label"] = False
        changed = rows.copy()
        changed.loc[~changed.finite_label, "target"] = 1e20
        model = fit_month(rows, ["x"], "2021-01-01")
        other = fit_month(changed, ["x"], "2021-01-01")
        self.assertEqual(model.metadata, other.metadata)
        probe = rows.tail(30).copy()
        expected = model.select(probe)
        probe[["target", "end"]] = np.nan
        probe["finite_label"] = False
        actual = model.select(probe)
        np.testing.assert_array_equal(expected.selected, actual.selected)
        np.testing.assert_array_equal(expected.prediction, actual.prediction)

    def test_opposite_directions_cannot_both_be_selected_with_duplicate_indices(self):
        rows = pd.DataFrame(dict(date=pd.to_datetime(["2021-01-01"] * 2), pair=["A/B"] * 2,
                                direction=[-1, 1], x=[.4, .8]), index=[0, 0])
        pipeline = make_pipeline("ridge", .1, 2).fit(rows[["x"]], [1., 2.])
        model = MonthlyRule(["x"], pipeline, 0., {})
        selection = model.select(rows)
        self.assertEqual(selection.selected.tolist(), [False, True])

    def test_skip_if_validation_payoffs_all_negative(self):
        dates = pd.bdate_range("2020-01-01", periods=40)
        oof = pd.DataFrame(dict(date=dates, pair="A/B", direction=1,
                               score=np.arange(40) + 1., target=-1.))
        hurdle, _ = tune_hurdle(oof, dates)
        self.assertTrue(np.isinf(hurdle))

    def test_spline_handles_constants_and_missing_columns(self):
        rows = fixture()
        model = fit_month(rows, ["x", "flag", "missing"], "2021-01-01", kind="spline")
        self.assertTrue(np.isfinite(model.predict_scores(rows)).all())

    def test_insufficient_data_and_target_predictor_are_rejected(self):
        rows = fixture().tail(10)
        rule = fit_month(rows, ["x"], "2021-03-01")
        self.assertFalse(rule.select(rows).selected.any())
        with self.assertRaises(ValueError):
            fit_month(fixture(), ["x", "target"], "2021-01-01")


if __name__ == "__main__":
    unittest.main()
