"""Boundary, causal-window, and fallback checks for direct threshold rules."""
import unittest

import numpy as np
import pandas as pd

from scripts.p25_threshold_policy import (ThresholdPolicy, policy_grid,
    policy_directions, proposal_mask, select_policy)


def books_fixture():
    dates = pd.bdate_range("2019-01-01", "2022-12-30")
    signal = np.sin(np.arange(len(dates))*.37)*.001
    policies = {key: policy_grid(False)[key] for key in ["q10", "q20", "q30"]}
    books = {key: pd.Series(signal+mean, index=dates)
             for key, mean in [("q10", .0002), ("q20", .0001), ("q30", 0.)]}
    decisions = {key: pd.DataFrame(dict(entry_date=dates[::7], status="entered"))
                 for key in policies}
    return books, decisions, policies


class ThresholdPolicyTests(unittest.TestCase):
    def test_exact_rank_boundaries(self):
        frame = pd.DataFrame(dict(iv_rank=[.20, .20001, .80, .80001, np.nan, -1, 2]))
        self.assertEqual(policy_directions(frame, ThresholdPolicy("q20", .20)).tolist(),
                         [1, 0, 0, -1, 0, 0, 0])

    def test_context_vix_boundary_and_unavailable(self):
        frame = pd.DataFrame(dict(iv_rank=[.15]*5, vix=[.19999, .20, .20001, np.nan, 0.]))
        self.assertEqual(policy_directions(frame, ThresholdPolicy("vix", .10, .30)).tolist(),
                         [0, 1, 1, 1, 1])

    def test_memory_thresholds_and_future_metadata_independence(self):
        frame = pd.DataFrame(dict(iv_rank=[.1]*6, direction=[1, 1, 1, 1, 1, -1],
            profit_91d=[1, 0, 1, 1, 1, 1], count_91d=[21, 21, 20, 21, 21, 21],
            profit_365d=[1, 1, 1, 0, 1, 1], count_365d=[63, 63, 63, 63, 62, 63]))
        policy = ThresholdPolicy("q20", .20)
        expected = proposal_mask(frame, policy)
        self.assertEqual(expected.tolist(), [True, False, False, False, False, False])
        frame["target"] = np.nan
        frame["end"] = pd.NaT
        frame["finite_label"] = False
        np.testing.assert_array_equal(expected, proposal_mask(frame, policy))

    def test_postcutoff_outcomes_and_entries_cannot_change_choice(self):
        books, decisions, policies = books_fixture()
        initial = select_policy(books, decisions, policies, "2022-01-01")
        self.assertEqual(initial["policy"], "q10")
        self.assertLess(pd.Timestamp(initial["latest_return"]), pd.Timestamp("2022-01-01"))
        for key in books:
            books[key].loc["2022":] = 1e8 if key == "q30" else -1e8
            decisions[key].loc[decisions[key].entry_date >= "2022-01-01", "status"] = "rejected"
        self.assertEqual(initial, select_policy(books, decisions, policies, "2022-01-01"))

    def test_minimum_sessions_and_entries_fallback(self):
        books, decisions, policies = books_fixture()
        short = select_policy(books, decisions, policies, "2019-06-01")
        self.assertEqual((short["policy"], short["status"]), ("q20", "fallback"))
        decisions = {key: frame.iloc[:1] for key, frame in decisions.items()}
        sparse = select_policy(books, decisions, policies, "2022-01-01")
        self.assertEqual((sparse["policy"], sparse["status"]), ("q20", "fallback"))

    def test_ties_favor_baseline_then_smaller_tail(self):
        books, decisions, policies = books_fixture()
        books = {key: books["q20"].copy() for key in books}
        self.assertEqual(select_policy(books, decisions, policies, "2022-01-01")["policy"], "q20")
        # A q20 candidate without enough entries cannot win the eligible tie.
        decisions["q20"] = decisions["q20"].iloc[:0]
        self.assertEqual(select_policy(books, decisions, policies, "2022-01-01")["policy"], "q10")

    def test_fixed_selection_calendar_and_empty_window(self):
        books, decisions, policies = books_fixture()
        fixed = select_policy(books, decisions, policies, "2022-01-01",
                              lookback_months=None, start="2019-01-01")
        self.assertEqual(fixed["candidates"][0]["sessions"], 783)
        self.assertEqual(fixed["candidates"][0]["first_return"], "2019-01-02T00:00:00")
        self.assertEqual(fixed["policy"], "q10")
        empty = select_policy(books, decisions, policies, "2018-01-01")
        self.assertEqual(empty["status"], "fallback")
        self.assertIsNone(empty["latest_return"])
        books["q30"] = books["q30"].iloc[1:]
        with self.assertRaises(ValueError):
            select_policy(books, decisions, policies, "2022-01-01", lookback_months=None)


if __name__ == "__main__":
    unittest.main()
