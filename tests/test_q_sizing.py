"""Accounting, observation and Bellman-boundary checks for isolated Q sizing."""
import unittest

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from scripts.run_q_sizing import (ACTIONS, COST, FEATURES, RISK_WEIGHTS, ZERO,
    action_values, available, choose_q, constrained_population, counterfactuals, fit_sequential,
    load_panel, paired_interval, simulate, state_features, transition, utility)
from scripts.run_first_pass import net_rewards, payoff_sharpe
from scripts.run_walk_forward import fold_data


def fixture(n=100):
    dates = pd.bdate_range("2006-01-02", periods=n)
    d = pd.DataFrame({"spread": 5., "amd_earnings_days_capped": 31.,
        "nvda_earnings_days_capped": 31., "trailing_rv_amd": 40.,
        "trailing_rv_nvda": 35., "iv_amd": 40., "iv_nvda": 35.,
        "long_reward": np.sin(np.arange(n)/5.)*3+1,
        "end": dates+pd.Timedelta(days=30)}, index=dates)
    return d


class NoLimit:
    def check(self):
        pass


class QSizingTests(unittest.TestCase):
    def test_original_terminal_formula_and_calendar_horizon(self):
        d = load_panel("data/current_dataset.csv")
        expected = (d.rv_amd-d.iv_amd)-(d.rv_nvda-d.iv_nvda)
        np.testing.assert_array_equal(d.long_reward, expected)
        self.assertTrue(((d.end-d.index).dt.days == 30).all())

    def test_features_do_not_read_future_outcomes_even_for_live_tickets(self):
        d = fixture(); date = d.index[2]
        row, entered = d.iloc[2].copy(), d.iloc[0].copy()
        before = state_features(row, date, [(entered, .5)], 5.)
        for r in (row, entered):
            r["long_reward"] = 1e9; r["rv_amd"] = -999.; r["rv_nvda"] = 999.
            r["fold"] = 2099; r["future_return"] = 1000.
        np.testing.assert_array_equal(before, state_features(row, date, [(entered, .5)], 5.))
        self.assertFalse(np.array_equal(state_features(row, date, [], 1.), state_features(row, date, [], 5.)))

    def test_budget_mask_and_illegal_action(self):
        d = fixture(); book = [(d.iloc[0], .75)]
        mask = available(book, 1.)
        np.testing.assert_array_equal(ACTIONS[mask], [-.25, 0, .25])
        with self.assertRaises(ValueError):
            transition(d, 1, book, .5, 1., RISK_WEIGHTS, 5.)

    def test_cost_paid_once_and_all_terminal_tickets_settle(self):
        d = fixture(1); d["long_reward"] = 4.
        book, reward, nxt, mask, live = transition(d, 0, [], .5, 1., RISK_WEIGHTS, 5.)
        net = .5*4.-COST*.5
        np.testing.assert_allclose(reward, [utility(net, risk, 5.) for risk in RISK_WEIGHTS])
        self.assertEqual(book, []); self.assertFalse(live)
        self.assertEqual(len(nxt), 219)

    def test_intermediate_reward_is_cost_only_before_maturity(self):
        d = fixture()
        book, reward, nxt, mask, live = transition(d, 0, [], .5, 1., RISK_WEIGHTS, 5.)
        np.testing.assert_allclose(reward, [-.25]*3)
        self.assertTrue(live); self.assertEqual(len(book), 1)

    def test_batched_counterfactuals_match_individual_transitions(self):
        d = fixture(50)
        for i in (0, 4, 22, 49):
            book = [] if i == 0 else [(d.iloc[max(0, i-3)], .75)]
            for budget in (1., 5., 10.):
                r, nx, nm, live, legal = counterfactuals(d, i, book, budget, RISK_WEIGHTS, 5.)
                for ai, action in enumerate(ACTIONS):
                    _, expected_r, expected_x, expected_mask, expected_live = transition(
                        d, i, book, action if legal[ai] else 0., budget, RISK_WEIGHTS, 5.)
                    np.testing.assert_allclose(r[:, ai], expected_r, atol=1e-6)
                    np.testing.assert_allclose(nx[ai], expected_x, atol=1e-7)
                    np.testing.assert_array_equal(nm[ai], expected_mask)
                    self.assertEqual(live[ai], expected_live)

    def test_full_replay_reconciles_risk_utility_and_capacity(self):
        d = fixture()
        decide = lambda i, x, mask, book: ACTIONS[np.flatnonzero(mask)[-1]]
        actions, audit = simulate(d, 1., decide, scale=5.)
        self.assertLessEqual(audit["peak_used"], 1.)
        self.assertTrue(audit["settled"])
        self.assertAlmostEqual(audit["net_payoff"], net_rewards(actions, d.long_reward, COST).sum())
        self.assertGreater(np.count_nonzero(actions), 1)

    def test_changing_unrealized_rewards_cannot_change_decisions(self):
        d = fixture()
        altered = d.copy(); altered["long_reward"] = -7*d.long_reward+123.
        decide = lambda i, x, mask, book: ACTIONS[np.flatnonzero(mask)[-1]]
        first, _ = simulate(d, 5., decide, scale=5.)
        second, _ = simulate(altered, 5., decide, scale=5.)
        np.testing.assert_array_equal(first, second)

    def test_constrained_cem_population_matches_independent_ticket_ledger(self):
        d = fixture(); leaves = np.zeros(len(d), dtype=int)
        policies = np.array([[1.], [-.5], [.25], [0.]])
        expected = []
        for policy in policies:
            def decide(i, x, mask, book):
                wanted = policy[0]
                free = 1.-sum(abs(a) for _, a in book)
                return np.sign(wanted)*np.floor(min(abs(wanted), free)*4+1e-6)/4
            actions, _ = simulate(d, 1., decide)
            expected.append(payoff_sharpe(net_rewards(actions, d.long_reward, COST)))
        np.testing.assert_allclose(constrained_population(d, leaves, policies, 1.), expected, atol=1e-12)

    def test_entry_action_values_recover_known_positive_direction(self):
        d = fixture(); d["long_reward"] = 3.
        tree = DecisionTreeRegressor(min_samples_leaf=20).fit(d[FEATURES], d.long_reward)
        ids = np.unique(tree.apply(d[FEATURES]))
        policy, q = action_values(d, tree, ids, 0., 1.)
        np.testing.assert_array_equal(policy, [1.])
        self.assertAlmostEqual(q[0, -1], 2.5)

    def test_bellman_terminal_has_no_future_bootstrap(self):
        # Every transition is terminal. Even repeated fitting must return the same value.
        x = np.arange(20, dtype=np.float32).reshape(-1, 1)
        rewards = np.zeros((20, 3, 9), np.float32); rewards[:, :, -1] = 3.
        nx = np.full((20, 9, 1), 1e6, np.float32)
        masks = np.ones((20, 9, 9), bool)
        live = np.zeros((20, 9), bool); legal = np.ones((20, 9), bool)
        model, _ = fit_sequential((x, rewards, nx, masks, live, legal), NoLimit(),
                                  iterations=3, trees=2, min_leaf=1)
        self.assertEqual(choose_q(model, [5.], np.ones(9, bool), 0), 8)
        self.assertAlmostEqual(model.predict([[5.]])[0, 8], 3.)
        only_flat = np.zeros(9, bool); only_flat[ZERO] = True
        self.assertEqual(choose_q(model, [5.], only_flat, 0), ZERO)

    def test_training_and_validation_outcomes_mature_before_next_split(self):
        d = load_panel("data/current_dataset.csv")
        for year in range(2010, 2026, 2):
            train, validation, test = fold_data(d, year)
            self.assertLess(train.end.max(), validation.index.min())
            self.assertLess(validation.end.max(), test.index.min())

    def test_identical_policies_have_zero_paired_difference(self):
        r = np.sin(np.arange(100))
        np.testing.assert_allclose(paired_interval(r, r, repeats=20), [0., 0.])


if __name__ == "__main__":
    unittest.main()
