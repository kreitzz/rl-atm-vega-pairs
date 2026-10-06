import unittest

import numpy as np
import pandas as pd

from scripts.run_long_tenor_policy_search import (
    completed_episodes, native_features, simulate_shadow_management, daily_sharpes)


class NativeLongPolicyTest(unittest.TestCase):
    def episode(self):
        n = 12
        return dict(dates=pd.bdate_range('2020-01-02', periods=n),
            state_known=np.ones(n, bool), can_exit=np.ones(n, bool),
            split_seen=np.zeros(n, bool), hedge=np.zeros((n, 2)),
            hprev=np.zeros((n, 2)), spot=np.ones((n, 2))*100,
            gross_training=np.ones(n), exit_cost_training=np.ones(n)*2)

    def test_original_expiry_purge_and_signal_lookback(self):
        cutoff = pd.Timestamp('2022-01-01')
        episodes = [dict(date=pd.Timestamp(date), expiry=pd.Timestamp(expiry))
                    for date, expiry in [('2020-02-01', '2021-12-31'),
                                         ('2021-01-01', '2022-01-01'),
                                         ('2019-12-31', '2021-06-01')]]
        self.assertEqual(completed_episodes(episodes, cutoff, 24), episodes[:1])

    def test_decision_fills_next_day_and_pays_exit_cost(self):
        ep = self.episode()
        reward = simulate_shadow_management(ep, np.array([[0.], [1.]]), np.zeros(12, int), 1.)
        np.testing.assert_array_equal(reward[:6, 0], np.ones(6))
        self.assertEqual(reward[6, 0], -1.)
        np.testing.assert_array_equal(reward[7:, 0], np.zeros(5))
        np.testing.assert_array_equal(reward[:, 1], np.ones(12))

    def test_failed_next_quote_retains_inventory_and_retries_on_cadence(self):
        ep = self.episode()
        ep['can_exit'][6] = False
        reward = simulate_shadow_management(ep, np.array([[0.]]), np.zeros(12, int), 1.)
        np.testing.assert_array_equal(reward[:11, 0], np.ones(11))
        self.assertEqual(reward[11, 0], -1.)

    def test_trim_cannot_reexpand_and_exit_cost_only_charged_once(self):
        ep = self.episode()
        states = np.zeros(12, int)
        states[10] = 1
        reward = simulate_shadow_management(ep, np.array([[.5, 1.]]), states, 1.)
        self.assertEqual(reward[6, 0], 0.)
        np.testing.assert_array_equal(reward[7:, 0], np.full(5, .5))

    def test_current_features_do_not_use_future_rv(self):
        date = pd.Timestamp('2020-01-02')
        held = [dict(secid=sid, quantity=qty, expiry=pd.Timestamp('2020-07-01'),
                     quote=dict(bs_iv=iv, delta=delta))
                for sid, qty, iv, delta in [(1, 1, .4, -.25), (2, -1, .3, -.25)]]
        rv = {(date, 1):(.09, .1), (date, 2):(.08, .09)}
        first = native_features(date, held, rv)
        rv[(date+pd.Timedelta(days=1), 1)] = (100., 100.)
        self.assertEqual(native_features(date, held, rv), first)

    def test_no_action_shadow_reproduces_full_exposure_with_hedge_cost(self):
        ep = self.episode()
        ep['hedge'][:, 0] = 2.
        ep['hprev'][1:, 0] = 2.
        actual = simulate_shadow_management(ep, np.array([[1.]]), np.zeros(12, int), 1.)[:, 0]
        expected = ep['gross_training'].copy()
        expected[0] -= .02
        np.testing.assert_allclose(actual, expected)

    def test_sharpe_invariant_to_positive_scale(self):
        returns = np.array([1., -2., 3., 2., -1.])[:, None]
        np.testing.assert_allclose(daily_sharpes(returns), daily_sharpes(returns*5))

    def test_long_rv_fallback_is_explicit_and_keeps_features_finite(self):
        from scripts.run_long_tenor_full_universe import add_slow_features
        date = pd.Timestamp('2020-01-02')
        values = dict(iv_mean_native=.35, iv_gap_native=.1, carry20=.01, carry60=.02)
        rv = {(date, 1):(.09, .1), (date, 2):(.08, .09)}
        result = add_slow_features(values, date, [1, 2], 1, date+pd.Timedelta(days=365), rv)
        self.assertEqual(result['carry252_available'], 0.)
        self.assertTrue(np.isfinite(result['carry_forecast']))

    def test_later_long_rv_does_not_change_current_forecast(self):
        from scripts import run_long_tenor_full_universe as full
        date = pd.Timestamp('2020-01-02')
        values = dict(iv_mean_native=.35, iv_gap_native=.1, carry20=.01, carry60=.02)
        rv = {(date, 1):(.09, .1), (date, 2):(.08, .09)}
        saved = dict(full.SLOW_RV)
        try:
            full.SLOW_RV = {(date, 1):.1, (date, 2):.09}
            first = full.add_slow_features(values, date, [1, 2], 1, date+pd.Timedelta(days=365), rv)
            full.SLOW_RV[date+pd.Timedelta(days=1), 1] = 1000.
            self.assertEqual(first, full.add_slow_features(values, date, [1, 2], 1, date+pd.Timedelta(days=365), rv))
        finally:
            full.SLOW_RV = saved

    def test_nonnegative_blend_diversifies_independent_profitable_sleeves(self):
        from scripts.report_long_tenor_search import optimal_nonnegative_weights
        returns = pd.DataFrame(dict(a=[2.,0.,2.,0.,1.], b=[2.,2.,0.,0.,1.]))
        weights = optimal_nonnegative_weights(returns)
        np.testing.assert_allclose(weights, [.5,.5], atol=1e-12)
        self.assertGreater(daily_sharpes((returns@weights).to_numpy()[:,None])[0],
                           daily_sharpes(returns[['a']].to_numpy())[0])

    def test_blend_search_cannot_short_a_losing_sleeve(self):
        from scripts.report_long_tenor_search import optimal_nonnegative_weights
        returns = pd.DataFrame(dict(a=[2.,0.,2.,0.,1.], b=[0.,0.,-2.,-2.,-1.]))
        weights = optimal_nonnegative_weights(returns)
        np.testing.assert_allclose(weights, [1.,0.], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
