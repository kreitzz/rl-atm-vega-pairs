"""Causal timing and fitted-Q boundary checks on small synthetic histories."""
import os
for _name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'ARROW_NUM_THREADS']:
    os.environ[_name] = '1'

from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from scripts.p25_rl.policy import (FEATURES, Planner, QPolicy, SPECS,
                                   fit_policy, transitions)


class ActionTree:
    def __init__(self, values):
        self.values = np.asarray(values, float)

    def predict(self, x):
        return self.values[np.asarray(x)[:, -4:].argmax(axis=1)]


class ActionForest:
    def __init__(self, tree_values):
        self.estimators_ = [ActionTree(values) for values in tree_values]

    def predict(self, x):
        return np.mean([tree.predict(x) for tree in self.estimators_], axis=0)


class ForbiddenExecutionContext:
    def __getitem__(self, key):
        raise AssertionError('Signal-close policy read current execution state')

    def get(self, key, default=None):
        raise AssertionError('Signal-close policy read current execution state')


def planner_inputs():
    dates = pd.bdate_range('2020-01-06', periods=5)
    rows = []
    for date in dates[:2]:
        for pair, a, b in [('a_b', 1, 2), ('c_d', 3, 4)]:
            rows.append(dict(date=date, pair=pair, secid_a=a, secid_b=b,
                             iv_mean=.3, oriented_iv_gap=.05, rank_strength=.8,
                             oriented_vrp_gap=.02, profit_91d=.2, vix=20.))
    cfg = SimpleNamespace(capital=10_000_000., max_gross_vega=200_000.)
    return pd.DataFrame(rows), pd.Series(0., index=dates), cfg, dates


def empty_close(cfg):
    return dict(nav=cfg.capital, return_=0.), [], {}, {}


def training_inputs():
    dates = pd.bdate_range('2020-01-06', periods=80)
    cutoff = dates[-1] + pd.offsets.BDay(2)
    rows = []
    for j, date in enumerate(dates):
        state = (np.sin(np.arange(len(FEATURES)) + j / 7) * .2).tolist()
        nxt = (np.sin(np.arange(len(FEATURES)) + (j + 1) / 7) * .2).tolist()
        for action in [0, 2]:
            reward = float((j % 7) - 3) * action / 2
            rows.append(dict(start=date, end=date + pd.offsets.BDay(1),
                             state=state, next_state=nxt, action=action,
                             reward=reward, squared_reward=reward**2,
                             sessions=1, terminal=False))
    daily = pd.DataFrame(dict(date=dates, bps=np.tile([-1., 0., 1., 2.], 20)))
    return pd.DataFrame(rows), daily, cutoff


class PolicyTests(unittest.TestCase):
    def test_entry_ignores_execution_state_and_freezes_diversification(self):
        signals, rates, cfg, dates = planner_inputs()
        planner = Planner(signals, rates, cfg, 'diversified')
        day, positions, _, global_risk = empty_close(cfg)
        pairs = {'a_b': {'active_cohorts': 1}, 'c_d': {'active_cohorts': 0}}
        planner.observe(dates[0], day, positions, pairs, global_risk)
        # An expiry/new fill between close and execution cannot revise the
        # already recorded diversification membership.
        pairs['a_b']['active_cohorts'] = 0
        pairs['c_d']['active_cohorts'] = 3
        forbidden = ForbiddenExecutionContext()
        self.assertEqual(planner.entry({'date': dates[0], 'pair': 'a_b'}, forbidden), 0)
        self.assertEqual(planner.entry({'date': dates[0], 'pair': 'c_d'}, forbidden), 4)
        self.assertEqual(planner.actions[dates[0]], 3)
        # Subsequent completed closes may choose different memberships.
        planner.observe(dates[1], day, positions, pairs, global_risk)
        self.assertEqual(planner.entry({'date': dates[1], 'pair': 'a_b'}, forbidden), 4)
        self.assertEqual(planner.entry({'date': dates[1], 'pair': 'c_d'}, forbidden), 0)

    def test_transitions_exclude_start_include_end_and_preserve_censoring(self):
        dates = pd.bdate_range('2020-01-06', periods=10)
        values = np.arange(10, dtype=float) - 4
        planner = SimpleNamespace(
            daily=[dict(date=date, bps=float(value)) for date, value in zip(dates, values)],
            events=[dict(date=dates[j], action=2,
                         state=np.full(len(FEATURES), j).tolist()) for j in [0, 3, 6]])
        records, daily = transitions(planner)
        self.assertEqual(len(records), 3)
        for j, (start, end) in enumerate([(0, 3), (3, 6), (6, 9)]):
            segment = values[start + 1:end + 1]
            row = records.iloc[j]
            self.assertEqual(row.reward, float(segment.sum()))
            self.assertEqual(row.squared_reward, float((segment**2).sum()))
            self.assertEqual(row.sessions, len(segment))
            self.assertEqual(row.start, dates[start])
            self.assertEqual(row.end, dates[end])
        self.assertEqual(records.terminal.tolist(), [False, False, True])
        self.assertEqual(records.iloc[0].next_state, planner.events[1]['state'])
        self.assertEqual(records.iloc[-1].next_state, [0.] * len(FEATURES))
        censored = records.loc[records.end.lt(dates[4])]
        self.assertEqual(len(censored), 1)
        self.assertFalse(censored.iloc[-1].terminal)
        self.assertEqual(censored.iloc[-1].next_state, planner.events[1]['state'])
        pd.testing.assert_series_equal(daily.bps, pd.Series(values, name='bps'))

    def test_ties_retain_full_and_conservative_rule_uses_tree_std(self):
        state = np.zeros(len(FEATURES))
        tied = QPolicy(ActionForest([[1., 1., 1., 1.]] * 4),
                       list(range(len(FEATURES))), [0, 1, 2], False, {})
        self.assertEqual(tied.choose(state), 2)
        negligible = deepcopy(tied)
        negligible.model = ActionForest([[1. + 1e-9, 0., 1., 0.]] * 4)
        self.assertEqual(negligible.choose(state), 2)
        # Mean skip advantage=2, sample SD=sqrt(16/3)>2. The standard
        # error would be half that and would incorrectly choose skip here.
        forest = ActionForest([[0., -.5, 0., 0.], [0., -.5, 0., 0.],
                               [4., -.5, 0., 0.], [4., -.5, 0., 0.]])
        ordinary = QPolicy(forest, list(range(len(FEATURES))), [0, 1, 2], False, {})
        conservative = QPolicy(forest, list(range(len(FEATURES))), [0, 1, 2], True, {})
        self.assertEqual(ordinary.choose(state), 0)
        self.assertEqual(conservative.choose(state), 2)

    def test_fit_uses_only_complete_past_endpoints_and_past_reward_scale(self):
        records, daily, cutoff = training_inputs()
        spec = deepcopy(SPECS['myopic_risk'])
        expected = fit_policy(records, daily, cutoff, spec)
        crossing = dict(records.iloc[-1])
        crossing.update(start=cutoff - pd.Timedelta(days=1), end=cutoff,
                        reward=1e12, squared_reward=1e24, terminal=True,
                        state=[1e6] * len(FEATURES), next_state=[1e6] * len(FEATURES))
        future = dict(crossing, start=cutoff, end=cutoff + pd.Timedelta(days=2))
        contaminated_records = pd.concat([records, pd.DataFrame([crossing, future])], ignore_index=True)
        contaminated_daily = pd.concat([daily, pd.DataFrame([
            dict(date=cutoff, bps=1e12),
            dict(date=cutoff + pd.Timedelta(days=1), bps=-1e12)])], ignore_index=True)
        actual = fit_policy(contaminated_records, contaminated_daily, cutoff, spec)
        self.assertEqual(actual.audit['transitions'], 160)
        self.assertEqual(actual.audit['distinct_market_dates'], 80)
        self.assertEqual(actual.audit['terminal_transitions'], 0)
        self.assertLess(pd.Timestamp(actual.audit['last_transition_end']), cutoff)
        self.assertEqual(actual.audit['daily_reward_scale_bps'], daily.bps.std(ddof=1))
        self.assertEqual(actual.audit, expected.audit)
        states = np.stack(records.state.map(np.asarray))
        np.testing.assert_array_equal(actual.values(states), expected.values(states))

    def test_q_learning_can_reserve_capacity_for_a_delayed_opportunity(self):
        # Supported counterfactual trajectories of a two-step capacity MDP.
        # Skip preserves one slot for a later reward of 8; half/full spend the
        # slot immediately and obtain only .5/1, leaving later reward zero.
        # All actions at each state have >=80 observations, so the production
        # minleaf30/depth3 learner can represent this temporal distinction.
        dates = pd.bdate_range('2020-01-06', periods=161)
        flat = np.zeros(len(FEATURES))
        available = flat.copy(); available[0] = 1.
        exhausted = flat.copy(); exhausted[0] = -1.
        rows = []
        for j in range(80):
            start, next_day, end = dates[2*j:2*j+3]
            for action, immediate in [(0, 0.), (1, .5), (2, 1.)]:
                successor = available if action == 0 else exhausted
                rows.append(dict(start=start, end=next_day, action=action,
                    state=flat.tolist(), next_state=successor.tolist(),
                    reward=immediate, squared_reward=immediate**2,
                    sessions=1, terminal=False))
                for state, later in [(available, 8.), (exhausted, 0.)]:
                    rows.append(dict(start=next_day, end=end, action=action,
                        state=state.tolist(), next_state=flat.tolist(),
                        reward=later, squared_reward=later**2,
                        sessions=1, terminal=True))
        records = pd.DataFrame(rows)
        daily = pd.DataFrame(dict(date=dates, bps=np.resize([-1., 1.], len(dates))))
        cutoff = dates[-1] + pd.offsets.BDay(1)
        sequential_spec = deepcopy(SPECS['q_capacity'])
        myopic_spec = dict(sequential_spec, gamma=0.)
        sequential = fit_policy(records, daily, cutoff, sequential_spec)
        myopic = fit_policy(records, daily, cutoff, myopic_spec)
        self.assertGreaterEqual(sequential.audit['distinct_market_dates'], 60)
        self.assertGreater(sequential.audit['transitions'], 100)
        self.assertEqual(myopic.choose(flat), 2)
        self.assertEqual(sequential.choose(flat), 0)
        values = sequential.values(flat)[0]
        self.assertGreater(values[0], values[2])


if __name__ == '__main__':
    unittest.main()
