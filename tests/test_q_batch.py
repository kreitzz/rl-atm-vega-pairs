import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.q_panel import HOLD
from scripts.run_q_batch import Encoder, evaluate, features, fit_q, transitions, market, slices, respect_gaps
from scripts.run_q_learning import Market, STAY, LONG, SHORT, EXIT, simulate


def panel(n=320):
    t = np.arange(n)
    return pd.DataFrame({
        'iv_amd': 40+np.sin(t/7), 'iv_nvda': 30+np.cos(t/9),
        'spread': 10+np.sin(t/7)-np.cos(t/9),
        'trailing_rv_amd': 35., 'trailing_rv_nvda': 28.,
        'amd_earnings_days_capped': 31, 'nvda_earnings_days_capped': 31,
        'lr_amd_next': .01*np.sin(t), 'lr_nvda_next': .01*np.cos(t),
        'cost': 1.+t/n, 'long_reward': np.sin(t/20),
    }, index=pd.bdate_range('2000-01-01', periods=n))


class BatchQTests(unittest.TestCase):
    def test_terminal_entry_pays_liquidation_and_has_no_bootstrap(self):
        d = panel()
        enc = Encoder(d, 'economic', True, True, 4)
        m = Market(d)
        sa, reward, nxt, live = transitions(m, enc.leaves(d), enc)
        i = 2*(len(d)-1)-1  # final row of flat LONG action block
        self.assertEqual(sa[i] % 4, LONG)
        self.assertFalse(live[i])
        t = len(d)-2
        self.assertAlmostEqual(reward[i], m.step(t, t, 1)-m.half[t]-m.half[t+1])

    def test_expiry_cost_charged_once_and_next_state_flat(self):
        d = panel()
        enc = Encoder(d, 'economic', True, False, 4)
        m = Market(d)
        sa, r, nxt, live = transitions(m, enc.leaves(d), enc)
        offset = 3*(m.n-1) + sum(2*(m.n-1-age) for age in range(1, HOLD-1))
        t = HOLD-1
        self.assertAlmostEqual(r[offset], m.step(t, 0, 1)-m.half[t+1])
        self.assertEqual(nxt[offset] % enc.width, 0)
        self.assertTrue(live[offset])

    def test_evaluation_reconciles_with_existing_simulator(self):
        d = panel()
        enc = Encoder(d, 'economic', True, False, 4)
        q = np.full((enc.nstates, 4), -np.inf)
        for s in range(enc.nstates):
            q[s, LONG if s % enc.width == 0 else STAY] = 1
        actual, inv, trades = evaluate(d, enc, q)
        expected, nt = simulate(Market(d), lambda t, pos, age, cum: LONG if not pos else STAY)
        np.testing.assert_allclose(actual, expected)
        self.assertEqual(trades, nt)
        # Explicit independent first trade accounting, including forced exit.
        m = Market(d)
        expected_total = 0.
        for e in range(0, m.n-1, HOLD):
            end = min(e+HOLD, m.n-1)
            expected_total += sum(m.step(t, e, 1) for t in range(e, end))-m.half[e]-m.half[end]
        self.assertAlmostEqual(actual.sum(), expected_total)

    def test_future_columns_are_not_policy_features(self):
        d = panel()
        altered = d.copy()
        for column in ('long_reward', 'rv_amd', 'rv_nvda', 'lr_amd_next', 'lr_nvda_next', 'end'):
            altered[column] = 999999.
        for mode in ('original', 'economic', 'earnings', 'spread'):
            pd.testing.assert_frame_equal(features(d, mode), features(altered, mode))
        enc = Encoder(d, 'economic', True, True, 4)
        np.testing.assert_array_equal(enc.leaves(d), enc.leaves(altered))

    def test_all_inventory_ages_covered_and_deterministic(self):
        d = panel()
        enc = Encoder(d, 'economic', True, False, 4)
        q1, diag = fit_q(Market(d), enc.leaves(d), enc, .9)
        q2, _ = fit_q(Market(d), enc.leaves(d), enc, .9)
        np.testing.assert_array_equal(q1, q2)
        self.assertEqual(diag['unvisited_states'], 0)
        self.assertLess(diag['residual'], 1e-9)
        self.assertEqual(enc.width, 1+2*(HOLD-1))
        self.assertFalse(np.isfinite(q1[0, EXIT]))
        self.assertFalse(np.isfinite(q1[1, LONG]))

    def test_gaps_split_training_and_evaluation_episodes(self):
        d = panel()
        d['calendar_position'] = np.r_[np.arange(160), np.arange(170, 330)]
        enc = Encoder(d, 'economic', True, False, 4)
        q = np.full((enc.nstates, 4), -np.inf)
        q[0, LONG] = 1
        q[1:, STAY] = 1
        combined = evaluate(d, enc, q)
        separate = [evaluate(d.iloc[s], enc, q) for s in slices(d)]
        np.testing.assert_allclose(combined[0], np.concatenate([x[0] for x in separate]))
        self.assertEqual(combined[2], sum(x[2] for x in separate))
        tr = transitions(market(d), enc.leaves(d), enc)
        expected = [transitions(Market(d.iloc[s]), enc.leaves(d.iloc[s]), enc) for s in slices(d)]
        for i in range(4):
            np.testing.assert_array_equal(tr[i], np.concatenate([x[i] for x in expected]))

    def test_labels_cannot_cross_a_missing_date(self):
        d = panel()
        calendar = pd.bdate_range('2000-01-01', periods=330)
        d.index = calendar[np.r_[np.arange(160), np.arange(170, 330)]]
        d['end'] = pd.Series(d.index, index=d.index).shift(-HOLD)
        with patch('scripts.run_q_batch.load_raw', return_value=(None, pd.DataFrame(index=calendar), None)):
            clean = respect_gaps(d, 'AMD', 'NVDA')
        self.assertTrue(clean.long_reward.iloc[139:160].isna().all())
        self.assertTrue(clean.end.iloc[139:160].isna().all())
        self.assertTrue(clean.long_reward.iloc[:139].notna().all())
        self.assertEqual(clean.end.iloc[138], d.index[159])


if __name__ == '__main__':
    unittest.main()
