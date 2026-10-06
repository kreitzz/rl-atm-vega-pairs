import unittest
import numpy as np
import pandas as pd

from scripts.tail_put25_pooled import state, FEATURES, rule_codes, completed_mask, aggregate_chunk, partition, optimize_book
from scripts.tail_put25_rule_sizing import sizing_labels, SIZE_CODES
from scripts.tail_put25_policy import CODES
from scripts.curve_trading.policy import fit


class PooledTailTest(unittest.TestCase):
    def fixture(self):
        cal = pd.bdate_range('2014-01-01', periods=350)
        a = pd.DataFrame(dict(date=cal, bs_iv=.3 + np.arange(len(cal)) * .0001,
            expiry=cal + pd.Timedelta(days=30), bs_vega=.1))
        b = a.copy(); b['bs_iv'] = .35
        returns = pd.DataFrame({0: .01, 1: .012}, index=cal)
        screen = pd.DataFrame(dict(beta=1., correlation=.9, eligible=True), index=cal)
        memory = pd.DataFrame({f'{p}_{s}_{d}d': 0. for p in ['profit', 'count']
            for s in ['short', 'long'] for d in [91, 365]}, index=cal)
        return cal, a, b, returns, screen, memory

    def test_missing_optional_features_do_not_block_pairs(self):
        cal, a, b, returns, screen, memory = self.fixture()
        returns[:] = np.nan
        panel = state(a, b, returns, screen, cal, memory, pd.Series(np.nan, index=cal), .65)
        self.assertEqual(panel.observed.sum(), len(cal))
        self.assertTrue(np.isfinite(panel.loc[panel.observed, FEATURES]).all().all())
        self.assertEqual(panel.rv_mean_available.sum(), 0)
        self.assertEqual(panel.vix_available.sum(), 0)
        self.assertGreater(rule_codes(panel).ne(0).sum(), 0)

    def test_future_quotes_cannot_change_earlier_states(self):
        cal, a, b, returns, screen, memory = self.fixture()
        x = state(a, b, returns, screen, cal, memory, pd.Series(.2, index=cal), .65)
        a.loc[300:, 'bs_iv'] = 4.
        y = state(a, b, returns, screen, cal, memory, pd.Series(.2, index=cal), .65)
        pd.testing.assert_frame_equal(x.iloc[:300], y.iloc[:300])

    def test_completed_outcomes_purged_at_fit_cutoff(self):
        dates = pd.to_datetime(['2017-01-01', '2017-12-01', '2018-01-01'])
        ends = pd.to_datetime(['2017-02-01', '2018-01-01', '2018-02-01'])
        np.testing.assert_array_equal(completed_mask(dates, ends, '2018-01-01'), [True, False, False])

    def test_chunked_cross_pair_book_matches_reference_optimizer(self):
        rng = np.random.default_rng(11)
        x = pd.DataFrame(rng.normal(size=(600, 3)))
        raw = rng.normal(.01, .1, size=(600, 9, 7)).astype(np.float32)
        rule = np.where(np.arange(600) % 2, CODES[1], CODES[-1])
        labels = sizing_labels(raw, rule)
        positions = np.arange(600) % 300
        ends = positions + 7
        baseline, score = fit(x, labels, positions, ends, SIZE_CODES)
        tree, leaves = partition(x, labels[:, -1].sum(axis=1) / 2)
        states = np.searchsorted(leaves, tree.apply(x))
        first = positions.min() + 1
        book = np.zeros((ends.max() - first + 1, len(leaves) * 5))
        for rows in [np.arange(0, 600, 2), np.arange(1, 600, 2)]:
            aggregate_chunk(book, states[rows], labels[rows], positions[rows], first)
        model, details = optimize_book(book, leaves, tree)
        np.testing.assert_array_equal(model[2], baseline[2])
        self.assertAlmostEqual(details['training_shadow_daily_sharpe'], score['training_shadow_daily_sharpe'], places=9)


if __name__ == '__main__':
    unittest.main()
