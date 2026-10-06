import unittest
import numpy as np
import pandas as pd
from scripts.trade_quality_ledger import allocate_cost, connected_episodes, reconstruct


class TradeQualityTest(unittest.TestCase):
    def test_netted_cost_allocation_preserves_actual_cost(self):
        cost = [.5, 0., .3]
        weights = np.array([[10., 20.], [10., 10.], [0., 10.]])
        x = allocate_cost(cost, weights)
        np.testing.assert_allclose(x.sum(axis=1), cost)
        self.assertAlmostEqual(x[0, 0], .5/3)
        self.assertEqual(x[1].sum(), 0.)
        self.assertEqual(x[2, 0], 0.)
        with self.assertRaises(AssertionError): allocate_cost([1.], [[0., 0.]])

    def test_episodes_join_transitively_without_merging_names(self):
        f = pd.DataFrame(dict(pair=['a', 'a', 'a', 'a', 'b'],
            entry_date=pd.to_datetime(['2020-01-01', '2020-01-10', '2020-01-25', '2020-03-01', '2020-01-01']),
            exit_date=pd.to_datetime(['2020-01-20', '2020-02-01', '2020-02-20', '2020-03-10', '2020-01-20'])))
        self.assertEqual(connected_episodes(f).tolist(), ['a#1', 'a#1', 'a#1', 'a#2', 'b#1'])

    def test_actual_overlapping_inventory_fallback_and_split_reconcile(self):
        from dataclasses import asdict, replace
        from tests.test_expiry_holding import fixture
        from scripts.option_flex.contract_engine import simulate
        from scripts.run_aligned_contract_learning import mark_book
        for split in [False, True]:
            with self.subTest(split=split):
                q, s, _, dates, cfg = fixture()
                cfg = replace(cfg, entry_vega_per_leg=20, max_gross_vega=1000, stock_cost_bps=1,
                    stock_borrow_rate=.003, option_penalty_vol_points=.1, support_integer_splits=split)
                s['rate'] = .04; q['bs_rate'] = .04
                if split:
                    use = q.date.ge(dates[5]) & q.secid.eq(1)
                    q.loc[use, ['bid', 'ask', 'strike', 'vega']] /= 2; q.loc[use, 'cfadj'] = 2
                    use = s.date.ge(dates[5]) & s.secid.eq(1)
                    s.loc[use, 'close'] /= 2; s.loc[use, 'cfadj'] = 2
                else: q.loc[q.date.eq(dates[3]), ['delta', 'vega', 'bs_iv']] = np.nan
                dividends = pd.DataFrame([dict(secid=1, ex_date=dates[4], declare_date=dates[0], amount=.1)])
                nav, trades, _ = simulate(q, s, pd.Series([1., -1.], index=dates[:2]), dates, [1, 2], cfg, dividends)
                mark_book(nav, pd.Series(.04, index=dates), cfg.capital)
                trades['cohort'] = pd.to_datetime(trades.cohort)
                legs, daily, financing, _, _, error = reconstruct(nav, trades, q, s, dividends, asdict(cfg))
                self.assertEqual(len(legs), 4)
                self.assertLess(error, 1e-6)
                self.assertAlmostEqual(legs.net_before_shared_financing.sum()+financing.sum(),
                    nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1], places=6)


if __name__ == '__main__': unittest.main()
