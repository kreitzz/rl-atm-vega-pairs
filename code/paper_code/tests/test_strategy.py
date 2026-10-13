"""Behavioral checks for timing, accounting and policy optimization."""
import unittest

import numpy as np
import pandas as pd

from paper_code.contracts import MarketRow, Settings, integer_size
from paper_code.ledger import simulate
from paper_code.metrics import mark_benchmark, verify_ledger
from paper_code.policy import completed_window, search_sizes
from paper_code.pricing import option_values
from paper_code.signals import profitability_memory, qualify


class SyntheticMarket:
    """Small two-name book with dividends and overlapping option cohorts."""
    def __init__(self):
        self.calendar = pd.bdate_range("2020-01-02", periods=18)
        self.rates = pd.Series(.04, index=self.calendar)
        self.divs = pd.DataFrame({
            "secid": [1, 2], "amount": [.50, .40],
            "declare_date": [self.calendar[0]] * 2,
            "ex_date": [self.calendar[4]] * 2,
        })

    def stock(self, date, sid):
        return 100. - ([.50, .40][sid - 1] if date >= self.calendar[4] else 0.), 1.

    def rate(self, date):
        return .04

    def dividend_pv(self, date, sid, expiry, rate):
        events = self.divs.loc[self.divs.secid.eq(sid) & self.divs.ex_date.gt(date)
                               & self.divs.ex_date.le(expiry)]
        return float((events.amount * np.exp(-rate * (events.ex_date - date).dt.days / 365)).sum())

    def quote(self, date, optionid):
        sid = optionid // 100
        expiry = self.calendar[10 + sid]
        if date >= expiry:
            return None
        spot, adjustment = self.stock(date, sid)
        pv = self.dividend_pv(date, sid, expiry, .04)
        price, delta, _, vega = option_values(spot - pv, 100., (expiry - date).days / 365,
                                             .04, .30, "P")
        return MarketRow(date=date, secid=sid, optionid=optionid, strike=100.,
                         expiry=expiry, cp_flag="P", multiplier=100., standard=True,
                         cfadj=adjustment, bid=float(price) - .01, ask=float(price) + .01,
                         delta=float(delta), vega=float(vega), bs_iv=.30,
                         bs_rate=.04, known_dividend_pv=pv, open_interest=1000)


class StrategyTests(unittest.TestCase):
    def test_unfinished_profit_cannot_change_memory(self):
        dates = pd.bdate_range("2020-01-01", periods=80)
        ends = dates + pd.Timedelta(days=30)
        rewards = np.ones((80, 2))
        entered = np.ones((80, 2), dtype=bool)
        before = profitability_memory(dates, ends, rewards, entered)
        cutoff = dates[40]
        rewards[ends >= cutoff] = -1e9
        after = profitability_memory(dates, ends, rewards, entered)
        pd.testing.assert_frame_equal(before.loc[:cutoff], after.loc[:cutoff])
        # The outcome finishing exactly at t is still excluded.
        self.assertEqual(before.loc[ends[0], "count_long_91d"], 0)

    def test_training_requires_completed_past_outcomes(self):
        rows = pd.DataFrame({
            "date": pd.to_datetime(["2017-10-01", "2017-12-01", "2018-01-01"]),
            "end": pd.to_datetime(["2017-11-01", "2018-01-01", "2018-02-01"]),
            "signal_eligible": True, "finite_label": True,
        })
        self.assertEqual(completed_window(rows, "2018-01-01").tolist(), [True, False, False])

    def test_future_labels_are_not_entry_inputs(self):
        rows = pd.DataFrame({"observed": [True], "eligible": [True],
                             "direction": [1], "count_91d": [21], "count_365d": [63],
                             "profit_91d": [1.], "profit_365d": [1.]})
        before = qualify(rows)
        rows["target"] = -1e12
        rows["end"] = pd.Timestamp("2099-01-01")
        pd.testing.assert_series_equal(before, qualify(rows))

    def test_search_contains_full_size_anchor_and_valid_actions(self):
        rng = np.random.default_rng(17)
        book = rng.normal(.01, .1, (150, 10))
        actions, log = search_sizes(book)
        self.assertTrue(np.isin(actions, [2, 3, 4]).all())
        self.assertGreaterEqual(log["training_shadow_daily_sharpe"],
                                log["baseline_shadow_daily_sharpe"])

    def test_integer_vega_matching_rejects_bad_imbalance(self):
        np.testing.assert_array_equal(integer_size([4.5, 3.], 450., .1), [100, 150])
        self.assertIsNone(integer_size([300., 100.], 450., .1))

    def test_shared_ledger_dividends_costs_and_repeated_entries(self):
        market = SyntheticMarket()
        signals = pd.DataFrame({
            "date": market.calendar[:2], "pair": "A/B", "wing": "P25",
            "secid_a": 1, "secid_b": 2, "optionid_a": 100, "optionid_b": 200,
            "base_level": 2, "direction": 1,
        })
        config = Settings(capital=10_000_000, max_gross_vega=200_000,
                          hold_to_expiry=True, risk_controls=True,
                          min_dte=1, max_dte=55, option_spread_fraction=0,
                          option_penalty_vol_points=.1, option_commission=.03)
        nav, trades, decisions, _ = simulate(
            market, signals, market.calendar, config, pair_capital=1_000_000,
            wing_capital=1_000_000)
        mark_benchmark(nav, market.rates, config.capital)
        verify_ledger(nav, trades, decisions, signals)
        self.assertEqual(decisions.status.tolist(), ["entered", "entered"])
        self.assertEqual(trades.loc[trades.reason.eq("entry"), "date"].min(), market.calendar[1])
        ex_date, prior = market.calendar[4], market.calendar[3]
        # Both hedges were outstanding before the ex-date. Reconstruct the
        # prior-close hedge from held contract counts and observed deltas.
        entries = trades.loc[trades.reason.eq("entry")]
        shares = {sid: -sum(row.quantity * 100 * market.quote(prior, row.optionid).delta
                           for row in entries.loc[entries.secid.eq(sid)].itertuples())
                  for sid in (1, 2)}
        self.assertAlmostEqual(nav.loc[ex_date, "dividend_cash"],
                               shares[1] * .50 + shares[2] * .40)
        self.assertAlmostEqual(nav.loc[ex_date, "hedge_gross_pnl"],
                               -shares[1] * .50 - shares[2] * .40)
        paid = trades.loc[trades.reason.eq("entry")]
        np.testing.assert_allclose(paid.commission, paid.quantity.abs() * .03)
        self.assertGreater(nav.stock_borrow_cost.sum(), 0)
        self.assertTrue(nav.cash.ge(0).all())


if __name__ == "__main__":
    unittest.main()
