"""Economic identity and causal Greek controls for the frozen transfer."""
import unittest

import numpy as np
import pandas as pd

from scripts.option_flex.black_scholes import option_values
from scripts.option_flex.contract_engine import MarketRow
from scripts.run_p25_call_transfer import TransferMarket, ZeroCarryMarket, period_metrics


class BudgetStub:
    def rest(self, started):
        pass


class PutStub:
    def __init__(self):
        self.dates = pd.bdate_range('2022-01-03', periods=3)
        self.rates = pd.Series(.02, index=self.dates)
        self.divs = pd.DataFrame(columns=['date'])

    def stock(self, date, sid):
        return 80., 1.

    def quote(self, date, oid):
        value, delta, _, vega = option_values(80., 100., 30/365, .02, .3, 'P')
        return MarketRow(date=date, optionid=1, secid=123, strike=100.,
                         expiry=pd.Timestamp('2022-02-04'), bs_iv=.3,
                         delta=float(delta), vega=float(vega), bid=float(value), ask=float(value),
                         bs_rate=.02, known_dividend_pv=0., bs_status='ok', cp_flag='P')

    def rate(self, date):
        return .02

    def dividend_pv(self, *args):
        return 0.


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.puts = PutStub()
        self.calls = pd.DataFrame([dict(date=d, optionid=2, put_optionid=1,
                                        put_bs_iv=iv, bid=1., ask=1.2)
                                   for d, iv in zip(self.puts.dates, [.3, np.nan, .9])])

    def test_parity_greeks_preserve_actual_call_price_and_vol_direction(self):
        market = TransferMarket(self.puts, self.calls, {2: 1}, BudgetStub(), mode='parity')
        date = self.puts.dates[0]
        call = market.quote(date, 2)
        put = self.puts.quote(date, 1)
        self.assertEqual(call.bid, 1.)
        self.assertEqual(call.ask, 1.2)
        self.assertAlmostEqual(call.delta-put.delta, 1.)
        self.assertEqual(call.vega, put.vega)

    def test_zero_carry_deep_itm_put_identity_and_no_future_iv(self):
        pmarket = ZeroCarryMarket(self.puts, self.calls, {2: 1}, BudgetStub())
        cmarket = ZeroCarryMarket(self.puts, self.calls, {2: 1}, BudgetStub(), synthetic_call=True)
        for date in self.puts.dates:
            p = pmarket.quote(date, 1)
            c = cmarket.quote(date, 2)
            self.assertGreater(c.bid, 0.)
            self.assertGreater(p.bid, 0.)
            self.assertAlmostEqual(c.bid-p.bid, 80.-100., places=12)
            self.assertAlmostEqual(c.delta-p.delta, 1., places=12)
            self.assertAlmostEqual(c.vega, p.vega, places=12)
        self.assertEqual(pmarket.quote(self.puts.dates[1], 1).bs_iv, .3)
        self.assertEqual(pmarket.quote(self.puts.dates[2], 1).bs_iv, .9)

    def test_period_pnl_preserves_carry_before_the_period(self):
        dates = pd.to_datetime(['2021-12-31', '2022-01-03', '2022-01-04'])
        nav = pd.DataFrame(dict(nav=[110., 115., 117.], cash_benchmark=[102., 103., 104.],
                                excess_return=[0., .02, .01]), index=dates)
        decisions = pd.DataFrame(dict(status=['entered'], entry_date=[dates[0]]))
        result = period_metrics(nav, decisions, '2022-01-01', '2022-12-31')
        self.assertEqual(result['profit_above_cash'], 5.)
        self.assertEqual(result['entries'], 0)


if __name__ == '__main__':
    unittest.main()
