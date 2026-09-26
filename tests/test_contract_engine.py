import unittest
import numpy as np
import pandas as pd

from scripts.contract_engine import Settings, UnresolvedPosition, choose_contract, integer_size, simulate
from scripts.run_contract_pilot import normalize_quotes


def fixture():
    dates = pd.bdate_range('2025-01-02', periods=5)
    rows, stocks = [], []
    for date in dates:
        for s in [1, 2]:
            rows.append(dict(date=date, secid=s, optionid=s, expiry=pd.Timestamp('2025-02-07'),
                             cp_flag='C', strike=100., bid=4.9, ask=5.1, delta=.5,
                             vega=.1, multiplier=100., cfadj=1., standard=True, open_interest=1000))
            stocks.append(dict(date=date, secid=s, close=100., cfadj=1., rate=0.))
    signals = pd.Series([1.], index=dates[:1])
    cfg = Settings(entry_vega_per_leg=10, max_gross_vega=100, holding_days=3,
                   stock_cost_bps=0, option_commission=0, stock_borrow_rate=0)
    return pd.DataFrame(rows), pd.DataFrame(stocks), signals, dates, cfg


class ContractEngineTests(unittest.TestCase):
    def test_units(self):
        raw = pd.DataFrame([dict(date='2025-01-02', exdate='2025-02-07', secid=1,
                                 optionid=1, strike_price=100000., vega=10., ss_flag='0')])
        q = normalize_quotes(raw).iloc[0]
        self.assertEqual(q.strike, 100)
        self.assertEqual(q.vega*100, 10)

    def test_bid_ask_roundtrip_and_cash_reconciliation(self):
        q,s,a,d,c = fixture()
        nav,t,dec = simulate(q,s,a,d,[1,2],c)
        self.assertEqual(dec.status.iloc[0], 'entered')
        self.assertEqual(t.date.iloc[0], d[1])
        # Two contracts, each crosses a 20-cent spread roundtrip.
        self.assertAlmostEqual(nav.nav.iloc[-1]-c.capital, -40)
        self.assertAlmostEqual(nav.option_execution_cost.sum(), 40)
        np.testing.assert_allclose(nav.nav, nav.cash+nav.stock_value+nav.option_value)
        self.assertEqual(nav.active_cohorts.iloc[-1], 0)

    def test_fixed_contract_and_hedge_offset(self):
        q,s,a,d,c = fixture()
        s.loc[(s.date >= d[2]) & s.secid.eq(1), 'close'] = 102
        q.loc[(q.date >= d[2]) & q.secid.eq(1), ['bid','ask']] += 1
        nav,t,_ = simulate(q,s,a,d,[1,2],c)
        self.assertAlmostEqual(nav.nav.iloc[-1]-c.capital, -40)
        self.assertEqual(set(t.optionid), {1,2})

    def test_missing_held_mark_fails(self):
        q,s,a,d,c = fixture()
        q = q[~(q.date.eq(d[2]) & q.optionid.eq(1))]
        with self.assertRaises(UnresolvedPosition):
            simulate(q,s,a,d,[1,2],c)

    def test_bad_entry_skips_both_legs(self):
        q,s,a,d,c = fixture()
        q = q[~(q.date.eq(d[1]) & q.optionid.eq(1))]
        nav,t,dec = simulate(q,s,a,d,[1,2],c)
        self.assertTrue(t.empty)
        self.assertEqual(dec.status.iloc[0], 'entry_quote_unavailable')
        self.assertEqual(nav.nav.iloc[-1], c.capital)

    def test_size_and_cap(self):
        self.assertIsNone(integer_size([10,100], 20, .1))
        self.assertIsNone(integer_size([10,18], 20, .05))
        np.testing.assert_array_equal(integer_size([10,20], 40, .1), [4,2])
        q,s,a,d,c = fixture()
        c.max_gross_vega = 19
        _,t,dec = simulate(q,s,a,d,[1,2],c)
        self.assertTrue(t.empty)
        self.assertEqual(dec.status.iloc[0], 'gross_vega_limit')

    def test_future_price_cannot_change_entry(self):
        q,s,a,d,c = fixture()
        _,t,dec = simulate(q,s,a,d,[1,2],c)
        q.loc[q.date >= d[2], ['bid','ask']] *= 2
        _,t2,dec2 = simulate(q,s,a,d,[1,2],c)
        pd.testing.assert_frame_equal(dec,dec2)
        pd.testing.assert_frame_equal(t[t.reason.eq('entry')],t2[t2.reason.eq('entry')])

    def test_adjustment_fails_when_held(self):
        q,s,a,d,c = fixture()
        q.loc[q.date.eq(d[2]) & q.secid.eq(1), 'cfadj'] = 2
        with self.assertRaises(UnresolvedPosition):
            simulate(q,s,a,d,[1,2],c)

    def test_known_dividend_closes_pair_before_exdate(self):
        q,s,a,d,c = fixture()
        events = pd.DataFrame([dict(secid=1, ex_date=d[3], declare_date=d[0], amount=1.)])
        nav,t,_ = simulate(q,s,a,d,[1,2],c,events)
        exits = t[t.reason.eq('pre_ex_dividend_exit')]
        self.assertEqual(len(exits),2)
        self.assertTrue(exits.date.eq(d[2]).all())
        self.assertEqual(nav.dividend_cash.sum(),0)

    def test_flat_cash_earns_interest(self):
        q,s,a,d,c = fixture()
        s['rate'] = .05
        a[:] = 0
        nav,_,_ = simulate(q,s,a,d,[1,2],c)
        expected = c.capital
        for gap in d.to_series().diff().dt.days.dropna():
            expected *= 1+.05*gap/365
        self.assertAlmostEqual(nav.nav.iloc[-1],expected)

    def test_crossed_quote_strict_and_explicit_sensitivity(self):
        q,s,a,d,c = fixture()
        q.loc[q.date.eq(d[2]) & q.optionid.eq(1), ['bid','ask']] = [5.1,4.9]
        with self.assertRaises(UnresolvedPosition):
            simulate(q,s,a,d,[1,2],c)
        c.repair_crossed_held_quotes = True
        nav,_,_ = simulate(q,s,a,d,[1,2],c)
        self.assertAlmostEqual(nav.nav.iloc[-1]-c.capital,-40)
        self.assertEqual(nav.repaired_held_quotes.sum(),1)

    def test_missing_greeks_liquidates_pair_at_observed_prices(self):
        q,s,a,d,c = fixture()
        q.loc[q.date.eq(d[2]) & q.optionid.eq(1), ['delta','vega']] = np.nan
        nav,t,_ = simulate(q,s,a,d,[1,2],c)
        self.assertEqual(t.reason.eq('missing_greeks_exit').sum(),2)
        self.assertAlmostEqual(nav.nav.iloc[-1]-c.capital,-40)


if __name__ == '__main__':
    unittest.main()
