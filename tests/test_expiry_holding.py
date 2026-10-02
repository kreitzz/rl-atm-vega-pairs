import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from scripts.contract_engine import Settings, simulate, UnresolvedPosition
from scripts.run_selected_pair_backtest import audit


def fixture():
    dates = pd.bdate_range('2025-01-02', '2025-02-10')
    quotes, stocks = [], []
    for day in dates:
        for sid in [1, 2]:
            expiry = pd.Timestamp('2025-02-01' if sid == 1 else '2025-02-07')
            stocks.append(dict(date=day, secid=sid, close=102. if sid == 1 else 100., cfadj=1., rate=0.))
            # Deliberately no quotes on either settlement date.
            if day < (pd.Timestamp('2025-01-31') if sid == 1 else expiry):
                quotes.append(dict(date=day, secid=sid, optionid=sid, expiry=expiry,
                    cp_flag='C', strike=100., bid=4.9, ask=5.1, delta=.5, vega=.1,
                    multiplier=100., cfadj=1., standard=True, open_interest=1000,
                    bs_iv=.4, bs_rate=0., known_dividend_pv=0.))
    cfg = Settings(hold_to_expiry=True, entry_vega_per_leg=10, max_gross_vega=100,
        holding_days=3, stock_cost_bps=0, option_spread_fraction=0,
        option_commission=.65, stock_borrow_rate=0)
    return pd.DataFrame(quotes), pd.DataFrame(stocks), pd.Series([1.], index=dates[:1]), dates, cfg


class ExpiryHoldingTests(unittest.TestCase):
    def test_independent_settlement_without_expiry_quotes(self):
        q,s,a,d,c = fixture()
        nav,t,dec = simulate(q,s,a,d,[1,2],c)
        exits=t[t.reason.eq('expiry_settlement')].set_index('secid')
        self.assertEqual(exits.loc[1,'date'], pd.Timestamp('2025-01-31'))
        self.assertEqual(exits.loc[2,'date'], pd.Timestamp('2025-02-07'))
        self.assertEqual(exits.loc[1,'price'], 2.)
        self.assertEqual(exits.loc[2,'price'], 0.)
        self.assertEqual(exits.commission.sum(), 0.)
        self.assertAlmostEqual(nav.nav.iloc[-1]-c.capital, 200-1.3)
        self.assertEqual(nav.loc['2025-01-31','hedge_1'], 0)
        self.assertNotEqual(nav.loc['2025-01-31','hedge_2'], 0)
        audit(nav,t,s,c,[1,2])

    def test_missing_greeks_use_past_iv_and_dividend_does_not_close(self):
        q,s,a,d,c=fixture()
        q.loc[q.date.eq(d[3]), ['delta','vega','bs_iv']]=np.nan
        div=pd.DataFrame([dict(secid=1, ex_date=d[4], declare_date=d[0], amount=.1)])
        nav,t,_=simulate(q,s,a,d,[1,2],c,div)
        self.assertEqual(nav.greek_fallbacks.sum(),2)
        self.assertEqual(set(t.reason),{'entry','expiry_settlement'})
        self.assertNotEqual(nav.dividend_cash.sum(),0)
        audit(nav,t,s,c,[1,2])

    def test_insufficient_runoff_blocks_entry(self):
        q,s,a,d,c=fixture()
        _,t,dec=simulate(q,s,a,d[:-3],[1,2],c)
        self.assertTrue(t.empty)
        self.assertEqual(dec.status.iloc[0],'insufficient_runoff')

    def test_interim_marks_do_not_change_terminal_cash_with_same_hedges(self):
        q,s,a,d,c=fixture()
        before,_,_=simulate(q,s,a,d,[1,2],c)
        q.loc[q.date.eq(d[5]) & q.secid.eq(1),['bid','ask']]+=20
        after,_,_=simulate(q,s,a,d,[1,2],c)
        self.assertNotEqual(before.loc[d[5],'nav'],after.loc[d[5],'nav'])
        self.assertAlmostEqual(before.nav.iloc[-1],after.nav.iloc[-1])

    def test_unsupported_distribution_remains_unresolved(self):
        q,s,a,d,c=fixture()
        q.loc[q.date.eq(d[3]),'bs_status']='unsupported_distribution'
        with self.assertRaisesRegex(UnresolvedPosition,'unsupported distribution'):
            simulate(q,s,a,d,[1,2],c)

    def test_risk_breach_blocks_entry_without_liquidating(self):
        q,s,a,d,c=fixture(); c.risk_controls=True; c.entry_vega_per_leg=11
        a=pd.Series([1.,1.],index=d[:2])
        def breaches(metrics,nav,cfg):
            # The first entry is feasible; a subsequent mark triggers a limit.
            return ['stress_loss'] if metrics['gross_vega'] and calls[0]>1 else []
        calls=[0]
        from scripts.contract_risk import book_risk
        def book(*args,**kwargs):
            if args[0]:calls[0]+=1
            return book_risk(*args,**kwargs)
        with patch('scripts.contract_risk.risk_breaches',side_effect=breaches), patch('scripts.contract_risk.book_risk',side_effect=book):
            nav,t,dec=simulate(q,s,a,d,[1,2],c)
        self.assertEqual(dec.status.tolist(),['entered','risk_deleveraging'])
        self.assertEqual(set(t.reason),{'entry','expiry_settlement'})
        self.assertTrue(nav.post_trade_risk_breach.any())
        audit(nav,t,s,c,[1,2])
