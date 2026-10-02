import unittest
import numpy as np
import pandas as pd
from scripts.contract_engine import Settings,simulate
from scripts.contract_risk import book_risk,risk_breaches
from tests.test_contract_engine import fixture


def risk_fixture():
    q,s,a,d,c=fixture()
    q['bs_iv']=.4;q['bs_rate']=0.;q['known_dividend_pv']=0.
    c.risk_controls=True
    return q,s,a,d,c


class ContractRiskTests(unittest.TestCase):
    def test_nav_changes_new_entry_sizes(self):
        q,s,a,d,c=risk_fixture()
        c.capital=1_000_000;c.entry_vega_per_leg=100;c.max_gross_vega=20000
        # Double equity with cash interest before the first entry, without any future marks.
        s['rate']=365.
        nav,trades,dec=simulate(q,s,a,d,[1,2],c)
        self.assertAlmostEqual(dec.sizing_nav.iloc[0],2_000_000.)
        self.assertEqual(trades[trades.reason.eq('entry')].quantity.abs().tolist(),[20,20])

    def test_entry_rejected_when_reserve_exceeds_post_cost_nav(self):
        q,s,a,d,c=risk_fixture()
        c.capital=100;c.max_gross_vega=100;c.max_reserve_fraction=.8
        nav,trades,dec=simulate(q,s,a,d,[1,2],c)
        self.assertTrue(trades.empty)
        self.assertEqual(dec.status.iloc[0],'risk_limit_entry')
        self.assertIn('capital_reserve',dec.risk_limits.iloc[0])

    def test_entry_budget_shrinks_after_equity_loss(self):
        q,s,a,d,c=risk_fixture()
        c.entry_vega_per_leg=100;c.max_gross_vega=20000
        s.loc[s.date.eq(d[0]),'rate']=-182.5
        nav,trades,dec=simulate(q,s,a,d,[1,2],c)
        self.assertAlmostEqual(dec.sizing_nav.iloc[0],500000.)
        self.assertEqual(trades[trades.reason.eq('entry')].quantity.abs().tolist(),[5,5])

    def test_existing_pair_is_closed_when_risk_limit_breaches(self):
        q,s,a,d,c=risk_fixture()
        c.capital=100000;c.max_gross_vega=100;c.holding_days=4
        # Same prices/deltas, but a large marked-vega jump violates the current-NAV cap.
        q.loc[q.date.ge(d[2]),'vega']=10.
        nav,trades,dec=simulate(q,s,a,d,[1,2],c)
        exits=trades[trades.reason.eq('risk_limit_exit')]
        self.assertEqual(set(exits.secid),{1,2})
        self.assertEqual(set(exits.date),{d[2]})
        self.assertEqual(nav.loc[d[2],'risk_exit_cohorts'],1)
        self.assertTrue(nav.loc[d[2],'pre_trade_risk_breach'])
        self.assertEqual(nav.loc[d[2],'active_cohorts'],0)
        self.assertTrue((nav.capital_reserve<=c.max_reserve_fraction*nav.nav+1e-8).all())
        np.testing.assert_allclose(nav.nav,nav.cash+nav.stock_value+nav.option_value)

    def test_independent_stress_does_not_cancel_equal_pair_vega(self):
        q,s,a,d,c=risk_fixture()
        rows={int(r.secid):r for _,r in q[q.date.eq(d[0])].iterrows()}
        positions=[dict(secid=1,quantity=1),dict(secid=2,quantity=-1)]
        risk=book_risk(positions,lambda p:rows[p['secid']],{1:100.,2:100.},d[0],c)
        self.assertGreater(risk['stress_loss'],0)
        self.assertEqual(risk['gross_vega'],20.)
        self.assertEqual(risk['gross_hedge_notional'],10000.)
        self.assertIn('stress_loss',risk_breaches(risk,1000,c))

    def test_margin_reserve_includes_option_premiums_and_hedges(self):
        q,s,a,d,c=risk_fixture();row=q.iloc[0]
        risk=book_risk([dict(secid=1,quantity=2)],lambda p:row,{1:100.,2:100.},d[0],c)
        self.assertAlmostEqual(risk['gross_option_premium'],1000.)
        self.assertAlmostEqual(risk['capital_reserve'],6000.)


if __name__=='__main__':unittest.main()
