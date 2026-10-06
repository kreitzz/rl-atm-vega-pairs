import unittest
import numpy as np
import pandas as pd

from scripts.contract_engine import Settings
from scripts.contract_q_engine import ContractMarket, WAIT, LONG, SHORT, EXIT, evaluate, FEATURE_NAMES
from scripts.run_contract_q import build_graph, FittedQ, summarize


def fixture(n=12):
    dates=pd.bdate_range('2025-01-02',periods=n)
    quotes,stocks=[],[]
    for date in dates:
        for sid in (1,2):
            quotes.append(dict(date=date,secid=sid,optionid=100+sid,expiry=pd.Timestamp('2025-02-14'),
                cp_flag='C',strike=100.,bid=4.9,ask=5.1,delta=.5,vega=.1,multiplier=100.,
                cfadj=1.,standard=True,open_interest=1000.,known_open_interest=1000.,
                impl_volatility=.5 if sid==1 else .4))
            stocks.append(dict(date=date,secid=sid,close=100.,cfadj=1.,rate=0.,rv=35.,corr=.6))
    cfg=Settings(capital=100000,entry_vega_per_leg=10,max_gross_vega=100,
                 holding_days=5,stock_borrow_rate=0,borrowing_spread=0)
    return pd.DataFrame(quotes),pd.DataFrame(stocks),dates,cfg


def enter_once(m,t,book):
    return LONG if t==0 else WAIT


class ContractQTests(unittest.TestCase):
    def test_next_session_fill_costs_integer_contracts_and_terminal_flat(self):
        q,s,d,c=fixture()
        m=ContractMarket(q,s,d,[1,2],c)
        nav,error=evaluate(m,enter_once)
        self.assertIsNone(error)
        entry=nav[nav.status.eq('entered')].iloc[0]
        self.assertEqual(entry.name,d[1])
        self.assertEqual(entry.qty_a,1)
        self.assertEqual(entry.qty_b,-1)
        self.assertEqual(entry.optionid_a,101)
        # Four option transactions: $40 spreads + $2.60 commissions; $2 stock costs.
        self.assertAlmostEqual(nav.nav.iloc[-1]-c.capital,-44.6)
        self.assertEqual(nav.position.iloc[-1],0)

    def test_delta_hedge_offsets_option_delta_move(self):
        q,s,d,c=fixture()
        c.stock_cost_bps=0
        s.loc[s.date.ge(d[2])&s.secid.eq(1),'close']=102
        q.loc[q.date.ge(d[2])&q.secid.eq(1),['bid','ask']]+=1
        nav,error=evaluate(ContractMarket(q,s,d,[1,2],c),enter_once)
        self.assertIsNone(error)
        self.assertAlmostEqual(nav.option_mark_pnl.sum(),100)
        self.assertAlmostEqual(nav.hedge_pnl.sum(),-100)
        self.assertAlmostEqual(nav.nav.iloc[-1]-c.capital,-42.6)

    def test_unfillable_entry_rejected_without_future_action_mask(self):
        q,s,d,c=fixture()
        q=q[~(q.date.eq(d[1])&q.optionid.eq(101))]
        m=ContractMarket(q,s,d,[1,2],c)
        self.assertIn(LONG,m.legal(0,None))
        nav,error=evaluate(m,enter_once)
        self.assertIsNone(error)
        self.assertEqual(nav.status.iloc[1],'entry_rejected')
        self.assertEqual(nav.nav.iloc[-1],c.capital)

    def test_missing_held_quote_is_incomplete_not_a_zero_return(self):
        q,s,d,c=fixture()
        q=q[~(q.date.eq(d[2])&q.optionid.eq(101))]
        m=ContractMarket(q,s,d,[1,2],c)
        nav,error=evaluate(m,enter_once)
        self.assertIsNotNone(error)
        self.assertEqual(len(nav),2)
        self.assertNotIn('excess_sharpe',summarize(nav,error,c))
        graph=build_graph(m)
        self.assertGreater(len(graph.errors),0)
        self.assertTrue(graph.legal[0,LONG])
        self.assertFalse(graph.resolved.all())

    def test_future_data_cannot_change_current_state_or_fill(self):
        q,s,d,c=fixture()
        a=ContractMarket(q,s,d,[1,2],c)
        q.loc[q.date.ge(d[3]),['bid','ask','impl_volatility']]*=2
        b=ContractMarket(q,s,d,[1,2],c)
        np.testing.assert_array_equal(a.features(0,None),b.features(0,None))
        at,ar,_=a.step(0,None,LONG)
        bt,br,_=b.step(0,None,LONG)
        self.assertEqual(at.optionids,bt.optionids)
        np.testing.assert_array_equal(at.quantity,bt.quantity)
        self.assertEqual(ar,br)

    def test_cash_benchmark_and_ticket_nav_recursion(self):
        q,s,d,c=fixture()
        s['rate']=.05
        m=ContractMarket(q,s,d,[1,2],c)
        nav,error=evaluate(m,lambda m,t,b: WAIT)
        self.assertIsNone(error)
        np.testing.assert_allclose(nav.nav,nav.cash_benchmark)
        self.assertTrue(nav.excess_return.eq(0).all())
        nav,error=evaluate(m,enter_once)
        self.assertIsNone(error)
        reconstructed=[c.capital]
        for r in nav.iloc[1:].itertuples():
            reconstructed.append(reconstructed[-1]*(1+r.cash_return)+r.excess_dollars)
        np.testing.assert_allclose(reconstructed,nav.nav)

    def test_terminal_liquidation_and_split_local_graph(self):
        q,s,d,c=fixture(5)
        c.holding_days=30
        m=ContractMarket(q,s,d,[1,2],c)
        nav,error=evaluate(m,enter_once)
        self.assertIsNone(error)
        self.assertEqual(nav.status.iloc[-1],'terminal_exit')
        self.assertAlmostEqual(nav.nav.iloc[-1]-c.capital,-44.6)
        g=build_graph(m)
        self.assertTrue((g.date_index[g.next_state]<len(d)).all())
        self.assertTrue((g.date_index[g.next_state[g.terminal]]==len(d)-1).all())
        self.assertEqual(g.x.shape[1],len(FEATURE_NAMES))
        model=FittedQ(iterations=2,min_leaf=2,trees=2).fit(g)
        nav,error=evaluate(m,model)
        self.assertIsNone(error)

    def test_common_expiry_and_known_liquidity(self):
        q,s,d,c=fixture()
        q.loc[q.secid.eq(2),'expiry']=pd.Timestamp('2025-02-13')
        m=ContractMarket(q,s,d,[1,2],c)
        self.assertNotIn(LONG,m.legal(0,None))
        q.loc[q.secid.eq(2),'expiry']=pd.Timestamp('2025-02-14')
        q['known_open_interest']=0.
        self.assertNotIn(LONG,ContractMarket(q,s,d,[1,2],c).legal(0,None))

    def test_known_dividend_and_missing_greeks_force_pair_exit(self):
        q,s,d,c=fixture()
        div=pd.DataFrame([dict(secid=1,declare_date=d[0],ex_date=d[3],amount=.1,ordinary=True)])
        nav,error=evaluate(ContractMarket(q,s,d,[1,2],c,div),enter_once)
        self.assertIsNone(error)
        self.assertEqual(nav.status.iloc[2],'dividend_exit')
        q.loc[q.date.eq(d[2])&q.secid.eq(1),'delta']=np.nan
        nav,error=evaluate(ContractMarket(q,s,d,[1,2],c),enter_once)
        self.assertIsNone(error)
        self.assertEqual(nav.status.iloc[2],'greeks_exit')

    def test_fill_ledger_reconstructs_financing_cash(self):
        q,s,d,c=fixture()
        nav,error=evaluate(ContractMarket(q,s,d,[1,2],c),enter_once)
        self.assertIsNone(error)
        cash=0.
        for r in nav[nav.status.isin(['entered','held','horizon_exit'])].itertuples():
            fees=(abs(r.trade_qty_a)+abs(r.trade_qty_b))*c.option_commission
            option_cash=sum(-qty*100*price for qty,price in
                            [(r.trade_qty_a,r.fill_price_a),(r.trade_qty_b,r.fill_price_b)] if qty)
            stock_cash=-r.hedge_trade_a*r.stock_close_a-r.hedge_trade_b*r.stock_close_b
            cash+=option_cash+stock_cash-fees-r.hedge_cost+r.interest-r.borrow+r.dividend_cash
            self.assertAlmostEqual(cash,r.ticket_cash)
        self.assertAlmostEqual(cash,nav.nav.iloc[-1]-c.capital)


if __name__=='__main__':
    unittest.main()
