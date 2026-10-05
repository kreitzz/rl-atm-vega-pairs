import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from scripts.contract_engine import Settings, execution_terms, simulate
from scripts.contract_learning_labels import ShadowLabels, LADDER
from scripts.run_selected_pair_backtest import audit
from scripts.reprice_contract_execution import reprice_cash
from tests.test_contract_engine import fixture


class MidExecutionTests(unittest.TestCase):
    def test_mid_roundtrip_charges_penalty_once_per_fill(self):
        q,s,a,d,c=fixture()
        c=replace(c,option_penalty_vol_points=.1,option_commission=.65)
        nav,trades,_=simulate(q,s,a,d,[1,2],c)
        np.testing.assert_allclose(trades.price,5.)
        np.testing.assert_allclose(trades.execution_penalty,1.)
        self.assertAlmostEqual(nav.option_model_penalty.sum(),4.)
        self.assertAlmostEqual(nav.option_execution_cost.sum(),6.6)
        self.assertAlmostEqual(nav.nav.iloc[-1]-c.capital,-6.6)
        audit(nav,trades,s,c,[1,2])

    def test_mid_zero_retains_commission(self):
        q,s,a,d,c=fixture();c=replace(c,option_penalty_vol_points=0,option_commission=.65)
        nav,trades,_=simulate(q,s,a,d,[1,2],c)
        self.assertAlmostEqual(nav.nav.iloc[-1]-c.capital,-2.6)
        self.assertEqual(trades.execution_penalty.sum(),0)

    def test_forced_exit_uses_previous_observed_vega(self):
        q,s,a,d,c=fixture();c=replace(c,option_penalty_vol_points=.1)
        q.loc[q.date.eq(d[2])&q.secid.eq(1),['vega','delta']]=np.nan
        nav,trades,_=simulate(q,s,a,d,[1,2],c)
        self.assertEqual(trades.reason.eq('missing_greeks_exit').sum(),2)
        self.assertAlmostEqual(nav.option_model_penalty.sum(),4.)
        audit(nav,trades,s,c,[1,2])

    def test_penalty_is_separate_and_does_not_make_price_negative(self):
        q,_,_,_,c=fixture();quote=q.iloc[0].to_dict();quote.update(bid=.04,ask=.06,vega=1.)
        price,penalty,cost=execution_terms(quote,-1,replace(c,option_penalty_vol_points=.1))
        self.assertEqual(price,.05);self.assertEqual(penalty,10);self.assertEqual(cost,10)

    def test_shadow_matches_engine_with_mid_penalty(self):
        q,s,_,d,c=fixture()
        c=replace(c,entry_vega_per_leg=100,max_gross_vega=1000,option_penalty_vol_points=.1,
                  option_commission=.65,stock_cost_bps=1,stock_borrow_rate=.003)
        s['rate']=.04
        selected=q[q.date.eq(d[0])][['date','secid','optionid']]
        dividends=pd.DataFrame(columns=['secid','ex_date','declare_date','amount'])
        label=ShadowLabels(q,s,selected,d,[1,2],c,dividends).label(0)
        for j,action in enumerate(LADDER):
            nav,_,_=simulate(q,s,pd.Series([action],index=d[:1]),d,[1,2],c,selections=selected)
            benchmark=c.capital*(1+.04*d.to_series().diff().dt.days.fillna(0)/365).cumprod()
            expected=nav.pnl-benchmark.diff().fillna(0);end=label['end_index']
            np.testing.assert_allclose(label['daily'][j,:end],expected.iloc[1:end+1]/c.entry_vega_per_leg,atol=1e-9)

    def test_invalid_mid_penalty_rejected(self):
        for value in [-1,np.nan,np.inf]:
            with self.assertRaises(ValueError):Settings(option_penalty_vol_points=value)

    def test_penalty_enters_pretrade_risk_check(self):
        q,s,a,d,c=fixture()
        # An otherwise eligible entry becomes unaffordable after its cash costs.
        for name,value in [('bs_iv',.4),('bs_rate',0.),('known_dividend_pv',0.)]:q[name]=value
        c=replace(c,risk_controls=True,option_penalty_vol_points=100000)
        nav,trades,dec=simulate(q,s,a,d,[1,2],c)
        self.assertEqual(dec.status.iloc[0],'risk_limit_entry')
        self.assertTrue(trades.empty)
        self.assertEqual(nav.nav.iloc[-1],c.capital)

    def test_fixed_trade_repricing_matches_full_ledger_with_borrowing(self):
        q,s,a,d,c=fixture();c=replace(c,capital=100,option_commission=.65)
        q.loc[q.secid.eq(1),['bid','ask']]+=1
        s['rate']=.04
        old,trades,_=simulate(q,s,a,d,[1,2],c)
        new,_,_=simulate(q,s,a,d,[1,2],replace(c,option_penalty_vol_points=0))
        self.assertLess(old.cash.iloc[1],0)
        old['cash_benchmark']=c.capital*(1+.04*d.to_series().diff().dt.days.fillna(0)/365).cumprod()
        commission=trades.groupby('date').commission.sum().reindex(d,fill_value=0)
        cost=old.option_execution_cost-commission
        result=reprice_cash(old,pd.Series(.04,index=d),cost,np.zeros(len(d)),c.capital,c.borrowing_spread)
        np.testing.assert_allclose(result.nav,new.nav,atol=1e-10)
        np.testing.assert_allclose(result.interest,new.interest,atol=1e-10)


if __name__=='__main__':unittest.main()
