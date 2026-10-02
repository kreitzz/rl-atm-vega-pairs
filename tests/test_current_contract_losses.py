import unittest
import numpy as np
import pandas as pd
from scripts.diagnose_current_contract_losses import attribute, complete_nav


class LossAttributionTest(unittest.TestCase):
    def fixture(self,split=False):
        dates=pd.to_datetime(['2020-01-02','2020-01-03','2020-01-06'])
        factors=[1,2 if split else 1,1]
        prices=[100,51 if split else 102,50.5 if split else 101]
        hedges=[-50,-110 if split else -55,0]
        commission=np.array([.65,0,1.3 if split else .65])
        penalty=np.array([2.,0,2.]);cost=np.array([.5,.051,.5555])
        borrow=np.array([0,5000*.003/365,5610*.003*3/365])
        option_gross=np.array([0,50,50]);hedge_gross=np.array([0,-100,55])
        pnl=option_gross+hedge_gross-commission-penalty-cost-borrow
        nav=pd.DataFrame(dict(nav=10000+pnl.cumsum(),pnl=pnl,cash_benchmark=10000.,interest=0.,
            option_value=[500,550,0],stock_value=np.array(hedges)*prices,option_execution_cost=commission+penalty,
            option_model_penalty=penalty,hedge_cost=cost,stock_borrow_cost=borrow,dividend_cash=0.,
            hedge_1=hedges,hedge_2=0.,split_factor_1=factors,split_factor_2=1.),index=dates)
        trades=[dict(date=dates[0],cohort='2020-01-01',secid=1,optionid=1,quantity=1,price=5.,commission=.65,execution_penalty=2,reason='entry'),
                dict(date=dates[2],cohort='2020-01-01',secid=1,optionid=1,quantity=-2 if split else -1,price=3. if split else 6.,commission=commission[-1],execution_penalty=2,reason='scheduled_exit')]
        if split:trades.append(dict(date=dates[1],cohort='2020-01-01',secid=1,optionid=1,quantity=1,price=0.,commission=0.,execution_penalty=0.,reason='split_adjustment'))
        stocks=pd.Series(prices+[50]*3,index=pd.MultiIndex.from_product([[1,2],dates],names=['secid','date']))
        cfg=dict(capital=10000,stock_cost_bps=1,stock_borrow_rate=.003)
        return nav,pd.DataFrame(trades),[1,2],stocks,pd.DataFrame(columns=['secid','ex_date','amount']),cfg

    def test_exact_option_hedge_fee_and_borrow_attribution(self):
        frame,legs,error=attribute(*self.fixture())
        self.assertAlmostEqual(frame.option_mid_pnl.sum(),100.)
        self.assertAlmostEqual(frame.stock_hedge_gross_pnl.sum(),-45.)
        self.assertAlmostEqual(frame.stock_execution.sum(),1.1065)
        self.assertLess(error,1e-8)
        self.assertAlmostEqual(sum(r['net_trading_before_financing'] for r in legs),frame.net_excess_pnl.sum())

    def test_split_does_not_create_a_stock_loss(self):
        frame,legs,error=attribute(*self.fixture(split=True))
        self.assertAlmostEqual(frame.stock_hedge_gross_pnl.sum(),-45.)
        self.assertAlmostEqual(frame.option_mid_pnl.sum(),100.)
        self.assertAlmostEqual(frame.option_commission.sum(),1.95)
        self.assertLess(error,1e-8)

    def test_cash_only_path_has_no_trading_attribution(self):
        nav=pd.DataFrame(dict(nav=[100.,101.],cash_benchmark=[100.,101.]),index=pd.date_range('2020-01-01',periods=2))
        restored=complete_nav(nav,100.)
        np.testing.assert_array_equal(restored.interest,[0,1])
        self.assertTrue(restored.option_execution_cost.eq(0).all())


if __name__=='__main__':unittest.main()
