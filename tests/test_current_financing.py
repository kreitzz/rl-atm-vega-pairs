import unittest
import numpy as np
import pandas as pd
from scripts.audit_current_financing import reprice_funding


class FundingTest(unittest.TestCase):
    def test_weekend_interest_and_borrow_reprice(self):
        dates=pd.to_datetime(['2024-01-04','2024-01-05','2024-01-08'])
        dt=np.array([0.,1/365,3/365]); rate=.04; capital=1_000_000; short=200_000
        cash=[800_000.]; interest=[0.]; borrow=short*.003*dt
        for i in range(1,3):
            interest.append(cash[-1]*rate*dt[i]);cash.append(cash[-1]+interest[-1]-borrow[i])
        nav=pd.DataFrame(dict(cash=cash,interest=interest,stock_borrow_cost=borrow,
            stock_value=200_000.,option_value=0.,cash_benchmark=capital*np.cumprod(1+rate*dt)),index=dates)
        nav['nav']=nav.cash+nav.stock_value
        baseline,exposure=reprice_funding(nav,capital,.003,.003,.01)
        np.testing.assert_allclose(baseline.nav,nav.nav,atol=1e-8,rtol=0)
        self.assertAlmostEqual(exposure['short_dollar_years'],short*4/365)
        stressed,_=reprice_funding(nav,capital,.003,.01,.01,.005,True)
        expected=(.007+.005)*short*(4/365)+(.007+.005)*short/365*.04*3/365
        self.assertAlmostEqual(nav.nav.iloc[-1]-stressed.nav.iloc[-1],expected,places=7)

    def test_positive_cash_can_still_need_collateral_funding(self):
        dates=pd.to_datetime(['2024-01-04','2024-01-05']); dt=1/365
        nav=pd.DataFrame(dict(cash=[100.,100.+4*dt-300*.003*dt],interest=[0.,4*dt],
            stock_borrow_cost=[0.,300*.003*dt],stock_value=[900.,900.],option_value=[0.,0.],
            cash_benchmark=[1000.,1000*(1+.04*dt)]),index=dates)
        nav['nav']=nav.cash+nav.stock_value
        plain,_=reprice_funding(nav,1000,.003,.003,.01)
        separated,exposure=reprice_funding(nav,1000,.003,.003,.01,segregate=True)
        self.assertAlmostEqual(plain.nav.iloc[-1]-separated.nav.iloc[-1],200*.01*dt)
        self.assertAlmostEqual(exposure['segregated_cash_loan_dollar_years'],200*dt)


if __name__=='__main__':unittest.main()
