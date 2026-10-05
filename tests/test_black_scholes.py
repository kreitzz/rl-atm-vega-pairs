import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.black_scholes import call_values, invert_calls, fit_quotes
from scripts.select_atm_contracts import select_candidates, load_calendar
from scripts.contract_engine import choose_contract, Settings
from tests.test_contract_engine import fixture


class BlackScholesTests(unittest.TestCase):
    def test_calendar_uses_stock_sessions_across_market_closure(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'underlying').mkdir()
            pd.DataFrame({'date':pd.to_datetime(['2012-10-26','2012-10-31','2012-10-26'])}).to_parquet(root/'underlying/year=2012.parquet')
            self.assertEqual(list(load_calendar(root)),list(pd.to_datetime(['2012-10-26','2012-10-31'])))

    def test_recover_iv_and_greeks(self):
        s=np.array([30.,100.,200.]); k=np.array([32.,100.,185.])
        t=np.array([21.,30.,45.])/365; r=.04; iv=np.array([.2,.5,1.2])
        price,delta,gamma,vega=call_values(s,k,t,r,iv)
        fitted=invert_calls(s,k,t,r,price)
        np.testing.assert_allclose(fitted[0],iv,atol=1e-10)
        eps=1e-3
        up=call_values(s+eps,k,t,r,iv)[0]; dn=call_values(s-eps,k,t,r,iv)[0]
        np.testing.assert_allclose(delta,(up-dn)/(2*eps),rtol=1e-6)
        np.testing.assert_allclose(gamma,(up-2*price+dn)/eps**2,rtol=1e-4)
        bump=1e-5
        np.testing.assert_allclose(vega,(call_values(s,k,t,r,iv+bump)[0]-call_values(s,k,t,r,iv-bump)[0])/(2*bump)/100,rtol=1e-6)

    def test_invalid_prices_are_not_clipped(self):
        result=invert_calls([100]*4,[90]*4,[.1,.1,.1,0],0,[5,100,110,15])[0]
        self.assertTrue(np.isnan(result).all())

    def test_selection_matches_simulator(self):
        q,s,a,dates,cfg=fixture()
        q=q[q.date.eq(dates[0]) & q.secid.eq(1)].copy()
        alternatives=pd.concat([q.assign(optionid=3,expiry=dates[1]+pd.Timedelta(days=30),strike=102),
                                q.assign(optionid=4,expiry=dates[1]+pd.Timedelta(days=30),strike=100),q])
        s=s[s.date.eq(dates[0])].assign(entry_date=dates[1])
        best=select_candidates(alternatives,s,cfg)
        self.assertEqual(best.optionid.iloc[0],4)
        self.assertEqual(best.optionid.iloc[0],choose_contract(alternatives,100,dates[1],cfg).optionid)

    def test_dividends_are_asof_and_vendor_fields_preserved(self):
        q,s,a,dates,cfg=fixture(); q=q.iloc[:1];s=s.iloc[:1]
        zero=pd.DataFrame(dict(date=[dates[0],dates[0]],days=[10,60],rate=[3.,5.]))
        events=pd.DataFrame([dict(secid=1,ex_date='2025-01-20',declare_date='2025-01-01',amount=1.,
             cancel_flag='0',distr_type='1',currency='USD',adj_factor=1,approx_flag='0',liquid_flag='0')])
        result=fit_quotes(q,s,zero,events)
        expected_rate=(3+2*(36-10)/50)/100
        self.assertAlmostEqual(result.bs_rate.iloc[0],expected_rate)
        self.assertAlmostEqual(result.known_dividend_pv.iloc[0],np.exp(-expected_rate*18/365))
        self.assertEqual(result.delta.iloc[0],.5)
        self.assertEqual(result.bs_status.iloc[0],'ok')
        events['declare_date']='2025-01-03'
        future=fit_quotes(q,s,zero,events)
        self.assertEqual(future.known_dividend_pv.iloc[0],0)
        self.assertNotEqual(future.bs_iv.iloc[0],result.bs_iv.iloc[0])
        events['declare_date']='1900-01-01'
        unknown=fit_quotes(q,s,zero,events)
        self.assertEqual(unknown.known_dividend_pv.iloc[0],0)
        self.assertAlmostEqual(unknown.bs_iv.iloc[0],future.bs_iv.iloc[0])
        events['declare_date']='2025-01-01';events['currency']='EUR'
        self.assertEqual(fit_quotes(q,s,zero,events).bs_status.iloc[0],'unsupported_distribution')


if __name__ == '__main__': unittest.main()
