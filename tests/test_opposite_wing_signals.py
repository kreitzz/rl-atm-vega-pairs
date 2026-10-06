import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from scripts.research_opposite_wing_signals import har_forecast,past_vol_beta,rule_codes,working_inputs,CORPORATE_EVENTS
from scripts.curve_trading.config import COMBOS


class OppositeSignalsTest(unittest.TestCase):
    def test_oih_reverse_split_uses_etf_identifier(self):
        event=[e for e in CORPORATE_EVENTS if e['effective']=='2020-04-15']
        self.assertEqual(len(event),1)
        self.assertEqual(event[0]['secid'],114726)
        self.assertLess(event[0]['announced'],event[0]['effective'])

    def test_foreign_cash_uses_only_fx_before_declaration(self):
        distribution=pd.DataFrame(dict(secid=[127149],declare_date=['2022-11-16'],
            ex_date=['2022-12-14'],cancel_flag=['0'],liquid_flag=['0'],approx_flag=['0'],
            adj_factor=[1.],amount=[.125],distr_type=['1'],currency=['CAD']))
        fx=pd.DataFrame(dict(observation_date=['2022-11-15','2022-11-16','2022-12-14'],
            DEXCAUS=[1.25,2.,3.]))
        inputs=(pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),distribution.iloc[:0].copy())
        with patch('scripts.research_opposite_wing_signals.frozen_working_inputs',return_value=inputs), \
             patch('scripts.research_opposite_wing_signals.pd.read_parquet',return_value=distribution), \
             patch('scripts.research_opposite_wing_signals.pd.read_csv',return_value=fx):
            _,_,_,cash=working_inputs([], [127149])
        self.assertAlmostEqual(cash.amount.iloc[0],.1)
        self.assertEqual(cash.currency.iloc[0],'USD')

    def test_har_cannot_use_future_returns_or_incomplete_targets(self):
        dates=pd.bdate_range('2012-01-02',periods=1200)
        rng=np.random.default_rng(2);returns=pd.Series(rng.normal(0,.02,len(dates)),index=dates)
        before,_,logs=har_forecast(returns)
        modified=returns.copy();modified.iloc[900:]=.3
        after,_,_=har_forecast(modified)
        pd.testing.assert_series_equal(before.iloc[:900],after.iloc[:900])
        for r in logs:self.assertLess(pd.Timestamp(r['last_target_end']),pd.Timestamp(r['fit']))

    def test_vol_beta_excludes_current_iv_shock(self):
        dates=pd.bdate_range('2012-01-02',periods=180)
        rng=np.random.default_rng(3);vix=pd.Series(.2+rng.normal(0,.01,180),index=dates)
        iv=.3+2*(vix-.2);before=past_vol_beta(iv,vix)
        iv.iloc[150:]=2.;after=past_vol_beta(iv,vix)
        pd.testing.assert_series_equal(before.iloc[:151],after.iloc[:151])

    def test_native_profitability_can_only_filter_matching_contract_type(self):
        dates=pd.bdate_range('2018-01-02',periods=4)
        panel=pd.DataFrame(dict(wing_iv_rank=[.1,.9,.5,.1],observed=True,eligible=True,legacy=True,
            profit_91d=[1.,-1.,1.,1.],profit_365d=1.,count_91d=21,count_365d=63),index=dates)
        codes=rule_codes(panel,'C',25,True);idx=COMBOS.index((30,'C',25,25))
        np.testing.assert_array_equal(codes,[1+idx*8+7,0,0,1+idx*8+7])
        panel.loc[dates[3],'legacy']=False
        self.assertEqual(rule_codes(panel,'C',25,False).iloc[3],0)


if __name__=='__main__':unittest.main()
