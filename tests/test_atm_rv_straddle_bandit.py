import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from scripts import atm_rv_straddle_bandit as study


class StraddleBanditTest(unittest.TestCase):
    def test_sampling_is_calendar_based_and_independent_of_future_profit(self):
        cal=pd.bdate_range('2020-01-01','2020-02-28')
        rows=pd.DataFrame([dict(date=d,secid=s,residual_memory=float(s),future_profit=-s)
                           for d in cal for s in range(10)])
        actual=study.sample_rows(rows,cal)
        self.assertEqual(len(actual),24)
        self.assertEqual(set(actual.secid),{0,4,9})
        expected=[d for _,g in pd.Series(cal,index=cal).groupby(cal.to_period('M')) for d in g.iloc[[0,5,10,15]]]
        self.assertEqual(set(actual.date),set(expected))
        rows['future_profit']=1e20
        pd.testing.assert_frame_equal(actual.drop(columns='future_profit'),study.sample_rows(rows,cal).drop(columns='future_profit'))

    def test_straddle_scaled_shadow_matches_separate_integer_ledgers(self):
        from scripts.option_flex.contract_engine import MarketRow,Settings
        from scripts.atm_rv_persistence_engine import simulate
        cal=pd.bdate_range('2020-01-02',periods=25)
        class Market:
            divs=pd.DataFrame([dict(secid=1,ex_date=cal[4],declare_date=cal[0],amount=.2)])
            def stock(self,date,sid):return 100.+3*np.sin(cal.get_loc(date)/4),1.
            def rate(self,date):return .05
            def quote(self,date,oid):
                if date>=cal[-1]:return None
                return MarketRow(date=date,optionid=oid,secid=1,strike=100.,expiry=cal[-1],cp_flag='C' if oid==1 else 'P',
                    cfadj=1.,standard=True,multiplier=100,bid=2.,ask=2.1,delta=.6 if oid==1 else -.4,vega=.1,
                    bs_iv=.3,bs_rate=.05,known_dividend_pv=0.,bs_status='ok',open_interest=1000)
        rows=pd.DataFrame([dict(date=cal[0],pair='single_1',wing='ATM_STRADDLE',secid_a=1,secid_b=1,
            optionid_a=1,optionid_b=2,base_level=4,direction=-1)])
        cfg=Settings(capital=10000.,entry_vega_per_leg=20.,hold_to_expiry=True,risk_controls=True,
            option_commission=.03,option_spread_fraction=0,option_penalty_vol_points=.1,
            max_gross_vega=1e8,max_hedge_leverage=100.,max_reserve_fraction=.99,max_stress_loss_fraction=.99,
            stock_reserve_fraction=.000001,spot_stress=.0001,vol_stress=.0001)
        cohort=f'{cal[0].date()}|single_1|ATM_STRADDLE'
        unit=simulate(Market(),rows,cal,cfg,10000.,straddle=True,locked_entry_counts={cohort:(1,1)})[0]
        counts=np.array([1,2,4]);rates=pd.Series(.05,index=cal)
        actual=study.bandit.scaled_cash_paths(unit,counts,rates,capital=10000.,borrowing_spread=cfg.borrowing_spread)
        benchmark=10000.*(1+rates.shift(1).fillna(0)*cal.to_series().diff().dt.days.fillna(0)/365).cumprod()
        for i,n in enumerate(counts):
            nav=simulate(Market(),rows,cal,cfg,10000.,straddle=True,locked_entry_counts={cohort:(int(n),int(n))})[0]
            np.testing.assert_allclose(actual[i],nav.pnl-benchmark.diff().fillna(0),atol=1e-8,rtol=0)


if __name__=='__main__':unittest.main()
