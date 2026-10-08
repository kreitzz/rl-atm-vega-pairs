"""Check the economic units, overlapping paths, and isolated fee bridge."""
import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
from scripts.diagnose_p25_generalization import aggregate, legacy_functions, base


class GeneralizationTests(unittest.TestCase):
    def test_overlapping_paths_use_calendar_and_reference_vega(self):
        calendar=pd.date_range('2020-01-01',periods=5)
        result=aggregate(np.array([[1.,2.,0.],[3.,4.,0.]]),[0,1],calendar)
        np.testing.assert_array_equal(result,[0.,450.,2250.,1800.,0.])
        masked=aggregate(np.array([[1.,2.,0.],[3.,4.,0.]]),[0,1],calendar,[False,True])
        np.testing.assert_array_equal(masked,[0.,0.,1350.,1800.,0.])

    def test_missing_outcomes_and_truncated_pnl_are_rejected(self):
        calendar=pd.date_range('2020-01-01',periods=3)
        with self.assertRaises(ValueError):aggregate(np.array([[np.nan]]),[0],calendar)
        with self.assertRaises(ValueError):aggregate(np.array([[1.,2.]]),[1],calendar)

    def test_commission_change_matches_contract_count_and_financing(self):
        UnitPaths,pair_labels=legacy_functions()
        dates=pd.date_range('2020-01-02',periods=4)
        quotes=pd.DataFrame(dict(date=dates,optionid=1,expiry=dates[-1],cp_flag='P',
            strike=100.,bid=1.99,ask=2.01,multiplier=100.,cfadj=1.,delta=-.25,
            vega=.1,open_interest=1000.,bs_iv=.2,bs_rate=.05,known_dividend_pv=0.,
            standard=True,bs_status='ok'))
        stock=pd.DataFrame(dict(date=dates,close=100.,cfadj=1.,rate=.05))
        selection=pd.DataFrame(dict(date=[dates[0]],optionid=[1],tenor=[30],cp_flag=['P'],bucket=[25]))
        cfg=base.Settings(capital=1e6,entry_vega_per_leg=450,hold_to_expiry=True,
                          min_dte=0,max_dte=55,option_spread_fraction=0,option_penalty_vol_points=.1)
        paths=[]
        for fee in [.65,.03]:
            settings=replace(cfg,option_commission=fee)
            unit=UnitPaths(quotes,stock,selection,dates,dates[:1],pd.DataFrame(),settings)
            daily,_,_,status=pair_labels(unit,unit,(30,'P',25,25),settings)
            self.assertEqual(status[0],'complete')
            paths.append(daily[0,-1]*450)
        difference=paths[1]-paths[0]
        # Full size is45 contracts per leg; fee savings subsequently earn cash interest.
        self.assertAlmostEqual(difference[0],90*.62,places=7)
        self.assertAlmostEqual(difference.sum(),90*.62*(1+.05/365)**2,places=7)


if __name__=='__main__':unittest.main()
