import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from scripts.curve_trading.config import config
from scripts.curve_trading.unit_paths import UnitPaths, pair_labels
from scripts.option_flex.black_scholes import option_values
from scripts.put75_cost_hurdle import with_training_commission


class CostHurdleTest(unittest.TestCase):
    def inputs(self, scale):
        cal = pd.bdate_range('2017-01-02', periods=30)
        stock = pd.DataFrame(dict(date=cal, close=np.linspace(100,103,len(cal))*scale,
                                  cfadj=1., rate=.03))
        rows = []
        for d,s in zip(cal,stock.close):
            t = (cal[-1]-d).days/365
            if t:
                price,delta,_,vega = option_values(s,110*scale,t,.03,.4,'P')
            else:
                price,delta,vega = max(110*scale-s,0),-1.,0.
            rows.append(dict(date=d,optionid=int(scale),secid=int(scale),expiry=cal[-1],
                             cp_flag='P',strike=110*scale,bid=max(float(price)-.01,0),
                             ask=float(price)+.01,delta=float(delta),vega=float(vega),
                             multiplier=100.,cfadj=1.,standard=True,open_interest=1000,
                             bs_iv=.4,bs_rate=.03,known_dividend_pv=0.,bs_status='ok'))
        selection = pd.DataFrame(dict(date=[cal[0]],tenor=[30],cp_flag=['P'],bucket=['75'],optionid=[int(scale)]))
        return pd.DataFrame(rows),stock,selection,cal,cal[:1],pd.DataFrame()

    def test_adjustment_matches_fresh_paths_and_exact_funding(self):
        for capital in [1_000_000,500]:
            cfg = replace(config(),option_commission=.65,capital=capital)
            a,b = [UnitPaths(*self.inputs(scale),cfg) for scale in [1,2]]
            original_long,original_short = a.flow_long.copy(),a.flow_short.copy()
            for commission in [1.,1.5]:
                adjusted = [with_training_commission(u,commission) for u in [a,b]]
                fresh = [UnitPaths(*self.inputs(scale),replace(cfg,option_commission=commission)) for scale in [1,2]]
                for x,y in zip(adjusted,fresh):
                    np.testing.assert_allclose(x.flow_long,y.flow_long,atol=1e-10,rtol=0)
                    np.testing.assert_allclose(x.flow_short,y.flow_short,atol=1e-10,rtol=0)
                x = pair_labels(*adjusted,(30,'P',75,75),replace(cfg,option_commission=commission))
                y = pair_labels(*fresh,(30,'P',75,75),replace(cfg,option_commission=commission))
                for actual,expected in zip(x,y):
                    if np.issubdtype(actual.dtype,np.number):
                        np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=0)
                    else:
                        np.testing.assert_array_equal(actual,expected)
            np.testing.assert_array_equal(a.flow_long,original_long)
            np.testing.assert_array_equal(a.flow_short,original_short)

    def test_invalid_hurdles_are_rejected(self):
        unit = UnitPaths(*self.inputs(1),replace(config(),option_commission=.65))
        for value in [-1,np.nan,np.inf]:
            with self.assertRaises(ValueError):
                with_training_commission(unit,value)


if __name__ == '__main__':
    unittest.main()
