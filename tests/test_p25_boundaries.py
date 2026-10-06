import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
from scripts.p25_boundaries import states,breaches
from scripts.curve_trading.config import config
from scripts.same_structure_smile import oriented_states


class BoundaryTest(unittest.TestCase):
    def test_twenty_boundary_reproduces_original_and_widening_reorients_memory(self):
        raw=pd.read_parquet('artifacts/contracts/same_structure_smile/P25_rows.parquet').iloc[:1000].copy()
        old=oriented_states(raw);new=states(raw,20)
        pd.testing.assert_frame_equal(new,old)
        wide=states(raw,40)
        newly=wide.direction.ne(0)&old.direction.eq(0)
        self.assertTrue(newly.any())
        expected=np.where(wide.direction>0,raw.profit_long_91d,raw.profit_short_91d)
        np.testing.assert_array_equal(wide.profit_91d,expected)
        np.testing.assert_array_equal(wide.label_row,raw.label_row)

    def test_disable_stress_only_preserves_other_caps(self):
        cfg=replace(config(),capital=1_000_000)
        metrics=dict(gross_vega=30_000,gross_hedge_notional=3_000_000,capital_reserve=900_000,stress_loss=60_000)
        self.assertEqual(breaches(metrics,1_000_000,cfg,'current'),['gross_vega','gross_hedge_notional','capital_reserve','stress_loss'])
        self.assertEqual(breaches(metrics,1_000_000,cfg,'no_stress'),['gross_vega','gross_hedge_notional','capital_reserve'])
        self.assertEqual(breaches(metrics,1_000_000,cfg,'no_caps'),[])

    def test_nonpositive_nav_remains_in_every_configuration(self):
        for mode in ['current','no_stress','no_caps']:
            self.assertEqual(breaches({},0,config(),mode),['nonpositive_nav'])


if __name__=='__main__':unittest.main()
