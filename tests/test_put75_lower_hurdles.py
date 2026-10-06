import importlib.util
from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from scripts.curve_trading.config import config
from scripts.curve_trading.unit_paths import UnitPaths, pair_labels
from scripts.put75_cost_hurdle import with_training_commission

spec = importlib.util.spec_from_file_location('hurdle_fixture',Path(__file__).with_name('test_put75_cost_hurdle.py'))
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


class LowerHurdleTest(unittest.TestCase):
    def test_lower_hurdles_match_fresh_paths_without_mutating_baseline(self):
        inputs = fixture.CostHurdleTest().inputs
        for capital in [1_000_000,500]:
            cfg = replace(config(),option_commission=.65,capital=capital)
            units = [UnitPaths(*inputs(scale),cfg) for scale in [1,2]]
            original = [(u.flow_long.copy(),u.flow_short.copy()) for u in units]
            baseline = pair_labels(*units,(30,'P',75,75),cfg)[0]
            for commission in [.03,.25,.50]:
                updated_cfg = replace(cfg,option_commission=commission)
                adjusted = [with_training_commission(u,commission) for u in units]
                fresh = [UnitPaths(*inputs(scale),updated_cfg) for scale in [1,2]]
                for x,y in zip(adjusted,fresh):
                    np.testing.assert_allclose(x.flow_long,y.flow_long,atol=1e-10,rtol=0)
                    np.testing.assert_allclose(x.flow_short,y.flow_short,atol=1e-10,rtol=0)
                x = pair_labels(*adjusted,(30,'P',75,75),updated_cfg)
                y = pair_labels(*fresh,(30,'P',75,75),updated_cfg)
                for actual,expected in zip(x,y):
                    if np.issubdtype(actual.dtype,np.number):
                        np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=0)
                    else:
                        np.testing.assert_array_equal(actual,expected)
                self.assertTrue((x[0].sum(axis=2)>=baseline.sum(axis=2)-1e-8).all())
            for unit,(long,short) in zip(units,original):
                np.testing.assert_array_equal(unit.flow_long,long)
                np.testing.assert_array_equal(unit.flow_short,short)


if __name__=='__main__':
    unittest.main()
