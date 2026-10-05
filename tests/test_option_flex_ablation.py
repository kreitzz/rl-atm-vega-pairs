import unittest
import numpy as np
import pandas as pd

from scripts.option_flex_ablation import fit, predict, eligibility, apply_gate, DEFAULT, VARIANTS, CC
from scripts.option_flex.policy import ACTIONS, FLAT, COMBOS, fit as full_fit, predict as full_predict, gate
from scripts.run_contract_walkforward import fit_rl, predict as old_predict


class AblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(14); n = 252
        cls.x = pd.DataFrame(rng.normal(size=(n, 3)), columns=['a', 'b', 'c'])
        cls.labels = rng.normal(size=(n, 33, 4)); cls.labels[:, FLAT] = 0
        cls.positions = np.arange(n); cls.ends = cls.positions+4

    def test_full_search_matches_frozen_33_action_implementation(self):
        a, da = full_fit(self.x, self.labels, self.positions, self.ends)
        b, db = fit(self.x, self.labels, self.positions, self.ends, DEFAULT)
        np.testing.assert_array_equal(full_predict(a, self.x), predict(b, self.x))
        self.assertEqual(da, db)

    def test_calls_only_matches_original_nine_action_search(self):
        a, da = fit_rl(self.x, self.labels[:, CC], self.positions, self.ends)
        b, db = fit(self.x, self.labels, self.positions, self.ends, VARIANTS['cc_original'])
        np.testing.assert_array_equal(old_predict(a, self.x), [ACTIONS[k][1] for k in predict(b, self.x)])
        self.assertEqual(da, db)

    def test_action_subsets_and_longer_search(self):
        for name in ['same_type', 'same_type_no_sizing']:
            model, _ = fit(self.x, self.labels, self.positions, self.ends, VARIANTS[name])
            self.assertTrue(np.isin(predict(model, self.x), VARIANTS[name]['allowed']).all())
        _, short = fit(self.x, self.labels, self.positions, self.ends, DEFAULT)
        _, long = fit(self.x, self.labels, self.positions, self.ends, dict(DEFAULT, generations=200))
        self.assertGreaterEqual(long['training_shadow_daily_sharpe'], short['training_shadow_daily_sharpe'])

    def test_original_and_own_memory_are_distinct_without_bypassing_availability(self):
        dates = pd.date_range('2020-01-01', periods=3); p = pd.DataFrame({'eligible': [True]*3}, index=dates)
        for c in COMBOS:
            p[c+'_available'] = True
            for side in ['short', 'long']:
                for days in [91, 365]:
                    p[f'{c}_profit_{side}_{days}d'] = 1.; p[f'{c}_count_{side}_{days}d'] = 100
                    p[f'profit_{side}_{days}d'] = 1.; p[f'count_{side}_{days}d'] = 100
        p.loc[dates[0], 'PC_profit_long_91d'] = -1
        p.loc[dates[1], 'PC_available'] = False
        codes = pd.Series(ACTIONS.index(('PC', 1.)), index=dates)
        own = apply_gate(codes, eligibility(p, DEFAULT))
        pd.testing.assert_series_equal(own, gate(codes, p))
        original = apply_gate(codes, eligibility(p, dict(DEFAULT, memory='original')))
        self.assertEqual(own.iloc[0], FLAT); self.assertNotEqual(original.iloc[0], FLAT)
        self.assertEqual(original.iloc[1], FLAT)
        restricted = apply_gate(codes, eligibility(p, VARIANTS['same_type']))
        self.assertTrue(restricted.eq(FLAT).all())


if __name__ == '__main__': unittest.main()
