import unittest

import numpy as np
import pandas as pd

from scripts.run_atm65_ablation import features as atm_features
from scripts.run_curve_trading import NEW_FEATURES, EXTRA, FEATURES
from scripts.retrain_atm_commission import fast_memory
from scripts.tail_put25_atm_features import states, columns


class TailATMFeaturesTest(unittest.TestCase):
    def fixture(self):
        dates = pd.bdate_range('2015-01-01', periods=300)
        memory = fast_memory(dates, dates + pd.Timedelta(days=30),
                             np.ones((300, 2)), np.ones((300, 2), bool))
        names = NEW_FEATURES + [c if c != 'commission_vol_roundtrip' else 'commission_vol_entry' for c in EXTRA]
        names += list(memory.columns) + FEATURES + ['vix_previous_close']
        panel = pd.DataFrame(1., index=dates, columns=names)
        panel['eligible'] = True
        panel['trailing_rv_amd'] = 25.
        panel['trailing_rv_nvda'] = 20.
        panel['employee_leadership_gap'] = .2
        panel['employee_leadership_available'] = 1.
        panel['iv_amd'] = 99.
        panel['iv_nvda'] = 99.
        def selection(iv, strike):
            return pd.DataFrame(dict(date=dates, bs_iv=iv, expiry=dates+pd.Timedelta(days=30),
                                     strike=strike, close=100., bs_vega=.05))
        a = selection(.3 + .01 * np.sin(np.arange(300)/10), 90.)
        b = selection(np.full(300, .25), 95.)
        return dates, panel, memory, a, b

    def test_put_contract_inputs_and_memory_replace_atm_values(self):
        dates, panel, memory, a, b = self.fixture()
        result = states(a, b, panel, dates, memory, .65)
        self.assertEqual(len(columns(result)), 47)
        self.assertEqual(len(columns(result, employee=True, rank=True)), 50)
        self.assertEqual(columns(result), atm_features(panel))
        np.testing.assert_allclose(result.spread, (a.bs_iv-b.bs_iv)*100)
        np.testing.assert_allclose(result.rv_adjusted_value, 5.-result.spread)
        np.testing.assert_allclose(result.commission_vol_entry, .26)
        np.testing.assert_allclose(result.moneyness_a, -.1)
        np.testing.assert_allclose(result.dte_a, 30)
        pd.testing.assert_frame_equal(result[memory.columns], memory)
        inherited = [c for c in FEATURES+['vix_previous_close', 'beta', 'correlation', 'employee_leadership_gap']]
        pd.testing.assert_frame_equal(result[inherited], panel[inherited])
        self.assertFalse(result.feature_available.iloc[:126].any())
        self.assertTrue(result.feature_available.iloc[126:].all())

    def test_future_quotes_and_outcomes_cannot_change_past_state(self):
        dates, panel, memory, a, b = self.fixture()
        original = states(a, b, panel, dates, memory, .65)
        changed = a.copy()
        changed.loc[200:, 'bs_iv'] = .9
        changed.loc[200:, 'strike'] = 20.
        after = states(changed, b, panel, dates, memory, .65)
        pd.testing.assert_frame_equal(original.iloc[:200], after.iloc[:200])
        ends = dates + pd.Timedelta(days=30)
        outcomes = np.ones((300, 2))
        altered = outcomes.copy()
        altered[ends >= dates[200]] = -1000.
        before = fast_memory(dates, ends, outcomes, np.ones((300, 2), bool))
        after = fast_memory(dates, ends, altered, np.ones((300, 2), bool))
        pd.testing.assert_frame_equal(before.iloc[:201], after.iloc[:201])

    def test_missing_signal_quote_is_not_filled_or_replaced_by_atm(self):
        dates, panel, memory, a, b = self.fixture()
        a = a.drop(index=200)
        result = states(a, b, panel, dates, memory, .65)
        self.assertTrue(np.isnan(result.loc[dates[200], 'iv_amd']))
        self.assertFalse(result.loc[dates[200], 'feature_available'])
        self.assertEqual(result.loc[dates[200], 'iv_nvda'], 25.)


if __name__ == '__main__':
    unittest.main()
