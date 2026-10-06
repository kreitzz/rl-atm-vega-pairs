import unittest
import numpy as np
import pandas as pd
from scripts.atm65_mined_features import downside_share, iv_factor_scores, add_signals, GROUPS


class MinedFeaturesTest(unittest.TestCase):
    def test_features_do_not_depend_on_future_prices_or_iv(self):
        dates = pd.bdate_range('2019-01-01', periods=360)
        rng = np.random.default_rng(42)
        stocks = pd.DataFrame(dict(date=dates, close=100*np.exp(np.cumsum(rng.normal(0, .01, 360))), cfadj=1.))
        iv = pd.DataFrame(.4+np.cumsum(rng.normal(0, .002, (360, 4)), axis=0), index=dates)
        a = downside_share(stocks, dates); b = iv_factor_scores(iv)
        future = stocks.copy(); future.loc[300:, 'close'] *= np.linspace(1, 2, 60)
        future_iv = iv.copy(); future_iv.iloc[300:] *= 2
        pd.testing.assert_series_equal(a.iloc[:300], downside_share(future, dates).iloc[:300])
        pd.testing.assert_frame_equal(b.iloc[:300], iv_factor_scores(future_iv).iloc[:300])

    def test_downside_share_and_split_adjustment(self):
        dates = pd.bdate_range('2020-01-01', periods=30)
        adjusted = 100*np.exp(np.arange(30)*-.01)
        prices = pd.DataFrame(dict(date=dates, close=adjusted, cfadj=1.))
        prices.loc[15:, 'close'] /= 2; prices.loc[15:, 'cfadj'] = 2
        self.assertTrue(downside_share(prices, dates).iloc[21:].eq(1).all())

    def test_leg_order_missing_flags_and_original_states(self):
        dates = pd.bdate_range('2021-01-01', periods=2)
        p = pd.DataFrame(dict(original=[1., 2.]), index=dates)
        a = pd.DataFrame(dict(downside_fraction=[.7, np.nan], iv_factor_reversion=[1., 2.]), index=dates)
        b = pd.DataFrame(dict(downside_fraction=[.2, .3], iv_factor_reversion=[3., np.nan]), index=dates)
        x = add_signals(p, a, b); reverse = add_signals(p, b, a)
        pd.testing.assert_frame_equal(x[p.columns], p)
        np.testing.assert_allclose(x.downside_fraction_gap, [.5, 0.])
        np.testing.assert_allclose(x.iv_factor_reversion_gap, [-2., 0.])
        for signal in ['downside_fraction', 'iv_factor_reversion']:
            np.testing.assert_array_equal(x[signal+'_available'], [1., 0.])
            np.testing.assert_allclose(reverse[signal+'_gap'], -x[signal+'_gap'])
        self.assertEqual(GROUPS['rl_both'], GROUPS['rl_downside']+GROUPS['rl_iv_factor'])

    def test_separate_outputs_and_three_rl_variants(self):
        from scripts import run_atm65_mined_signals as runner
        runner.configure()
        self.assertEqual(runner.MODELS, ['rl_downside', 'rl_iv_factor', 'rl_both'])
        self.assertNotEqual(runner.OUT, runner.EMPLOYEE)


if __name__ == '__main__':
    unittest.main()
