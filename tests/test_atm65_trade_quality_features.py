import unittest
import numpy as np
import pandas as pd
from scripts.atm65_trade_quality_features import nonoverlap_samples, name_memory, add_signals, gamma_per_vega
from scripts.option_flex.black_scholes import option_values


class TradeQualityFeaturesTest(unittest.TestCase):
    def examples(self):
        dates = pd.to_datetime(['2020-01-01', '2020-01-05', '2020-02-01', '2020-03-01', '2020-04-01'])
        return pd.DataFrame(dict(date=dates, secid=1, optionid=np.arange(5), entry=dates+pd.Timedelta(days=1),
            end=pd.to_datetime(['2020-01-20', '2020-01-25', '2020-02-20', '2020-03-20', '2020-04-20']),
            long_net=[1., 999., 2., 3., 4.], short_net=[-2., 999., -3., -4., -5.]))

    def test_duplicate_and_overlapping_contracts_do_not_inflate_memory(self):
        raw = self.examples()
        chosen = nonoverlap_samples(pd.concat([raw, raw], ignore_index=True))
        self.assertEqual(chosen.optionid.tolist(), [0, 2, 3, 4])
        cal = pd.to_datetime(['2020-04-20', '2020-04-21', '2021-04-21'])
        memory = name_memory(chosen, cal)
        np.testing.assert_array_equal(memory.blocks365, [3., 4., 0.])
        np.testing.assert_array_equal(memory.available, [0., 1., 0.])
        self.assertEqual(memory.long_mean365.iloc[1], 2.5)
        self.assertAlmostEqual(memory.long_uncertainty365.iloc[1], np.std([1,2,3,4], ddof=1)/2)

    def test_future_labels_and_same_day_completion_excluded(self):
        raw = nonoverlap_samples(self.examples())
        times = pd.to_datetime(['2020-04-20', '2020-04-21'])
        before = name_memory(raw, times)
        changed = raw.copy(); changed.loc[changed.end.ge(times[0]), ['long_net', 'short_net']] = 10000.
        pd.testing.assert_frame_equal(before.iloc[:1], name_memory(changed, times).iloc[:1])
        self.assertNotEqual(before.long_mean365.iloc[1], name_memory(changed, times).long_mean365.iloc[1])

    def test_missing_outcome_keeps_interval_occupied(self):
        raw = self.examples(); raw.loc[0, ['long_net', 'short_net']] = np.nan
        chosen = nonoverlap_samples(raw)
        self.assertNotIn(1, chosen.optionid.tolist())
        memory = name_memory(chosen, pd.to_datetime(['2020-04-21']))
        self.assertEqual(memory.blocks365.iloc[0], 3.)
        self.assertEqual(memory.available.iloc[0], 0.)

    def test_regime_causality_gamma_order_and_original_states(self):
        rng = np.random.default_rng(42); dates = pd.bdate_range('2018-01-01', periods=450)
        panel = pd.DataFrame(dict(iv_amd=30+np.cumsum(rng.normal(0,.2,450)),
            iv_nvda=35+np.cumsum(rng.normal(0,.2,450))), index=dates)
        a = name_memory(self.examples().iloc[:0], dates); a['gamma_per_vega'] = 3.
        b = a.copy(); b['gamma_per_vega'] = 2.
        x = add_signals(panel, a, b); reverse = add_signals(panel, b, a)
        np.testing.assert_allclose(x.gamma_log_ratio_a_b, -reverse.gamma_log_ratio_a_b)
        pd.testing.assert_frame_equal(x[panel.columns], panel)
        changed = panel.copy(); changed.iloc[400:] *= 10
        pd.testing.assert_frame_equal(x.iloc[:400], add_signals(changed, a, b).iloc[:400])

    def test_gamma_per_vega_uses_observed_dividend_adjustment(self):
        q = pd.DataFrame(dict(date=[pd.Timestamp('2020-01-01')], expiry=[pd.Timestamp('2020-01-31')],
            known_dividend_pv=[2.], strike=[100.], bs_rate=[.02], bs_iv=[.3], cp_flag=['C']))
        _, _, g, v = option_values(98., 100., 30/365, .02, .3)
        q['vega'] = v
        self.assertAlmostEqual(gamma_per_vega(q, [100.])[0], g*10000/v)


if __name__ == '__main__': unittest.main()
