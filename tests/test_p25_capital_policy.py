import unittest
import numpy as np
from scripts.option_flex.contract_engine import Settings
from scripts.uniform_smile_engine import capital_context
from scripts.p25_capital_policy import FEATURES, scale, constant_weights


class CapitalPolicyTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Settings()
        self.row = dict(secid_a=1, secid_b=2, learned_level=4)

    def context(self, **kwargs):
        args = dict(row=self.row, pair_metrics={'stress_loss': 25_000},
                    global_metrics={'capital_reserve': 400_000, 'stress_loss': 25_000,
                                    'gross_hedge_notional': 1_000_000},
                    names={1: 450, 2: 900}, equity=1_000_000, cash=1_000_000,
                    benchmark=1_000_000, relative_peak=1., cfg=self.cfg,
                    pair_capital=1_000_000)
        args.update(kwargs)
        return capital_context(**args)

    def test_capital_usage_is_relative_to_the_relevant_limit(self):
        context = self.context()
        for name in ['reserve_utilization', 'stress_utilization', 'hedge_utilization',
                     'pair_stress_utilization']:
            self.assertAlmostEqual(context[name], .5)
        self.assertEqual(context['name_inventory_units'], 2.)
        self.assertEqual(context['excess_drawdown_fraction'], 0.)

    def test_drawdown_removes_cash_benchmark_and_uses_prior_peak(self):
        context = self.context(equity=1_045_000, benchmark=1_100_000,
                               relative_peak=1., cash=-104_500)
        self.assertAlmostEqual(context['excess_drawdown_fraction'], .05)
        self.assertAlmostEqual(context['borrowing_fraction'], .1)
        self.assertEqual(context['prior_equity'], 1_045_000)

    def test_constant_controls_do_not_change_with_inventory(self):
        for multiplier in [.5, 1., 2., 4.]:
            self.assertEqual(scale(constant_weights(multiplier), self.row, self.context()), multiplier)
            self.assertEqual(scale(constant_weights(multiplier), self.row,
                                   self.context(names={1: 9000, 2: 4000})), multiplier)

    def test_same_signal_can_change_size_when_capital_is_consumed(self):
        weights = np.zeros(len(FEATURES)); weights[0] = 1.; weights[2] = -1.5
        spare = self.context(global_metrics={})
        consumed = self.context(global_metrics={'stress_loss': 50_000})
        self.assertEqual(scale(weights, self.row, spare), 2.)
        self.assertEqual(scale(weights, self.row, consumed), 1.)


if __name__ == '__main__':
    unittest.main()
