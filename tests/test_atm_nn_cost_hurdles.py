import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from scripts.run_atm_nn_cost_hurdles import fit_fixed_nn, predict_fixed_nn
from scripts.curve_trading.policy import ATM


class NNBenchmarkTests(unittest.TestCase):
    def test_fitted_nn_recovers_net_profitable_direction(self):
        x = pd.DataFrame({'signal': np.linspace(-1, 1, 40)})
        codes = [0, *range(1 + ATM * 8, 1 + (ATM + 1) * 8)]
        labels = np.zeros((40, 9, 3))
        labels[:, 1, 0] = -7.; labels[:, -1, 0] = 11.
        model, _ = fit_fixed_nn(x, labels, np.arange(40), np.arange(40), codes)
        self.assertTrue((predict_fixed_nn(model, x) == 1 + ATM * 8 + 7).all())

    def test_directional_target_excludes_prepended_cash(self):
        labels = np.zeros((5, 9, 3)); labels[:, 1] = -7; labels[:, -1] = 11
        x = pd.DataFrame({'signal': range(5)})
        with patch('scripts.run_atm_nn_cost_hurdles.fit_nn', return_value=('model', {'converged': True})) as fit:
            fit_fixed_nn(x, labels, np.arange(5), np.arange(5), [0, *range(1 + ATM * 8, 1 + (ATM + 1) * 8)])
        target = fit.call_args.args[1]
        np.testing.assert_array_equal(target[:, 0].sum(axis=1), np.full(5, -21))
        np.testing.assert_array_equal(target[:, -1].sum(axis=1), np.full(5, 33))

    def test_nn_maps_only_full_size_or_cash_to_contract_actions(self):
        with patch('scripts.run_atm_nn_cost_hurdles.predict_direction', return_value=np.array([-1., 0., 1.])):
            codes = predict_fixed_nn(None, pd.DataFrame({'signal': [1, 2, 3]}))
        np.testing.assert_array_equal(codes, [1 + ATM * 8, 0, 1 + ATM * 8 + 7])
