import unittest

from scripts.p25_rl.netting import potential_netting


def order(pair, quantity, commission=.03, penalty=.4):
    return dict(date='2020-01-07', optionid=123, reason='entry', pair=pair,
                quantity=quantity, price=2., commission=abs(quantity) * commission,
                execution_penalty=abs(quantity) * penalty)


class NettingTests(unittest.TestCase):
    def test_cancelled_both_sides_save_only_saved_fees_and_penalty(self):
        rows, summary = potential_netting([order('a_b', 3), order('a_c', -2)])
        self.assertEqual(rows[0]['matched_contracts_per_side'], 2)
        self.assertEqual(summary['cancelled_order_contracts'], 4)
        self.assertAlmostEqual(summary['potential_commission_savings'], .12)
        self.assertAlmostEqual(summary['potential_penalty_savings'], 1.6)
        self.assertAlmostEqual(summary['potential_total_savings'], 1.72)
        self.assertAlmostEqual(summary['share_of_paid_explicit_option_cost'], .8)

    def test_different_pair_constraint_and_inconsistent_terms_prevent_cancellation(self):
        _, same_pair = potential_netting([order('a_b', 3), order('a_b', -3)])
        self.assertEqual(same_pair['cancelled_order_contracts'], 0)
        rows, mismatch = potential_netting([order('a_b', 3), order('a_c', -3, commission=.04)])
        self.assertFalse(rows[0]['identical_per_contract_terms'])
        self.assertEqual(mismatch['potential_total_savings'], 0)

    def test_matching_finds_all_three_pair_cycle_cancellations(self):
        orders = [order(pair, sign * 2) for pair in ['a_b', 'a_c', 'a_d'] for sign in [1, -1]]
        _, summary = potential_netting(orders)
        self.assertEqual(summary['matched_contracts_per_side'], 6)
        self.assertEqual(summary['cancelled_order_contracts'], 12)


if __name__ == '__main__':
    unittest.main()
