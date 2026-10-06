import unittest
import pandas as pd
from scripts.p25_signal_transfer import choose_existing_first, transfer_rows


class TransferTest(unittest.TestCase):
    def test_new_contracts_only_fill_missing_days_and_keep_oi100(self):
        old = pd.DataFrame({'date': pd.to_datetime(['2020-01-02']), 'cp_flag': ['P'], 'bucket': ['75'], 'optionid': [1]})
        new = pd.DataFrame({'date': pd.to_datetime(['2020-01-02', '2020-01-03', '2020-01-06']),
                            'cp_flag': ['P']*3, 'bucket': ['75']*3, 'optionid': [2, 3, 4], 'open_interest': [100, 100, 10]})
        picked = choose_existing_first(old, new, 'P75')
        self.assertEqual(picked.optionid.tolist(), [1, 3])
        self.assertEqual(picked.contract_source.tolist(), ['existing', 'additional'])

    def test_p25_is_unchanged_and_missing_wing_contracts_do_not_drop_decisions(self):
        anchor = pd.DataFrame({'date': pd.to_datetime(['2020-01-02', '2020-01-03']), 'pair': ['1__2']*2,
                               'wing': ['P25']*2, 'secid_a': [1]*2, 'secid_b': [2]*2,
                               'optionid_a': [11, 12], 'optionid_b': [21, 22], 'direction': [1, -1], 'base_level': [2, 4]})
        got = transfer_rows(anchor, {}, 'P25')
        pd.testing.assert_frame_equal(got[anchor.columns], anchor)
        choices = pd.DataFrame({'optionid': [99], 'contract_source': ['existing']}, index=pd.to_datetime(['2020-01-02']))
        got = transfer_rows(anchor, {(1, 'P75'): choices, (2, 'P75'): choices}, 'P75')
        pd.testing.assert_frame_equal(got[['date', 'pair', 'direction', 'base_level']], anchor[['date', 'pair', 'direction', 'base_level']])
        self.assertTrue(got.loc[1, ['optionid_a', 'optionid_b']].isna().all())


if __name__ == '__main__':
    unittest.main()
