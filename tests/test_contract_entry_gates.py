import unittest
import numpy as np
import pandas as pd
from scripts.test_contract_entry_gates import gate_actions, metrics


class EntryGatesTest(unittest.TestCase):
    def test_directional_memory_and_missing_state(self):
        dates=pd.date_range('2020-01-01',periods=4)
        actions=pd.Series([.25,-.75,1.,-1.],index=dates)
        states=pd.DataFrame(dict(profit_long_365d=[1]*3,profit_short_365d=[-1,2,2],
            count_long_365d=[63,63,62],count_short_365d=[63]*3,
            profit_long_91d=[-1]*3,profit_short_91d=[1]*3,
            count_long_91d=[21]*3,count_short_91d=[21]*3),index=dates[:3])
        np.testing.assert_array_equal(gate_actions(actions,states,'memory_year'),[.25,-.75,0,0])
        np.testing.assert_array_equal(gate_actions(actions,states,'memory_both'),[0,-.75,0,0])

    def test_future_states_do_not_change_prior_gate(self):
        dates=pd.date_range('2020-01-01',periods=4)
        actions=pd.Series([1.,-.5,.25,1.],index=dates)
        states=pd.DataFrame({'commission_vol_roundtrip':[.3,.5,np.nan,.4]},index=dates)
        first=gate_actions(actions,states,'commission_budget')
        states.loc[dates[-1],'commission_vol_roundtrip']=99
        second=gate_actions(actions,states,'commission_budget')
        pd.testing.assert_series_equal(first.iloc[:-1],second.iloc[:-1])
        np.testing.assert_array_equal(first,[1,0,0,1])

    def test_later_period_includes_first_day_return(self):
        dates=pd.to_datetime(['2021-12-30','2021-12-31','2022-01-03','2022-01-04'])
        book=pd.DataFrame({'nav':[100,100,110,121],'cash_benchmark':[100]*4},index=dates)
        result={r['period']:r for r in metrics(book)}
        self.assertAlmostEqual(result['later_2022_2025']['total_return'],.21)


if __name__=='__main__': unittest.main()
