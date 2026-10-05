import unittest
from dataclasses import replace
import numpy as np
import pandas as pd

from scripts.batched_contract_labels import build_labels
from scripts.contract_learning_labels import ShadowLabels
from tests.test_contract_engine import fixture


class BatchedLabelsTests(unittest.TestCase):
    def compare(self,case):
        q,s,a,d,c=fixture();c=replace(c,option_penalty_vol_points=.1,repair_crossed_held_quotes=True,support_integer_splits=True)
        div=pd.DataFrame(columns=['secid','ex_date','declare_date','amount'])
        if case=='missing':q=q[~(q.date.eq(d[2])&q.optionid.eq(1))]
        if case=='greeks':q.loc[q.date.eq(d[2])&q.optionid.eq(1),'vega']=np.nan
        if case=='crossed':q.loc[q.date.eq(d[2])&q.optionid.eq(1),['bid','ask']]=[6.,4.]
        if case=='split':
            mask=q.date.ge(d[2])&q.secid.eq(1)
            q.loc[mask,['bid','ask','strike','vega']]/=2;q.loc[mask,'cfadj']=2
            mask=s.date.ge(d[2])&s.secid.eq(1)
            s.loc[mask,'close']/=2;s.loc[mask,'cfadj']=2
        if case=='dividend':div=pd.DataFrame([dict(secid=1,declare_date=d[0],ex_date=d[3],amount=.2)])
        selected=q[['date','secid','optionid']]
        maker=ShadowLabels(q,s,selected,d,[1,2],c,div)
        positions=np.arange(len(d));result=build_labels(maker,positions,batch_size=3)
        slow=[maker.label(i) for i in positions]
        np.testing.assert_allclose(result[0],np.stack([v['daily'] for v in slow]),atol=1e-8,rtol=0,equal_nan=True)
        np.testing.assert_array_equal(result[1],[v['end_index'] for v in slow])
        np.testing.assert_allclose(result[2],np.stack([v['entry_vega'] for v in slow]))
        np.testing.assert_array_equal(result[3],[v['status'] for v in slow])

    def test_cash_and_integer_actions(self):self.compare('normal')
    def test_missing_held_quote(self):self.compare('missing')
    def test_missing_greek_exit(self):self.compare('greeks')
    def test_crossed_quote_repair(self):self.compare('crossed')
    def test_split_quantity_and_hedge(self):self.compare('split')
    def test_dividend_early_exit(self):self.compare('dividend')


if __name__=='__main__':unittest.main()
