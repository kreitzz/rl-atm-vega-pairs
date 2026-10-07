import unittest
import pandas as pd
from scripts.run_p25_memory_ablation import eligible


class MemoryGateTests(unittest.TestCase):
    def rows(self):
        return pd.DataFrame(dict(observed=[True]*5,eligible=[True]*5,context=[True]*5,
            direction=[1,-1,1,1,0],count_91d=[21,21,20,0,100],count_365d=[63,63,62,0,100],
            profit_91d=[1,-1,1,0,1],profit_365d=[1,1,1,0,1],finite_label=[False]*5))

    def test_declared_nested_gates(self):
        rows=self.rows()
        self.assertEqual(eligible(rows,'retained_baseline').tolist(),[True,False,False,False,False])
        self.assertEqual(eligible(rows,'counts_only').tolist(),[True,True,False,False,False])
        self.assertEqual(eligible(rows,'no_memory_gate').tolist(),[True,True,True,True,False])

    def test_future_training_labels_do_not_gate_proposals(self):
        rows=self.rows()
        before=eligible(rows,'no_memory_gate')
        rows['finite_label']=True
        rows['target']=[-1e9]*len(rows)
        rows['end']=pd.Timestamp('2099-01-01')
        pd.testing.assert_series_equal(before,eligible(rows,'no_memory_gate'))

    def test_other_screens_remain_required(self):
        for field in ['observed','eligible','context']:
            rows=self.rows()
            rows[field]=False
            self.assertFalse(eligible(rows,'no_memory_gate').any())


if __name__=='__main__':unittest.main()
