import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from scripts.tech_etf_market import augment
from scripts.run_contract_surface_features import FEATURES
from scripts.build_contract_surface import VALUES


class TechSurfaceTest(unittest.TestCase):
    def test_etf_surface_replaces_missing_values_and_pair_differences(self):
        dates=pd.date_range('2020-01-02',periods=2)
        base=pd.DataFrame(0.,index=dates,columns=FEATURES)
        for v in VALUES:
            base[f'surface_{v}_b']=[2.,3.]
            base[f'surface_{v}_b_available']=1.
        surface=pd.DataFrame({v:[5.,np.nan] for v in VALUES},index=dates)
        with patch('scripts.tech_etf_market.stock_augment',return_value=base),patch('scripts.tech_etf_market.pd.read_parquet',return_value=surface):
            result=augment({'secids':[151720,101121]},pd.DataFrame(index=dates),dates)
        for v in VALUES:
            self.assertEqual(result[f'surface_{v}_difference'].tolist(),[3.,0.])
            self.assertEqual(result[f'surface_{v}_difference_available'].tolist(),[1.,0.])
            self.assertEqual(result[f'surface_{v}_a_available'].tolist(),[1.,0.])
