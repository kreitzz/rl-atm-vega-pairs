import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from scripts.run_vega_portfolio import run

class PortfolioLedgerTest(unittest.TestCase):
    def test_constant_marks_lose_exactly_costs_and_borrow(self):
        cal=pd.bdate_range('2020-01-02',periods=5)
        spots=pd.DataFrame(10.,index=cal,columns=[1,2]);adj=spots/10;rates=pd.Series(0.,index=cal)
        qs={};candidates=[]
        for sid,score in [(1,4.),(2,-4.)]:
            q=dict(optionid=sid,secid=sid,expiry=pd.Timestamp('2020-02-03'),strike=10.,bid=.95,ask=1.05,
                   delta=.5,gamma=.1,vega=.1,bs_iv=.4,bs_rate=0.,known_dividend_pv=0.,cfadj=1.,standard=True,
                   multiplier=100,cp_flag='C',open_interest=1000)
            qs[sid]={(d,sid):q for d in cal}
            candidates.append(dict(q,score=score,sector='a',event=0.))
        data=(cal,spots,adj,rates,{}, {cal[0]:pd.DataFrame(candidates)}, {s:pd.Series(31.,index=cal) for s in [1,2]},2)
        with tempfile.TemporaryDirectory() as td,patch('scripts.run_vega_portfolio.OUT',Path(td)),patch('scripts.run_vega_portfolio.quotes',side_effect=lambda sid,year:qs[sid]):
            result=run('forecast',data)
            trades=pd.read_csv(Path(td)/'forecast'/'trades.csv')
            same=run('forecast',data,entry_scales={cal[0]:1.},quiet=True,persist=False)
            pd.testing.assert_frame_equal(result,same,check_exact=True)
            flat=run('forecast',data,entry_scales={cal[0]:0.},quiet=True,persist=False)
            self.assertTrue(flat.nav.eq(1e6).all())
            self.assertTrue(flat.option_cost.eq(0).all())
            half=run('forecast',data,entry_scales={cal[0]:.5},quiet=True,persist=False)
            self.assertLess(half.gross_vega.max(),result.gross_vega.max())
        # Each fixture contract has $10 vega: $1 penalty plus $0.03 commission.
        self.assertAlmostEqual(result.option_cost.sum(),trades.quantity.abs().sum()*1.03,places=6)
        self.assertGreater(result.names.max(),0)
        self.assertEqual(result.names.iloc[-1],0)
        costs=result.option_cost.sum()+result.stock_cost.sum()+result.borrow.sum()
        self.assertAlmostEqual(result.nav.iloc[-1],1e6-costs,places=6)
        self.assertLess(result.ledger_residual.abs().max(),1e-6)
        self.assertLess(result.net_vega.abs().max(),1e-6)

if __name__=='__main__':unittest.main()
