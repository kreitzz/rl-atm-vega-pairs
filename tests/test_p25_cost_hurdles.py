import unittest
from types import SimpleNamespace
from dataclasses import replace
import numpy as np
import pandas as pd
from scripts.curve_trading.config import config, SLOTS
from scripts.p25_cost_hurdles import reprice_labels, directional_targets


def fake_unit(cal, long, short, mark, vega):
    lookup = np.full((len(cal),len(SLOTS)),-1,int); lookup[0,SLOTS.index((30,'P',25))]=0
    width=config().max_dte+2
    def path(values):
        result=np.zeros((2,width));result[0,:len(values)]=values;return result
    return SimpleNamespace(cfg=replace(config(),option_commission=.65),calendar=cal,dates=cal,
        width=width,positions=np.arange(len(cal)),lookup=lookup,flow_long=path(long),flow_short=path(short),mark=path(mark),
        entry_vega=np.array([vega,0.]),entered=np.array([True,False]),resolved=np.ones(2,bool),
        ends=np.array([2,0]),stock_bad_prefix=np.zeros(len(cal)+1,int),rates=np.full(len(cal),.1),
        ids=lambda t,cp,b:lookup[:,SLOTS.index((t,cp,b))])


class CostTest(unittest.TestCase):
    def test_lower_fee_changes_both_directions_and_carries_financing(self):
        cal=pd.bdate_range('2020-01-01',periods=3)
        a=fake_unit(cal,[-100.65,120], [99.35,-120], [100,0],10.)
        b=fake_unit(cal,[-80.65,85], [79.35,-85], [80,0],10.)
        high,ends,vegas,status=reprice_labels(a,b,.65)
        low,new_ends,new_vegas,new_status=reprice_labels(a,b,.03)
        np.testing.assert_array_equal(ends,new_ends);np.testing.assert_array_equal(vegas,new_vegas)
        np.testing.assert_array_equal(status,new_status)
        np.testing.assert_allclose((low-high)[0,[0,7],0],90*.62/450,atol=1e-12)
        np.testing.assert_allclose((low-high)[0,[0,7],1],90*.62*(.1/365)/450,atol=1e-12)
        self.assertEqual(a.cfg.option_commission,.65)
        self.assertEqual(a.flow_long[0,0],-100.65)

    def test_full_size_target_uses_the_frozen_proposal_direction(self):
        labels=np.arange(2*9*3).reshape(2,9,3)
        np.testing.assert_array_equal(directional_targets(labels,[1,-1]),[labels[0,8].sum(),labels[1,1].sum()])


if __name__=='__main__':unittest.main()
