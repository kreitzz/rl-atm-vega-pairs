import unittest
import numpy as np
import pandas as pd

from scripts import uniform_smile_policy as p
from scripts.run_uniform_smile import aggregate
from scripts.tail_put25_pooled import completed_mask


class UniformSmileTest(unittest.TestCase):
    def test_optional_absence_keeps_observed_states_for_every_wing(self):
        cal=pd.bdate_range('2016-01-01',periods=300)
        a=pd.DataFrame({'date':cal,'bs_iv':np.linspace(.2,.5,len(cal)),'bs_vega':np.nan,'expiry':cal+pd.Timedelta(days=30)})
        b=a.copy();b['bs_iv']=.3
        returns=pd.DataFrame(0.,index=cal,columns=['a','b'])
        screen=pd.DataFrame({'beta':1.,'correlation':.8,'eligible':True},index=cal)
        memory=pd.DataFrame(0.,index=cal,columns=[f'{prefix}_{side}_{days}d' for days in [91,365] for side in ['short','long'] for prefix in ['profit','count']])
        panels=[p.states(a,b,returns,screen,cal,memory,pd.Series(np.nan,index=cal),.65) for _ in range(3)]
        for panel in panels:
            self.assertEqual(int(panel.observed.sum()),len(cal))
            self.assertTrue(np.isfinite(panel[p.FEATURES]).all().all())
            self.assertEqual(panel.vix_available.sum(),0)
            self.assertTrue(p.candidate_mask(panel,'rl_entry_sized').all())
        pd.testing.assert_frame_equal(panels[0],panels[1]);pd.testing.assert_frame_equal(panels[1],panels[2])

    def test_action_paths_and_decoding_preserve_direction_and_sizes(self):
        labels=np.broadcast_to(np.arange(9)[None,:,None],(2,9,3)).copy();direction=np.array([1,-1])
        reward=p.reward_choices(labels,direction,'spread_floor')
        np.testing.assert_array_equal(reward[:,:,0],[[6,7,8],[3,2,1]])
        d,s=p.decode(np.array([2,4]),direction,'spread_floor')
        np.testing.assert_array_equal(d,direction);np.testing.assert_array_equal(s,[2,4])
        d,s=p.decode(np.array([0,3,6]),np.ones(3),'rl_entry_sized_full')
        np.testing.assert_array_equal(d,[0,-1,1]);np.testing.assert_array_equal(s,[0,4,4])

    def test_global_fallback_and_memory_rule_are_explicit(self):
        rows=pd.DataFrame({'rule_direction':[1,-1,0]})
        for side in ['short','long']:
            for days in [91,365]:rows[f'count_{side}_{days}d']=100;rows[f'profit_{side}_{days}d']=1.
        rows.loc[1,'profit_short_91d']=-1
        self.assertEqual(p.candidate_mask(rows,'spread_floor').tolist(),[True,True,False])
        self.assertEqual(p.candidate_mask(rows,'memory_floor').tolist(),[True,False,False])
        self.assertEqual(p.apply({'tree':None},rows,'rl_entry_sized').tolist(),[8,1,0])

    def test_aggregation_respects_cash_action_and_trade_start(self):
        book=np.zeros((4,6));labels=np.array([[[0.,0.],[1.,2.],[2.,4.]]])
        aggregate(book,np.array([1]),labels,np.array([3]),first=3)
        np.testing.assert_array_equal(book[:,3],0)
        np.testing.assert_array_equal(book[:,4],[0,1,2,0])
        np.testing.assert_array_equal(book[:,5],[0,2,4,0])

    def test_training_never_uses_a_same_cutoff_expiry(self):
        dates=pd.to_datetime(['2020-01-01']*2);ends=pd.to_datetime(['2020-01-31','2020-02-01'])
        self.assertEqual(completed_mask(dates,ends,'2020-02-01').tolist(),[True,False])
