"""Known-close timing, flat continuation and management-Q cutoff checks."""
import os
for _name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS','ARROW_NUM_THREADS']:
    os.environ[_name]='1'

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.p25_rl.management import (COHORT_COLUMN,FEATURES,Planner,QPolicy,
    fit_policy,transitions,variance_states)
from scripts.option_flex.contract_engine import Settings


class ActionForest:
    def __init__(self,values):self.action_values=np.asarray(values,float)
    def predict(self,x):return self.action_values[np.asarray(x)[:,-3:].argmax(axis=1)]


class BackupForest:
    targets=[]
    def __init__(self,**kwargs):pass
    def fit(self,x,y,sample_weight=None):
        type(self).targets.append(np.asarray(y).copy())
        return self
    def predict(self,x):
        # Incorrectly admitting close/trim in cash would bootstrap value100;
        # the sole legal cash hold has value5.
        return np.array([5.,50.,100.])[np.asarray(x)[:,-3:].argmax(axis=1)]


def holding(date,sid=1,quantity=4):
    q=dict(bid=1.,ask=1.1,bs_iv=.3,vega=.2)
    return dict(secid=sid,quantity=quantity,initial_quantity=quantity,
        cohort='cohort',expiry=date+pd.Timedelta(days=30),quote=q,
        entry_iv=.25,last_vega=.2,management_fraction=1.)


class ManagementTests(unittest.TestCase):
    def test_variance_current_prefix_ignores_future_stock_observations(self):
        dates=pd.bdate_range('2018-01-02',periods=100)
        spots=pd.DataFrame({1:100*np.exp(np.sin(np.arange(100)/11)*.1)},index=dates)
        adjust=pd.DataFrame(1.,index=dates,columns=[1])
        original=variance_states(SimpleNamespace(spots=spots,adjustments=adjust))
        future=spots.copy();future.iloc[80:]*=1000
        altered=variance_states(SimpleNamespace(spots=future,adjustments=adjust))
        for half in [20,60]:
            pd.testing.assert_frame_equal(original[half].iloc[:80],altered[half].iloc[:80])
        self.assertTrue(np.isfinite(original[20].iloc[70,0]))

    def test_close_decision_frozen_and_flat_only_hold(self):
        dates=pd.bdate_range('2020-01-02',periods=3)
        cfg=Settings(capital=10_000_000,max_gross_vega=200_000,
                     option_penalty_vol_points=.1,option_commission=.03)
        rv={half:pd.DataFrame(.1,index=dates,columns=[1]) for half in [20,60]}
        policy=QPolicy(ActionForest([0.,1.,2.]),{})
        planner=Planner(pd.Series(0.,index=dates),cfg,rv,model=policy)
        planner.observe(dates[0],dict(nav=cfg.capital,return_=0.),
                        [holding(dates[0])],{}, {})
        self.assertEqual(planner.actions[dates[0]],2)
        # Callback contents cannot revise the completed-close instruction.
        self.assertEqual(planner.manage(dates[0],[{'future_quote':1e99}]),0.)
        planner.observe(dates[1],dict(nav=cfg.capital,return_=0.),[],{}, {})
        self.assertEqual(planner.actions[dates[1]],0)
        self.assertEqual(policy.choose(np.zeros(len(FEATURES))),0)
        self.assertFalse(planner.events[-1]['eligible_decision'])
        tied=QPolicy(ActionForest([1.,1.,1.]),{})
        state=np.zeros(len(FEATURES));state[COHORT_COLUMN]=1
        self.assertEqual(tied.choose(state),0)

    def test_daily_flat_gaps_are_nonterminal_and_rewards_reconcile(self):
        dates=pd.bdate_range('2020-01-02',periods=5)
        events=[]
        for date,cohorts in zip(dates,[1,0,0,1,0]):
            state=np.zeros(len(FEATURES));state[COHORT_COLUMN]=cohorts
            events.append(dict(date=date,action=0,state=state.tolist(),held_cohorts=cohorts))
        planner=SimpleNamespace(events=events,daily=[
            dict(date=date,bps=value) for date,value in zip(dates,[0.,2.,0.,-1.,3.])])
        rows,daily=transitions(planner)
        self.assertEqual(rows.reward.tolist(),[2.,0.,-1.,3.])
        self.assertEqual(rows.terminal.tolist(),[False,False,False,True])
        self.assertFalse(rows.iloc[1].held)
        self.assertFalse(rows.iloc[1].terminal)
        self.assertTrue(rows.iloc[2].next_held)
        self.assertEqual(rows.reward.sum(),daily.bps.iloc[1:].sum())
        bad=SimpleNamespace(events=events[:-1],daily=planner.daily[:-1])
        with self.assertRaises(ValueError):transitions(bad)

    def test_fit_cash_continuation_and_future_cutoff(self):
        dates=pd.bdate_range('2018-01-02',periods=121)
        rows=[]
        state=np.zeros(len(FEATURES));state[COHORT_COLUMN]=1
        for start,end in zip(dates[:-1],dates[1:]):
            for action in [0,1,2]:
                rows.append(dict(start=start,end=end,action=action,
                    state=state.tolist(),next_state=np.zeros(len(FEATURES)).tolist(),
                    held=True,next_held=False,reward=1.,squared_reward=1.,
                    sessions=1,terminal=False))
        cutoff=dates[-1]+pd.offsets.BDay(1)
        poison=dict(rows[-1],start=cutoff-pd.Timedelta(days=1),end=cutoff,
                    reward=1e99,squared_reward=1e198,terminal=True)
        frame=pd.DataFrame(rows+[poison])
        daily=pd.DataFrame(dict(date=list(dates)+[cutoff],
                               bps=list(np.resize([-1.,1.],len(dates)))+[1e99]))
        BackupForest.targets=[]
        with patch('scripts.p25_rl.management.ActionForest',BackupForest):
            result=fit_policy(frame,daily,cutoff,dict(gamma=.99,penalty=0.))
        self.assertEqual(result.audit['transitions'],360)
        self.assertLess(pd.Timestamp(result.audit['last_transition_end']),cutoff)
        self.assertEqual(result.audit['terminal_transitions'],0)
        np.testing.assert_array_equal(BackupForest.targets[0],np.ones(360))
        np.testing.assert_allclose(BackupForest.targets[1],np.full(360,1.+.99*5))
        self.assertLess(result.audit['daily_reward_scale_bps'],2.)


if __name__=='__main__':unittest.main()
