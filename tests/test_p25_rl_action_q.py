"""Action-specific Q must learn delayed value and respect the past cutoff."""
import os
for _key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS','ARROW_NUM_THREADS']:
    os.environ[_key]='1'

from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.p25_rl.action_q import FEATURES,SPECS,fit_policy


def training_inputs():
    dates=pd.bdate_range('2018-01-02',periods=81)
    rows=[]
    for j,date in enumerate(dates[:-1]):
        state=(np.sin(np.arange(len(FEATURES))+j/7)*.2).tolist()
        nxt=(np.sin(np.arange(len(FEATURES))+(j+1)/7)*.2).tolist()
        for action in [0,1,2]:
            reward=float(j%7-3)*action/2
            rows.append(dict(start=date,end=dates[j+1],state=state,next_state=nxt,
                action=action,reward=reward,squared_reward=reward**2,
                sessions=1,terminal=False))
    daily=pd.DataFrame(dict(date=dates,bps=np.resize([-1.,0.,1.,2.],len(dates))))
    return pd.DataFrame(rows),daily,dates[-1]+pd.offsets.BDay(1)


class FixedPredictionForest:
    targets=[]
    def __init__(self,**kwargs):pass
    def fit(self,x,y,sample_weight=None):
        type(self).targets.append(np.asarray(y).copy());return self
    def predict(self,x):return np.full(len(x),100.)


class ActionQTests(unittest.TestCase):
    def test_sequential_value_reserves_capacity_for_delayed_reward(self):
        dates=pd.bdate_range('2018-01-02',periods=161)
        flat=np.zeros(len(FEATURES))
        available=flat.copy();available[0]=1.
        exhausted=flat.copy();exhausted[0]=-1.
        rows=[]
        for j in range(80):
            start,next_day,end=dates[2*j:2*j+3]
            for action,immediate in [(0,0.),(1,.5),(2,1.)]:
                successor=available if action==0 else exhausted
                rows.append(dict(start=start,end=next_day,action=action,
                    state=flat.tolist(),next_state=successor.tolist(),
                    reward=immediate,squared_reward=immediate**2,
                    sessions=1,terminal=False))
                for state,later in [(available,8.),(exhausted,0.)]:
                    rows.append(dict(start=next_day,end=end,action=action,
                        state=state.tolist(),next_state=flat.tolist(),
                        reward=later,squared_reward=later**2,
                        sessions=1,terminal=True))
        records=pd.DataFrame(rows)
        daily=pd.DataFrame(dict(date=dates,bps=np.resize([-1.,1.],len(dates))))
        cutoff=dates[-1]+pd.offsets.BDay(1)
        sequential=fit_policy(records,daily,cutoff,deepcopy(SPECS['action_q_capacity']))
        myopic_spec=dict(SPECS['action_q_capacity'],gamma=0.)
        myopic=fit_policy(records,daily,cutoff,myopic_spec)
        self.assertEqual(myopic.choose(flat),2)
        self.assertEqual(sequential.choose(flat),0)
        self.assertGreater(sequential.values(flat)[0,0],sequential.values(flat)[0,2])
        self.assertEqual(sequential.model.present_actions_,[0,1,2])
        self.assertEqual(sequential.audit['action_support'],{'0':240,'1':240,'2':240})

    def test_future_rewards_and_crossing_endpoints_cannot_change_fit(self):
        records,daily,cutoff=training_inputs()
        spec=deepcopy(SPECS['action_myopic_risk'])
        expected=fit_policy(records,daily,cutoff,spec)
        crossing=dict(records.iloc[-1])
        crossing.update(start=cutoff-pd.Timedelta(days=1),end=cutoff,
            state=[1e6]*len(FEATURES),next_state=[1e6]*len(FEATURES),
            reward=1e12,squared_reward=1e24,terminal=True)
        future=dict(crossing,start=cutoff,end=cutoff+pd.offsets.BDay(1))
        altered_records=pd.concat([records,pd.DataFrame([crossing,future])],ignore_index=True)
        altered_daily=pd.concat([daily,pd.DataFrame([
            dict(date=cutoff,bps=1e12),
            dict(date=cutoff+pd.offsets.BDay(1),bps=-1e12)])],ignore_index=True)
        actual=fit_policy(altered_records,altered_daily,cutoff,spec)
        self.assertEqual(actual.audit,expected.audit)
        self.assertEqual(actual.audit['transitions'],240)
        self.assertEqual(actual.audit['terminal_transitions'],0)
        self.assertLess(pd.Timestamp(actual.audit['last_transition_end']),cutoff)
        self.assertEqual(actual.audit['daily_reward_scale_bps'],daily.bps.std(ddof=1))
        states=np.stack(records.state.map(np.asarray))
        np.testing.assert_array_equal(actual.values(states),expected.values(states))

    def test_terminal_continuation_is_zero_and_discount_uses_sessions(self):
        records,daily,cutoff=training_inputs()
        records['reward']=1.;records['squared_reward']=1.
        records['sessions']=3;records['terminal']=True
        FixedPredictionForest.targets=[]
        with patch('scripts.p25_rl.action_q.ActionForest',FixedPredictionForest):
            fitted=fit_policy(records,daily,cutoff,deepcopy(SPECS['action_q_capacity']))
        self.assertEqual(fitted.audit['terminal_transitions'],len(records))
        np.testing.assert_array_equal(FixedPredictionForest.targets[1],np.ones(len(records)))
        records['terminal']=False
        FixedPredictionForest.targets=[]
        with patch('scripts.p25_rl.action_q.ActionForest',FixedPredictionForest):
            fit_policy(records,daily,cutoff,deepcopy(SPECS['action_q_capacity']))
        np.testing.assert_allclose(FixedPredictionForest.targets[1],
                                  np.full(len(records),1.+.99**3*100.))


if __name__=='__main__':unittest.main()
