import copy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.atm_p25_allocation_q import (N,FEATURES,contract_counts,drift,fit,
    market_features,month_path,solve_fqi,target_weights)


class AllocationQTest(unittest.TestCase):
    def test_actions_balance_risk_without_leverage_and_include_cash(self):
        targets=target_weights(.001,.02)
        np.testing.assert_allclose(targets.sum(axis=1),1)
        self.assertTrue((targets>=0).all())
        np.testing.assert_allclose(targets[0],[0,0,1])
        np.testing.assert_allclose(targets[1],[1,0,0])
        np.testing.assert_allclose(targets[-1],[0,1,0])
        self.assertAlmostEqual(targets[3,0]*.001,targets[3,1]*.02)

    def test_first_session_return_precedes_execution_and_both_sides_pay(self):
        gross=np.array([[1.10,.9,1],[1.,1.10,1.]])
        returns,end,cost=month_path(gross,np.zeros(2),np.array([.01,.01]),
                                  np.array([1.,0,0]),np.array([0.,1,0]))
        np.testing.assert_allclose(returns,[1.1*.98-1,.10])
        self.assertAlmostEqual(cost,.02)
        np.testing.assert_allclose(end,[0,1,0])

    def test_post_rebalance_drift_is_the_action_dependent_next_state(self):
        gross=np.array([[1.1,.9,1],[.8,1.2,1],[1.02,1.03,1.]])
        target=np.array([.8,.2,0])
        expected=drift(target,gross[1:])
        for previous in [[1,0,0],[0,0,1],[0,1,0]]:
            _,end,_=month_path(gross,np.zeros(3),np.array([.001,.001]),previous,target)
            np.testing.assert_allclose(end,expected)

    def test_bellman_continuation_changes_q_values(self):
        x=np.eye(2);rewards=np.array([0.,1.])
        next_x=np.array([[[0.,1.]],[[1.,0.]]])
        immediate=solve_fqi(x,rewards,next_x,0.,0.,100)
        sequential=solve_fqi(x,rewards,next_x,.8,0.,100)
        np.testing.assert_allclose(immediate,[0,1])
        np.testing.assert_allclose(sequential,[.8/(1-.8**2),1/(1-.8**2)],atol=1e-8)

    def test_contract_counts_include_only_known_completed_flows(self):
        dates=pd.bdate_range('2020-01-02',periods=4)
        trades=pd.DataFrame(dict(date=[dates[1],dates[3]],cohort=['a','a'],optionid=[1,1],quantity=[2,-2]))
        np.testing.assert_array_equal(contract_counts(trades,dates),[0,2,2,0])

    def test_future_market_returns_do_not_change_earlier_features(self):
        dates=pd.bdate_range('2020-01-02',periods=160)
        rng=np.random.default_rng(11)
        def book():
            r=rng.normal(.0001,.001,len(dates))
            return pd.DataFrame(dict(excess_return=r,nav=100*(1+r).cumprod(),
                cash_benchmark=100.,capital_reserve=10.),index=dates)
        a,b=book(),book()
        vix=pd.DataFrame(dict(DATE=dates,CLOSE=20.))
        with patch('scripts.atm_p25_allocation_q.pd.read_csv',return_value=vix):
            first=market_features(a,b)
            changed=b.copy();changed.loc[dates[100]:,'excess_return']=-.8
            changed.loc[dates[100]:,'nav']=1.
            second=market_features(a,changed)
        pd.testing.assert_frame_equal(first.iloc[:100],second.iloc[:100])

    def test_current_and_future_unfinished_month_rewards_never_fit(self):
        targets=target_weights(.001,.02);months=[]
        for i,cutoff in enumerate(pd.date_range('2018-01-01',periods=14,freq='MS')):
            paths=np.empty((N,N,3))
            for action in range(N):paths[:,action]=[.0001*(action+1),-.00005*action,.0002]
            months.append(dict(cutoff=cutoff,end=cutoff+pd.offsets.BMonthEnd(),
                feature_date=cutoff-pd.Timedelta(days=1),ready=True,
                features=np.sin(np.arange(len(FEATURES))+i),before=targets,after=targets,paths=paths))
        original=fit(dict(months=months),12,24,1.,.9)
        changed=copy.deepcopy(months)
        changed[12]['paths'][:]=1000.;changed[13]['paths'][:]=-1000.
        repeated=fit(dict(months=changed),12,24,1.,.9)
        self.assertEqual(original,repeated)
        self.assertLess(pd.Timestamp(original['last_training_end']),months[12]['cutoff'])


if __name__=='__main__':unittest.main()
