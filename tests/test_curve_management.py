import copy
import tempfile
from pathlib import Path
import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from tests.test_expiry_holding import fixture
from scripts.curve_management.config import COMBOS,ACTIONS,ATM,UNIVERSES
from scripts.curve_management.data import Paths,Replay
from scripts.curve_management.engine import simulate
from scripts.curve_management.learning import apply_stops,exit_stops,shortlist
from scripts.option_flex.contract_engine import simulate as original
from scripts.run_selected_pair_backtest import audit

class CurveManagementTests(unittest.TestCase):
    def setup_market(self):
        q,s,signals,cal,cfg=fixture()
        cfg=replace(cfg,min_dte=7,max_dte=105,option_penalty_vol_points=.1,stock_cost_bps=1,
                    stock_borrow_rate=.003,borrowing_spread=.01)
        s['rate']=.04;q['bs_rate']=.04
        selection=q[['date','secid','optionid','cp_flag']].assign(tenor=30,bucket='atm')
        panel=pd.DataFrame(dict(eligible=True,beta=1.,correlation=1.,vix_previous_close=20.),index=signals.index)
        return q,s,signals,cal,cfg,selection,panel

    def test_compact_market_and_disabled_management_reproduce_original(self):
        q,s,a,cal,cfg,selection,_=self.setup_market()
        div=pd.DataFrame(columns=['secid','ex_date','declare_date','amount'])
        codes=pd.Series([ACTIONS.index((ATM,1.))],index=a.index)
        actual=Replay(q,s,selection,[1,2],div,cfg).run(codes,cal)
        expected=original(q,s,a,cal,[1,2],cfg,div,selection)
        for left,right in zip(actual,expected):pd.testing.assert_frame_equal(left,right)

    def test_exit_labels_match_actual_liquidation_financing_and_hedge_costs(self):
        q,s,a,cal,cfg,selection,panel=self.setup_market()
        div=pd.DataFrame(columns=['secid','ex_date','declare_date','amount'])
        with tempfile.TemporaryDirectory() as temp:
            data=Paths(panel,q,s,selection,div,cal,[1,2],cfg,Path(temp)/'labels.dat')
            original_labels=data.labels[ATM].copy()
            self.assertFalse(data.labels[ATM].flags.writeable)
            tree=DecisionTreeRegressor(max_leaf_nodes=2).fit(np.zeros((64,31)),np.zeros(64))
            leaves=np.unique(tree.apply(np.zeros((1,31))))
            data.subset([ATM],(tree,leaves,np.ones(len(leaves),bool)))
            np.testing.assert_array_equal(data.labels[ATM],original_labels)
            daily=data.labels[ATM][:,7];cost=data.stop_cost(ATM)[:,7]
            stop=np.array([3])  # Decision at entry+2, execution entry+3.
            predicted=apply_stops(daily,cost,stop,data.positions,data.ends[ATM],cal,data.rates)
            nav,trades,_=simulate(q,s,a,cal,[1,2],cfg,div,selection,cohort_exit_schedule={cal[3]:{str(cal[0].date())}})
            audit(nav,trades,s,cfg,[1,2])
            from scripts.replay_frozen_atm_commission import cash_book
            reference=cash_book(cal,pd.Series(.04,index=cal),cfg.capital)
            end=int(data.ends[ATM,0]);expected=(nav.nav.iloc[end]-reference.nav.iloc[end])/cfg.entry_vega_per_leg
            self.assertAlmostEqual(float(predicted.sum()),expected,places=4)
            closed=trades[trades.reason.eq('learned_policy_exit')]
            self.assertEqual(set(closed.secid),{1,2});self.assertTrue(closed.date.eq(cal[4]).all())

    def test_close_is_next_session_and_future_states_cannot_change_earlier_decision(self):
        x=np.zeros((1,12,2),np.float32);valid=np.ones((1,12),bool);x[0,4,0]=1
        train=np.array([[0,0]]*64+[[1,0]]*64);tree=DecisionTreeRegressor(max_leaf_nodes=2).fit(train,np.r_[np.zeros(64),np.ones(64)])
        leaves=np.unique(tree.apply(train));close=np.array([False,True]);model=(tree,leaves,close)
        first=exit_stops(model,x,valid,np.array([0]),np.array([12]))
        altered=x.copy();altered[:,8:]=999
        np.testing.assert_array_equal(first,exit_stops(model,altered,valid,np.array([0]),np.array([12])))
        self.assertEqual(first[0],5)

    def test_multiple_expiries_and_strikes_preserve_each_cohort_inventory(self):
        q,s,a,cal,cfg,selection,_=self.setup_market()
        cal=pd.bdate_range(cal[0],'2025-04-10')
        s=pd.concat([part.set_index('date').reindex(cal).ffill().assign(secid=sid)
                     .rename_axis('date').reset_index() for sid,part in s.groupby('secid')],ignore_index=True)
        rows=[]
        for day in cal:
            for sid in [1,2]:
                row=q[q.secid.eq(sid)].iloc[0].to_dict()
                row.update(date=day,optionid=sid+100,
                           expiry=pd.Timestamp('2025-04-02' if sid==1 else '2025-04-08'),
                           strike=95. if sid==1 else 105.,bid=8.9,ask=9.1)
                if day<row['expiry']:rows.append(row)
        long=pd.DataFrame(rows)
        selected_long=long[['date','secid','optionid','cp_flag']].assign(tenor=90)
        selected_long['bucket']=np.where(selected_long.secid.eq(1),'75','25')
        short=q.copy();short['optionid']+=200
        short['expiry']=pd.to_datetime(np.where(short.secid.eq(1),'2025-01-24','2025-01-27'))
        short=short[short.date.lt(short.expiry)]
        selected_short=short[['date','secid','optionid','cp_flag']].assign(tenor=14,bucket='atm')
        both=pd.concat([q,long,short],ignore_index=True)
        selections=pd.concat([selection,selected_long,selected_short],ignore_index=True)
        combo=COMBOS.index((90,'C',75,25));near=COMBOS.index((14,'C','atm','atm'))
        codes=pd.Series([ACTIONS.index((ATM,1.)),ACTIONS.index((combo,-1.)),ACTIONS.index((near,1.))],index=cal[[0,2,5]])
        div=pd.DataFrame(columns=['secid','ex_date','declare_date','amount'])
        nav,trades,dec=Replay(both,s,selections,[1,2],div,cfg).run(codes,cal,{cal[8]:{str(cal[0].date())}})
        first=trades[trades.cohort.eq(str(cal[0].date()))];second=trades[trades.cohort.eq(str(cal[2].date()))]
        third=trades[trades.cohort.eq(str(cal[5].date()))]
        self.assertEqual(set(first.optionid),{1,2});self.assertEqual(set(second.optionid),{101,102})
        self.assertEqual(set(third.optionid),{201,202});self.assertEqual(nav.active_cohorts.max(),3)
        self.assertFalse(second.reason.eq('learned_policy_exit').any())
        self.assertEqual(set(second[second.reason.eq('expiry_settlement')].date),{pd.Timestamp('2025-04-02'),pd.Timestamp('2025-04-08')})
        self.assertEqual(set(third[third.reason.eq('expiry_settlement')].date),{pd.Timestamp('2025-01-24'),pd.Timestamp('2025-01-27')})
        self.assertEqual(nav.loc['2025-03-03','active_cohorts'],1)
        self.assertTrue(trades.groupby(['cohort','optionid']).quantity.sum().eq(0).all())
        audit(nav,trades,s,cfg,[1,2])

    def test_contract_universes_and_training_only_shortlist(self):
        self.assertEqual(len(UNIVERSES['atm30']),1);self.assertEqual(len(UNIVERSES['tenors']),3)
        self.assertEqual(len(UNIVERSES['smile']),9);self.assertEqual(len(UNIVERSES['both']),27)
        self.assertEqual({c[0] for c in COMBOS},{14,30,90});self.assertTrue(all(c[1]=='C' for c in COMBOS))
        cal=pd.bdate_range('2011-01-01','2020-12-31');dates=cal[::10];n=len(dates)
        panel=pd.DataFrame({'eligible':True},index=dates);positions=cal.get_indexer(dates)
        labels=np.zeros((27,n,8,2),np.float32);ends=np.tile(np.minimum(positions+2,len(cal)-1),(27,1));gates=np.ones((27,n,2),bool)
        winner=COMBOS.index((90,'C',75,25));labels[winner,:,7,0]=1+np.sin(np.arange(n))*.1
        selected,records=shortlist(2018,UNIVERSES['both'],panel,labels,ends,gates,cal)
        self.assertIn(winner,selected)
        labels[:,dates>='2017-01-01']=1e9
        again,new=shortlist(2018,UNIVERSES['both'],panel,labels,ends,gates,cal)
        self.assertEqual(selected,again);self.assertEqual(records,new)

    def test_learned_exit_can_stop_forecast_losses_and_hold_forecast_gains(self):
        from types import SimpleNamespace
        from scripts.curve_management.learning import fit_exit
        n=64;w=6;cal=pd.bdate_range('2012-01-02',periods=460);positions=np.arange(n)*7
        panel=pd.DataFrame({'eligible':True},index=cal[positions])
        x=np.zeros((n,w,2),np.float32);x[:,:,0]=np.where(np.arange(n)%2,1.,-1.)[:,None]
        valid=np.ones((n,w),bool);valid[:,-1]=False
        labels=np.zeros((27,n,8,w),np.float32)
        labels[ATM,:,0,0]=-.02;labels[ATM,:,7,0]=-.02
        labels[ATM,:,7,-1]=x[:,0,0];labels[ATM,:,0,-1]=-x[:,0,0]
        def direction(values,side):
            adjusted=values.copy();adjusted[:,:,0]*=1. if side else -1.
            return np.concatenate([adjusted,np.full((*values.shape[:2],1),1. if side else -1.)],axis=2)
        data=SimpleNamespace(panel=panel,cal=cal,positions=positions,rates=np.zeros(len(cal)),labels=labels,
            ends=np.tile(positions+w,(27,1)),entered=np.ones((27,n),bool),
            held_features=lambda combo:(x,valid),stop_cost=lambda combo:np.full((n,8,w),.01,np.float32),direction_features=direction)
        model,detail=fit_exit(data,np.ones(n,bool),[ATM],cal[-1])
        self.assertIsNotNone(model)
        long=exit_stops(model,direction(x,1),valid,positions,data.ends[ATM])
        short=exit_stops(model,direction(x,0),valid,positions,data.ends[ATM])
        self.assertTrue((long[x[:,0,0]<0]<w-1).all());self.assertTrue((long[x[:,0,0]>0]==w-1).all())
        self.assertTrue((short[x[:,0,0]>0]<w-1).all());self.assertTrue((short[x[:,0,0]<0]==w-1).all())
        self.assertLess(pd.Timestamp(detail['exit_last_outcome']),cal[-1])

if __name__=='__main__':unittest.main()
