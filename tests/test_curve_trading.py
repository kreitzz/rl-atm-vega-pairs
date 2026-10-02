import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
from tests.test_expiry_holding import fixture
from scripts.curve_trading.config import COMBOS, ACTIONS, LADDER
from scripts.curve_trading.unit_paths import UnitPaths, pair_labels
from scripts.option_flex.contract_learning_labels import ShadowLabels
from scripts.option_flex.batched_contract_labels import build_labels


class CurveTradingTests(unittest.TestCase):
    def compare(self, cp='C', split=False, fallback=False, missing=False, capital=1_000_000):
        q,s,_,cal,cfg=fixture()
        cfg=replace(cfg,capital=capital,entry_vega_per_leg=80,max_gross_vega=1000,
                    option_penalty_vol_points=.1,stock_cost_bps=1,stock_borrow_rate=.003,
                    borrowing_spread=.01,support_integer_splits=True)
        q['cp_flag']=cp;q['delta']=.5 if cp=='C' else -.5
        q['bs_rate']=.04;s['rate']=.04
        s.loc[s.secid.eq(1),'close']=98.
        if split:
            use=q.date.ge(cal[5])&q.secid.eq(1)
            q.loc[use,['bid','ask','strike','vega']]/=2;q.loc[use,'cfadj']=2.
            use=s.date.ge(cal[5])&s.secid.eq(1)
            s.loc[use,'close']/=2;s.loc[use,'cfadj']=2.
        if fallback:q.loc[q.date.eq(cal[3]),['delta','vega','bs_iv']]=np.nan
        if missing:q=q[~(q.date.eq(cal[4])&q.secid.eq(1))].copy()
        div=pd.DataFrame([dict(secid=1,ex_date=cal[6],declare_date=cal[0],amount=.1)])
        sel=q[['date','secid','optionid','cp_flag']].copy();sel['tenor']=30
        sel['bucket']=np.where(sel.secid.eq(1),'25','75')
        dates=cal[:2]
        units=[UnitPaths(q[q.secid.eq(sid)],s[s.secid.eq(sid)],sel[sel.secid.eq(sid)],cal,dates,
                         div[div.secid.eq(sid)],cfg) for sid in [1,2]]
        fast,ends,vegas,status=pair_labels(*units,(30,cp,25,75),cfg)
        old=ShadowLabels(q,s,sel,cal,[1,2],cfg,div)
        expected,e,v,st=build_labels(old,np.array([0,1]),batch_size=2)
        keep=[0,1,2,3,5,6,7,8]
        np.testing.assert_allclose(fast,expected[:,keep],atol=1e-8,equal_nan=True)
        np.testing.assert_allclose(vegas,v[:,keep])
        if not missing:np.testing.assert_array_equal(ends,e)
        self.assertEqual(status[0],'unresolved' if missing else 'complete')

    def test_lifecycle_matches_reference(self):
        for cp in ['C','P']:
            for kwargs in [{},{'split':True},{'fallback':True},{'missing':True},{'capital':10}]:
                with self.subTest(cp=cp,**kwargs):self.compare(cp,**kwargs)

    def test_action_space(self):
        self.assertEqual(len(COMBOS),216)
        self.assertEqual(len(ACTIONS),1729)
        self.assertEqual({c[0] for c in COMBOS},{14,30,45})
        self.assertTrue(any(c[2]!=c[3] for c in COMBOS))
        self.assertTrue(all(c[1] in ['C','P'] for c in COMBOS))

    def test_memory_is_strictly_completed_and_matches_reference(self):
        from scripts.contract_learning_labels import profitability_memory
        from scripts.curve_trading.policy import memory_gates
        dates=pd.bdate_range('2019-01-01',periods=400)
        ends=dates+pd.to_timedelta(np.arange(400)%40+7,unit='D')
        rewards=np.column_stack([np.sin(np.arange(400)/30),np.cos(np.arange(400)/20)])
        entered=np.ones_like(rewards,dtype=bool);entered[::7]=False;rewards[::19]=np.nan
        memory=profitability_memory(dates,ends,rewards,entered)
        expected=np.column_stack([(memory[f'profit_{side}_91d'].gt(0)&memory[f'count_{side}_91d'].ge(21)&
                                   memory[f'profit_{side}_365d'].gt(0)&memory[f'count_{side}_365d'].ge(63))
                                  for side in ['short','long']])
        np.testing.assert_array_equal(memory_gates(dates,ends,rewards,entered),expected)
        changed=rewards.copy();changed[ends>=dates[250]]=1e9
        np.testing.assert_array_equal(memory_gates(dates,ends,changed,entered)[:251],expected[:251])

    def test_search_can_select_different_strikes_and_tenor(self):
        from scripts.curve_trading.policy import ATM,fit,predict
        n=126;combo=COMBOS.index((45,'P',25,75));codes=[0]+list(range(1+ATM*8,1+(ATM+1)*8))+list(range(1+combo*8,1+(combo+1)*8))
        labels=np.zeros((n,len(codes),3));labels[:,-1,0]=1+np.sin(np.arange(n))*.1
        x=pd.DataFrame({'state':np.zeros(n)})
        model,_=fit(x,labels,np.arange(n),np.arange(n)+3,codes)
        self.assertTrue((predict(model,x)==codes[-1]).all())

    def test_replay_uses_independent_strikes_same_type(self):
        from scripts.curve_trading.policy import Replay
        from scripts.option_flex.contract_engine import simulate
        q,s,a,cal,cfg=fixture();q['cp_flag']='P';q['delta']=-.5
        sel=q[['date','secid','optionid','cp_flag']].copy();sel['tenor']=14;sel['bucket']=np.where(sel.secid.eq(1),'25','75')
        div=pd.DataFrame(columns=['secid','ex_date','declare_date','amount'])
        combo=COMBOS.index((14,'P',25,75));codes=pd.Series([ACTIONS.index((combo,1.))],index=a.index)
        actual=Replay(q,s,sel,[1,2],div,cfg).run(codes,cal)
        expected=simulate(q,s,a,cal,[1,2],cfg,div,sel)
        for left,right in zip(actual,expected):pd.testing.assert_frame_equal(left,right)
        empty,empty_trades,_=Replay(q,s,sel,[1,2],div,cfg).run(codes.iloc[:0],cal)
        self.assertTrue(empty_trades.empty)
        np.testing.assert_allclose(empty.nav,cfg.capital)

    def test_selection_tolerance_and_exact_atm(self):
        from scripts.prepare_curve_contracts import choose
        date=pd.Timestamp('2020-01-02');expiry=pd.Timestamp('2020-01-31')
        anchor=pd.DataFrame([dict(date=date,secid=1,expiry=expiry,anchor_id=3,anchor_strike=100.,tenor=30)])
        q=pd.DataFrame([dict(date=date,secid=1,expiry=expiry,optionid=i,strike=k,bs_status='ok',bs_vega=.1,
                            bs_delta=delta,relative_width=.1,open_interest=200) for i,k,delta in
                           [(1,110,.24),(2,111,.26),(3,100,.52),(4,90,.74),(5,120,.02)]])
        choices=choose(q,anchor,'C').set_index('bucket')
        self.assertEqual(choices.loc['atm','optionid'],3)
        self.assertEqual(choices.loc['25','optionid'],1)
        self.assertEqual(choices.loc['75','optionid'],4)
        self.assertNotIn('10',choices.index)
        self.assertNotIn('90',choices.index)

    def test_shortlist_does_not_use_validation_or_test_returns(self):
        from scripts.run_curve_trading import annual_shortlist
        from scripts.curve_trading.policy import ATM
        cal=pd.bdate_range('2011-01-01','2020-12-31')
        dates=cal[::10];n=len(dates);positions=cal.get_indexer(dates)
        panel=pd.DataFrame({'eligible':True},index=dates)
        labels=np.zeros((216,n,8,2),np.float32);ends=np.tile(np.minimum(positions+2,len(cal)-1),(216,1))
        gates=np.ones((216,n,2),bool);chosen_combo=COMBOS.index((45,'P',25,75))
        labels[chosen_combo,:,7,0]=1+np.sin(np.arange(n))*.1
        selected,records=annual_shortlist(2018,panel,labels,ends,gates,cal)
        self.assertEqual(selected,[ATM,chosen_combo])
        labels[:,dates>='2017-01-01']=1e8
        again,new_records=annual_shortlist(2018,panel,labels,ends,gates,cal)
        self.assertEqual(selected,again);self.assertEqual(records,new_records)

    def test_gate_uses_chosen_combination_direction(self):
        from scripts.curve_trading.policy import gate
        dates=pd.bdate_range('2020-01-01',periods=2);panel=pd.DataFrame({'eligible':True},index=dates)
        available=np.ones((216,2),bool);gates=np.ones((216,2,2),bool)
        c=COMBOS.index((14,'P',25,75));gates[c,0,1]=False
        codes=[ACTIONS.index((c,1.)),ACTIONS.index((c,-1.))]
        self.assertEqual(gate(codes,dates,panel,gates,available).tolist(),[0,codes[1]])


if __name__=='__main__':unittest.main()
