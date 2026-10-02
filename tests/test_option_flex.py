import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
from scripts.option_flex.black_scholes import option_values,invert_options
from scripts.option_flex.contract_engine import simulate,valid_greeks
from scripts.option_flex.contract_learning_labels import ShadowLabels,LADDER
from scripts.option_flex.batched_contract_labels import build_labels
from scripts.option_flex.policy import Replay,ACTIONS,FLAT
from scripts.run_aligned_contract_learning import mark_book
from scripts.run_selected_pair_backtest import audit
from tests.test_expiry_holding import fixture


class OptionFlexTests(unittest.TestCase):
    def test_put_parity_iv_and_finite_difference_greeks(self):
        s=np.array([80.,100.,130.]);k=100.;t=.15;r=.05;v=.4
        c=option_values(s,k,t,r,v,'C');p=option_values(s,k,t,r,v,'P')
        np.testing.assert_allclose(c[0]-p[0],s-k*np.exp(-r*t))
        np.testing.assert_allclose(c[1]-p[1],1.)
        np.testing.assert_allclose(c[2:],p[2:])
        np.testing.assert_allclose(invert_options(s,k,t,r,p[0],'P')[0],v,atol=1e-10)
        e=.001;up=option_values(s+e,k,t,r,v,'P')[0];dn=option_values(s-e,k,t,r,v,'P')[0]
        np.testing.assert_allclose(p[1],(up-dn)/(2*e),rtol=1e-6)
        self.assertTrue(valid_greeks(dict(cp_flag='P',delta=-.5,vega=.1)))
        self.assertFalse(valid_greeks(dict(cp_flag='P',delta=.5,vega=.1)))

    def compare(self,combo,fallback=False,risk=False,split=False):
        q,s,a,d,c=fixture();c=replace(c,entry_vega_per_leg=80,max_gross_vega=1000,option_penalty_vol_points=.1,
            stock_cost_bps=1,stock_borrow_rate=.003,risk_controls=risk)
        for sid,cp in zip([1,2],combo):q.loc[q.secid.eq(sid),'cp_flag']=cp;q.loc[q.secid.eq(sid),'delta']=.5 if cp=='C' else -.5
        s['rate']=.04;q['bs_rate']=.04
        s.loc[s.secid.eq(1),'close']=98.
        if split:
            c=replace(c,support_integer_splits=True)
            use=q.date.ge(d[5])&q.secid.eq(1)
            q.loc[use,['bid','ask','strike','vega']]/=2;q.loc[use,'cfadj']=2.
            use=s.date.ge(d[5])&s.secid.eq(1)
            s.loc[use,'close']/=2;s.loc[use,'cfadj']=2.
        if fallback:q.loc[q.date.eq(d[3]),['delta','vega','bs_iv']]=np.nan
        div=pd.DataFrame(columns=['secid','ex_date','declare_date','amount']);selected=q[['date','secid','optionid']]
        maker=ShadowLabels(q,s,selected,d,[1,2],c,div)
        daily,ends,vegas,status=build_labels(maker,np.array([0]),batch_size=1)
        self.assertEqual(status[0],'complete')
        for j,action in enumerate(LADDER):
            nav,trades,dec=simulate(q,s,pd.Series([action],index=d[:1]),d,[1,2],c,div,selected)
            mark_book(nav,pd.Series(.04,index=d),c.capital);audit(nav,trades,s,c,[1,2])
            excess=nav.pnl-nav.cash_benchmark.diff().fillna(0)
            np.testing.assert_allclose(daily[0,j,:ends[0]],excess.iloc[1:ends[0]+1]/c.entry_vega_per_leg,atol=1e-8)
            if action:
                settlements=trades[trades.reason.eq('expiry_settlement')].set_index('secid')
                self.assertEqual(settlements.loc[1,'price'],(1. if split else 2.) if combo[0]=='P' else 0.)
        codes=pd.Series([ACTIONS.index((combo,1.))],index=d[:1])
        replay=Replay(q,s,{combo:selected},[1,2],div,c)
        rn,_,_=replay.run(codes,d)
        expected,_,_=simulate(q,s,a,d,[1,2],c,div,selected)
        pd.testing.assert_frame_equal(rn,expected)

    def test_all_combinations_lifecycle_and_labels(self):
        for combo in ['CC','CP','PC','PP']:
            with self.subTest(combo=combo):self.compare(combo)

    def test_put_fallback_and_risk_pricing(self):self.compare('PP',fallback=True,risk=True)

    def test_put_split_preserves_cashflows(self):self.compare('PP',split=True)

    def test_calls_only_engine_is_unchanged(self):
        from scripts.contract_engine import simulate as reference
        q,s,a,d,c=fixture()
        left=simulate(q,s,a,d,[1,2],c);right=reference(q,s,a,d,[1,2],c)
        for x,y in zip(left,right):pd.testing.assert_frame_equal(x,y)

    def test_independent_leg_selection_and_vega_match(self):
        q,s,a,d,c=fixture();put=q.copy();put['cp_flag']='P';put['delta']=-.5;put['optionid']+=10
        quotes=pd.concat([q,put],ignore_index=True)
        selected=pd.concat([put[put.secid.eq(1)],q[q.secid.eq(2)]])[['date','secid','optionid']]
        replay=Replay(quotes,s,{'PC':selected},[1,2],pd.DataFrame(columns=['secid','ex_date','declare_date','amount']),c)
        nav,t,dec=replay.run(pd.Series([ACTIONS.index(('PC',1.))],index=d[:1]),d)
        entry=t[t.reason.eq('entry')].set_index('secid')
        self.assertEqual(entry.loc[1,'optionid'],11);self.assertEqual(entry.loc[2,'optionid'],2)
        self.assertGreater(entry.loc[1,'quantity'],0);self.assertLess(entry.loc[2,'quantity'],0)
        self.assertGreater(nav.hedge_1.iloc[1],0);self.assertGreater(nav.hedge_2.iloc[1],0)
        entered=dec[dec.status.eq('entered')].iloc[0]
        self.assertLessEqual(abs(entered.vega_a-entered.vega_b)/max(entered.vega_a,entered.vega_b),c.max_entry_vega_imbalance)
        self.assertEqual(len(ACTIONS),33);self.assertEqual(ACTIONS[FLAT],('CC',0.))

    def test_sharpe_search_can_select_puts(self):
        from scripts.option_flex.policy import fit,predict
        n=126;x=pd.DataFrame({'state':np.zeros(n)});daily=np.zeros((n,33,2))
        daily[:,32,0]=1+np.sin(np.arange(n))*.1
        model,_=fit(x,daily,np.arange(n),np.arange(n)+2)
        self.assertTrue((predict(model,x)==32).all())

    def test_memory_gate_uses_selected_combination_and_direction(self):
        from scripts.option_flex.policy import gate,COMBOS
        index=pd.date_range('2020-01-01',periods=2);p=pd.DataFrame({'eligible':[True,True]},index=index)
        for c in COMBOS:
            p[c+'_available']=True
            for side in ['long','short']:
                for days in [91,365]:
                    p[f'{c}_profit_{side}_{days}d']=1.;p[f'{c}_count_{side}_{days}d']=100.
        p['CP_profit_long_91d']=-1.
        codes=pd.Series([ACTIONS.index(('CP',1.)),ACTIONS.index(('PP',-.75))],index=index)
        result=gate(codes,p)
        self.assertEqual(result.iloc[0],FLAT);self.assertEqual(result.iloc[1],codes.iloc[1])

    def test_delta_hedged_put_call_equivalence_without_carry_or_costs(self):
        q,s,a,d,c=fixture();c=replace(c,option_commission=0,stock_cost_bps=0,stock_borrow_rate=0,
            entry_vega_per_leg=80,max_gross_vega=1000)
        for sid in [1,2]:
            mask=s.secid.eq(sid);s.loc[mask,'close']=100+np.sin(np.arange(mask.sum())/3)*(6 if sid==1 else 3)
        paths=[]
        for combo in ['CC','CP','PC','PP']:
            own=q.copy()
            for idx,row in own.iterrows():
                cp=combo[int(row.secid)-1];spot=s.loc[s.date.eq(row.date)&s.secid.eq(row.secid),'close'].iloc[0]
                price,delta,gamma,vega=option_values(spot,row.strike,(row.expiry-row.date).days/365,0,.4,cp)
                own.loc[idx,['cp_flag','bid','ask','delta','vega']]=[cp,price,price,delta,vega]
            nav,trades,_=simulate(own,s,a,d,[1,2],c)
            self.assertTrue(trades.reason.eq('entry').any())
            paths.append(nav.nav.to_numpy())
        for path in paths[1:]:np.testing.assert_allclose(path,paths[0],atol=1e-8,rtol=0)
