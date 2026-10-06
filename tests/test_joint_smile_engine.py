import importlib.util
from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from scripts.curve_trading.config import config
from scripts.option_flex.contract_engine import MarketRow
from scripts.option_flex.pooled_tail_engine import simulate as original
from scripts.joint_smile_engine import simulate

spec=importlib.util.spec_from_file_location('joint_fixture',Path(__file__).with_name('test_put75_cost_hurdle.py'))
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)


class Market:
    def __init__(self,q,s,divs):
        self.q={(r.date,int(r.optionid)):MarketRow(r._asdict()) for r in q.itertuples(index=False)}
        self.s={(r.date,int(r.secid)):(r.close,r.cfadj) for r in s.itertuples(index=False)}
        self.divs=divs
    def quote(self,d,o):return self.q.get((d,o))
    def stock(self,d,s):return self.s[d,s]
    def rate(self,d):return .03
    def dividend_pv(self,*args):return 0.


class JointEngineTest(unittest.TestCase):
    def inputs(self):
        inputs=fixture.CostHurdleTest().inputs
        a,b=[inputs(scale) for scale in [1,2]]
        q=pd.concat([a[0],b[0]],ignore_index=True)
        s=pd.concat([a[1].assign(secid=1),b[1].assign(secid=2)],ignore_index=True)
        sel=pd.concat([a[2].assign(secid=1),b[2].assign(secid=2)],ignore_index=True)
        divs=pd.DataFrame(columns=['secid','ex_date','declare_date','amount'])
        return q,s,sel,a[3],divs

    def signals(self,cal,direction=1,level=3):
        return pd.DataFrame([dict(date=cal[0],pair='1__2',wing='P75',secid_a=1,secid_b=2,
                                  optionid_a=1,optionid_b=2,direction=direction,base_level=level)])

    def test_one_pair_matches_frozen_engine_in_both_directions(self):
        q,s,sel,cal,divs=self.inputs();cfg=replace(config(),option_commission=.03)
        for direction in [-1,1]:
            signals=self.signals(cal,direction)
            actions=pd.Series(direction*.75,index=cal[:1])
            old,t,d=original(q,s,actions,cal,[1,2],cfg,divs,sel)
            new,nt,nd,audit=simulate(Market(q,s,divs),signals,cal,cfg,cfg.capital)
            for col in ['nav','cash','option_value','stock_value','gross_vega','net_vega','hedge_cost','interest','stock_borrow_cost']:
                np.testing.assert_allclose(old[col],new[col],atol=1e-7,rtol=0)
            self.assertEqual(len(t),len(nt));self.assertEqual(nd.status.iloc[0],'entered')
            self.assertLess(audit['max_ledger_error'],1e-7)

    def test_same_contract_shared_hedge_offsets_across_pair_labels(self):
        q,s,sel,cal,divs=self.inputs();cfg=replace(config(),option_commission=.03)
        signals=pd.concat([self.signals(cal,1,4),self.signals(cal,-1,4).assign(pair='reverse')],ignore_index=True)
        nav,trades,decisions,audit=simulate(Market(q,s,divs),signals,cal,cfg,2*cfg.capital)
        self.assertEqual(decisions.status.eq('entered').sum(),2)
        np.testing.assert_allclose(nav.gross_hedge_notional,0.,atol=1e-7)
        np.testing.assert_allclose(nav.hedge_cost,0.,atol=1e-7)
        self.assertGreater(nav.option_execution_cost.sum(),0)
        self.assertLess(audit['max_ledger_error'],1e-7)

    def test_policy_uses_previous_close_inventory_and_cannot_reverse(self):
        q,s,sel,cal,divs=self.inputs();cfg=replace(config(),option_commission=.03)
        rows=self.signals(cal,1,4)
        rows=pd.concat([rows,rows.assign(date=cal[1])],ignore_index=True)
        seen=[]
        def policy(row,context):
            seen.append(context.copy())
            return 4 if context['pair_inventory_cohorts']==0 else 0
        nav,trades,decisions,audit=simulate(Market(q,s,divs),rows,cal,cfg,cfg.capital,policy)
        self.assertEqual(decisions.level.tolist(),[4,0])
        self.assertEqual(seen[0]['pair_inventory_cohorts'],0)
        self.assertGreater(seen[1]['pair_inventory_cohorts'],0)
        self.assertLess(audit['max_ledger_error'],1e-7)

    def test_missing_selected_contract_is_kept_as_unavailable_decision(self):
        q,s,sel,cal,divs=self.inputs();cfg=replace(config(),option_commission=.03)
        signals=self.signals(cal);signals['optionid_a']=np.nan
        nav,trades,decisions,audit=simulate(Market(q,s,divs),signals,cal,cfg,cfg.capital)
        self.assertEqual(decisions.status.tolist(),['no_eligible_contract'])
        self.assertTrue(trades.empty)
        self.assertEqual(nav.active_cohorts.max(),0)

    def test_native_wing_limits_preserve_reservations_with_shared_cash(self):
        q,s,sel,cal,divs=self.inputs()
        cfg=replace(config(),option_commission=.03,capital=3_000_000,max_gross_vega=2700)
        signals=pd.concat([self.signals(cal,1,4),self.signals(cal,1,4).assign(date=cal[1])],ignore_index=True)
        _,_,pooled,_=simulate(Market(q,s,divs),signals,cal,cfg,cfg.capital)
        _,_,native,_=simulate(Market(q,s,divs),signals,cal,cfg,cfg.capital,wing_capital=1_000_000)
        self.assertEqual(pooled.status.eq('entered').sum(),2)
        self.assertEqual(native.status.eq('entered').sum(),1)
        signals.loc[1,'wing']='P25'
        _,_,separate,_=simulate(Market(q,s,divs),signals,cal,cfg,cfg.capital,wing_capital=1_000_000)
        self.assertEqual(separate.status.eq('entered').sum(),2)


if __name__=='__main__':unittest.main()
