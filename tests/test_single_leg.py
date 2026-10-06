import unittest
import tempfile
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
from tests.test_expiry_holding import fixture
from scripts.option_flex.contract_engine import simulate as paired
from scripts.option_flex.single_leg_engine import simulate,WEIGHTS
from scripts.curve_trading.unit_paths import UnitPaths
from scripts.single_leg_policy import single_labels,add_vrp,single_conditions,fit,predict
from scripts.run_selected_pair_backtest import audit
from scripts.run_aligned_contract_learning import mark_book


class SingleLegTest(unittest.TestCase):
    def test_pair_actions_preserve_reference_nav_and_trades(self):
        q,s,_,cal,cfg=fixture()
        cfg=replace(cfg,entry_vega_per_leg=80,max_gross_vega=1000,stock_cost_bps=1)
        for code in range(1,9):
            old=paired(q,s,pd.Series([WEIGHTS[code][0]],index=cal[:1]),cal,[1,2],cfg)
            new=simulate(q,s,pd.Series([code],index=cal[:1]),cal,[1,2],cfg)
            pd.testing.assert_frame_equal(old[0],new[0][old[0].columns])
            pd.testing.assert_frame_equal(old[1],new[1])

    def test_single_inventory_own_expiry_and_shadow_rewards(self):
        q,s,_,cal,cfg=fixture();s['rate']=.04;q['bs_rate']=.04
        cfg=replace(cfg,entry_vega_per_leg=80,max_gross_vega=1000,stock_cost_bps=1,
                    stock_borrow_rate=.003,option_penalty_vol_points=.1)
        div=pd.DataFrame(columns=['secid','ex_date','declare_date','amount'])
        selection=q[['date','secid','optionid','cp_flag']].copy();selection['tenor']=30;selection['bucket']='atm'
        for leg in [0,1]:
            sid=leg+1
            unit=UnitPaths(q[q.secid.eq(sid)],s[s.secid.eq(sid)],selection[selection.secid.eq(sid)],cal,cal[:1],div,cfg)
            daily,ends,_,_=single_labels(unit,cfg)
            for offset in range(8):
                code=9+leg*8+offset
                nav,trades,decisions=simulate(q,s,pd.Series([code],index=cal[:1]),cal,[1,2],cfg,div,selection)
                mark_book(nav,pd.Series(.04,index=cal),cfg.capital);audit(nav,trades,s,cfg,[1,2])
                self.assertEqual(trades.secid.unique().tolist(),[sid])
                self.assertTrue(nav[f'hedge_{2-leg}'].eq(0).all())
                self.assertEqual(decisions.entry_kind.iloc[0],'single_a' if leg==0 else 'single_b')
                expected=nav.pnl-nav.cash_benchmark.diff().fillna(0)
                np.testing.assert_allclose(daily[0,offset,:ends[0]],expected.iloc[1:ends[0]+1]/cfg.entry_vega_per_leg,atol=1e-8)
                self.assertEqual(trades[trades.reason.eq('expiry_settlement')].date.iloc[0],
                    pd.Timestamp('2025-01-31' if leg==0 else '2025-02-07'))

    def test_net_vega_entry_limit_blocks_single(self):
        q,s,_,cal,cfg=fixture()
        _,trades,decisions=simulate(q,s,pd.Series([16],index=cal[:1]),cal,[1,2],cfg,max_single_net_vega=5.)
        self.assertTrue(trades.empty)
        self.assertEqual(decisions.status.iloc[0],'single_net_vega_limit')

    def test_single_result_saves_with_its_own_inventory_audit(self):
        from scripts.run_atm65_single_leg import save_result
        q,s,_,cal,cfg=fixture()
        cfg=replace(cfg,entry_vega_per_leg=80,stock_cost_bps=1)
        nav,trades,decisions=simulate(q,s,pd.Series([16],index=cal[:1]),cal,[1,2],cfg)
        mark_book(nav,pd.Series(0.,index=cal),cfg.capital)
        with tempfile.TemporaryDirectory() as directory:
            result=save_result(Path(directory),nav,trades,decisions,cfg,s,dict(pair='fixture',label='A/B',secids=[1,2]))
            self.assertEqual(result['status'],'complete')
            self.assertEqual(result['entries'],1)
            self.assertLess(result['ledger_error'],1e-7)

    def test_historical_and_absolute_conditions_are_distinct(self):
        p=pd.DataFrame(dict(vrp_z_a=[-1.,1.],vrp_z_b=[-1.,1.],vrp_z_a_available=1.,vrp_z_b_available=1.,
            vrp_level_a=.1,vrp_level_b=.1,iv_amd=30.,iv_nvda=30.,trailing_rv_amd=20.,trailing_rv_nvda=20.))
        cheap,rich=single_conditions(p,'historical')
        np.testing.assert_array_equal(cheap,[True,False]);np.testing.assert_array_equal(rich,[False,True])
        cheap,rich=single_conditions(p,'absolute')
        self.assertFalse(cheap.any());self.assertTrue(rich.all())

    def test_normalization_uses_only_previous_history(self):
        dates=pd.bdate_range('2020-01-02',periods=160)
        p=pd.DataFrame(dict(iv_amd=30+np.sin(np.arange(160)),iv_nvda=35+np.cos(np.arange(160)),
            trailing_rv_amd=20.,trailing_rv_nvda=22.),index=dates)
        left=add_vrp(p);p.loc[dates[141]:,'iv_amd']=1000.
        right=add_vrp(p);pd.testing.assert_frame_equal(left.iloc[:141],right.iloc[:141])
        self.assertTrue(left.vrp_z_a_available.iloc[:126].eq(0).all())
        self.assertEqual(left.vrp_z_a_available.iloc[126],1.)
        premium=(left.iv_amd/100)**2-(left.trailing_rv_amd/100)**2
        self.assertAlmostEqual(left.vrp_z_a.iloc[126],(premium.iloc[126]-premium.iloc[:126].mean())/premium.iloc[:126].std())

    def test_sharpe_optimizer_can_choose_single_actions(self):
        n=126;x=pd.DataFrame({'constant':np.zeros(n)});daily=np.zeros((n,25,2))
        daily[:,16,0]=1+.1*np.sin(np.arange(n))
        model,_=fit(x,daily,np.arange(n),np.arange(n)+2,list(range(25)))
        self.assertTrue((predict(model,x)==16).all())
        control,_=fit(x,daily,np.arange(n),np.arange(n)+2,list(range(9)))
        self.assertTrue((predict(control,x)==0).all())


if __name__=='__main__':unittest.main()
