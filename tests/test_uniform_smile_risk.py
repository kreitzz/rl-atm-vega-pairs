import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
import importlib.util
from pathlib import Path

from scripts.uniform_smile_risk import book_risk
from scripts.option_flex.contract_risk import book_risk as original
from scripts.option_flex.contract_engine import MarketRow
from scripts.option_flex.black_scholes import option_values
from scripts.curve_trading.config import config


class UniformRiskTest(unittest.TestCase):
    def test_random_mixed_cohorts_and_incremental_trials_match_original(self):
        rng=np.random.default_rng(51);date=pd.Timestamp('2020-06-01');cfg=config()
        spots={1:60.,2:120.,3:200.};positions=[];quotes={}
        for n in range(60):
            sid=int(rng.integers(1,4));expiry=date+pd.Timedelta(days=int(rng.integers(7,55)));strike=spots[sid]*rng.uniform(.75,1.25)
            cp='C' if n%2 else 'P';iv=rng.uniform(.15,1.5);rate=.03;pv=.2
            mid,delta,_,vega=option_values(spots[sid]-pv,strike,(expiry-date).days/365,rate,iv,cp)
            quotes[n]=MarketRow(delta=float(delta),vega=float(vega),bs_iv=iv,bs_rate=rate,known_dividend_pv=pv,expiry=expiry,strike=strike,cp_flag=cp,bid=max(0.,float(mid)-.01),ask=float(mid)+.01)
            positions.append(dict(secid=sid,cohort=str(n),optionid=n,quantity=int(rng.choice([-100,-5,3,25])),strike=strike,cfadj=1.,expiry=expiry))
        cache={}
        for count in [0,2,8,30,60,17,60]:
            a=original(positions[:count],lambda p:quotes[p['optionid']],spots,date,cfg,{})
            b=book_risk(positions[:count],lambda p:quotes[p['optionid']],spots,date,cfg,cache)
            self.assertEqual(a.keys(),b.keys())
            for key in a:
                if key=='target_hedge':np.testing.assert_allclose(list(a[key].values()),list(b[key].values()),rtol=1e-12,atol=1e-8)
                else:np.testing.assert_allclose(a[key],b[key],rtol=1e-12,atol=1e-8)
        changed=positions[0].copy();changed['quantity']*=2
        a=original([changed],lambda p:quotes[p['optionid']],spots,date,cfg,{})
        b=book_risk([changed],lambda p:quotes[p['optionid']],spots,date,cfg,cache)
        self.assertAlmostEqual(a['gross_vega'],b['gross_vega'])

    def test_entire_shared_cash_replay_matches_original_risk(self):
        spec=importlib.util.spec_from_file_location('uniform_engine_fixture',Path(__file__).with_name('test_joint_smile_engine.py'))
        fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
        q,s,sel,cal,divs=fixture.JointEngineTest().inputs()
        from scripts import joint_smile_engine as engine
        signals=fixture.JointEngineTest().signals(cal,1,4)
        signals=pd.concat([signals,signals.assign(date=cal[1],direction=-1)],ignore_index=True)
        market=fixture.Market(q,s,divs);cfg=replace(config(),option_commission=.03)
        a,ta,da,_=engine.simulate(market,signals,cal,cfg,cfg.capital)
        from scripts import uniform_smile_engine as engine
        saved=engine.book_risk
        try:
            engine.book_risk=book_risk
            b,tb,db,_=engine.simulate(market,signals,cal,cfg,cfg.capital)
        finally:engine.book_risk=saved
        for col in ['nav','cash','gross_vega','net_vega','gross_hedge_notional','stress_loss','hedge_cost','stock_borrow_cost']:
            np.testing.assert_allclose(a[col],b[col],rtol=0,atol=1e-7)
        pd.testing.assert_frame_equal(ta,tb);pd.testing.assert_frame_equal(da,db)

    def test_declared_unsupported_distribution_exits_both_legs_causally(self):
        spec=importlib.util.spec_from_file_location('distribution_fixture',Path(__file__).with_name('test_joint_smile_engine.py'))
        fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
        q,s,sel,cal,divs=fixture.JointEngineTest().inputs()
        from scripts.uniform_smile_engine import simulate
        signals=fixture.JointEngineTest().signals(cal,1,4)
        mask=q.date.eq(cal[2])&q.secid.eq(1)
        q.loc[mask,'bs_status']='unsupported_distribution'
        q.loc[mask,['delta','vega','bs_iv']]=np.nan
        market=fixture.Market(q,s,divs);cfg=replace(config(),option_commission=.03)
        nav,trades,decisions,_=simulate(market,signals,cal,cfg,cfg.capital)
        exits=trades.loc[trades.reason.eq('announced_corporate_action_exit')]
        self.assertEqual(len(exits),2)
        self.assertTrue(exits.date.eq(cal[2]).all())
        self.assertEqual(set(exits.secid),{1,2})
        self.assertEqual(nav.loc[cal[2],'gross_vega'],0)
        self.assertLess(nav.ledger_error.abs().max(),1e-7)

    def test_same_day_event_not_used_before_announced_date(self):
        spec=importlib.util.spec_from_file_location('same_day_fixture',Path(__file__).with_name('test_joint_smile_engine.py'))
        fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
        q,s,sel,cal,divs=fixture.JointEngineTest().inputs()
        from scripts.uniform_smile_engine import simulate
        signals=fixture.JointEngineTest().signals(cal,1,4);cfg=replace(config(),option_commission=.03)
        event=dict(secid=1,announced=cal[2],effective=cal[2],exit_same_day=True)
        nav,trades,_,_=simulate(fixture.Market(q,s,divs),signals,cal,cfg,cfg.capital,corporate_events=[event])
        self.assertGreater(nav.loc[cal[1],'gross_vega'],0)
        self.assertTrue(trades.loc[trades.reason.eq('announced_corporate_action_exit'),'date'].eq(cal[2]).all())
        event['announced']=cal[3]
        nav,trades,_,_=simulate(fixture.Market(q,s,divs),signals,cal,cfg,cfg.capital,corporate_events=[event])
        self.assertFalse(trades.reason.eq('announced_corporate_action_exit').any())

    def test_known_effective_day_cannot_reopen_after_prior_close_exit(self):
        spec=importlib.util.spec_from_file_location('blackout_fixture',Path(__file__).with_name('test_joint_smile_engine.py'))
        fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
        q,s,sel,cal,divs=fixture.JointEngineTest().inputs()
        from scripts.uniform_smile_engine import simulate
        signals=fixture.JointEngineTest().signals(cal,1,4)
        signals=pd.concat([signals,signals.assign(date=cal[2])],ignore_index=True)
        event=dict(secid=1,announced=cal[1],effective=cal[3])
        nav,trades,decisions,_=simulate(fixture.Market(q,s,divs),signals,cal,replace(config(),option_commission=.03),config().capital,corporate_events=[event])
        self.assertEqual(decisions.status.tolist(),['entered','announced_corporate_action_no_entry'])
        self.assertTrue(trades.loc[trades.reason.eq('announced_corporate_action_exit'),'date'].eq(cal[2]).all())

    def test_hess_deliverable_spot_is_same_session_and_not_backdated(self):
        from scripts.uniform_smile_data import ReplayMarket
        m=ReplayMarket.__new__(ReplayMarket)
        dates=pd.to_datetime(['2025-07-17','2025-07-18','2025-07-21'])
        m.spots=pd.DataFrame({101322:[153.,np.nan,np.nan],102968:[149.,150.,151.]},index=dates)
        m.adjustments=pd.DataFrame({101322:[1.,np.nan,np.nan]},index=dates)
        self.assertEqual(m.stock(dates[0],101322),(153.,1.))
        self.assertEqual(m.stock(dates[1],101322),(153.75,1.))
        self.assertTrue(np.isnan(m.stock(dates[2],101322)[0]))
