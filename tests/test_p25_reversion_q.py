import unittest

import numpy as np
import pandas as pd

from scripts import p25_reversion_q as policy


class ReversionQTest(unittest.TestCase):
    def test_future_prices_do_not_change_state_history(self):
        rng=np.random.default_rng(7)
        spread=pd.Series(rng.normal(0,.02,500).cumsum(),index=pd.bdate_range('2010-01-01',periods=500))
        original=policy.spread_states(spread)
        altered=spread.copy();altered.iloc[300:]=rng.normal(10,50,200)
        pd.testing.assert_frame_equal(original.iloc[:300],policy.spread_states(altered).iloc[:300])
        pd.testing.assert_frame_equal(original.iloc[:300],policy.spread_states(spread.iloc[:300]))

    def test_failed_excursion_is_reported_instead_of_counted_as_fast_reversion(self):
        x=pd.Series(np.r_[np.sin(np.arange(200))*.01,np.ones(150)],index=pd.bdate_range('2010-01-01',periods=350))
        states=policy.spread_states(x)
        self.assertGreater(states.failed_or_censored_fraction.iloc[230],0)
        self.assertGreater(states.active_excursion_age.iloc[230],0)
        self.assertEqual(states.excursion_count.iloc[230],1)
        self.assertTrue(np.isnan(states.reversion_days.iloc[230]))

    def test_training_requires_scheduled_expiry_before_fit(self):
        rows=pd.DataFrame(dict(date=pd.to_datetime(['2017-12-01','2017-12-02','2017-12-03']),
            end=pd.to_datetime(['2017-12-29','2018-01-01','2018-01-02']),finite_label=True))
        np.testing.assert_array_equal(policy.causal_rows(rows,pd.Timestamp('2018-01-01')),[0])
        rows.loc[2,'end']=pd.Timestamp('2025-01-01')
        np.testing.assert_array_equal(policy.causal_rows(rows,pd.Timestamp('2018-01-01')),[0])

    def test_fine_state_borrows_available_coarse_pattern(self):
        # Two histories ending in the same last move share the backoff state.
        state_a=(2*3+1)*3+2
        state_b=(6*3+1)*3+2
        means,support=policy.pooled_mean(np.array([state_a]),np.array([2.]),policy.BASE_STATES,policy.base_backoff,policy.BACKOFF_BASE)
        self.assertAlmostEqual(means[state_b],2.)
        self.assertEqual(support[state_b],1.)

    def test_fixed_entry_menu_cannot_choose_half_size_or_negative_expected_value(self):
        model=dict(entry_q=np.zeros((policy.BASE_STATES,6)),entry_n=np.ones((policy.BASE_STATES,6)))
        model['entry_q'][0]=[10,9,1,8,7,2]
        direction,level=policy.entry_actions(model,np.array([0,1]),sized=False)
        np.testing.assert_array_equal(direction,[1,1]);np.testing.assert_array_equal(level,[4,0])
        direction,level=policy.entry_actions(model,np.array([0]),sized=True)
        np.testing.assert_array_equal(direction,[-1]);np.testing.assert_array_equal(level,[2])

    def test_close_requires_support_and_strict_advantage(self):
        model=dict(close_q=np.array([5.,1.,2.]),hold_q=np.array([0.,1.,1.]),close_n=np.array([0.,10.,10.]))
        np.testing.assert_array_equal(policy.should_close(model,[0,1,2]),[False,False,True])

    def test_bellman_continuation_reaches_entry_value(self):
        s0=int(policy.held_state(0,0,2));s1=int(policy.held_state(1,1,2))
        kwargs=dict(entry_base=np.array([0]),held=np.array([s0,s1]),following=np.array([s1,s1]),
            reward=np.array([1.,4.]),close_reward=np.array([.8,3.8]),executable_close=np.array([True,True]),
            terminal=np.array([False,True]),entry_reward=np.zeros((1,6)),
            entry_next=np.full((1,6),s0),entry_valid=np.ones((1,6),bool),risk_penalty=0)
        model=policy.fit_q(**kwargs)
        self.assertGreater(model['entry_q'][0,2],4.)
        kwargs['reward']=np.array([1.,1.]);kwargs['close_reward']=np.array([.8,.8])
        smaller=policy.fit_q(**kwargs)
        self.assertGreater(model['entry_q'][0,2],smaller['entry_q'][0,2])

    def test_shadow_liquidation_cost_matches_shared_ledger_change(self):
        from dataclasses import replace
        from tests.test_joint_smile_engine import JointEngineTest,Market
        from scripts.curve_trading.config import config
        from scripts.uniform_smile_engine import simulate
        from scripts.run_p25_reversion_q import exit_adjustments
        fixture=JointEngineTest()
        q,s,_,cal,divs=fixture.inputs()
        q=q.assign(bs_status='ok')
        rows=fixture.signals(cal,1,4).assign(position=0)
        cfg=replace(config(),option_commission=.65)
        class Cache:
            def get(self,sid):
                quotes=q.loc[q.secid.eq(sid)].set_index(['date','optionid'])
                stock=s.loc[s.secid.eq(sid)].set_index('date').reindex(cal)
                return quotes,stock
        extra,can_exit,valid=exit_adjustments(rows,Cache(),cal)
        self.assertTrue(valid[0,5]);self.assertTrue(can_exit[0,1])
        hold=simulate(Market(q,s,divs),rows,cal,cfg,cfg.capital)
        close=simulate(Market(q,s,divs),rows,cal,cfg,cfg.capital,management_policy=lambda date,held:0.)
        delta=hold[0].pnl.loc[cal[2]]-close[0].pnl.loc[cal[2]]
        np.testing.assert_allclose(float(extra[0,5,1])*450,delta,atol=1e-5,rtol=0)

    def test_failed_exits_retain_reward_and_continuation(self):
        s0=int(policy.held_state(0,0,2));s1=int(policy.held_state(1,1,2))
        model=policy.fit_q(entry_base=np.array([0]),held=np.array([s0,s1]),following=np.array([s1,s1]),
            reward=np.array([1.,4.]),close_reward=np.array([-100.,-100.]),executable_close=np.array([False,False]),
            terminal=np.array([False,True]),entry_reward=np.zeros((1,6)),
            entry_next=np.full((1,6),s0),entry_valid=np.ones((1,6),bool),risk_penalty=0)
        self.assertGreater(model['close_q'][s0],4.)
        np.testing.assert_allclose(model['close_q'],model['hold_q'],atol=1e-12)
        self.assertFalse(policy.should_close(model,[s0])[0])

    def test_known_expiry_prevents_an_unpaired_next_session_exit(self):
        from scripts.run_p25_reversion_q import paired_exit_survives
        cal=pd.bdate_range('2020-01-01',periods=5)
        held=[dict(exit_date=cal[2]),dict(exit_date=cal[4])]
        self.assertTrue(paired_exit_survives(cal[0],held,cal))
        self.assertFalse(paired_exit_survives(cal[1],held,cal))

    def test_unfillable_entry_counts_as_zero_not_a_success_conditioned_sample(self):
        model=dict(hold_q=np.full(policy.HELD_STATES,10.),close_q=np.zeros(policy.HELD_STATES),
            close_n=np.ones(policy.HELD_STATES),reward_scale=1.,risk_penalty=0.)
        valid=np.ones((2,6),bool);valid[1,0]=False
        policy.fit_entry_head(model,np.array([0,0]),np.zeros((2,6)),np.zeros((2,6),int),valid)
        self.assertAlmostEqual(model['entry_q'][0,0],4.95)
        self.assertAlmostEqual(model['entry_q'][0,1],9.9)


if __name__=='__main__':unittest.main()
