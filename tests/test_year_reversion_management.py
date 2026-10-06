import unittest
import numpy as np
import pandas as pd

from scripts import year_reversion_management as q
from scripts.p25_reversion_q import spread_states


class YearManagementTest(unittest.TestCase):
    def episode(self):
        n=12
        return dict(dates=pd.bdate_range('2020-01-01',periods=n),state_known=np.ones(n,bool),
            states=np.tile([256*9+5,1,0],(n,1)),can_exit=np.ones(n,bool),can_trim=np.ones(n,bool),
            gross_training=np.ones(n),entry_vega=1.,hedge=np.zeros((n,2)),
            hprev=np.zeros((n,2)),spot=np.ones((n,2))*100,exit_cost_training=np.ones(n)*2)

    def test_online_features_match_causal_offline_history(self):
        values=pd.Series(np.random.default_rng(5).normal(size=350).cumsum()/100)
        offline=spread_states(values)
        history=q.SpreadHistory()
        online=[history.update(v) for v in values]
        np.testing.assert_array_equal([r['paper_base'] for r in online],offline.base_state)
        np.testing.assert_allclose([r['reversion_days'] for r in online],offline.reversion_days,equal_nan=True)

    def test_future_spread_never_changes_prefix(self):
        values=np.random.default_rng(7).normal(size=200)
        a=q.SpreadHistory(); b=q.SpreadHistory()
        first=[a.update(v)['paper_base'] for v in values]
        values[100:]+=1000
        second=[b.update(v)['paper_base'] for v in values]
        self.assertEqual(first[:100],second[:100])

    def test_close_keeps_fill_day_pnl_and_pays_once(self):
        self.assertEqual(q.reward_step(self.episode(),5,10,1.,0.),(-1.,0.))

    def test_failed_close_retains_pnl_and_inventory(self):
        ep=self.episode(); ep['can_exit'][6]=False
        self.assertEqual(q.reward_step(ep,5,10,1.,0.),(5.,1.))

    def test_unavailable_integer_trim_is_not_invented(self):
        ep=self.episode(); ep['can_trim'][6]=False
        self.assertEqual(q.reward_step(ep,5,10,1.,.5),(5.,1.))

    def test_available_half_receives_half_after_fill(self):
        self.assertEqual(q.reward_step(self.episode(),5,10,1.,.5),(2.,.5))

    def test_prefix_purge_excludes_future_reward(self):
        ep=self.episode(); cutoff=ep['dates'][11]
        a=q.causal_prefix(q.pack(q.transitions(ep,0)),cutoff)
        ep['gross_training'][11]=1e6
        b=q.causal_prefix(q.pack(q.transitions(ep,0)),cutoff)
        np.testing.assert_array_equal(a['reward'],b['reward'])
        self.assertTrue((a['end']<cutoff.value).all())

    def test_encoded_state_backoff_stays_in_bounds(self):
        s=np.array([[0,0,0],[q.PAPER_BASE-1,3,2]])
        for v in q.VARIANTS:
            state=q.encode(s,[1.,.5],v)
            self.assertTrue((state>=0).all() and (state<q.dimensions(v)).all())
            self.assertTrue((q.backoff(state,v)>=0).all())

    def test_q_mask_never_reexpands_and_ties_hold(self):
        states=np.array([[0,0,0],[0,0,0]])
        for v in q.VARIANTS:
            model=dict(q=np.zeros((q.dimensions(v),3)),support=np.ones((q.dimensions(v),3)),variant=v)
            np.testing.assert_array_equal(q.choose(model,states,[1.,.5]),[1.,.5])
            model['q'][:,2]=100.
            self.assertEqual(q.choose(model,states[:1],[.5])[0],.5)

    def test_fitted_q_bellman_values_finite(self):
        records=[]
        for i in range(20):records.extend(q.transitions(self.episode(),i))
        data=q.pack(records)
        for v in q.VARIANTS:
            model=q.fit_q(data,v,iterations=3)
            self.assertTrue(np.isfinite(model['q']).all())


if __name__=='__main__':unittest.main()
