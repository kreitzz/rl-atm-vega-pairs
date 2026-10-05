import unittest
import numpy as np
from scripts.run_contract_episodes import discounted_step,episode_value,fit_exits,fit_entries,episodes
from scripts.contract_q_engine import LONG,SHORT,WAIT,EXIT,ContractMarket,evaluate
from tests.test_contract_q import fixture


class Constant:
    def __init__(self,value): self.value=value
    def predict(self,x): return np.tile(self.value,(len(x),1))


class EpisodeTests(unittest.TestCase):
    def test_actual_contract_episode_matches_cash_discounted_account(self):
        quotes,stocks,dates,cfg=fixture()
        stocks['rate']=.04
        market=ContractMarket(quotes,stocks,dates,[1,2],cfg)
        class Features:
            def features(self,m,t,book): return m.features(t,book)
        rows=episodes(market,Features(),3)
        row=next(r for r in rows if r['t']==0 and r['action']==LONG)
        def policy(m,t,book):
            if t==0: return LONG
            if book is not None and t-book.entry>=2: return EXIT
            return WAIT
        nav,error=evaluate(market,policy)
        self.assertIsNone(error)
        discounted=(nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1])/(nav.cash_benchmark.iloc[-1]/cfg.capital)
        self.assertAlmostEqual(episode_value(row),discounted,places=7)
        self.assertEqual(len(row['steps']),3)
        self.assertTrue(all(r['t']+4<market.n for r in rows))

    def test_cash_discounted_rewards_telescope(self):
        # An opening cost of 10 and next-session reward 25; cash grows 1% then 2%.
        row=dict(reward=-10.,growth=1.01,steps=[dict(terminal=True,exit_reward=25.,growth=1.02)])
        self.assertAlmostEqual(episode_value(row),(-10*1.02+25)/(1.01*1.02))
        self.assertEqual(discounted_step(0,1.02),0)

    def test_realized_exit_labels_do_not_select_hindsight_best(self):
        row=dict(reward=-2.,growth=1.,steps=[
            dict(age=0,x=np.zeros(2),terminal=False,exit_reward=-5.,hold_reward=1.,growth=1.),
            dict(age=1,x=np.zeros(2),terminal=True,exit_reward=20.,hold_reward=20.,growth=1.)])
        self.assertEqual(episode_value(row),19)
        self.assertEqual(episode_value(row,{0:Constant([2.,1.])}),-7)

    def test_predictable_profitable_path_is_recovered_after_costs(self):
        rows=[]
        for i in range(200):
            for action in (LONG,SHORT):
                payoff=20. if action==LONG else -20.
                rows.append(dict(action=action,x=np.array([1.,0.]),reward=-2.,growth=1.,steps=[
                    dict(age=0,x=np.array([float(action),0.]),terminal=False,exit_reward=-2.,
                         hold_reward=1.,growth=1.,date=i),
                    dict(age=1,x=np.array([float(action),1.]),terminal=True,
                         exit_reward=payoff,hold_reward=payoff,growth=1.,date=i+1)]))
        exits=fit_exits(rows,2)
        self.assertGreater(exits[0].predict([[1.,0.]])[0,1],exits[0].predict([[1.,0.]])[0,0])
        policy=fit_entries(rows,None,2,exits)
        q=policy.entry_values(np.array([1.,0.]))
        self.assertEqual(q[0],0.)
        self.assertEqual(int(np.argmax(q)),LONG)
        self.assertGreater(q[LONG],0)
        self.assertLess(q[SHORT],0)


if __name__=='__main__': unittest.main()
