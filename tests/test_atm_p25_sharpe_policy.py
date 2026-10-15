import copy
import unittest
import numpy as np
import pandas as pd
from scripts import atm_p25_sharpe_policy as s


class SharpePolicyTests(unittest.TestCase):
    def test_target_risk_share(self):
        for r in [.1,1,10]:
            for share in [.2,.5,.8]:
                w=s.target(r,share)
                self.assertAlmostEqual(w.sum(),1.)
                self.assertAlmostEqual(w[1]*r/(w[0]+w[1]*r),share)

    def test_tree_assigns_every_observation_with_minimum_support(self):
        rng=np.random.default_rng(5);x=rng.normal(size=(36,6));y=x[:,0]-x[:,1]
        leaves=s.partition(x,y,'tree3');states=s.contexts(x,leaves)
        self.assertTrue((np.bincount(states)>=6).all())
        self.assertLessEqual(len(leaves),3)

    def test_sequential_search_matches_brute_daily_sharpe(self):
        rng=np.random.default_rng(5);months=[]
        for i in range(12):
            paths=rng.normal(.001,.01,(3,3,21))
            months.append(dict(paths=paths,sums=paths.sum(axis=2),squares=(paths**2).sum(axis=2),cash=np.zeros(21)))
        states=np.arange(12)%2
        choices,score,baseline,objective,n=s.search(months,states,2)
        actions=choices[states];p=1;paths=[]
        for m,a in zip(months,actions):paths.append(m['paths'][p,a]);p=a
        self.assertAlmostEqual(score,s.sharpe(np.concatenate(paths)))
        self.assertAlmostEqual(objective,score-.05*np.mean(actions!=1))
        self.assertGreaterEqual(objective,baseline-1e-12)
        self.assertEqual(n,9)

    def test_unfinished_month_never_fits(self):
        rng=np.random.default_rng(5);months=[]
        for date in pd.date_range('2018-01-01',periods=14,freq='MS'):
            paths=rng.normal(.001,.01,(3,3,21))
            months.append(dict(cutoff=date,end=date+pd.offsets.BMonthEnd(),ready=True,state=rng.normal(size=6),
                paths=paths,sums=paths.sum(axis=2),squares=(paths**2).sum(axis=2),cash=np.zeros(21)))
        original=s.fit(dict(months=months),12,24,'tree3')
        changed=copy.deepcopy(months)
        for m in changed[12:]:m['paths'][:]=100;m['state'][:]=100;m['sums'][:]=100;m['squares'][:]=100
        repeated=s.fit(dict(months=changed),12,24,'tree3')
        self.assertEqual(original,repeated)
        self.assertLess(pd.Timestamp(original['last_training_end']),months[12]['cutoff'])


if __name__=='__main__':unittest.main()
