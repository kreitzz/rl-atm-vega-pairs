import unittest
import numpy as np
from scripts.portfolio_vega import allocate,constraint_matrix

class PortfolioAllocatorTests(unittest.TestCase):
    def setUp(self):
        self.m=np.array([[17.,23.,31.,19.],[1000.]*4,[600.]*4,[150.]*4,[2.]*4])
        self.sign=np.array([1,1,-1,-1]);self.sectors=['a','b','a','b'];self.events=np.zeros(4)

    def check(self,q):
        gross=self.m[0]@q;net=(self.m[0]*self.sign)@q
        self.assertLessEqual(abs(net),max(10,.01*gross)+1e-8)
        a,b=constraint_matrix(self.m,self.sign,self.sectors,self.events,1e6)
        self.assertTrue((a@q<=b+1e-7).all())
        self.assertTrue((q>=0).all())
        self.assertTrue((q==q.astype(int)).all())

    def test_optimizer_balances_integer_contracts(self):
        q=allocate(self.m,self.sign,self.sectors,self.events,1e6,np.array([3,2,-3,-2]),'forecast')
        self.assertGreater(q.sum(),0);self.check(q)

    def test_small_contract_caps_are_hard_limits(self):
        limits=np.array([3,7,4,6])
        for mode in ['forecast','equal_rank']:
            q=allocate(self.m,self.sign,self.sectors,self.events,1e6,np.array([30,20,-30,-20]),mode,max_contracts=limits)
            self.assertTrue((q<=limits).all());self.check(q)

    def test_no_edge_means_cash(self):
        q=allocate(self.m,self.sign,self.sectors,self.events,1e6,np.zeros(4),'forecast')
        self.assertEqual(q.sum(),0)

    def test_daily_rebalance_and_rank_respect_risk(self):
        for target in [None,np.array([2000.,2000.,2000.,2000.])]:
            q=allocate(self.m,self.sign,self.sectors,self.events,1e6,np.array([3,2,-3,-2]),'equal_rank',target=target)
            self.assertGreater(q.sum(),0);self.check(q)

if __name__=='__main__':unittest.main()
