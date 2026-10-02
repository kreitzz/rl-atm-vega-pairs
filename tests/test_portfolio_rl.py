import unittest
import numpy as np
import pandas as pd
from scripts.portfolio_rl_policy import completed_mask,FEATURES,fit_predict
from scripts.portfolio_vega import allocate

class PortfolioRLTests(unittest.TestCase):
    def test_completed_months_purged_before_signal(self):
        d=pd.DataFrame(dict(date=pd.to_datetime(['2021-10-01','2021-11-01','2021-12-01']),end=pd.to_datetime(['2021-11-01','2021-12-01','2022-01-03'])))
        self.assertEqual(completed_mask(d,pd.Timestamp('2021-12-31')).tolist(),[True,True,False])

    def test_all_zero_budget_is_flat(self):
        m=np.array([[10.,10.],[100.,100.],[100.,100.],[100.,100.],[1.,1.]])
        q=allocate(m,np.array([-1,1]),['a','a'],np.zeros(2),1e6,np.array([-1.,1.]),'forecast',target=np.zeros(2))
        np.testing.assert_array_equal(q,[0,0])

    def test_policy_ignores_future_rewards(self):
        dates=pd.date_range('2020-01-01',periods=24,freq='MS')
        train=pd.DataFrame({'date':dates,'end':dates+pd.Timedelta(days=20)})
        for j,f in enumerate(FEATURES):train[f]=np.sin(np.arange(24)+j)
        test=train.iloc[[-1]].copy();test['signal_date']=pd.Timestamp('2022-01-01')
        paths={}
        for i,d in enumerate(dates):
            days=pd.bdate_range(d,periods=10)
            daily=(.0001+.0002*np.sin(np.arange(10)+i))[:,None]*np.array([0,.25,.5,.75,1])[None,:]
            paths[d]=(days,daily)
        a=fit_predict(train,test,paths)
        paths[pd.Timestamp('2023-01-01')]=(pd.bdate_range('2023-01-01',periods=10),np.ones((10,5))*999)
        b=fit_predict(train,test,paths)
        self.assertEqual(a,b)
        self.assertIn(a[0],[0,.25,.5,.75,1])

if __name__=='__main__':unittest.main()
