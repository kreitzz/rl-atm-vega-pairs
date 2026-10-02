import unittest
import numpy as np
import pandas as pd
from scripts.atm65_mining_features import engineer,names,DYNAMICS,GAMMA,thin_completed

class MiningFeatureTests(unittest.TestCase):
    def test_compact_cache_gamma_ratio_matches_bs_with_cash_dividends(self):
        from scripts.black_scholes import call_values
        date=pd.Timestamp('2020-01-02');q=[];s=[];selection=[];ratios=[]
        for sid,spot,pv,iv in [(1,100.,2.,.4),(2,80.,.5,.25)]:
            _,_,gamma,vega=call_values(spot-pv,spot,30/365,.02,iv)
            ratios.append(float(gamma*spot**2/vega))
            q.append(dict(date=date,secid=sid,optionid=sid,bs_iv=iv,known_dividend_pv=pv,
                          vega=float(vega),bid=2.,ask=2.1,open_interest=500,expiry=date+pd.Timedelta(days=30)))
            s.append(dict(date=date,secid=sid,close=spot,cfadj=1))
            selection.append(dict(date=date,secid=sid,optionid=sid))
        panel=pd.DataFrame({'beta':[1.]},index=[date])
        result=engineer(panel,pd.DataFrame(q),pd.DataFrame(s),pd.DataFrame(selection),[1,2])
        self.assertAlmostEqual(result.mine_gamma_risk_gap.iloc[0],ratios[0]-ratios[1])

    def test_future_market_changes_cannot_change_earlier_features(self):
        dates=pd.bdate_range('2020-01-01',periods=100);q=[];s=[];selection=[]
        for i,date in enumerate(dates):
            for sid in [1,2]:
                s.append(dict(date=date,secid=sid,close=100*np.exp(.001*i+.01*np.sin(i/sid)),cfadj=1))
                q.append(dict(date=date,secid=sid,optionid=sid,bs_iv=.3+.01*np.sin(i/sid),bs_gamma=.02,
                              vega=.1,bid=4.9,ask=5.1,open_interest=500,expiry=date+pd.Timedelta(days=30)))
                selection.append(dict(date=date,secid=sid,optionid=sid))
        q=pd.DataFrame(q);s=pd.DataFrame(s);selection=pd.DataFrame(selection)
        # Occasional missing daily selections should not destroy a 63-session
        # correlation window, and fewer than 32 paired observations cannot pass.
        selection=selection[~selection.date.isin(dates[::7])]
        panel=pd.DataFrame({'beta':1.,'unchanged_signal':np.arange(100)},index=dates)
        first=engineer(panel,q,s,selection,[1,2])
        self.assertEqual(first.mine_iv_correlation_63_available.iloc[:32].sum(),0)
        self.assertGreater(first.mine_iv_correlation_63_available.iloc[45:].mean(),.9)
        future=q.date>dates[75];q.loc[future,'bs_iv']*=4
        s.loc[s.date>dates[75],'close']*=3
        second=engineer(panel,q,s,selection,[1,2])
        pd.testing.assert_frame_equal(first.loc[:dates[75]],second.loc[:dates[75]])
        self.assertTrue(np.isfinite(first[names(DYNAMICS+GAMMA)].to_numpy()).all())
        pd.testing.assert_frame_equal(first[panel.columns],panel)

    def test_completed_thinning_keeps_independent_pair_cohorts(self):
        dates=pd.bdate_range('2020-01-01',periods=6)
        ends=dates+pd.Timedelta(days=3)
        chosen=thin_completed(dates,ends,np.ones(6,bool))
        self.assertTrue(all(dates[b]>ends[a] for a,b in zip(chosen,chosen[1:])))

if __name__=='__main__':unittest.main()
