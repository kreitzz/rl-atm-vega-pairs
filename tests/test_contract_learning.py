import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.contract_engine import simulate, UnresolvedPosition, verified_integer_split
from scripts.contract_learning_labels import ShadowLabels, LADDER, profitability_memory, beta_screen
from scripts.run_contract_walkforward import daily_moments, search_policy, train_walkforward, NEW_FEATURES
from scripts.run_selected_pair_backtest import audit
from tests.test_contract_engine import fixture


class LearningTests(unittest.TestCase):
    def test_shadow_matches_cash_engine_all_actions(self):
        q,s,a,d,c=fixture()
        c=replace(c,entry_vega_per_leg=100, max_gross_vega=1000, option_spread_fraction=.5,
                  option_commission=.65, stock_cost_bps=1, stock_borrow_rate=.003)
        s['rate']=.04
        # Moving stocks, prices and deltas exercise daily hedge/cash accounting.
        for i,date in enumerate(d):
            s.loc[s.date.eq(date)&s.secid.eq(1),'close']+=i*2
            q.loc[q.date.eq(date)&q.secid.eq(1),['bid','ask']]+=i*.9
            q.loc[q.date.eq(date)&q.secid.eq(1),'delta']+=i*.02
        selected=q[q.date.eq(d[0])][['date','secid','optionid']]
        dividends=pd.DataFrame(columns=['secid','ex_date','declare_date','amount'])
        result=ShadowLabels(q,s,selected,d,[1,2],c,dividends).label(0)
        self.assertEqual(result['status'],'complete')
        for j,action in enumerate(LADDER):
            nav,_,_=simulate(q,s,pd.Series([action],index=d[:1]),d,[1,2],c,selections=selected)
            cash_ret=.04*d.to_series().diff().dt.days.fillna(0)/365
            benchmark=c.capital*(1+cash_ret).cumprod()
            excess=nav.pnl-benchmark.diff().fillna(0)
            end=result['end_index']
            np.testing.assert_allclose(result['daily'][j,:end],excess.iloc[1:end+1]/c.entry_vega_per_leg,atol=1e-9)

    def test_unresolved_shadow_is_not_zero(self):
        q,s,a,d,c=fixture();selected=q[q.date.eq(d[0])][['date','secid','optionid']]
        q=q[~(q.date.eq(d[2])&q.optionid.eq(1))]
        div=pd.DataFrame(columns=['secid','ex_date','declare_date','amount'])
        result=ShadowLabels(q,s,selected,d,[1,2],c,div).label(0)
        self.assertEqual(result['status'],'unresolved_contract')
        self.assertTrue(np.isnan(result['daily']).all())

    def test_memory_waits_until_after_exit_and_expires(self):
        dates=pd.to_datetime(['2020-02-01','2020-02-02','2021-02-02'])
        ends=pd.to_datetime(['2020-02-01','2021-03-01'])
        rewards=np.array([[2.,-3.],[999.,999.]])
        state=profitability_memory(dates,ends,rewards,np.ones((2,2),bool))
        self.assertEqual(state.iloc[0].count_long_91d,0)
        self.assertEqual(state.iloc[1].profit_long_91d,-3)
        self.assertEqual(state.iloc[1].profit_short_365d,2)
        self.assertEqual(state.iloc[2].count_short_365d,0)

    def test_beta_screen_uses_preceding_close_and_freezes(self):
        dates=pd.bdate_range('2020-01-01',periods=100)
        rng=np.random.default_rng(3);b=rng.normal(0,.01,len(dates));r=pd.DataFrame({'a':b,'b':b},index=dates)
        before=beta_screen(r,dates,window=20)
        first=pd.Timestamp('2020-03-02');r.loc[first:,'a']=rng.normal(0,1,len(r.loc[first:]))
        after=beta_screen(r,dates,window=20)
        pd.testing.assert_frame_equal(before.loc[:'2020-03-31'],after.loc[:'2020-03-31'])
        self.assertAlmostEqual(after.loc[first,'beta'],1)
        self.assertTrue(after.loc[first,'eligible'])

    def test_tighter_beta_range_excludes_both_tails(self):
        dates=pd.bdate_range('2020-01-01',periods=100)
        b=np.random.default_rng(3).normal(0,.01,len(dates))
        for beta,eligible in [(0.84,False),(0.86,True),(1.,True),(1.14,True),(1.16,False)]:
            with self.subTest(beta=beta):
                r=pd.DataFrame({'a':beta*b,'b':b},index=dates)
                screen=beta_screen(r,dates,window=20,beta_tolerance=.15)
                self.assertEqual(bool(screen.loc['2020-03-02','eligible']),eligible)
                self.assertTrue(beta_screen(r,dates,window=20).loc['2020-03-02','eligible'])

    def test_daily_moments_include_overlap(self):
        labels=np.zeros((2,9,3)); labels[0,8]=[1,2,0];labels[1,8]=[3,4,0]
        mean,cov=daily_moments(np.array([0,1]),labels,np.array([0,1]),np.array([2,3]),2)
        # Both long: daily PnL [1,5,4], not the two cohort totals [3,7].
        chosen=[8,17]
        self.assertAlmostEqual(mean[chosen].sum(),10/3)
        self.assertAlmostEqual(cov[np.ix_(chosen,chosen)].sum(),np.var([1,5,4],ddof=1))

    def test_policy_stays_flat_when_every_trade_loses(self):
        mean=np.full(9,-1.);mean[4]=0
        policy,score=search_policy(mean,np.eye(9),1)
        self.assertEqual(policy[0],4);self.assertEqual(score,0)

    def test_split_preserves_wealth_and_inventory(self):
        q,s,a,d,c=fixture();c=replace(c,support_integer_splits=True)
        baseline,_,_=simulate(q,s,a,d,[1,2],c)
        mask=q.date.ge(d[2])&q.secid.eq(1)
        q.loc[mask,['bid','ask','strike','vega']]/=2;q.loc[mask,'cfadj']=2
        mask=s.date.ge(d[2])&s.secid.eq(1)
        s.loc[mask,'close']/=2;s.loc[mask,'cfadj']=2
        nav,trades,_=simulate(q,s,a,d,[1,2],c)
        np.testing.assert_allclose(nav.nav,baseline.nav,atol=1e-8)
        self.assertEqual(trades.reason.eq('split_adjustment').sum(),1)
        self.assertTrue(trades.groupby('optionid').quantity.sum().eq(0).all())
        audit(nav,trades,s,c,[1,2])
        selected=q[q.date.eq(d[0])][['date','secid','optionid']]
        div=pd.DataFrame(columns=['secid','ex_date','declare_date','amount'])
        label=ShadowLabels(q,s,selected,d,[1,2],c,div).label(0)
        self.assertEqual(label['status'],'complete')
        end=label['end_index']
        np.testing.assert_allclose(label['daily'][-1,:end],nav.pnl.iloc[1:end+1]/c.entry_vega_per_leg,atol=1e-9)
        q.loc[q.date.ge(d[2])&q.secid.eq(1),'strike']=60
        with self.assertRaises(UnresolvedPosition):simulate(q,s,a,d,[1,2],c)

    def test_split_allows_only_half_cent_strike_rounding(self):
        q,_,_,_,_=fixture();old=q.iloc[0].to_dict();old['strike']=797.5
        new=dict(old,strike=199.38,cfadj=4)
        self.assertTrue(verified_integer_split(old,new,4))
        new['strike']=199.39
        self.assertFalse(verified_integer_split(old,new,4))

    def test_monthly_fit_excludes_overlapping_future_outcomes(self):
        calendar=pd.bdate_range('2010-01-01','2013-03-29')
        dates=calendar[:-30];panel=pd.DataFrame(1.,index=dates,columns=NEW_FEATURES)
        positions=np.arange(len(dates));ends=calendar[positions+22]
        labels=np.ones((len(dates),9,32))
        start=pd.Timestamp('2013-01-01');stop=pd.Timestamp('2013-02-15')
        fit_inputs=[]
        def fake_fit(x,*args):
            fit_inputs.append(x.index.max())
            return None,{}
        with patch('scripts.run_contract_walkforward.fit_rl',side_effect=fake_fit), \
             patch('scripts.run_contract_walkforward.fit_nn',side_effect=fake_fit), \
             patch('scripts.run_contract_walkforward.predict',side_effect=lambda model,x,neural:np.ones(len(x))):
            actions,logs=train_walkforward(panel,labels,positions,ends,pd.Series(True,index=calendar),start,stop,calendar)
        self.assertTrue((pd.to_datetime(logs.last_outcome)<pd.to_datetime(logs.fit)).all())
        self.assertEqual(logs.policy.eq('annual').sum(),1)
        self.assertEqual(logs.policy.eq('base').sum(),2)
        self.assertLess(fit_inputs[0],pd.Timestamp('2012-12-01'))


if __name__=='__main__':unittest.main()
