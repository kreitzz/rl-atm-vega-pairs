import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from scripts.year_prefix_engine import simulate
from scripts.run_year_cheap_vega import cheap_masks,signal_rows,carry_state,write_transitions,transition_dtype,fit_carry_stream,sparse_book_risk,quarter_done,canonical_quote
from tests.test_p25_management import ManagementTest


class StandaloneVegaTest(unittest.TestCase):
    def test_read_only_mapped_variance_matches_original_with_gaps_and_splits(self):
        import tempfile
        import json
        from pathlib import Path
        from types import SimpleNamespace
        from scripts.cheap_vega_runtime import load_variance
        from scripts.run_year_reversion_management import input_variance
        dates=pd.bdate_range('2010-01-04',periods=600)
        rng=np.random.default_rng(71)
        spots=pd.DataFrame(np.exp(np.cumsum(rng.normal(0,.01,(600,3)),axis=0))*100,index=dates,columns=[1,2,3])
        adjust=spots*0+1;adjust.loc[dates[250]:,2]=2;spots.loc[dates[250]:,2]/=2
        spots.loc[dates[300:304],3]=np.nan
        expected=input_variance(SimpleNamespace(spots=spots,adjustments=adjust))
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);cache=folder/'market_cache';cache.mkdir()
            for half,minobs in [(20,20),(60,60),(252,126)]:
                actual=np.empty(spots.shape)
                for j,sid in enumerate(spots.columns):
                    close=(spots[sid]*adjust[sid]).where(spots[sid].gt(0)&adjust[sid].gt(0))
                    r=np.log(close/close.shift(1))
                    actual[:,j]=252*r.pow(2).ewm(halflife=half,min_periods=minobs,adjust=False).mean()
                np.testing.assert_array_equal(actual,expected[half].to_numpy())
                np.save(cache/f'variance_{half}.npy',actual)
            (cache/'metadata.json').write_text(json.dumps(dict(sources={},secids=[1,2,3],dates=list(map(str,dates)))))
            mapped=load_variance(None,folder)
            for half in expected:
                self.assertFalse(mapped[half].values.flags.writeable)
                for date in dates[126::41]:
                    for sid in [1,2,3]:self.assertEqual(mapped[half].at[date,sid],expected[half].at[date,sid])

    def test_vendor_identity_alias_preserves_observed_quote_and_contract(self):
        from scripts.option_flex.contract_engine import MarketRow
        date=pd.Timestamp('2022-07-22')
        q=MarketRow(optionid=135980064,date=date,secid=147220,strike=50,cp_flag='P',
                    expiry=pd.Timestamp('2023-01-20'),standard=True,multiplier=100,cfadj=1,
                    bid=4.3,ask=4.8,delta=-.334286,vega=.1384322)
        result=canonical_quote(144532048,date,{135980064:q})
        self.assertEqual(result['optionid'],144532048);self.assertEqual(result['vendor_optionid'],135980064)
        for key in ['bid','ask','delta','vega']:self.assertEqual(result[key],q[key])
        self.assertEqual(q.optionid,135980064)
        self.assertIsNone(canonical_quote(144532048,pd.Timestamp('2022-07-21'),{135980064:q}))
        self.assertIsNone(canonical_quote(144532048,date,{}))
        for key,value in [('strike',49),('standard',False),('multiplier',50)]:
            wrong=q.copy();wrong[key]=value
            with self.assertRaises(AssertionError):canonical_quote(144532048,date,{135980064:wrong})
        other=q.copy();other['optionid']=999;other['strike']=60
        family={123:dict(new_optionid=999,first_new_id='2022-07-22',strike=60)}
        linked=canonical_quote(123,date,{999:other},family)
        self.assertEqual(linked['vendor_optionid'],999);self.assertEqual(linked['strike'],60)

    def test_extended_missing_quote_stops_without_deleting_held_trade(self):
        from scripts.option_flex.contract_engine import UnresolvedPosition
        market,rows,cal,cfg=self.setup_book()
        for date in cal[2:4]:del market.q[date,1]
        with self.assertRaisesRegex(UnresolvedPosition,'held quote gap exceeds'):
            simulate(market,rows,cal,cfg,cfg.capital,standalone=True,max_stale_quote_sessions=1)

    def test_short_quote_gap_can_recover_under_gap_limit(self):
        market,rows,cal,cfg=self.setup_book();del market.q[cal[2],1]
        nav,trades,_,_=simulate(market,rows,cal,cfg,cfg.capital,standalone=True,max_stale_quote_sessions=1)
        self.assertGreater(nav.causal_held_mark_fallbacks.sum(),0)
        ManagementTest().reconcile(nav,trades,cfg)

    def test_weekend_quarter_resume_skips_completed_trading_sessions(self):
        cal=pd.bdate_range('2023-09-01','2023-12-29')
        through=pd.Timestamp('2023-09-30')
        self.assertFalse(quarter_done(cal,None,through))
        self.assertFalse(quarter_done(cal,dict(previous_date=pd.Timestamp('2023-09-28')),through))
        self.assertTrue(quarter_done(cal,dict(previous_date=pd.Timestamp('2023-09-29')),through))
        self.assertTrue(quarter_done(cal,dict(previous_date=cal[-1]),pd.Timestamp('2023-12-31')))

    def setup_book(self):
        market,rows,cal,cfg=ManagementTest().setup_book()
        rows=rows.copy();rows['secid_b']=rows.secid_a;rows['optionid_b']=np.nan
        return market,rows,cal,cfg

    def test_standalone_has_one_long_entry_and_reconciles(self):
        market,rows,cal,cfg=self.setup_book()
        nav,trades,decisions,audit=simulate(market,rows,cal,cfg,cfg.capital,standalone=True)
        entry=trades[trades.reason.eq('entry')]
        self.assertEqual(decisions.status.iloc[0],'entered')
        self.assertEqual(len(entry),1);self.assertGreater(entry.quantity.iloc[0],0)
        self.assertTrue(nav.net_vega.ge(0).all())
        self.assertAlmostEqual(nav.net_vega.iloc[1],nav.gross_vega.iloc[1])
        ManagementTest().reconcile(nav,trades,cfg)

    def test_sparse_risk_preserves_entire_portfolio_replay(self):
        from scripts import year_prefix_engine as engine
        from scripts.uniform_smile_risk import book_risk
        market,rows,cal,cfg=self.setup_book();original=engine.book_risk
        try:
            engine.book_risk=book_risk
            expected=simulate(market,rows,cal,cfg,cfg.capital,standalone=True)
            engine.book_risk=sparse_book_risk
            actual=simulate(market,rows,cal,cfg,cfg.capital,standalone=True)
        finally:engine.book_risk=original
        for left,right in zip(expected[:3],actual[:3]):pd.testing.assert_frame_equal(left,right)

    def test_sparse_risk_handles_many_inactive_names_and_signed_positions(self):
        from scripts.uniform_smile_risk import book_risk
        market,rows,cal,cfg=self.setup_book();date=cal[1]
        spots={sid:market.stock(date,sid)[0] for sid in [1,2]}
        spots.update({sid:100. for sid in range(10,310)})
        held=[]
        for sid,qty in [(1,3),(1,-1),(2,-2)]:
            quote=market.quote(date,sid)
            held.append(dict(secid=sid,cohort=str(len(held)),optionid=sid,quantity=qty,
                             strike=quote.strike,cfadj=quote.cfadj,expiry=quote.expiry))
        for positions in [[],held[:1],held]:
            expected=book_risk(positions,lambda p:market.quote(date,p['optionid']),spots,date,cfg)
            actual=sparse_book_risk(positions,lambda p:market.quote(date,p['optionid']),spots,date,cfg)
            self.assertEqual(actual['target_hedge'],expected['target_hedge'])
            for key in expected.keys()-{'target_hedge'}:
                self.assertAlmostEqual(actual[key],expected[key],places=8)

    def test_standalone_trim_executes_next_close_once(self):
        market,rows,cal,cfg=self.setup_book()
        nav,trades,_,audit=simulate(market,rows,cal,cfg,cfg.capital,standalone=True,
                                   management_policy=lambda date,held:.5)
        trims=trades[trades.reason.eq('learned_policy_trim')]
        entry=trades[trades.reason.eq('entry')]
        self.assertEqual(len(trims),1);self.assertEqual(trims.date.iloc[0],cal[2])
        self.assertEqual(entry.quantity.iloc[0]+trims.quantity.iloc[0],np.floor(entry.quantity.iloc[0]*.5))
        ManagementTest().reconcile(nav,trades,cfg)

    def test_standalone_close_uses_actual_quote_and_waits_if_missing(self):
        market,rows,cal,cfg=self.setup_book();del market.q[cal[2],1]
        nav,trades,_,audit=simulate(market,rows,cal,cfg,cfg.capital,standalone=True,
                                   management_policy=lambda date,held:0.)
        exits=trades[trades.reason.eq('learned_policy_exit')]
        self.assertEqual(len(exits),1);self.assertEqual(exits.date.iloc[0],cal[3])
        self.assertEqual(audit['management_log'][0]['status'],'no_executable_exit_quote')
        ManagementTest().reconcile(nav,trades,cfg)

    def test_duplicate_names_or_short_entries_rejected(self):
        market,rows,cal,cfg=self.setup_book()
        with self.assertRaises(ValueError):simulate(market,pd.concat([rows,rows]),cal,cfg,cfg.capital,standalone=True)
        rows['direction']=-1
        with self.assertRaises(ValueError):simulate(market,rows,cal,cfg,cfg.capital,standalone=True)

    def test_historical_cheapness_does_not_imply_absolute_positive_carry(self):
        names=pd.DataFrame(dict(date=pd.to_datetime(['2020-01-02']*3),secid=[1,2,3],optionid=[1,2,3],
                               bs_iv=[.4,.4,.4],forecast_variance=[.2,.1,.2],carry_forecast=[.05,-.05,.05],
                               carry_z=[1.,1.,-1.],expiry=pd.to_datetime(['2021-01-15']*3)))
        masks=cheap_masks(names)
        np.testing.assert_array_equal(masks['absolute'],[True,False,True])
        np.testing.assert_array_equal(masks['historical'],[True,True,False])
        np.testing.assert_array_equal(masks['both'],[True,False,False])
        rows=signal_rows(names,'both');self.assertEqual(len(rows),1)
        self.assertEqual(rows.secid_a.iloc[0],rows.secid_b.iloc[0]);self.assertTrue(rows.optionid_b.isna().all())

    def test_future_rv_cannot_change_past_carry(self):
        dates=pd.bdate_range('2020-01-02',periods=2)
        held=[dict(secid=1,quote=dict(bs_iv=.4,bs_status='fitted'),expiry=pd.Timestamp('2021-01-02'),entry_date=dates[0])]
        variance={k:pd.DataFrame({1:[.2,.3]},index=dates) for k in [20,60,252]}
        first=carry_state(dates[0],held,variance)
        for values in variance.values():values.loc[dates[1],1]=1e6
        self.assertEqual(first,carry_state(dates[0],held,variance))

    def test_disk_transitions_match_original_arrays(self):
        import tempfile
        from scripts import year_reversion_management as q
        records=[(1,10,20,np.array([0,1,2]),1.,.5,-.3,np.array([0,2,2]),.5,False,.98),
                 (2,20,30,np.array([0,2,2]),.5,0.,.4,np.array([0,3,2]),0.,True,.98)]
        with tempfile.TemporaryFile() as handle:
            self.assertEqual(write_transitions(handle,records),2)
            handle.seek(0);stored=np.fromfile(handle,dtype=transition_dtype())
        for key,value in q.pack(records).items():np.testing.assert_array_equal(stored[key],value)

    def test_bounded_carry_fit_matches_original_and_excludes_future(self):
        import tempfile
        from pathlib import Path
        from scripts import year_reversion_management as q
        rng=np.random.default_rng(137)
        records=[]
        for cohort in range(24):
            for step in range(12+cohort):
                start=pd.Timestamp('2019-01-02')+pd.Timedelta(days=step)
                state=np.array([0,rng.integers(4),rng.integers(3)])
                following=np.array([0,rng.integers(4),rng.integers(3)])
                for fraction in [1.,.5]:
                    for action in q.ACTIONS[q.ACTIONS<=fraction]:
                        records.append((cohort,start.value,(start+pd.Timedelta(days=5)).value,
                            state,fraction,action,rng.normal(),following,action,
                            action==0 or step==11+cohort,.98))
        cutoff=pd.Timestamp('2020-01-01')
        expected=q.fit_q(q.causal_prefix(q.pack(records),cutoff),'carry')
        records.append((999,cutoff.value,cutoff.value+100,np.array([0,1,1]),1.,1.,1e9,np.array([0,1,1]),1.,False,.98))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'records.bin'
            with path.open('wb') as handle:write_transitions(handle,records)
            actual,note=fit_carry_stream(path,cutoff,batch_size=73)
        self.assertEqual(note['cohorts'],24)
        self.assertLess(pd.Timestamp(note['last_outcome']),cutoff)
        np.testing.assert_allclose(actual['q'],expected['q'],rtol=1e-11,atol=1e-11)
        np.testing.assert_allclose(actual['support'],expected['support'],rtol=1e-11,atol=1e-11)
        np.testing.assert_array_equal(q.choose(actual,q.pack(records)['state'],q.pack(records)['fraction']),
                                      q.choose(expected,q.pack(records)['state'],q.pack(records)['fraction']))


if __name__=='__main__':unittest.main()
