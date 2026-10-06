import unittest
import pandas as pd
from scripts.run_p25_long_expiries import select_long


class LongSelectionTest(unittest.TestCase):
    def test_parity_market_roots_do_not_contaminate_each_other(self):
        from unittest.mock import patch
        from pathlib import Path
        from scripts.audit_long_expiry_parity import BoundMarket,source
        original=source.long.base.cached.OUT
        base=source.long.base.Market
        with patch.object(base,'__init__',lambda self:None),patch.object(base,'quote',lambda self,date,oid:source.long.base.cached.OUT):
            a=BoundMarket(Path('first'));b=BoundMarket(Path('second'))
            self.assertEqual(a.quote(None,1),Path('first'))
            self.assertEqual(b.quote(None,1),Path('second'))
            self.assertEqual(a.quote(None,1),Path('first'))
            self.assertEqual(source.long.base.cached.OUT,original)

    def test_same_strike_call_put_hedged_ledger_matches_without_costs(self):
        import numpy as np
        from dataclasses import replace
        from tests.test_joint_smile_engine import Market
        from scripts.option_flex.black_scholes import option_values
        from scripts.curve_trading.config import config
        from scripts.uniform_smile_engine import simulate
        cal=pd.bdate_range('2020-01-02',periods=15);stocks=[]
        for j,date in enumerate(cal):
            for sid,initial in [(1,100.),(2,150.)]:
                stocks.append(dict(date=date,secid=sid,close=initial*(1+.035*np.sin(j*.8+sid)),cfadj=1.))
        stocks=pd.DataFrame(stocks);divs=pd.DataFrame(columns=['secid','ex_date','declare_date','amount']);books=[]
        cfg=replace(config(),capital=10_000_000,entry_vega_per_leg=50,max_gross_vega=200_000,option_commission=0.,stock_cost_bps=0.,stock_borrow_rate=0.,option_penalty_vol_points=0.)
        for cp in ['P','C']:
            quotes=[]
            for r in stocks.itertuples():
                if r.date==cal[-1]:continue
                strike={1:95.,2:140.}[r.secid];sigma=.30+.025*np.sin((r.date-cal[0]).days*.2+r.secid)
                price,delta,gamma,vega=option_values(r.close,strike,(cal[-1]-r.date).days/365,0.,sigma,cp)
                quotes.append(dict(date=r.date,secid=r.secid,optionid=r.secid,expiry=cal[-1],cp_flag=cp,strike=strike,bid=float(price),ask=float(price),delta=float(delta),vega=float(vega),bs_iv=sigma,bs_rate=0.,known_dividend_pv=0.,bs_status='ok',standard=True,multiplier=100.,cfadj=1.,open_interest=1000))
            market=Market(pd.DataFrame(quotes),stocks,divs);market.rate=lambda date:0.
            rows=pd.DataFrame([dict(date=cal[0],pair='1__2',wing=cp,secid_a=1,secid_b=2,optionid_a=1,optionid_b=2,direction=1,base_level=4)])
            nav,trades,decisions,_=simulate(market,rows,cal,cfg,cfg.capital)
            self.assertEqual(decisions.status.tolist(),['entered'])
            self.assertTrue(trades.groupby('optionid').quantity.sum().eq(0).all())
            books.append(nav)
        np.testing.assert_allclose(books[0].nav,books[1].nav,atol=1e-7,rtol=0)

    def test_corporate_event_closes_pair_before_deliverable_changes(self):
        from tests.test_p25_management import ManagementTest
        from scripts.uniform_smile_engine import simulate
        fixture=ManagementTest();market,rows,cal,cfg=fixture.setup_book()
        for (date,oid),q in market.q.items():
            if date>=cal[3] and oid==1:q['standard']=False
        event=dict(secid=1,announced=str(cal[0].date()),effective=str(cal[3].date()))
        nav,trades,_,_=simulate(market,rows,cal,cfg,cfg.capital,corporate_events=[event])
        exits=trades.loc[trades.reason.eq('announced_corporate_action_exit')]
        self.assertEqual(len(exits),2)
        self.assertTrue(exits.date.eq(cal[2]).all())
        fixture.reconcile(nav,trades,cfg)

    def test_call_and_put_selections_are_separate(self):
        date=pd.Timestamp('2020-01-02');next_date=pd.Timestamp('2020-01-03')
        rows=[dict(date=date,secid=1,optionid=oid,expiry=next_date+pd.Timedelta(days=180),cp_flag=cp,bs_status='ok',bs_vega=.5,standard=True,multiplier=100,bid=2.,ask=2.1,open_interest=1000,bs_delta=delta) for oid,cp,delta in [(10,'C',.25),(20,'P',-.25)]]
        q=pd.DataFrame(rows);following=pd.Series([next_date],index=[date])
        self.assertEqual(select_long(q,following,180,'C').optionid.tolist(),[10])
        self.assertEqual(select_long(q,following,180,'P').optionid.tolist(),[20])

    def test_expiry_precedes_delta_fit_and_future_quote_does_not_select(self):
        date=pd.Timestamp('2020-01-02');next_date=pd.Timestamp('2020-01-03')
        rows=[]
        for oid,dte,delta in [(1,180,-.28),(2,200,-.25),(3,180,-.5),(4,410,-.25)]:
            rows.append(dict(date=date,secid=1,optionid=oid,expiry=next_date+pd.Timedelta(days=dte),cp_flag='P',bs_status='ok',bs_vega=.5,standard=True,multiplier=100,bid=2.,ask=2.1,open_interest=1000,bs_delta=delta))
        q=pd.DataFrame(rows);following=pd.Series([next_date],index=[date])
        choice=select_long(q,following,180)
        self.assertEqual(choice.optionid.tolist(),[1])
        future=q.iloc[[1]].assign(date=next_date,bid=100.,ask=101.)
        self.assertEqual(select_long(pd.concat([q,future]),following,180).optionid.tolist(),[1])

    def test_one_year_bounds_are_contract_expiry_not_training_history(self):
        date=pd.Timestamp('2020-01-02');next_date=pd.Timestamp('2020-01-03')
        rows=[]
        for oid,dte in [(1,365),(2,180),(3,401)]:
            rows.append(dict(date=date,secid=1,optionid=oid,expiry=next_date+pd.Timedelta(days=dte),cp_flag='P',bs_status='ok',bs_vega=.5,standard=True,multiplier=100,bid=2.,ask=2.1,open_interest=1000,bs_delta=-.25))
        self.assertEqual(select_long(pd.DataFrame(rows),pd.Series([next_date],index=[date]),365).optionid.tolist(),[1])


if __name__=='__main__':unittest.main()
