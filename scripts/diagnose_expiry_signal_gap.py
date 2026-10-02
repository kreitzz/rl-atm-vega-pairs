"""Matched expiry trades: proxy marks, exact hedge cashflows and fixed-book costs."""
from dataclasses import asdict
import json
from pathlib import Path
import resource
import time
import numpy as np
import pandas as pd
from scripts.black_scholes import call_values
from scripts.contract_engine import PreparedMarket, valid_greeks
from scripts.run_broad_contract_replay import Market,ROOT
from scripts.select_atm_contracts import load_calendar
from scripts.run_aligned_contract_learning import asset_paths
from scripts.diagnose_current_contract_losses import attribute,complete_nav,read_csv,COMPONENTS
from scripts.reprice_contract_execution import reprice_cash
from scripts.run_selected_pair_backtest import scores
from scripts.run_contract_walkforward import digest

REPORT=Path('reports/expiry_signal_gap')
POLICIES={
    'frozen_rl':('artifacts/contracts/expiry_holding','surface_filtered_rl'),
    'trained_rl':('artifacts/contracts/expiry_trained','rl_sized'),
    'frozen_nn':('artifacts/contracts/expiry_holding','surface_filtered_nn'),
    'trained_nn':('artifacts/contracts/expiry_trained','nn_fixed'),
}


def leg_path(trades,market,cal,divs,asset):
    """Exact gross leg P&L; independent fixed-entry-IV delta hedge diagnostic."""
    entry=trades[trades.reason.eq('entry')].iloc[0]
    close=trades[trades.reason.eq('expiry_settlement')].iloc[0]
    sid=int(entry.secid);oid=int(entry.optionid);qty=float(entry.quantity)
    q0=market.quotes.loc[(entry.date,oid)];sigma0=float(q0.bs_iv)
    days=cal[(cal>=entry.date)&(cal<=close.date)]
    divcash={}
    for e in divs[divs.secid.eq(sid)].itertuples():
        ix=cal.searchsorted(e.ex_date)
        if ix<len(cal):divcash[cal[ix]]=divcash.get(cal[ix],0.)+e.amount
    actual=fixed=0.;last_sigma=sigma0;hedges={};pnl={};logs=[];squares=[]
    gamma_num=gamma_den=0.;fallbacks=0;wide=0
    for i,date in enumerate(days):
        sr=market.stocks.loc[(date,sid)];spot=float(sr.close)
        if i:
            ratio=float(sr.cfadj)/prev_adj
            move=ratio*spot-prev_spot
            actual+=prev_hedge*move+prev_hedge*ratio*divcash.get(date,0.)
            fixed+=prev_fixed*move+prev_fixed*ratio*divcash.get(date,0.)
            hedge_day=prev_hedge*move+prev_hedge*ratio*divcash.get(date,0.)
            elapsed=(date-days[i-1]).days/365
            gamma_num+=abs(prev_qty)*100*prev_gamma*move**2
            gamma_den+=abs(prev_qty)*100*prev_gamma*prev_effective**2*elapsed
            qty*=ratio
            logs.append(np.log(ratio*spot/prev_spot));squares.append((ratio*spot/prev_spot-1)**2)
        else:hedge_day=0.
        if date==close.date:
            assert np.isclose(qty,-close.quantity)
            option_value=qty*100*close.price
            delta=fixed_delta=0.;gamma=0.;effective=spot
        else:
            q=market.quotes.loc[(date,oid)]
            sigma=float(q.get('bs_iv',np.nan))
            if valid_greeks(q) and np.isfinite(sigma) and sigma>0:
                delta=float(q.delta);last_sigma=sigma
            else:
                delta=float(call_values(spot-q.known_dividend_pv,q.strike,(q.expiry-date).days/365,q.bs_rate,last_sigma)[1])
                fallbacks+=1
            effective=spot-q.known_dividend_pv
            _,fixed_delta,gamma,_=call_values(effective,q.strike,(q.expiry-date).days/365,q.bs_rate,sigma0)
            option_value=qty*100*(q.bid+q.ask)/2
            wide+=int((abs(q.ask-q.bid)/((q.bid+q.ask)/2))>.25)
        if i:pnl[date]=option_value-prev_value+hedge_day
        else:pnl[date]=0.
        hedges[date]=-qty*100*delta
        prev_hedge=-qty*100*delta;prev_fixed=-qty*100*fixed_delta
        prev_spot=spot;prev_adj=float(sr.cfadj);prev_value=option_value
        prev_qty=qty;prev_gamma=float(gamma);prev_effective=effective
    option_pnl=float(-entry.quantity*100*entry.price-close.quantity*100*close.price)
    actual+=option_pnl;fixed+=option_pnl
    np.testing.assert_allclose(sum(pnl.values()),actual,atol=1e-6,rtol=0)
    signal=pd.Timestamp(entry.cohort)
    old_iv=float(asset.iv.get(signal,np.nan));old_rv=float(asset.forward_rv.get(signal,np.nan))
    signed_vega=entry.quantity*100*q0.vega
    rv=100*np.sqrt(252*np.mean(np.square(logs)))
    year_fraction=(close.date-entry.date).days/365
    record=dict(secid=sid,optionid=oid,entry_date=entry.date,exit_date=close.date,
        quantity=entry.quantity,signed_entry_vega=signed_vega,entry_dte=(q0.expiry-entry.date).days,
        entry_delta=q0.delta,standardized_iv=old_iv,entry_iv=100*sigma0,
        signal_contract_iv=100*market.quotes.loc[(signal,oid)].bs_iv,
        proxy_forward_rv=old_rv,expiry_rv=rv,
        calendar_arithmetic_rv=100*np.sqrt(sum(squares)/year_fraction),
        gamma_weighted_rv=100*np.sqrt(gamma_num/gamma_den) if gamma_den>0 else np.nan,
        old_proxy_gross=signed_vega*(old_rv-old_iv),
        contract_iv_proxy_gross=signed_vega*(old_rv-100*sigma0),
        expiry_rv_proxy_gross=signed_vega*(rv-100*sigma0),
        fixed_iv_hedge_gross=fixed,actual_gross=actual,
        gamma_variance_approx=.5*np.sign(entry.quantity)*(gamma_num-sigma0**2*gamma_den),
        option_gross=option_pnl,stock_gross=actual-option_pnl,
        fallbacks=fallbacks,wide_quote_days=wide)
    return record,pd.Series(hedges),pd.Series(pnl)


def main():
    REPORT.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    new=pd.read_csv('reports/expiry_trained/status.csv');old=pd.read_csv('reports/expiry_holding/status.csv')
    failed=set(new.loc[new.status.ne('complete'),'pair'])|set(old.loc[old.status.ne('complete'),'pair'])
    members=set(old.pair)-failed
    defs=[d for d in json.loads(Path('reports/aligned_contract_learning/universe.json').read_text()) if d['pair'] in members]
    cfg=json.loads(Path('reports/expiry_trained/manifest.json').read_text())['settings']
    cal=load_calendar(ROOT);cal=cal[(cal>='2018-01-01')&(cal<='2025-08-29')]
    market=Market(cal);stock_prices=market.stocks.set_index(['date','secid']).close
    legs=[];pairs=[];cohorts=[];worst=[];decisions=[];source_hashes={}
    totals={name:pd.DataFrame(0.,index=cal,columns=COMPONENTS) for name in POLICIES}
    books={};max_error=0.
    for i,d in enumerate(defs):
        prepared=None;assets=None
        for name,(root,sub) in POLICIES.items():
            folder=Path(root)/d['pair']/sub
            nav=complete_nav(pd.read_parquet(folder/'daily_nav.parquet'),cfg['capital'])
            t=read_csv(folder/'trades.csv',['date']);dec=read_csv(folder/'decisions.csv',['signal_date','entry_date'])
            source_hashes[str(folder/'daily_nav.parquet')]=digest(folder/'daily_nav.parquet')
            attrib,_,err=attribute(nav,t,d['secids'],stock_prices,market.divs,cfg)
            max_error=max(max_error,err);totals[name]+=attrib
            result=attrib.sum().to_dict();gross=result['option_mid_pnl']+result['stock_hedge_gross_pnl']+result['dividends']
            entered=dec[dec.status.eq('entered')] if len(dec) else dec
            pairs.append(dict(policy=name,pair=d['pair'],label=d['label'],entries=len(entered),gross=gross,**result))
            if len(dec):
                for reason,count in dec.status.value_counts().items():decisions.append(dict(policy=name,reason=reason,count=count))
            for date,r in attrib.nsmallest(3,'net_excess_pnl').iterrows():
                worst.append(dict(policy=name,pair=d['pair'],label=d['label'],date=date,**r.to_dict()))
            for stage,saving in {'net':np.zeros(len(nav)),'no_option_costs':nav.option_execution_cost,
                'no_execution_costs':nav.option_execution_cost+nav.hedge_cost,
                'no_execution_or_borrow':nav.option_execution_cost+nav.hedge_cost+nav.stock_borrow_cost}.items():
                book=reprice_cash(nav,market.rates,saving,np.zeros(len(nav)),cfg['capital'],cfg['borrowing_spread'])
                if stage=='net':np.testing.assert_allclose(book.nav,nav.nav,atol=1e-7,rtol=0)
                key=(name,stage)
                if key not in books:books[key]=book[['nav','cash_benchmark']].copy()
                else:books[key]+=book[['nav','cash_benchmark']]
            if not len(t):continue
            if prepared is None:
                q,s,selection,divs=market.pair(d['secids']);prepared=PreparedMarket(q,s,selection)
                assets={sid:pd.read_parquet(path,columns=['iv','forward_rv']) for sid,path in zip(d['secids'],asset_paths(d))}
            reconstructed={sid:pd.Series(0.,index=cal) for sid in d['secids']}
            reconstructed_pnl=pd.Series(0.,index=cal)
            for cohort,group in t.groupby('cohort'):
                own=[];tail_gross=0.
                first_expiry=group.loc[group.reason.eq('expiry_settlement'),'date'].min()
                for sid,leg in group.groupby('secid'):
                    row,hedge,pnl=leg_path(leg,prepared,cal,divs,assets[sid])
                    row.update(policy=name,pair=d['pair'],label=d['label'],cohort=cohort)
                    legs.append(row);own.append(row)
                    tail_gross+=pnl.loc[pnl.index>first_expiry].sum()
                    reconstructed[sid].loc[hedge.index]+=hedge
                    reconstructed_pnl.loc[pnl.index]+=pnl
                decision=entered[entered.signal_date.eq(pd.Timestamp(cohort))].iloc[0]
                values=['old_proxy_gross','contract_iv_proxy_gross','expiry_rv_proxy_gross','fixed_iv_hedge_gross','actual_gross','gamma_variance_approx']
                cohorts.append(dict(policy=name,pair=d['pair'],label=d['label'],cohort=cohort,
                    action=decision.action,mean_entry_vega=(decision.vega_a+decision.vega_b)/2,
                    entry_date=decision.entry_date,exit_date=max(x['exit_date'] for x in own),
                    expiry_gap_days=(max(x['exit_date'] for x in own)-first_expiry).days,
                    after_first_expiry_gross=tail_gross,
                    entry_option_cost=group.commission.sum()+group.execution_penalty.sum(),
                    **{v:sum(x[v] for x in own) for v in values}))
            for sid,h in reconstructed.items():np.testing.assert_allclose(h,nav[f'hedge_{sid}'],atol=1e-7,rtol=0)
            np.testing.assert_allclose(reconstructed_pnl,attrib.option_mid_pnl+attrib.stock_hedge_gross_pnl+attrib.dividends,atol=1e-6,rtol=0)
        if i%25==0:print(f'Attributed {i+1}/{len(defs)} pairs',flush=True)
    pd.DataFrame(legs).to_csv(REPORT/'trade_legs.csv',index=False)
    pd.DataFrame(cohorts).to_csv(REPORT/'cohorts.csv',index=False)
    pd.DataFrame(pairs).to_csv(REPORT/'pairs.csv',index=False)
    pd.DataFrame(worst).sort_values('net_excess_pnl').groupby('policy',sort=False).head(20).to_csv(REPORT/'worst_days.csv',index=False)
    pd.DataFrame(decisions).groupby(['policy','reason'])['count'].sum().to_csv(REPORT/'entry_decisions.csv')
    annual=[]
    for name,frame in totals.items():
        frame.to_csv(REPORT/f'{name}_daily_attribution.csv')
        for year,g in frame.groupby(frame.index.year):annual.append(dict(policy=name,year=year,**g.sum().to_dict()))
    pd.DataFrame(annual).to_csv(REPORT/'annual_components.csv',index=False)
    rows=[]
    for (name,stage),book in books.items():
        book['return_']=book.nav.pct_change().fillna(0);book['excess_return']=book.return_-book.cash_benchmark.pct_change().fillna(0)
        rows.append(dict(policy=name,stage=stage,**scores(book,len(defs)*cfg['capital'])))
    pd.DataFrame(rows).to_csv(REPORT/'cost_bridge.csv',index=False)
    (REPORT/'manifest.json').write_text(json.dumps(dict(pairs=len(defs),settings=cfg,max_ledger_error=max_error,
        elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        sources=source_hashes,code_sha256=digest(__file__)),indent=2)+'\n')
    print(pd.DataFrame(cohorts).groupby('policy')[['old_proxy_gross','contract_iv_proxy_gross','expiry_rv_proxy_gross','fixed_iv_hedge_gross','actual_gross']].sum().to_string())


if __name__=='__main__':main()
