"""Monthly ATM-call portfolio construction, daily integer vega/delta balancing."""
from pathlib import Path
from functools import lru_cache
import json
import resource
import warnings
import numpy as np
import pandas as pd
from scripts.forecast_contract_variance import forecast,MODELS
from scripts.discover_contract_signals import digest
from scripts.contract_engine import Settings,valid_quote,verified_integer_split,execution_terms
from scripts.black_scholes import call_values
from scripts.portfolio_vega import allocate,exposures,constraint_matrix,CAPITAL,OPTION_COMMISSION

OUT=Path('artifacts/contracts/vega_portfolio');REPORT=Path('reports/vega_portfolio')
ROOT=Path('data/option_contracts');CACHE=Path('data/contract_history_shared')
CFG=Settings(option_commission=OPTION_COMMISSION,option_penalty_vol_points=.1,option_spread_fraction=0.)
COLS=['date','optionid','secid','expiry','strike','bid','ask','delta','gamma','vega','bs_iv','bs_rate','known_dividend_pv','cfadj','standard','multiplier','cp_flag','open_interest','bs_delta','bs_gamma','bs_vega']

@lru_cache(maxsize=35)
def quotes(sid,year):
    p=CACHE/str(year)/f'{sid}.parquet'
    if not p.exists():return {}
    q=pd.read_parquet(p,columns=COLS)
    for greek in ['delta','gamma','vega']:q[greek]=q['bs_'+greek]
    return {(r['date'],int(r['optionid'])):r for r in q.to_dict('records')}


def eligible(d,date):
    if 'screen_quarters' in d:return str(date.to_period('Q')) in d['screen_quarters']
    return date.year in {y+k for y in d['screen_years'] for k in [0,1]}


def setup():
    definitions=json.loads(Path('artifacts/contracts/curve90/inputs.json').read_text())['definitions']
    ids=sorted({s for d in definitions for s in d['secids']})
    pieces=[]
    for year in range(2019,2026):
        pieces.append(pd.read_parquet(ROOT/'underlying'/f'year={year}.parquet',columns=['date','secid','close','cfadj'],filters=[('secid','in',ids)]))
    stocks=pd.concat(pieces);cal=pd.DatetimeIndex(stocks.date.unique()).sort_values()
    spots=stocks.pivot(index='date',columns='secid',values='close').reindex(cal)
    adj=stocks.pivot(index='date',columns='secid',values='cfadj').reindex(cal)
    square=np.log((spots*adj)/(spots*adj).shift())**2
    rv={n:252*square.rolling(n,min_periods=n).mean() for n in [5,21,63]}
    evalcal=cal[(cal>='2020-01-01')&(cal<='2025-08-29')]
    first=evalcal.to_series().groupby(evalcal.to_period('M')).first()
    signal_dates=pd.DatetimeIndex([cal[cal.get_loc(d)-1] for d in first])
    selections=[]
    for year in sorted(signal_dates.year.unique()):
        days=signal_dates[signal_dates.year==year].tolist()
        for sid in ids:
            p=CACHE/str(year)/f'{sid}_selection.parquet'
            if p.exists():selections.append(pd.read_parquet(p,columns=COLS,filters=[('date','in',days)]))
        print(f'Loaded monthly selections {year}',flush=True)
    sel=pd.concat(selections)
    for greek in ['delta','gamma','vega']:sel[greek]=sel['bs_'+greek]
    assets=json.loads(Path('artifacts/broad_pairs/assets.json').read_text())
    countdown={};assetmap={a['secid']:a for a in assets}
    for sid in ids:
        if sid in assetmap:
            p=Path('artifacts/broad_pairs/assets')/(assetmap[sid]['asset']+'.parquet')
            countdown[sid]=pd.read_parquet(p,columns=['earnings']).earnings
    sector={}
    # The cached historical screen supplies SIC industry groups, not GICS sectors.
    for d in definitions:
        value=str(d.get('industry','unknown'))
        group=value[:2] if value.isdigit() else 'unknown'
        for sid in d['secids']:
            if sid in sector and sector[sid]!=group:sector[sid]='unknown'
            else:sector[sid]=group
    zero=pd.read_parquet(ROOT/'zero_curve.parquet');zero['date']=pd.to_datetime(zero.date);zero['distance']=abs(zero.days-30)
    zero=zero.sort_values(['date','distance']).drop_duplicates('date').sort_values('date')
    rates=pd.merge_asof(pd.DataFrame({'date':cal}),zero[['date','rate']],on='date',direction='backward',tolerance=pd.Timedelta(days=7)).set_index('date').rate/100
    if rates.reindex(evalcal).isna().any():raise ValueError('Missing rates')
    div=pd.read_parquet(ROOT/'distributions.parquet')
    div=div[div.cancel_flag.astype(str).eq('0')&div.distr_type.astype(str).eq('1')&div.currency.eq('USD')&div.adj_factor.eq(1)&div.approx_flag.astype(str).eq('0')&div.liquid_flag.astype(str).eq('0')]
    divs={}
    for r in div.itertuples():
        i=cal.searchsorted(pd.Timestamp(r.ex_date))
        if i<len(cal):divs[(cal[i],r.secid)]=divs.get((cal[i],r.secid),0.)+r.amount
    train=pd.read_parquet('artifacts/contracts/variance_forecast/legs.parquet')
    monthly={};audit=[]
    for date,signal in zip(first,signal_dates):
        members={sid for d in definitions if eligible(d,signal) for sid in d['secids']}
        q=sel[sel.date.eq(signal)&sel.secid.isin(members)].copy()
        q=q[q.apply(lambda r:valid_quote(r,True,CFG),axis=1)]
        q=q[(q.expiry-date).dt.days.between(21,45)&q.delta.between(.25,.75)]
        q=q[[bool(spots.loc[signal,s]>=5) for s in q.secid]]
        if q.empty:monthly[date]=q;continue
        q['log_ivvar']=np.log(q.bs_iv**2);q['log_dte']=np.log((q.expiry-date).dt.days)
        for n in [5,21,63]:q[f'log_rv{n}']=[np.log(max(rv[n].loc[signal,s],1e-8)) for s in q.secid]
        q=q.replace([np.inf,-np.inf],np.nan).dropna(subset=MODELS['har_iv'])
        cutoff=signal.to_period('Q').start_time
        tr=train[(train.end<cutoff)&(train.date>=cutoff-pd.DateOffset(years=5))]
        if len(tr)<500 or q.empty:monthly[date]=q.iloc[:0];continue
        q['forecast']=forecast(tr,q,MODELS['har_iv'])
        q['score']=100*(q.forecast-q.bs_iv**2)/(2*q.bs_iv)
        q['score']-=q.score.median()
        q['sector']=[sector.get(s,'unknown') for s in q.secid]
        q['event']=[float(countdown.get(s,pd.Series(dtype=float)).get(signal,0.)<=7) for s in q.secid]
        # No label resolution / future exit availability filter for candidates.
        q=pd.concat([q[q.score>0].nlargest(15,'score'),q[q.score<0].nsmallest(15,'score')])
        monthly[date]=q
        audit.append(dict(date=date,signal_date=signal,fit_cutoff=cutoff,training_end=tr.end.max(),training_rows=len(tr),candidates=len(q)))
    pd.DataFrame(audit).to_csv(REPORT/'fit_audit.csv',index=False)
    return evalcal,spots,adj,rates,divs,monthly,countdown,len(ids)


def mark(p,date,spots,adjustments):
    sid=p['secid'];oid=p['optionid'];q=quotes(sid,date.year).get((date,oid))
    if q is None:raise ValueError(f'Missing held quote {sid}/{oid} on {date.date()}')
    q=dict(q);spot=spots.loc[date,sid]
    if not np.isfinite(spot) or spot<=0:raise ValueError(f'Missing held stock {sid} {date}')
    if 0<=q['ask']<q['bid']:q['bid'],q['ask']=q['ask'],q['bid']
    if not valid_quote(q):raise ValueError(f'Invalid held quote {sid}/{oid} {date}')
    ratio=adjustments.loc[date,sid]/p['stock_adj']
    if not np.isclose(ratio,1):
        if not verified_integer_split(p,q,ratio):raise ValueError(f'Unsupported held split {sid} {date}')
        p['quantity']*=int(round(ratio));p['max_contracts']*=int(round(ratio));p['hedge']*=ratio;p['prev_spot']/=ratio
        p.update(strike=q['strike'],cfadj=q['cfadj'],stock_adj=adjustments.loc[date,sid])
    if q['expiry']!=p['expiry'] or not np.isclose(q['strike'],p['strike'],atol=.005) or q['cfadj']!=p['cfadj']:raise ValueError('Unsupported option adjustment')
    fallback=False
    if not np.isfinite([q['bs_iv'],q['delta'],q['vega'],q['gamma']]).all() or q['bs_iv']<=0:
        q['bs_iv']=p['last_iv'];t=(q['expiry']-date).days/365
        _,q['delta'],q['gamma'],q['vega']=call_values(spot-q['known_dividend_pv'],q['strike'],t,q['bs_rate'],q['bs_iv'])
        fallback=True
    p['last_iv']=q['bs_iv']
    return q,float(spot),fallback


def run(mode,data,entry_scales=None,quiet=False,persist=True):
    cal,spots,adj,rates,divs,monthly,countdown,_=data
    def event_risk(sid,date):
        days=countdown.get(sid,pd.Series(dtype=float)).get(date,np.nan)
        return float(not np.isfinite(days) or days<=7)
    cash=CAPITAL;benchmark=CAPITAL;navprev=CAPITAL;book=[];daily=[];trades=[];holdings=[];previous=None;close_date=None
    for date in cal:
        interest=borrow=dividend=opcost=stockcost=optionpnl=hedgepnl=0.;fallbacks=0
        if previous is not None:
            dt=(date-previous).days/365;r=rates.loc[previous]
            interest=cash*(r+(.01 if cash<0 else 0))*dt;cash+=interest
            benchmark*=1+r*dt
        marks=[]
        for p in book:
            p['event']=event_risk(p['secid'],date)
            q,spot,fallback=mark(p,date,spots,adj);fallbacks+=fallback;marks.append((q,spot))
            dt=(date-previous).days/365
            cost=max(-p['hedge'],0)*p['prev_spot']*.003*dt
            payment=p['hedge']*divs.get((date,p['secid']),0.)
            cash+=payment-cost;borrow+=cost;dividend+=payment
            optionpnl+=p['quantity']*100*(q['bid']+q['ask'])/2-p['prev_option_value']
            hedgepnl+=p['hedge']*(spot-p['prev_spot'])
        before=cash+sum(p['quantity']*100*(q['bid']+q['ask'])/2+p['hedge']*spot for p,(q,spot) in zip(book,marks))
        expected=navprev+interest-borrow+dividend+optionpnl+hedgepnl
        if not np.isclose(before,expected,rtol=0,atol=1e-5):raise AssertionError('Pretrade ledger mismatch')
        reset=date in monthly or (close_date is not None and date>=close_date) or date==cal[-1]
        if reset:
            for p,(q,spot) in zip(book,marks):
                qty=-p['quantity'];mid,penalty,_=execution_terms(q,qty,CFG)
                fee=abs(qty)*CFG.option_commission;cash-=qty*100*mid+penalty+fee;opcost+=penalty+fee
                hc=abs(p['hedge'])*spot*.0001;cash+=p['hedge']*spot-hc;stockcost+=hc
                trades.append(dict(date=date,secid=p['secid'],optionid=p['optionid'],quantity=qty,reason='close',cost=penalty+fee))
            book=[];marks=[];close_date=None
        if date in monthly and date!=cal[-1]:
            qframe=monthly[date]
            entry_scale=1. if entry_scales is None else float(entry_scales.get(date,0.))
            if not 0<=entry_scale<=1:raise ValueError('Portfolio scale outside [0, 1]')
            if entry_scale==0:qframe=qframe.iloc[:0]
            for row in qframe.to_dict('records'):
                sid=int(row['secid']);q=quotes(sid,date.year).get((date,int(row['optionid'])))
                spot=spots.loc[date,sid]
                if q is None or not valid_quote(q,True,CFG) or not np.isfinite([q['bs_iv'],q['gamma'],spot]).all():continue
                if any(q[k]!=row[k] for k in ['strike','expiry','cfadj']):continue
                if not 21<=(q['expiry']-date).days<=45:continue
                if not np.isfinite(spot) or spot<5 or not .25<=q['delta']<=.75:continue
                p=dict(max_contracts=min(100,int(.1*row['open_interest'])),secid=sid,optionid=int(q['optionid']),quantity=0,hedge=0.,strike=q['strike'],expiry=q['expiry'],cfadj=q['cfadj'],stock_adj=adj.loc[date,sid],prev_spot=spot,last_iv=q['bs_iv'],sign=int(np.sign(row['score'])),score=row['score'],sector=row['sector'],event=event_risk(sid,date))
                book.append(p);marks.append((q,float(spot)))
            if book:
                m=np.array([exposures(q,s,date,p['sign']) for p,(q,s) in zip(book,marks)]).T
                sign=np.array([p['sign'] for p in book]);sectors=[p['sector'] for p in book];events=np.array([p['event'] for p in book]);score=np.array([p['score'] for p in book])
                qty=allocate(m,sign,sectors,events,.99*before,score,mode,max_contracts=[p['max_contracts'] for p in book])
                if entry_scale<1:
                    qty=allocate(m,sign,sectors,events,.99*before,score,mode,target=qty*m[0]*entry_scale,max_contracts=[p['max_contracts'] for p in book])
                for i,p in enumerate(book):p['target_vega']=qty[i]*m[0,i]
            else:qty=np.array([],dtype=int)
            reason='open'
        elif book:
            m=np.array([exposures(q,s,date,p['sign']) for p,(q,s) in zip(book,marks)]).T
            qty=allocate(m,np.array([p['sign'] for p in book]),[p['sector'] for p in book],np.array([p['event'] for p in book]),.99*before,np.array([p['score'] for p in book]),mode,target=[p['target_vega'] for p in book],max_contracts=[p['max_contracts'] if .15<=q['delta']<=.85 and valid_quote(q,True,CFG) else min(p['max_contracts'],abs(p['quantity'])) for p,(q,_) in zip(book,marks)]);reason='rebalance'
        else:qty=np.array([],dtype=int);reason='flat'
        retained=[];newmarks=[]
        for i,(p,(q,spot)) in enumerate(zip(book,marks)):
            desired=int(qty[i])*p['sign'];change=desired-p['quantity']
            if change:
                mid,penalty,_=execution_terms(q,change,CFG);fee=abs(change)*CFG.option_commission
                cash-=change*100*mid+penalty+fee;opcost+=penalty+fee
                trades.append(dict(date=date,secid=p['secid'],optionid=p['optionid'],quantity=change,reason=reason,cost=penalty+fee))
            hedge=-desired*100*q['delta'];delta=hedge-p['hedge'];cost=abs(delta)*spot*.0001
            cash-=delta*spot+cost;stockcost+=cost
            p.update(quantity=desired,hedge=hedge,prev_spot=spot,prev_option_value=desired*100*(q['bid']+q['ask'])/2)
            if desired:
                retained.append(p);newmarks.append((q,spot))
                holdings.append(dict(date=date,secid=p['secid'],optionid=p['optionid'],quantity=desired,vega=desired*100*q['vega'],sector=p['sector']))
        book=retained;marks=newmarks
        if book and close_date is None:
            expiry=min(p['expiry'] for p in book)
            close_date=cal[max(0,cal.searchsorted(expiry,side='right')-2)] if expiry<=cal[-1] else cal[-1]
        nav=cash+sum(p['prev_option_value']+p['hedge']*spot for p,(q,spot) in zip(book,marks))
        residual=nav-(before-opcost-stockcost)
        if abs(residual)>1e-5:raise AssertionError(f'Posttrade ledger mismatch {residual}')
        gross=sum(abs(p['quantity'])*100*q['vega'] for p,(q,s) in zip(book,marks));net=sum(p['quantity']*100*q['vega'] for p,(q,s) in zip(book,marks))
        if abs(net)>max(10.,.01*gross)+1e-7:raise AssertionError('Vega neutrality failed')
        if book:
            m=np.array([exposures(q,s,date,p['sign']) for p,(q,s) in zip(book,marks)]).T
            aa,bb=constraint_matrix(m,np.array([p['sign'] for p in book]),[p['sector'] for p in book],np.array([p['event'] for p in book]),before)
            if np.any(aa@np.array([abs(p['quantity']) for p in book])>bb+1e-5):raise AssertionError('Risk limit violation')
        rf=0 if previous is None else rates.loc[previous]*(date-previous).days/365
        daily.append(dict(date=date,nav=nav,cash_benchmark=benchmark,return_=nav/navprev-1,excess_return=nav/navprev-1-rf,
                          gross_vega=gross,net_vega=net,names=len(book),option_cost=opcost,stock_cost=stockcost,borrow=borrow,interest=interest,dividends=dividend,option_pnl=optionpnl,hedge_pnl=hedgepnl,ledger_residual=residual,greek_fallbacks=fallbacks))
        if nav<=0:
            folder=OUT/mode;folder.mkdir(exist_ok=True)
            pd.DataFrame(daily).to_csv(folder/'failed_nav.csv',index=False)
            pd.DataFrame(trades).to_csv(folder/'failed_trades.csv',index=False)
            raise ValueError(f'Portfolio insolvent at {date}; no valid full-period Sharpe')
        previous=date;navprev=nav
        if date in monthly and not quiet:print(f'{mode} {date.date()}: {len(book)} names, gross vega {gross:,.0f}, NAV {nav:,.0f}',flush=True)
    assert not book
    frame=pd.DataFrame(daily).set_index('date')
    if persist:
        folder=OUT/mode;folder.mkdir(exist_ok=True)
        frame.to_csv(folder/'nav.csv');pd.DataFrame(trades,columns=['date','secid','optionid','quantity','reason','cost']).to_csv(folder/'trades.csv',index=False);pd.DataFrame(holdings,columns=['date','secid','optionid','quantity','vega','sector']).to_csv(folder/'holdings.csv',index=False)
    return frame


def metrics(f):
    e=f.excess_return;v=e.std()
    return dict(annualized_excess_sharpe=float(np.sqrt(252)*e.mean()/v) if v>1e-12 else None,
                return_percent=float(((1+f.return_).prod()-1)*100),excess_pnl=float(f.nav.iloc[-1]-f.cash_benchmark.iloc[-1]),
                max_drawdown_percent=float((f.nav/f.nav.cummax()-1).min()*100),active_days=int(f.names.gt(0).sum()),
                option_cost=float(f.option_cost.sum()),hedge_cost=float(f.stock_cost.sum()),max_abs_net_vega=float(f.net_vega.abs().max()))


def main():
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    warnings.filterwarnings('ignore',message='Unrecognized options detected')
    (OUT/'status.json').write_text(json.dumps(dict(status='running',option_commission_per_contract=CFG.option_commission))+'\n')
    data=setup();results=[];annual=[]
    for mode in ['forecast','equal_rank']:
        try:
            f=run(mode,data);r=dict(model=mode,status='complete',**metrics(f));results.append(r)
            for year,g in f.groupby(f.index.year):annual.append(dict(model=mode,year=year,**metrics(g)))
        except Exception as exc:
            (OUT/'status.json').write_text(json.dumps(dict(status='incomplete',model=mode,error=str(exc)),indent=2)+'\n');raise
        quotes.cache_clear()
    pd.DataFrame(results).to_csv(REPORT/'summary.csv',index=False);pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False)
    status=dict(status='complete',option_commission_per_contract=CFG.option_commission,universe_names=data[-1],results=results,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                code_hashes={str(p):digest(p) for p in [Path(__file__),Path('scripts/portfolio_vega.py'),Path('scripts/forecast_contract_variance.py')]})
    (OUT/'status.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status,indent=2),flush=True)

if __name__=='__main__':main()
