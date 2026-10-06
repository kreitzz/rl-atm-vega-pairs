"""Bounded cached-data preparation for frozen-policy joint sizing."""
from collections import defaultdict,OrderedDict
from dataclasses import replace
import gc
import json
from pathlib import Path
import resource

import numpy as np
import pandas as pd

from scripts.run_put75_cost_hurdles import Sources
from scripts.run_atm65_volume_oi import employee_roster,baseline_actions,baseline_results
from scripts.run_atm65_ablation import read_actions,write_json
from scripts.run_contract_walkforward import digest
from scripts.run_broad_contract_replay import ROOT,Market
from scripts.curve_trading.config import DATA,ACTIONS,COMBOS,config,RUNOFF
from scripts.curve_trading.unit_paths import UnitPaths,pair_labels
from scripts.put75_cost_hurdle import with_training_commission
from scripts.run_smile_sizing import scoped_inputs
from scripts.smile_sizing_policy import directional_labels
from scripts.replay_frozen_atm_commission import cash_book
from scripts.run_selected_pair_backtest import scores
from scripts.option_flex.contract_engine import MarketRow

OUT=Path('artifacts/contracts/joint_smile')
REPORT=Path('reports/joint_smile')
WINGS=['ATM','P25','P75']
SLOTS={'ATM':('C','atm'),'P25':('P','25'),'P75':('P','75')}
HURDLES={'ATM':.65,'P25':.65,'P75':.50}
INVENTORY=['pair_inventory_vega','pair_inventory_hedge','pair_inventory_cohorts','global_inventory_vega','global_inventory_hedge']
CORE=['iv_mean','relative_iv_cheapness','rank_strength','beta','correlation','vix',
      'profit_91d','count_91d','profit_365d','count_365d','strategy_sharpe','strategy_volatility','strategy_correlation']
FEATURES=CORE+[c+'_available' for c in CORE]+INVENTORY


def progress(stage,**details):
    write_json(OUT/'progress.json',{**details,'stage':stage,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


def registry():
    s=Sources()
    for study in ['tail_put25_residual_rl','put75_lower_hurdles','put75_cost_hurdles']:
        s.expected.update(json.loads(Path(f'reports/{study}/output_hashes.json').read_text()))
    s.expected.update(json.loads(Path('config/atm30_fee065_employee_2026-10-01_checkpoint.json').read_text())['files'])
    if (OUT/'sources.json').exists():
        for path,value in json.loads((OUT/'sources.json').read_text()).items():
            assert digest(path)==value,f'Joint source changed: {path}'
            s[path]=value
            if path in s.expected:
                assert value==s.expected[path];s.verified.add(path)
    return s


def frozen_paths(d,wing,employees,sources):
    pair=d['pair']
    if wing=='ATM':
        root=baseline_actions(pair,'rl',pair in employees).parent
        full=baseline_results(pair,'rl','full',pair in employees)
        state=Path('artifacts/contracts/atm65_employee_sentiment')/pair/'states.parquet'
    elif wing=='P25':
        root=Path('artifacts/contracts/tail_put25_residual_rl')/pair/'context_residual_rl';full=root/'full'
        state=Path('artifacts/contracts/tail_put25_pooled')/pair/'states.parquet'
    else:
        root=Path('artifacts/contracts/put75_lower_hurdles/pairs')/pair/'050';full=root/'full'
        if (root/'reference.json').exists():
            full=Path(json.loads(sources.track(root/'reference.json').read_text())['source'])
        state=Path('artifacts/contracts/smile_sizing/pairs')/pair/'P75/features.parquet'
    return root/'actions.csv',full,state


def state_table(panel,actions,wing,cal):
    p=panel.reindex(cal);code=actions.reindex(cal,fill_value=0)
    cp,bucket=SLOTS[wing];bucket=int(bucket) if bucket!='atm' else bucket
    for c in code.loc[code.ne(0)].unique():assert COMBOS[ACTIONS[int(c)][0]]==(30,cp,bucket,bucket)
    level=np.array([int(round(abs(ACTIONS[int(c)][1])*4)) if c else 0 for c in code])
    direction=np.array([int(np.sign(ACTIONS[int(c)][1])) if c else 0 for c in code])
    x=pd.DataFrame(index=cal)
    if wing=='ATM':
        observed=np.isfinite(p.iv_amd)&np.isfinite(p.iv_nvda)&p.iv_amd.gt(0)&p.iv_nvda.gt(0)
        mean=(p.iv_amd+p.iv_nvda)/2;gap=p.iv_amd-p.iv_nvda
        x['iv_mean']=mean;x['relative_iv_cheapness']=-direction*gap/mean
        rank=panel.spread.rolling(252,min_periods=63).rank(pct=True).reindex(cal)
        x['rank_strength']=(rank-.5).abs()*2
        x['vix']=p.vix_previous_close/100
        for days in [91,365]:
            for prefix in ['profit','count']:
                x[f'{prefix}_{days}d']=np.where(direction>0,p[f'{prefix}_long_{days}d'],p[f'{prefix}_short_{days}d'])
                if prefix=='profit':x[f'{prefix}_{days}d']/=100
    else:
        observed=p.observed.fillna(False)
        x['iv_mean']=p.iv_mean
        x['relative_iv_cheapness']=p.relative_iv_cheapness if wing=='P75' else -p.oriented_iv_gap
        x['rank_strength']=p.rank_strength;x['vix']=p.vix
        for days in [91,365]:
            for prefix in ['profit','count']:
                x[f'{prefix}_{days}d']=p[f'{prefix}_{days}d']
    for c in ['beta','correlation']:x[c]=p[c]
    for c in ['strategy_sharpe','strategy_volatility','strategy_correlation']:x[c]=np.nan
    for c in CORE:
        x[c]=x[c].replace([np.inf,-np.inf],np.nan)
        x[c+'_available']=x[c].notna().astype(float);x[c]=x[c].fillna(0.)
    x['base_code']=code;x['base_level']=level;x['direction']=direction
    x['observed']=np.asarray(observed,bool)
    x=x.loc[x.observed&(x.index<=cal[-1]-pd.Timedelta(days=RUNOFF))]
    assert np.isfinite(x[CORE]).all().all()
    return x


def load_reference(defs,market,sources):
    cal=market.calendar[market.calendar>='2018-01-01'];cash=cash_book(cal,market.rates,1_000_000)
    employees=employee_roster();sources.track(Path('reports/atm65_employee_sentiment/coverage.csv'))
    required=defaultdict(set);frames=[];coverage=[]
    books={w:cash[['nav','cash_benchmark']]*len(defs) for w in WINGS}
    global_inventory=pd.DataFrame(0.,index=cal,columns=['gross_vega','gross_hedge_notional'])
    for n,d in enumerate(defs,1):
        pair_inv=pd.DataFrame(0.,index=cal,columns=['gross_vega','gross_hedge_notional','active_cohorts'])
        local=[]
        for wing in WINGS:
            action_path,full,state=frozen_paths(d,wing,employees,sources)
            status=json.loads(sources.track(full/'status.json').read_text());assert status['status']=='complete'
            if action_path.exists():actions=read_actions(sources.track(action_path))
            else:
                assert status.get('cash_only') and not status['entries']
                actions=pd.Series(0,index=cal,dtype=int)
            if status.get('cash_only'):
                nav=cash
            else:
                nav=pd.read_parquet(sources.track(Path(status.get('nav_source',full/'daily_nav.parquet'))))
                assert nav.index.equals(cal)
                pair_inv+=nav[['gross_vega','gross_hedge_notional','active_cohorts']]
            books[wing]+=nav[['nav','cash_benchmark']]-cash[['nav','cash_benchmark']]
            if state.exists():
                panel=pd.read_parquet(sources.track(state));x=state_table(panel,actions,wing,cal)
            else:
                assert not actions.ne(0).any();x=pd.DataFrame(columns=FEATURES+['base_code','base_level','direction','observed'],index=pd.DatetimeIndex([]))
            assert int(x.base_level.gt(0).sum())==int(actions.reindex(cal,fill_value=0).ne(0).sum()),f'Frozen proposals omitted: {d["pair"]} {wing}'
            coverage.append(dict(pair=d['pair'],label=d['label'],wing=wing,observed_states=len(x),frozen_proposals=int(actions.ne(0).sum()),frozen_entries=status['entries']))
            x['pair']=d['pair'];x['wing']=wing;x['secid_a']=d['secids'][0];x['secid_b']=d['secids'][1]
            for sid in d['secids']:required[sid,wing].update(x.index[x.base_level.gt(0)].tolist())
            local.append(x)
        global_inventory+=pair_inv[['gross_vega','gross_hedge_notional']]
        for x in local:
            x['pair_inventory_vega']=pair_inv.gross_vega.reindex(x.index)/60_000
            x['pair_inventory_hedge']=pair_inv.gross_hedge_notional.reindex(x.index)/6_000_000
            x['pair_inventory_cohorts']=pair_inv.active_cohorts.reindex(x.index)/10
            frames.append(x)
        if n%40==0:print(f'Joint reference {n}/{len(defs)}',flush=True)
    pnl=pd.DataFrame({w:(b.nav-b.cash_benchmark).diff().fillna(0.) for w,b in books.items()})
    for x in frames:
        if not len(x):continue
        wing=x.wing.iloc[0];r=pnl[wing];other=pnl.drop(columns=wing).sum(axis=1)
        sd=r.rolling(126,min_periods=63).std()
        for col,series in [('strategy_volatility',sd/450),('strategy_sharpe',np.sqrt(252)*r.rolling(126,min_periods=63).mean()/sd.replace(0,np.nan)),
                           ('strategy_correlation',r.rolling(126,min_periods=63).corr(other))]:
            values=series.reindex(x.index).replace([np.inf,-np.inf],np.nan)
            x[col]=values.fillna(0);x[col+'_available']=values.notna().astype(float)
        x['global_inventory_vega']=global_inventory.gross_vega.reindex(x.index)/(60_000*len(defs))
        x['global_inventory_hedge']=global_inventory.gross_hedge_notional.reindex(x.index)/(6_000_000*len(defs))
    state=pd.concat([x for x in frames if len(x)]).sort_index(kind='stable');state.index.name='date'
    assert np.isfinite(state[FEATURES]).all().all()
    state.reset_index().to_parquet(OUT/'states.parquet',index=False)
    pd.DataFrame(coverage).to_csv(REPORT/'coverage.csv',index=False)
    combined=sum(books.values())
    combined['return_']=combined.nav.pct_change().fillna(0)
    combined['excess_return']=combined.return_-combined.cash_benchmark.pct_change().fillna(0)
    result=scores(combined,3_000_000*len(defs))
    np.testing.assert_allclose(result['annualized_excess_sharpe'],.7723634057993916,atol=1e-10,rtol=0)
    combined.to_csv(REPORT/'frozen_separate_accounts_portfolio.csv')
    write_json(REPORT/'reference_audit.json',dict(status='complete',annualized_excess_sharpe=result['annualized_excess_sharpe'],pairs=len(defs),coverage_rows=len(coverage),proposals=int(state.base_level.gt(0).sum()),observed_states=len(state)))
    write_json(OUT/'required_dates.json',[dict(secid=int(s),wing=w,dates=[str(d.date()) for d in sorted(ds)]) for (s,w),ds in required.items() if ds])
    progress('reference_complete',proposals=int(state.base_level.gt(0).sum()),observed_states=len(state))


def prepare_assets(market,sources):
    required=json.loads((OUT/'required_dates.json').read_text());by_sid=defaultdict(dict)
    for r in required:by_sid[r['secid']][r['wing']]=pd.to_datetime(r['dates'])
    folder=OUT/'market/assets';folder.mkdir(parents=True,exist_ok=True)
    records=[]
    for n,(sid,dates) in enumerate(sorted(by_sid.items()),1):
        stamp=folder/f'{sid}.json'
        if stamp.exists():records.append(json.loads(stamp.read_text()));continue
        parts=[]
        for wing,days in dates.items():
            cp,bucket=SLOTS[wing]
            for year in sorted(set(days.year)):
                path=DATA/'selection'/cp/str(year)/f'{sid}.parquet'
                if path.exists():
                    frame=pd.read_parquet(sources.track(path));parts.append(frame.loc[frame.tenor.eq(30)&frame.bucket.eq(bucket)&frame.date.isin(days)])
        sel=pd.concat(parts,ignore_index=True).drop_duplicates(['date','tenor','cp_flag','bucket']) if parts else pd.DataFrame(columns=['date','expiry','tenor','cp_flag','bucket','optionid'])
        parts=[]
        for cp,rows in sel.groupby('cp_flag',sort=False):
            ids=rows.optionid.unique().tolist()
            for year in sorted(set(market.calendar[market.calendar>='2018-01-01'].year)):
                path=DATA/'history'/cp/str(year)/f'{sid}.parquet'
                if path.exists():parts.append(pd.read_parquet(sources.track(path),filters=[('optionid','in',ids)]))
        q=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
        stock=market.stocks.loc[market.stocks.secid.eq(sid)].set_index('date').reindex(market.calendar)
        stock['secid']=sid;stock['rate']=market.rates;stock.index.name='date'
        divs=market.divs.loc[market.divs.secid.eq(sid)]
        q,s,sel,divs=scoped_inputs([dict(q=q,s=stock.reset_index(),sel=sel,divs=divs)],[sid])
        q.to_parquet(folder/f'{sid}_quotes.parquet',index=False);s.to_parquet(folder/f'{sid}_stocks.parquet',index=False)
        sel.to_parquet(folder/f'{sid}_selection.parquet',index=False);divs.to_parquet(folder/f'{sid}_dividends.parquet',index=False)
        record=dict(secid=sid,quotes=len(q),selections=len(sel),wings=sorted(dates))
        write_json(stamp,record);records.append(record);write_json(OUT/'sources.json',sources)
        progress('prepare_assets',asset=n,assets=len(by_sid),secid=sid)
        if n%10==0:print(f'Joint assets {n}/{len(by_sid)}: {sid}',flush=True)
        del q,s,sel,divs,stock,parts;gc.collect()
    pd.DataFrame(records).to_csv(REPORT/'assets.csv',index=False)


class UnitCache:
    def __init__(self,market):self.market=market;self.cache={}
    def retain(self,ids):
        for sid in list(self.cache):
            if sid not in ids:del self.cache[sid]
        gc.collect()
    def asset(self,sid):
        if sid in self.cache:return self.cache[sid]
        folder=OUT/'market/assets'
        q,s,sel,divs=[pd.read_parquet(folder/f'{sid}_{kind}.parquet') for kind in ['quotes','stocks','selection','dividends']]
        dates=self.market.calendar[self.market.calendar>='2018-01-01']
        unit=UnitPaths(q,s,sel,self.market.calendar,dates,divs,replace(config(),option_commission=.65))
        self.cache[sid]=(unit,sel);return unit,sel


def prepare_labels(defs,market):
    state=pd.read_parquet(OUT/'states.parquet');candidate=state.loc[state.base_level.gt(0)].copy()
    dates=market.calendar[market.calendar>='2018-01-01'];loader=UnitCache(market);pieces=[]
    for n,d in enumerate(defs,1):
        rows=candidate.loc[candidate.pair.eq(d['pair'])]
        if rows.empty:continue
        folder=OUT/'pairs'/d['pair'];folder.mkdir(parents=True,exist_ok=True)
        if (folder/'prepared.json').exists():
            pieces.append(pd.read_parquet(folder/'training_rows.parquet'));continue
        loader.retain(d['secids']);assets=[loader.asset(s) for s in d['secids']]
        local=[]
        for wing,g in rows.groupby('wing',sort=False):
            cp,bucket=SLOTS[wing];units=[a[0] for a in assets]
            if HURDLES[wing]!=.65:units=[with_training_commission(u,HURDLES[wing]) for u in units]
            daily,ends,vegas,status=pair_labels(*units,(30,cp,int(bucket) if bucket!='atm' else bucket,int(bucket) if bucket!='atm' else bucket),replace(config(),option_commission=HURDLES[wing]))
            indices=dates.get_indexer(pd.DatetimeIndex(g.date));assert (indices>=0).all()
            labels=directional_labels(daily[indices],np.where(g.direction.to_numpy()>0,.1,.9)).astype(np.float32)
            meta=g.copy().reset_index(drop=True);meta['position']=market.calendar.get_indexer(pd.DatetimeIndex(meta.date))
            meta['end']=market.calendar[ends[indices]];meta['label_row']=np.arange(len(meta))
            meta['finite_label']=np.isfinite(labels).all(axis=(1,2))&(status[indices]=='complete')&(vegas[indices,0]>0)&(vegas[indices,-1]>0)
            meta['target']=labels[:,4].sum(axis=1);meta['training_hurdle']=HURDLES[wing]
            meta['optionid_a']=np.nan;meta['optionid_b']=np.nan
            for column,(unit,sel) in zip(['optionid_a','optionid_b'],assets):
                chosen=sel.loc[sel.tenor.eq(30)&sel.cp_flag.eq(cp)&sel.bucket.eq(bucket)].set_index('date').optionid
                meta[column]=chosen.reindex(pd.DatetimeIndex(meta.date)).to_numpy()
            np.save(folder/f'{wing}_labels.npy',labels);local.append(meta)
            del daily,labels,units
        complete=pd.concat(local,ignore_index=True);complete.to_parquet(folder/'training_rows.parquet',index=False);pieces.append(complete)
        write_json(folder/'prepared.json',dict(pair=d['pair'],rows=len(complete),finite_rows=int(complete.finite_label.sum())))
        progress('prepare_labels',pair=n,roster=len(defs));print(f'Joint labels {n}/{len(defs)} {d["label"]}: {len(complete)} proposals',flush=True)
        del assets,local,complete;gc.collect()
    loader.cache.clear();meta=pd.concat(pieces,ignore_index=True)
    # Entry availability/future label resolution never remove evaluation proposals.
    meta.to_parquet(OUT/'training_rows.parquet',index=False)
    write_json(REPORT/'label_audit.json',dict(rows=len(meta),finite_rows=int(meta.finite_label.sum()),unresolved_or_no_entry_rows=int((~meta.finite_label).sum()),evaluation_proposals_filtered_by_outcome=False))


def compact_market(market):
    folder=OUT/'market/assets';quotes=[];stocks=[];divs=[]
    ids=sorted(pd.read_csv(REPORT/'assets.csv').secid.astype(int))
    for sid in ids:
        stocks.append(pd.read_parquet(folder/f'{sid}_stocks.parquet'));divs.append(pd.read_parquet(folder/f'{sid}_dividends.parquet'))
    pd.concat(stocks,ignore_index=True).to_parquet(OUT/'market/stocks.parquet',index=False)
    pd.concat(divs,ignore_index=True).to_parquet(OUT/'market/dividends.parquet',index=False)
    market.rates.rename('rate').to_frame().to_parquet(OUT/'market/rates.parquet')
    del stocks,divs;gc.collect()
    for year in sorted(set(market.calendar[market.calendar>='2018-01-01'].year)):
        path=OUT/'market'/f'{year}_quotes.parquet'
        if path.exists():continue
        parts=[]
        for sid in ids:
            q=pd.read_parquet(folder/f'{sid}_quotes.parquet',filters=[('date','>=',pd.Timestamp(year,1,1)),('date','<',pd.Timestamp(year+1,1,1))])
            if len(q):parts.append(q)
        q=pd.concat(parts,ignore_index=True)
        assert not q.duplicated(['date','optionid']).any()
        q.to_parquet(path,index=False);print(f'Joint compact market {year}: {len(q):,} quote rows',flush=True)
        del parts,q;gc.collect()


class SharedMarket:
    def __init__(self):
        s=pd.read_parquet(OUT/'market/stocks.parquet')
        self.spots=s.pivot(index='date',columns='secid',values='close')
        self.adjustments=s.pivot(index='date',columns='secid',values='cfadj')
        self.rates=pd.read_parquet(OUT/'market/rates.parquet').rate
        self.divs=pd.read_parquet(OUT/'market/dividends.parquet')
        self.cache=OrderedDict()
    def stock(self,date,sid):return self.spots.at[date,sid],self.adjustments.at[date,sid]
    def rate(self,date):return float(self.rates.loc[date])
    def quote(self,date,oid):
        year=date.year
        if year not in self.cache:
            q=pd.read_parquet(OUT/'market'/f'{year}_quotes.parquet')
            self.cache[year]={(r['date'],int(r['optionid'])):MarketRow(r) for r in q.to_dict('records')}
            while len(self.cache)>2:self.cache.popitem(last=False)
        self.cache.move_to_end(year)
        return self.cache[year].get((date,int(oid)))
    def dividend_pv(self,date,sid,expiry,rate):
        d=self.divs.loc[self.divs.secid.eq(sid)&self.divs.declare_date.le(date)&self.divs.ex_date.gt(date)&self.divs.ex_date.le(expiry)]
        return float((d.amount*np.exp(-rate*(d.ex_date-date).dt.days/365)).sum())
