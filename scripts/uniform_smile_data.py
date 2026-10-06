"""Causal, streaming preparation of common ATM/P25/P75 research inputs."""
from collections import OrderedDict
from dataclasses import replace
import gc
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd

from scripts.curve_trading.config import DATA,config,RUNOFF
from scripts.curve_trading.unit_paths import UnitPaths,pair_labels
from scripts.run_broad_contract_replay import ROOT
from scripts.run_smile_sizing import scoped_inputs
from scripts.retrain_atm_commission import fast_memory
from scripts.run_tail_put25_pooled import prior_vix
from scripts.run_aligned_contract_learning import asset_paths
from scripts.contract_learning_labels import beta_screen
from scripts.run_atm65_ablation import read_actions,write_json
from scripts.option_flex.contract_engine import MarketRow
from scripts import uniform_smile_policy as policy

OUT=Path('artifacts/contracts/uniform_smile');REPORT=Path('reports/uniform_smile')
WINGS={'ATM':('C','atm'),'P25':('P','25'),'P75':('P','75')}
FEE=.65


def progress(stage,**details):
    record=dict(stage=stage,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    record.update(details);write_json(OUT/'progress.json',record)


def assets(defs,market,sources):
    folder=OUT/'market/assets';folder.mkdir(parents=True,exist_ok=True)
    ids=sorted({s for d in defs for s in d['secids']});records=[]
    for n,sid in enumerate(ids,1):
        stamp=folder/f'{sid}.json'
        if stamp.exists():records.append(json.loads(stamp.read_text()));continue
        selections=[];quotes=[]
        for cp,buckets in [('C',['atm']),('P',['25','75'])]:
            pieces=[]
            for year in market.calendar.year.unique():
                path=DATA/'selection'/cp/str(year)/f'{sid}.parquet'
                if path.exists():
                    s=pd.read_parquet(sources.track(path));pieces.append(s.loc[s.tenor.eq(30)&s.bucket.isin(buckets)])
            if not pieces:continue
            sel=pd.concat(pieces,ignore_index=True);selections.append(sel);oids=sel.optionid.unique().tolist()
            for year in market.calendar.year.unique():
                path=DATA/'history'/cp/str(year)/f'{sid}.parquet'
                if oids and path.exists():quotes.append(pd.read_parquet(sources.track(path),filters=[('optionid','in',oids)]))
        columns=['date','expiry','secid','tenor','cp_flag','bucket','optionid','bs_iv','bs_vega']
        sel=pd.concat(selections,ignore_index=True) if selections else pd.DataFrame(columns=columns)
        for c in ['date','expiry']:sel[c]=pd.to_datetime(sel[c])
        q=pd.concat(quotes,ignore_index=True) if quotes else pd.DataFrame(columns=['secid','bs_status'])
        stock=market.stocks.loc[market.stocks.secid.eq(sid)].set_index('date').reindex(market.calendar)
        stock['secid']=sid;stock['rate']=market.rates;stock.index.name='date'
        divs=market.divs.loc[market.divs.secid.eq(sid)]
        q,s,sel,divs=scoped_inputs([dict(q=q,s=stock.reset_index(),sel=sel,divs=divs)],[sid])
        # Features and labels use the same scoped/refitted IV and Greeks.
        if len(q) and len(sel):
            lookup=q.set_index(['date','optionid']);keys=pd.MultiIndex.from_frame(sel[['date','optionid']])
            for target,source in [('bs_iv','bs_iv'),('bs_vega','vega'),('bs_delta','delta')]:sel[target]=lookup[source].reindex(keys).to_numpy()
        for kind,frame in [('quotes',q),('stocks',s),('selection',sel),('dividends',divs)]:frame.to_parquet(folder/f'{sid}_{kind}.parquet',index=False)
        record=dict(secid=sid,quotes=len(q),selections=len(sel));records.append(record);write_json(stamp,record)
        write_json(OUT/'sources.json',sources);progress('assets',asset=n,assets=len(ids),secid=sid)
        if n%10==0:print(f'Uniform assets {n}/{len(ids)}',flush=True)
        del q,s,sel,stock,quotes,selections,pieces;gc.collect()
    pd.DataFrame(records).to_csv(REPORT/'assets.csv',index=False)


class Loader:
    def __init__(self,market):self.market=market;self.cache={}
    def retain(self,ids):
        for sid in list(self.cache):
            if sid not in ids:del self.cache[sid]
        gc.collect()
    def asset(self,sid):
        if sid in self.cache:return self.cache[sid]
        folder=OUT/'market/assets';q,s,sel,divs=[pd.read_parquet(folder/f'{sid}_{kind}.parquet') for kind in ['quotes','stocks','selection','dividends']]
        unit=UnitPaths(q,s,sel,self.market.calendar,self.market.calendar,divs,replace(config(),option_commission=FEE)) if len(q) else None
        returns=(s.set_index('date').close*s.set_index('date').cfadj).pct_change(fill_method=None)
        result=dict(unit=unit,sel=sel,returns=returns);self.cache[sid]=result;return result


def prepare_pairs(defs,market,sources):
    cal=market.calendar;loader=Loader(market);vix=prior_vix(cal);sources.track(Path('data/raw/VIX_History.csv'));coverage=[]
    for n,d in enumerate(defs,1):
        folder=OUT/'pairs'/d['pair'];folder.mkdir(parents=True,exist_ok=True)
        stamp=folder/'prepared.json'
        if stamp.exists():coverage.extend(json.loads(stamp.read_text()));continue
        loader.retain(d['secids']);a,b=[loader.asset(sid) for sid in d['secids']]
        returns=[]
        for path,asset in zip(asset_paths(d),[a,b]):
            if path.exists():returns.append(pd.read_parquet(sources.track(path),columns=['ret']).ret.reindex(cal))
            else:returns.append(asset['returns'])
        returns=pd.concat(returns,axis=1);returns.columns=d['secids']
        screen=beta_screen(returns,cal,beta_tolerance=.15)
        native=Path('artifacts/contracts/tail_put25_strategy')/d['pair']/'states.parquet'
        legacy=Path('artifacts/contracts/tail_calendar_ablation')/d['pair']/'legacy_calendar_memory/actions.csv'
        pre=pd.read_parquet(sources.track(native),columns=[]).index if native.exists() else pd.DatetimeIndex([])
        post=read_actions(sources.track(legacy)).index if legacy.exists() else pd.DatetimeIndex([])
        context=np.where(cal<'2018-01-01',cal.isin(pre),cal.isin(post));rows=[]
        for wing,(cp,bucket) in WINGS.items():
            chosen=[x['sel'].loc[x['sel'].cp_flag.eq(cp)&x['sel'].bucket.eq(bucket)] for x in [a,b]]
            memory=pd.DataFrame(0.,index=cal,columns=[f'{prefix}_{side}_{days}d' for days in [91,365] for side in ['short','long'] for prefix in ['profit','count']])
            daily=np.zeros((len(cal),8,config().max_dte+2),np.float32);ends=np.arange(len(cal));vegas=np.zeros((len(cal),8));status=np.full(len(cal),'no_entry',object)
            if a['unit'] is not None and b['unit'] is not None:
                daily,ends,vegas,status=pair_labels(a['unit'],b['unit'],(30,cp,int(bucket) if bucket!='atm' else bucket,int(bucket) if bucket!='atm' else bucket),replace(config(),option_commission=FEE))
                reward=np.divide(daily[:,[0,-1]].sum(axis=2)*450,vegas[:,[0,-1]],out=np.full((len(cal),2),np.nan),where=vegas[:,[0,-1]]>0)
                memory=fast_memory(cal,cal[ends],reward,vegas[:,[0,-1]]>0)
            panel=policy.states(*chosen,returns,screen,cal,memory,vix,FEE);panel['context']=context
            panel['candidate']=panel.observed&panel.eligible&panel.context
            test=panel.observed&(cal>='2018-01-01')&(cal<=cal[-1]-pd.Timedelta(days=RUNOFF))
            panel.loc[test].reset_index().rename(columns={'index':'date'}).to_parquet(folder/f'{wing}_observed.parquet',index=False)
            idx=np.flatnonzero(panel.candidate);meta=panel.iloc[idx].copy();meta['date']=meta.index;meta['position']=idx
            meta['end']=cal[ends[idx]];meta['label_row']=np.arange(len(meta))
            labels=np.zeros((len(idx),9,daily.shape[-1]),np.float32);labels[:,1:]=daily[idx]
            meta['finite_label']=np.isfinite(labels).all(axis=(1,2))&(status[idx]=='complete')&(vegas[idx,0]>0)&(vegas[idx,-1]>0)
            meta['target_long']=labels[:,8].sum(axis=1);meta['target_short']=labels[:,1].sum(axis=1)
            meta['pair']=d['pair'];meta['wing']=wing;meta['secid_a']=d['secids'][0];meta['secid_b']=d['secids'][1]
            for col,sel in zip(['optionid_a','optionid_b'],chosen):meta[col]=sel.set_index('date').optionid.reindex(meta.index).to_numpy()
            meta.reset_index(drop=True).to_parquet(folder/f'{wing}_rows.parquet',index=False);np.save(folder/f'{wing}_labels.npy',labels)
            rows.append(dict(pair=d['pair'],label=d['label'],wing=wing,observed_test_rows=int(test.sum()),
                             candidates=len(meta),finite_training_candidates=int(meta.finite_label.sum()),
                             evaluated_candidates=int(((meta.date>='2018-01-01')&(meta.date<=cal[-1]-pd.Timedelta(days=RUNOFF))).sum())))
            del panel,meta,labels,daily,memory;gc.collect()
        write_json(stamp,rows);coverage.extend(rows);write_json(OUT/'sources.json',sources);progress('prepare_pairs',pair=n,roster=len(defs))
        if n%20==0:print(f'Uniform pair labels {n}/{len(defs)} {d["label"]}',flush=True)
    loader.cache.clear();gc.collect();pd.DataFrame(coverage).to_csv(REPORT/'coverage.csv',index=False)
    for wing in WINGS:
        rows=pd.concat([pd.read_parquet(OUT/'pairs'/d['pair']/f'{wing}_rows.parquet') for d in defs],ignore_index=True)
        rows.to_parquet(OUT/f'{wing}_rows.parquet',index=False)


def compact(defs,market):
    folder=OUT/'market/assets';ids=sorted({s for d in defs for s in d['secids']});needed={s:set() for s in ids}
    for wing in WINGS:
        rows=pd.read_parquet(OUT/f'{wing}_rows.parquet',columns=['date','secid_a','secid_b','optionid_a','optionid_b'])
        rows=rows.loc[(rows.date>='2018-01-01')&(rows.date<=market.calendar[-1]-pd.Timedelta(days=RUNOFF))]
        for suffix in ['a','b']:
            for sid,g in rows.groupby('secid_'+suffix):needed[sid].update(g['optionid_'+suffix].dropna().astype(int))
    for kind in ['stocks','dividends']:
        pd.concat([pd.read_parquet(folder/f'{sid}_{kind}.parquet') for sid in ids],ignore_index=True).to_parquet(OUT/'market'/f'{kind}.parquet',index=False)
    market.rates.rename('rate').to_frame().to_parquet(OUT/'market/rates.parquet')
    for year in sorted(set(market.calendar[market.calendar>='2018-01-01'].year)):
        path=OUT/'market'/f'{year}_quotes.parquet'
        if path.exists():continue
        parts=[]
        for sid in ids:
            if not needed[sid]:continue
            parts.append(pd.read_parquet(folder/f'{sid}_quotes.parquet',filters=[('date','>=',pd.Timestamp(year,1,1)),('date','<',pd.Timestamp(year+1,1,1)),('optionid','in',sorted(needed[sid]))]))
        q=pd.concat(parts,ignore_index=True);assert not q.duplicated(['date','optionid']).any();q.to_parquet(path,index=False)
        print(f'Uniform compact {year}: {len(q):,} quote rows',flush=True);del q,parts;gc.collect()


class ReplayMarket:
    def __init__(self):
        stock=pd.read_parquet(OUT/'market/stocks.parquet');self.spots=stock.pivot(index='date',columns='secid',values='close')
        self.adjustments=stock.pivot(index='date',columns='secid',values='cfadj');self.rates=pd.read_parquet(OUT/'market/rates.parquet').rate
        self.divs=pd.read_parquet(OUT/'market/dividends.parquet');self.years=OrderedDict();self.days=OrderedDict();self.reported_year=None;self.variant='';self.started=time.monotonic()
    def stock(self,date,sid):
        # OCC56907 was public before the July18 open. The study exits all
        # Hess paired inventory that session; it never carries this proxy.
        if sid==101322 and date==pd.Timestamp('2025-07-18'):
            return 1.025*self.spots.at[date,102968],self.adjustments.at[pd.Timestamp('2025-07-17'),sid]
        return self.spots.at[date,sid],self.adjustments.at[date,sid]
    def rate(self,date):
        if self.reported_year!=date.year:
            self.reported_year=date.year
            progress('replay',variant=self.variant,date=str(date.date()),variant_elapsed_seconds=time.monotonic()-self.started)
            print(f'Uniform replay {self.variant}: {date.date()} ({time.monotonic()-self.started:.1f}s)',flush=True)
        return float(self.rates.loc[date])
    def quote(self,date,oid):
        if date not in self.days:
            year=date.year
            if year not in self.years:
                q=pd.read_parquet(OUT/'market'/f'{year}_quotes.parquet').set_index('date',drop=False).sort_index()
                self.years[year]=q
                while len(self.years)>2:self.years.popitem(last=False)
            self.years.move_to_end(year);q=self.years[year]
            daily=q.loc[[date]] if date in q.index else q.iloc[:0]
            self.days[date]={int(r['optionid']):MarketRow(r) for r in daily.to_dict('records')}
            while len(self.days)>2:self.days.popitem(last=False)
        self.days.move_to_end(date);return self.days[date].get(int(oid))
    def dividend_pv(self,date,sid,expiry,rate):
        d=self.divs.loc[self.divs.secid.eq(sid)&self.divs.declare_date.le(date)&self.divs.ex_date.gt(date)&self.divs.ex_date.le(expiry)]
        return float((d.amount*np.exp(-rate*(d.ex_date-date).dt.days/365)).sum())
