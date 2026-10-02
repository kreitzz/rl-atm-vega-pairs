"""Same-type independently selected strikes at 90D only; serial checkpoint worker."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import lru_cache
import gc
import json
import mmap
from pathlib import Path
import resource
import tempfile
import time
import numpy as np
import pandas as pd
from scripts.curve90.config import DATA,OUTPUT,REPORT,COMBOS,ACTIONS,LADDER,RUNOFF,config
from scripts.curve90.unit_paths import UnitPaths,pair_labels
from scripts.curve90.labels import LabelStore
from scripts.curve90.policy import ATM,Replay,fit,predict,gate,memory_gates
from scripts.run_tenor_comparison import OUTPUT as BASE, LABELBASE, definitions, NEW_FEATURES, EXTRA, FEATURES
from scripts.run_aligned_contract_learning import mark_book,screen_dates,asset_paths
from scripts.run_broad_contract_replay import Market,ROOT
from scripts.select_atm_contracts import load_calendar
from scripts.option_flex.contract_engine import UnresolvedPosition
from scripts.run_selected_pair_backtest import audit,scores
from scripts.run_contract_walkforward import digest

VARIANTS=['atm_control','curve']
SHORTLIST=8
_market=None

def discard_label_pages(labels):
    """Drop clean file-backed pages from worker RSS; keep every label on disk."""
    labels.flush()
    mapping=getattr(labels,'_mmap',None)
    if mapping is not None and hasattr(mapping,'madvise') and hasattr(mmap,'MADV_DONTNEED'):
        mapping.madvise(mmap.MADV_DONTNEED)

class CurveMarket(Market):
    def __init__(self,calendar,calls_only=False):
        super().__init__(calendar);self.calls_only=calls_only

    @lru_cache(maxsize=2)
    def curve_asset(self,sid):
        def read(kind):
            paths=[DATA/kind/cp/str(y)/f'{sid}.parquet' for cp in (['C'] if self.calls_only else ['C','P']) for y in sorted(set(self.calendar.year))]
            pieces=[pd.read_parquet(p) for p in paths if p.exists()]
            return pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame()
        q,selection=read('history'),read('selection')
        if q.empty:raise ValueError(f'No curve history for {sid}')
        assert not q.duplicated(['date','optionid']).any()
        stock=self.stocks[self.stocks.secid.eq(sid)].set_index('date').reindex(self.calendar)
        stock['secid']=sid;stock['rate']=self.rates;stock.index.name='date'
        return q,stock.reset_index(),selection

    def curve_pair(self,secids):
        parts=[self.curve_asset(sid) for sid in secids]
        return (*[pd.concat([p[k] for p in parts],ignore_index=True) for k in range(3)],self.divs[self.divs.secid.isin(secids)])

def create_labels(q,s,selection,divs,cal,panel,cfg,path,combos=None):
    secids=list(s.secid.unique());assert len(secids)==2
    units=[UnitPaths(q[q.secid.eq(sid)],s[s.secid.eq(sid)],selection[selection.secid.eq(sid)],cal,panel.index,
                     divs[divs.secid.eq(sid)],cfg) for sid in secids]
    n=len(panel);width=cfg.max_dte+2
    labels=LabelStore(path,shape=(len(COMBOS),n,8,width))
    ends=np.zeros((len(COMBOS),n),int);gates=np.zeros((len(COMBOS),n,2),bool)
    available=np.zeros((len(COMBOS),n),bool);status_rows=[]
    for combo in (range(len(COMBOS)) if combos is None else combos):
        description=COMBOS[combo]
        daily,e,vegas,status=pair_labels(*units,description,cfg)
        labels[combo]=daily;ends[combo]=e
        rewards=np.divide(daily[:,[0,-1]].sum(axis=2)*cfg.entry_vega_per_leg,vegas[:,[0,-1]],
                          out=np.full((n,2),np.nan),where=vegas[:,[0,-1]]>0)
        gates[combo]=memory_gates(panel.index,cal[e],rewards,vegas[:,[0,-1]]>0)
        tenor,cp,a,b=description
        # Signal-close selection only. Never gate deployment on future execution/label status.
        available[combo]=(units[0].ids(tenor,cp,a)>=0)&(units[1].ids(tenor,cp,b)>=0)
        status_rows.append(dict(combo=combo,tenor=tenor,cp=cp,bucket_a=a,bucket_b=b,
            available=int(available[combo].sum()),**{str(k):int(v) for k,v in pd.Series(status).value_counts().items()}))
        if combo%12==11:discard_label_pages(labels)
        if combo%36==35:print(f'  labels {combo+1}/{len(COMBOS)}',flush=True)
    discard_label_pages(labels);del units;gc.collect()
    return labels,ends,gates,available,pd.DataFrame(status_rows)

def annual_shortlist(year,panel,labels,ends,gates,cal):
    """Rank every combination using only five years before validation starts.

    Fixed full-size directional shadow daily Sharpe; at least 21 executed
    memory-permitted cohorts. Preserve exact ATM30 calls regardless of rank.
    """
    cutoff=pd.Timestamp(year-1,1,1);start=cutoff-pd.DateOffset(years=5)
    positions=cal.get_indexer(panel.index);last=cal.searchsorted(cutoff)-1;first=cal.searchsorted(start)
    records=[];eligible=panel.eligible.to_numpy(bool)
    for combo in range(len(COMBOS)):
        daily=np.asarray(labels[combo][:,[0,-1]],dtype=float)
        use=(panel.index>=start)&(cal[ends[combo]]<cutoff)&np.isfinite(daily).all(axis=(1,2))&eligible
        result=[];counts=[]
        for direction in range(2):
            rows=np.flatnonzero(use&gates[combo,:,direction]&np.any(daily[:,direction]!=0,axis=1));counts.append(len(rows))
            book=np.zeros(max(1,last-first+1));days=positions[rows,None]+1+np.arange(daily.shape[2])[None,:]-first
            good=(days>=0)&(days<len(book));np.add.at(book,days[good],daily[rows,direction,:][good])
            sd=book.std(ddof=1)
            result.append(float(np.sqrt(252)*book.mean()/sd) if len(rows)>=21 and sd>1e-12 else None)
        vals=[x for x in result if x is not None]
        records.append(dict(year=year,combo=combo,score=max(vals) if vals else None,
                            short_score=result[0],long_score=result[1],short_count=counts[0],long_count=counts[1],
                            ranking_start=str(start.date()),ranking_cutoff=str(cutoff.date())))
        discard_label_pages(labels)
    ranked=sorted([r for r in records if r['score'] is not None],key=lambda r:(-r['score'],r['combo']))
    chosen=[ATM]+[r['combo'] for r in ranked if r['combo']!=ATM][:SHORTLIST-1]
    for row in records:row['selected']=row['combo'] in chosen
    return chosen,records

def load_subset(labels,ends,combos):
    n=labels.shape[1];w=labels.shape[-1];daily=np.zeros((n,1+8*len(combos),w),dtype=np.float32)
    codes=[0]
    for j,c in enumerate(combos):daily[:,1+j*8:1+(j+1)*8]=labels[c];codes.extend(range(1+c*8,1+(c+1)*8))
    discard_label_pages(labels)
    return daily,ends[combos].max(axis=0),codes

def train(d,panel,labels,ends,gates,available,cal,replay,rates,folder,features,variant):
    start=pd.Timestamp('2018-01-01');stop=cal[-1]-pd.Timedelta(days=RUNOFF)
    wanted=(panel.index>=start)&(panel.index<=stop)&screen_dates(d,panel.index)
    actions=pd.Series(0,index=panel.index[wanted],dtype=int,name='code');logs=[];ranking=[];year=None
    positions=cal.get_indexer(panel.index);cfg=replay.cfg
    for month in pd.period_range(start,stop,freq='M'):
        cutoff=month.start_time;test=(actions.index>=cutoff)&(actions.index<(month+1).start_time)
        if not test.any():continue
        if year!=month.year:
            year=month.year
            combos,rank=([ATM],[]) if variant=='atm_control' else annual_shortlist(year,panel,labels,ends,gates,cal)
            ranking.extend(rank);daily,finish,codes=load_subset(labels,ends,combos)
            finite=np.isfinite(daily).all(axis=(1,2));completion=cal[finish]
            pd.DataFrame(ranking).to_csv(folder/'shortlists.csv',index=False)
        validation_start=cutoff-pd.DateOffset(years=1)
        use=(panel.index>=validation_start-pd.DateOffset(years=5))&(completion<validation_start)&finite
        val_cal=cal[(cal>=validation_start)&(cal<cutoff)]
        if use.sum()<504 or len(val_cal)<200:
            logs.append(dict(fit=str(cutoff.date()),status='insufficient_training',training_rows=int(use.sum())));continue
        val=(panel.index>=val_cal[0])&(panel.index<=val_cal[-1]-pd.Timedelta(days=RUNOFF))
        has_opportunity=any((panel.eligible.to_numpy()&available[c]&gates[c].any(axis=1)&val).any() for c in combos)
        info=dict(fit=str(cutoff.date()),training_rows=int(use.sum()),combos=json.dumps(combos),
                  last_training_outcome=str(completion[use].max().date()),validation_start=str(val_cal[0].date()),
                  validation_end=str(val_cal[-1].date()))
        if not has_opportunity:
            logs.append(dict(info,status='flat_no_validation_opportunities'));continue
        train_labels=daily[use].copy();train_labels[~panel.loc[use,'eligible'].to_numpy(bool)]=0
        model,details=fit(panel.loc[use,features],train_labels,positions[use],finish[use],codes)
        proposal=gate(predict(model,panel.loc[val,features]),panel.index[val],panel,gates,available)
        error=None;score=None;permitted=False
        try:
            nav,t,_=replay.run(proposal,val_cal);mark_book(nav,rates,cfg.capital);audit(nav,t,replay.s,cfg,d['secids'])
            score=scores(nav,cfg.capital)['annualized_excess_sharpe'];permitted=score is not None and score>0
        except (UnresolvedPosition,ValueError) as exc:error=str(exc)
        if permitted:
            recent=(panel.index>=cutoff-pd.DateOffset(years=5))&(completion<cutoff)&finite
            if recent.sum()<504:permitted=False;error='insufficient recent training'
            else:
                own=daily[recent].copy();own[~panel.loc[recent,'eligible'].to_numpy(bool)]=0
                model,refit=fit(panel.loc[recent,features],own,positions[recent],finish[recent],codes)
                details.update(recent_training_rows=int(recent.sum()),last_recent_outcome=str(completion[recent].max().date()),
                               recent_training_shadow_daily_sharpe=refit['training_shadow_daily_sharpe'])
                dates=actions.index[test]
                actions.loc[dates]=gate(predict(model,panel.loc[dates,features]),dates,panel,gates,available)
        logs.append(dict(info,status='enabled' if permitted else 'flat_validation_gate',validation_sharpe=score,validation_error=error,**details))
        pd.DataFrame(logs).to_csv(folder/'fits.csv',index=False);actions.to_csv(folder/'actions.partial.csv')
        if month.month in [1,7]:print(f'  {d["label"]} {variant} {month}: {logs[-1]["status"]}',flush=True)
    return actions,pd.DataFrame(logs)

def worker(d,variants=None):
    global _market
    prior=json.loads((BASE/d['pair']/'30d/status.json').read_text());assert prior['status']=='complete'
    variants=VARIANTS if variants is None else variants
    pending=[v for v in variants if not (OUTPUT/d['pair']/v/'status.json').exists()]
    if not pending:return
    for variant in pending:(OUTPUT/d['pair']/variant).mkdir(parents=True,exist_ok=True)
    if prior.get('structural_cash'):
        for variant in pending:
            row=dict(pair=d['pair'],label=d['label'],variant=variant,status='complete',entries=0,structural_cash=True,nav_source=prior['nav_source'])
            (OUTPUT/d['pair']/variant/'status.json').write_text(json.dumps(row,indent=2)+'\n')
        return
    started=time.monotonic();cal=load_calendar(ROOT);cal=cal[(cal>='2010-01-01')&(cal<='2025-08-29')];cfg=config()
    calls_only=variants==['atm_control']
    if _market is None:_market=CurveMarket(cal,calls_only=calls_only)
    q,s,selection,divs=_market.curve_pair(d['secids'])
    # Every pair runs in its own process; the source-asset cache is never reused.
    _market.curve_asset.cache_clear();gc.collect()
    if calls_only:
        selection=selection[selection.tenor.eq(90)&selection.bucket.eq('atm')&selection.cp_flag.eq('C')]
        q=q[q.optionid.isin(selection.optionid)].copy()
    panel=pd.read_parquet(BASE/d['pair']/'30d/states.parquet')
    panel=panel.loc[panel.index<=cal[-1]-pd.Timedelta(days=RUNOFF)].copy()
    features=NEW_FEATURES+[c if c!='commission_vol_roundtrip' else 'commission_vol_entry' for c in EXTRA]
    features += [c for c in panel if c.startswith(('profit_','count_'))]+FEATURES+['vix_previous_close']
    folder=OUTPUT/d['pair'];panel.to_parquet(folder/'states.parquet')
    print(f'{d["label"]}: {len(panel)} states, {len(q):,} held quotes; building labels',flush=True)
    with tempfile.TemporaryDirectory(prefix='curve_labels_',dir=OUTPUT) as temp:
        labels,ends,gates,available,status=create_labels(q,s,selection,divs,cal,panel,cfg,Path(temp)/'labels.dat',combos=[ATM] if calls_only else None)
        suffix='_control' if calls_only else ''
        status.to_csv(folder/f'label_status{suffix}.csv',index=False)
        np.savez_compressed(folder/f'gates{suffix}.npz',gates=gates,available=available,ends=ends,dates=panel.index.values)
        replay=Replay(q,s,selection,d['secids'],divs,cfg)
        for variant in pending:
            dest=folder/variant
            actions,fits=train(d,panel,labels,ends,gates,available,cal,replay,_market.rates,dest,features,variant)
            actions.to_csv(dest/'actions.csv');fits.to_csv(dest/'fits.csv',index=False)
            row=dict(pair=d['pair'],label=d['label'],variant=variant,state_rows=len(panel),nonzero_signals=int(actions.ne(0).sum()))
            try:
                nav,t,dec=replay.run(actions,cal[cal>='2018-01-01']);mark_book(nav,_market.rates,cfg.capital)
                checks=audit(nav,t,s,cfg,d['secids']);assert t.empty or set(t.reason).issubset({'entry','split_adjustment','expiry_settlement'})
                nav.to_parquet(dest/'daily_nav.parquet');t.to_csv(dest/'trades.csv',index=False);dec.to_csv(dest/'decisions.csv',index=False)
                row.update(status='complete',entries=int(dec.status.eq('entered').sum()) if len(dec) else 0,**scores(nav,cfg.capital),**checks)
            except (UnresolvedPosition,ValueError) as exc:row.update(status='incomplete',error=str(exc))
            row.update(worker_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,elapsed_seconds=time.monotonic()-started)
            (dest/'status.json').write_text(json.dumps(row,indent=2)+'\n')
            print(f'{d["label"]} {variant}: {row["status"]}, {row.get("entries",0)} entries, {row["elapsed_seconds"]:.1f}s',flush=True)
        del labels,replay
    gc.collect()

def control_worker(d):return worker(d,['atm_control'])

def verify_existing(d):
    """Retrain the original pilot and require exact output equivalence."""
    global OUTPUT,_market
    baseline=OUTPUT
    try:
        with tempfile.TemporaryDirectory(prefix='verify_memory_',dir=baseline) as temp:
            OUTPUT=Path(temp);_market=None;worker(d)
            for variant in VARIANTS:
                old=baseline/d['pair']/variant;new=OUTPUT/d['pair']/variant
                pd.testing.assert_frame_equal(pd.read_parquet(old/'daily_nav.parquet'),pd.read_parquet(new/'daily_nav.parquet'),check_exact=True)
                for name in ['actions.csv','trades.csv','decisions.csv','fits.csv']:
                    assert (old/name).read_bytes()==(new/name).read_bytes(),(variant,name)
                left=json.loads((old/'status.json').read_text());right=json.loads((new/'status.json').read_text())
                for key in ['worker_peak_rss_bytes','elapsed_seconds']:
                    left.pop(key);right.pop(key)
                assert left==right,(variant,'status')
            peaks={v:json.loads((OUTPUT/d['pair']/v/'status.json').read_text())['worker_peak_rss_bytes'] for v in VARIANTS}
    finally:OUTPUT=baseline;_market=None
    (OUTPUT/'memory_equivalence.json').write_text(json.dumps(dict(pair=d['pair'],status='exact_match',peak_rss_bytes=peaks),indent=2)+'\n')
    print('Memory optimization: exact pilot output equivalence passed',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--pair',required=True);p.add_argument('--verify-existing',action='store_true');args=p.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    d=next(d for d in definitions() if d['pair']==args.pair)
    if args.verify_existing:verify_existing(d)
    else:worker(d)

if __name__=='__main__':main()
