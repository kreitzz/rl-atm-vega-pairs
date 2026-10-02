"""Bounded selection/history preparation, safe to resume as put partitions arrive."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
from pathlib import Path
import resource
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scripts.curve_trading.config import DATA,TENORS,BUCKETS,DELTA_TOLERANCE,slot_name
from scripts.run_contract_pilot import normalize_quotes
from scripts.option_flex.black_scholes import fit_quotes
from scripts.run_contract_walkforward import digest
from scripts.run_broad_contract_replay import ROOT

KEYS=['date','secid','tenor','bucket']
HISTORY_COLUMNS=['date','secid','optionid','expiry','cp_flag','strike','bid','ask','delta','vega','multiplier','cfadj','standard',
    'open_interest','bs_iv','bs_rate','known_dividend_pv','bs_status']

@lru_cache(maxsize=1)
def common():return pd.read_parquet(ROOT/'zero_curve.parquet'),pd.read_parquet(ROOT/'distributions.parquet')
@lru_cache(maxsize=1)
def stocks(year):
    s=pd.read_parquet(ROOT/'underlying'/f'year={year}.parquet',columns=['date','secid','close']);s['date']=pd.to_datetime(s.date);return s

def anchors(job):
    pieces=[]
    for tenor in TENORS:
        root=Path('data/contract_history_shared') if tenor==30 else Path('data/tenor_comparison')/f'{tenor}d'
        for sid in job['secids']:
            p=root/str(job['year'])/f'{sid}_selection.parquet'
            if p.exists():pieces.append(pd.read_parquet(p,columns=['date','secid','expiry','strike','optionid']).assign(tenor=tenor))
    if not pieces:return pd.DataFrame(columns=['date','secid','expiry','strike','optionid','tenor'])
    return pd.concat(pieces,ignore_index=True).rename(columns={'strike':'anchor_strike','optionid':'anchor_id'})

def reduce_choices(p):
    return p.sort_values(KEYS+['distance','relative_width','open_interest','optionid'],ascending=[True]*6+[False,True]).drop_duplicates(KEYS).reset_index(drop=True)

def choose(q,a,cp):
    q=q[q.bs_status.eq('ok')&q.bs_vega.gt(0)&q.bs_delta.abs().between(0,1)].merge(a,on=['date','secid','expiry'],how='inner',validate='many_to_many')
    pieces=[]
    for bucket in BUCKETS:
        if bucket=='atm':
            use=q.optionid.eq(q.anchor_id) if cp=='C' else q.strike.eq(q.anchor_strike)
            p=q[use].copy();p['distance']=0.
        else:
            distance=(q.bs_delta.abs()-bucket/100).abs();p=q[distance.le(DELTA_TOLERANCE)].copy();p['distance']=distance.loc[p.index]
        p['bucket']=str(bucket);pieces.append(p)
    return reduce_choices(pd.concat(pieces,ignore_index=True))

def worker(task):
    cp,stage,job=task;year=job['year'];root=ROOT if cp=='C' else DATA/'puts_raw';source=root/job['path']
    meta=DATA/'metadata'/stage/cp/str(year)/(source.stem+'.json');meta.parent.mkdir(parents=True,exist_ok=True)
    if meta.exists():
        r=json.loads(meta.read_text())
        for p,h in r['files'].items():assert digest(p)==h,p
        return r
    start=time.monotonic();zero,divs=common();s=stocks(year);files={};count=0
    if stage=='select':
        a=anchors(job);best=pd.DataFrame()
        if len(a):
            expiries=a[['date','secid','expiry']].drop_duplicates()
            for batch in pq.ParquetFile(source).iter_batches(batch_size=32768,use_threads=False):
                q=normalize_quotes(batch.to_pandas());assert q.cp_flag.eq(cp).all()
                mid=(q.bid+q.ask)/2;q['relative_width']=(q.ask-q.bid)/mid
                q=q[q.standard&q.multiplier.eq(100)&q.bid.gt(0)&q.ask.ge(q.bid)&q.open_interest.ge(100)&q.relative_width.le(.25)]
                q=q.merge(expiries,on=['date','secid','expiry'],how='inner',validate='many_to_one')
                if q.empty:continue
                q=fit_quotes(q,s,zero,divs);p=choose(q,a,cp)
                if len(p):best=reduce_choices(pd.concat([best,p],ignore_index=True))
                if len(best)>150000:raise MemoryError('Curve choice partition bound exceeded')
            if len(best):
                for sid,g in best.groupby('secid'):
                    p=DATA/'selection'/cp/str(year)/f'{int(sid)}.parquet';p.parent.mkdir(parents=True,exist_ok=True)
                    g.to_parquet(p,index=False);files[str(p)]=digest(p);count+=len(g)
    else:
        wanted=set()
        for yr in [year-1,year]:
            for sid in job['secids']:
                p=DATA/'selection'/cp/str(yr)/f'{sid}.parquet'
                if p.exists():wanted.update(pd.read_parquet(p,columns=['optionid']).optionid)
        writers={};paths={}
        try:
            if wanted:
                for batch in pq.ParquetFile(source).iter_batches(batch_size=16384,use_threads=False):
                    raw=batch.to_pandas();raw=raw[raw.optionid.isin(wanted)]
                    if raw.empty:continue
                    q=fit_quotes(normalize_quotes(raw),s,zero,divs)
                    q['delta']=q.bs_delta;q['vega']=q.bs_vega;q=q[HISTORY_COLUMNS]
                    for sid,g in q.groupby('secid'):
                        sid=int(sid);p=DATA/'history'/cp/str(year)/f'{sid}.parquet';p.parent.mkdir(parents=True,exist_ok=True)
                        table=pa.Table.from_pandas(g,preserve_index=False)
                        if sid not in writers:
                            paths[sid]=p;writers[sid]=pq.ParquetWriter(p.with_suffix('.partial'),table.schema,compression='zstd')
                        writers[sid].write_table(table);count+=len(g)
        finally:
            for w in writers.values():w.close()
        for p in paths.values():p.with_suffix('.partial').replace(p);files[str(p)]=digest(p)
    result=dict(cp=cp,stage=stage,source=str(source),source_hash=digest(source),files=files,rows=count,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    meta.write_text(json.dumps(result,indent=2)+'\n');print(f'{stage} {cp} {year} {source.stem}: {count:,}, {time.monotonic()-start:.1f}s',flush=True)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--cp',choices=['C','P'],required=True);p.add_argument('--stage',choices=['select','history'],required=True);args=p.parse_args()
    jobs=json.loads((DATA/'puts_raw/requests.json').read_text());root=ROOT if args.cp=='C' else DATA/'puts_raw'
    jobs=sorted(jobs,key=lambda j:(j['year'],j['path']))
    ready=[j for j in jobs if (root/j['path']).exists() and (root/j['path']).with_suffix('.json').exists()]
    if args.stage=='history':
        for j in jobs:
            p=DATA/'metadata/select'/args.cp/str(j['year'])/(Path(j['path']).stem+'.json')
            if not p.exists():raise ValueError('Complete all selection partitions before history extraction')
    signature=dict(code={p:digest(p) for p in [__file__,'scripts/curve_trading/config.py','scripts/option_flex/black_scholes.py','scripts/run_contract_pilot.py']},
        jobs=jobs,call_manifest=digest('data/contract_history_shared/manifest.json'),tenor_manifest=digest('data/tenor_comparison/manifest.json'))
    fingerprint=DATA/'preparation_inputs.json'
    if fingerprint.exists():assert json.loads(fingerprint.read_text())==signature,'Preparation inputs changed'
    else:fingerprint.write_text(json.dumps(signature,indent=2)+'\n')
    records=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for i,row in enumerate(pool.map(worker,[(args.cp,args.stage,j) for j in ready]),1):
            records.append(row)
            (DATA/f'{args.cp}_{args.stage}_progress.json').write_text(json.dumps(dict(completed=i,ready=len(ready),total=len(jobs)),indent=2)+'\n')
    if len(ready)==len(jobs):
        (DATA/f'{args.cp}_{args.stage}_manifest.json').write_text(json.dumps(dict(inputs=digest(fingerprint),partitions=records),indent=2)+'\n')
    print(f'{args.cp} {args.stage}: {len(ready)}/{len(jobs)} ready partitions processed',flush=True)

if __name__=='__main__':main()
