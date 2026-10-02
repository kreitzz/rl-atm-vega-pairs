"""Stream cached chains into 14D/45D selections and their complete held-ID histories."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from functools import lru_cache
import json
from pathlib import Path
import resource
import time
import pandas as pd
import pyarrow.parquet as pq
from scripts.select_atm_contracts import select_candidates, load_calendar, SORT, ASC
from scripts.run_contract_pilot import normalize_quotes
from scripts.black_scholes import fit_quotes
from scripts.train_expiry_contracts import settings
from scripts.run_broad_contract_replay import ROOT
from scripts.run_contract_walkforward import digest
from scripts.download_matched_puts import definitions

OUTPUT=Path('data/tenor_comparison')
TENORS={14:(7,21),30:(21,45),45:(35,55)}

def config(tenor):
    lo,hi=TENORS[tenor]
    return replace(settings(),target_dte=tenor,min_dte=lo,max_dte=hi)

@lru_cache(maxsize=1)
def common():
    cal=load_calendar(ROOT)
    return pd.read_parquet(ROOT/'zero_curve.parquet'),pd.read_parquet(ROOT/'distributions.parquet'),pd.Series(cal[1:].to_numpy(),index=cal[:-1])

@lru_cache(maxsize=1)
def stocks(year):
    s=pd.read_parquet(ROOT/'underlying'/f'year={year}.parquet')
    s['date']=pd.to_datetime(s.date);s['entry_date']=s.date.map(common()[2]);return s

def worker(task):
    stage,job=task;year=job['year'];source=ROOT/job['path']
    meta=OUTPUT/'metadata'/stage/str(year)/(source.stem+'.json');meta.parent.mkdir(parents=True,exist_ok=True)
    if meta.exists():
        old=json.loads(meta.read_text())
        for p,h in old['outputs'].items(): assert digest(p)==h,p
        return old
    started=time.monotonic();zero,divs,_=common();s=stocks(year);outputs={}
    if stage=='select':
        best={t:[] for t in [14,45]}
        for batch in pq.ParquetFile(source).iter_batches(batch_size=10000,use_threads=False):
            q=normalize_quotes(batch.to_pandas())
            for t in best:
                c=select_candidates(q,s,config(t))
                if len(c):best[t].append(c)
                if len(best[t])>=20:
                    best[t]=[pd.concat(best[t],ignore_index=True).sort_values(SORT,ascending=ASC).drop_duplicates(['secid','date'])]
        count=0
        for t,parts in best.items():
            if not parts:continue
            selected=pd.concat(parts,ignore_index=True).sort_values(SORT,ascending=ASC).drop_duplicates(['secid','date'])
            selected=fit_quotes(selected,s,zero,divs);count+=len(selected)
            for sid,g in selected.groupby('secid'):
                p=OUTPUT/f'{t}d'/str(year)/f'{int(sid)}_selection.parquet';p.parent.mkdir(parents=True,exist_ok=True)
                g.to_parquet(p,index=False);outputs[str(p)]=digest(p)
    else:
        wanted=set()
        for t in [14,45]:
            for yr in [year-1,year]:
                for sid in job['secids']:
                    p=OUTPUT/f'{t}d'/str(yr)/f'{sid}_selection.parquet'
                    if p.exists():wanted.update(pd.read_parquet(p,columns=['optionid']).optionid)
        chunks=[];count=0
        if wanted:
            for batch in pq.ParquetFile(source).iter_batches(batch_size=10000,use_threads=False):
                frame=batch.to_pandas();frame=frame[frame.optionid.isin(wanted)]
                if len(frame):chunks.append(frame);count+=len(frame)
                if count>750000:raise MemoryError(f'Partition exceeded 750000 held quotes: {source}')
        if chunks:
            q=normalize_quotes(pd.concat(chunks,ignore_index=True));del chunks
            pieces=[fit_quotes(q.iloc[i:i+25000],s,zero,divs) for i in range(0,len(q),25000)]
            q=pd.concat(pieces,ignore_index=True);del pieces
            for field in ['delta','vega','impl_volatility']:q['vendor_'+field]=q[field]
            q['delta']=q.bs_delta;q['vega']=q.bs_vega;q['impl_volatility']=q.bs_iv
            for sid,g in q.groupby('secid'):
                p=OUTPUT/'history'/str(year)/f'{int(sid)}.parquet';p.parent.mkdir(parents=True,exist_ok=True)
                assert not g.duplicated(['date','optionid']).any()
                g.to_parquet(p,index=False);outputs[str(p)]=digest(p)
    row=dict(stage=stage,source=str(source),rows=count,outputs=outputs,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    meta.write_text(json.dumps(row,indent=2)+'\n')
    print(f'{stage} {year} {source.stem}: {count:,} rows, {time.monotonic()-started:.1f}s',flush=True)
    return row

def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    ids={sid for d in definitions() for sid in d['secids']}
    jobs=sorted([r for r in json.loads((ROOT/'requests.json').read_text()) if r['kind']=='quotes' and ids.intersection(r['secids'])],key=lambda j:(j['year'],j['path']))
    ownership=[(j['year'],sid) for j in jobs for sid in j['secids']]
    assert len(ownership)==len(set(ownership)), 'Overlapping source partition ownership'
    inputs=dict(tenors={t:asdict(config(t)) for t in TENORS},jobs=jobs,
        code={p:digest(p) for p in [__file__,'scripts/select_atm_contracts.py','scripts/black_scholes.py','scripts/run_contract_pilot.py','scripts/contract_engine.py','scripts/train_expiry_contracts.py']},
        sources={str(p):digest(p) for p in [ROOT/'requests.json',ROOT/'zero_curve.parquet',ROOT/'distributions.parquet',*[ROOT/'underlying'/f'year={y}.parquet' for y in sorted({j['year'] for j in jobs})]]},
        raw_quotes={str(ROOT/j['path']):digest(ROOT/j['path']) for j in jobs})
    p=OUTPUT/'inputs.json'
    if p.exists():assert json.loads(p.read_text())==json.loads(json.dumps(inputs)),'Preparation inputs changed'
    else:p.write_text(json.dumps(inputs,indent=2)+'\n')
    records=[];started=time.monotonic()
    for stage in ['select','history']:
        with ProcessPoolExecutor(max_workers=2) as pool:
            for i,row in enumerate(pool.map(worker,[(stage,j) for j in jobs]),1):
                records.append(row)
                (OUTPUT/'progress.json').write_text(json.dumps(dict(stage=stage,completed=i,total=len(jobs),elapsed_seconds=time.monotonic()-started),indent=2)+'\n')
    (OUTPUT/'manifest.json').write_text(json.dumps(dict(inputs=digest(p),partitions=records,elapsed_seconds=time.monotonic()-started),indent=2)+'\n')

if __name__=='__main__':main()
