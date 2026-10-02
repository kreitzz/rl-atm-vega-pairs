"""Same-expiry BS-delta smile from cached calls; bracketed interpolation only."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import resource
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scripts.black_scholes import fit_quotes
from scripts.run_contract_pilot import normalize_quotes
from scripts.run_contract_walkforward import digest
from scripts.run_broad_contract_replay import ROOT
from scripts.select_atm_contracts import load_calendar

OUTPUT=Path('data/rich_smile')
TARGETS=[.1,.25,.5,.75,.9]
MAX_GAP=.15
VALUES=['down10','down25','up25','up10','rr25','bf25','down_tail','up_tail']
KEYS=['date','secid','target','side']

def reduce_brackets(frame):
    if frame.empty:return frame
    return frame.sort_values(KEYS+['distance','relative_width','optionid']).drop_duplicates(KEYS).reset_index(drop=True)

def bracket_candidates(q):
    parts=[]
    q=q[q.bs_status.eq('ok')&q.bs_delta.between(0,1)&q.bs_iv.gt(0)].copy()
    for target in TARGETS:
        for side in ['lo','hi']:
            use=q.bs_delta.le(target) if side=='lo' else q.bs_delta.ge(target)
            p=q.loc[use,['date','secid','optionid','bs_delta','bs_iv','relative_width']].copy()
            p['target']=target;p['side']=side;p['distance']=(p.bs_delta-target).abs()
            parts.append(p)
    return reduce_brackets(pd.concat(parts,ignore_index=True))

def interpolate(points,selected):
    base=selected.set_index(['date','secid'])[['bs_iv','expiry']].rename(columns={'bs_iv':'atm_iv'})
    for target in TARGETS:
        p=points[points.target.eq(target)]
        lo=p[p.side.eq('lo')].set_index(['date','secid']);hi=p[p.side.eq('hi')].set_index(['date','secid'])
        ld=lo.bs_delta.reindex(base.index);hd=hi.bs_delta.reindex(base.index)
        lv=lo.bs_iv.reindex(base.index);hv=hi.bs_iv.reindex(base.index)
        gap=hd-ld;weight=(target-ld)/gap.replace(0,np.nan)
        iv=(lv+(hv-lv)*weight).where(gap.ne(0),lv)
        valid=ld.le(target)&hd.ge(target)&gap.between(0,MAX_GAP)&iv.gt(0)
        base[f'iv{round(target*100)}']=iv.where(valid)
        base[f'gap{round(target*100)}']=gap.where(valid)
    atm=base.atm_iv.where(base.atm_iv.gt(0))
    for name,col in [('down10','iv90'),('down25','iv75'),('up25','iv25'),('up10','iv10')]:base[name]=base[col]/atm-1
    base['rr25']=(base.iv25-base.iv75)/atm
    base['bf25']=((base.iv25+base.iv75)/2-base.iv50)/atm
    base['down_tail']=(base.iv90-base.iv75)/atm
    base['up_tail']=(base.iv10-base.iv25)/atm
    return base.replace([np.inf,-np.inf],np.nan).reset_index()

def build(job):
    year=job['year'];source=ROOT/job['path'];folder=OUTPUT/str(year);folder.mkdir(exist_ok=True)
    path=folder/source.name;meta=path.with_suffix('.json')
    if meta.exists():
        row=json.loads(meta.read_text());assert digest(path)==row['sha256'];return row
    selected=pd.read_parquet(Path('data/atm_30d')/str(year)/source.name)
    stocks=pd.read_parquet(ROOT/'underlying'/f'year={year}.parquet',columns=['date','secid','close'])
    stocks['date']=pd.to_datetime(stocks.date)
    zero=pd.read_parquet(ROOT/'zero_curve.parquet');divs=pd.read_parquet(ROOT/'distributions.parquet')
    points=pd.DataFrame();fitted=0;start=time.monotonic()
    for batch in pq.ParquetFile(source).iter_batches(batch_size=32768,use_threads=False):
        q=normalize_quotes(batch.to_pandas());mid=(q.bid+q.ask)/2
        q['relative_width']=(q.ask-q.bid)/mid
        good=q.standard&q.multiplier.eq(100)&q.cp_flag.eq('C')&q.bid.gt(0)&q.ask.ge(q.bid)&q.open_interest.ge(100)&q.relative_width.le(.25)
        q=q[good].merge(selected[['date','secid','expiry']],on=['date','secid','expiry'],how='inner',validate='many_to_one')
        if q.empty:continue
        q=fit_quotes(q,stocks,zero,divs);fitted+=len(q)
        new=bracket_candidates(q)
        if len(new):points=reduce_brackets(pd.concat([points,new],ignore_index=True))
        if len(points)>100000:raise MemoryError('Smile bracket bound exceeded')
    if points.empty:points=pd.DataFrame(columns=KEYS+['bs_delta','bs_iv'])
    values=interpolate(points,selected);values.to_parquet(path,index=False)
    row=dict(source=str(source),output=str(path),sha256=digest(path),fitted_quotes=fitted,rows=len(values),
             valid={v:int(values[v].notna().sum()) for v in VALUES},peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    meta.write_text(json.dumps(row,indent=2)+'\n')
    print(f'{year} {source.stem}: {fitted:,} fits, {len(values):,} dates, {time.monotonic()-start:.1f}s',flush=True)
    return row

def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    alljobs=[j for j in json.loads((ROOT/'requests.json').read_text()) if j['kind']=='quotes']
    jobs=sorted([j for j in alljobs if (Path('data/atm_30d')/str(j['year'])/Path(j['path']).name).exists()],key=lambda j:(j['year'],j['path']))
    files=[ROOT/j['path'] for j in jobs]+[Path('data/atm_30d')/str(j['year'])/Path(j['path']).name for j in jobs]
    files += [ROOT/'underlying'/f'year={y}.parquet' for y in sorted({j['year'] for j in jobs})]+[ROOT/'zero_curve.parquet',ROOT/'distributions.parquet']
    inputs=dict(jobs=jobs,targets=TARGETS,max_gap=MAX_GAP,values=VALUES,
                code={p:digest(p) for p in [__file__,'scripts/black_scholes.py','scripts/run_contract_pilot.py']},sources={str(p):digest(p) for p in files})
    path=OUTPUT/'inputs.json'
    if path.exists():assert json.loads(path.read_text())==inputs,'Smile inputs changed'
    else:path.write_text(json.dumps(inputs,indent=2)+'\n')
    records=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for i,row in enumerate(pool.map(build,jobs),1):
            records.append(row)
            (OUTPUT/'progress.json').write_text(json.dumps(dict(completed=i,total=len(jobs),elapsed_seconds=time.monotonic()-start),indent=2)+'\n')
    # Only small daily feature panels, never full chains, are combined here.
    panels=pd.concat([pd.read_parquet(r['output']) for r in records],ignore_index=True)
    assert not panels.duplicated(['date','secid']).any()
    folder=OUTPUT/'assets';folder.mkdir(exist_ok=True);assets={}
    for sid,g in panels.groupby('secid'):
        p=folder/f'{int(sid)}.parquet';g.set_index('date').sort_index().to_parquet(p);assets[str(p)]=digest(p)
    (OUTPUT/'manifest.json').write_text(json.dumps(dict(inputs=digest(path),partitions=records,assets=assets,
                elapsed_seconds=time.monotonic()-start),indent=2)+'\n')

if __name__=='__main__':main()
