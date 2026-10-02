"""Bounded call-chain surface extraction using observed midpoints, no extrapolation."""
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

ROOT=Path('data/tech_etf/contracts')
OUTPUT=Path('data/tech_etf/surface')
VALUES=['downside_skew','smile_curvature','term_slope','term_bend']
KEYS=['date','secid','slot']


def reduce_points(frame):
    if frame.empty:return frame
    return frame.sort_values(KEYS+['distance','strike_distance','expiry','strike','optionid']).drop_duplicates(KEYS).reset_index(drop=True)


def candidates(raw,selected,stocks):
    q=normalize_quotes(raw)
    mid=(q.bid+q.ask)/2
    good=(q.standard&q.multiplier.eq(100)&q.cp_flag.eq('C')&q.bid.gt(0)&q.ask.ge(q.bid)&
          ((q.ask-q.bid)/mid).le(.25)&q.open_interest.ge(100))
    q=q[good].merge(selected[['date','secid','expiry']].rename(columns={'expiry':'middle_expiry'}),on=['date','secid'],how='inner',validate='many_to_one')
    q=q.merge(stocks[['date','secid','close']],on=['date','secid'],how='left',validate='many_to_one')
    q['dte']=(q.expiry-q.date).dt.days
    q['strike_distance']=(q.strike/q.close-1).abs()
    parts=[]
    for slot,target in [('lower',.75),('upper',.25)]:
        use=q.expiry.eq(q.middle_expiry)&q.delta.sub(target).abs().le(.10)
        p=q[use].copy();p['slot']=slot;p['distance']=p.delta.sub(target).abs();parts.append(p)
    for slot,target,lo,hi in [('front',15,8,22),('back',45,35,55)]:
        maturity=q.expiry.lt(q.middle_expiry) if slot=='front' else q.expiry.gt(q.middle_expiry)
        p=q[q.dte.between(lo,hi)&maturity&q.strike_distance.le(.1)].copy()
        p['slot']=slot;p['distance']=p.dte.sub(target).abs();parts.append(p)
    return reduce_points(pd.concat(parts,ignore_index=True).drop(columns=['close']))


def surface_values(selected,points):
    """All IVs in vol points; term slope per 30 calendar days."""
    base=selected.set_index(['date','secid'])[['bs_iv','expiry','strike']].rename(columns={'bs_iv':'atm_iv','expiry':'atm_expiry','strike':'atm_strike'})
    base['atm_iv']*=100
    base['atm_dte']=(base.atm_expiry-base.index.get_level_values('date')).dt.days
    for slot in ['lower','upper','front','back']:
        p=points[points.slot.eq(slot)].set_index(['date','secid'])
        for source,target in [('bs_iv','iv'),('bs_delta','delta'),('expiry','expiry'),('strike','strike')]:
            base[f'{slot}_{target}']=p[source] if source in p else np.nan
        base[f'{slot}_iv']*=100
        base[f'{slot}_dte']=(pd.to_datetime(base[f'{slot}_expiry'])-base.index.get_level_values('date')).dt.days
    lower=(base.lower_expiry.eq(base.atm_expiry)&base.lower_delta.sub(.75).abs().le(.10)&base.lower_strike.lt(base.atm_strike))
    upper=(base.upper_expiry.eq(base.atm_expiry)&base.upper_delta.sub(.25).abs().le(.10)&base.upper_strike.gt(base.atm_strike))
    base['downside_skew']=(base.lower_iv-base.atm_iv).where(lower)
    base['smile_curvature']=((base.lower_iv+base.upper_iv)/2-base.atm_iv).where(lower&upper)
    bracket=base.front_dte.lt(base.atm_dte)&base.back_dte.gt(base.atm_dte)
    span=base.back_dte-base.front_dte
    base['term_slope']=((base.back_iv-base.front_iv)*30/span).where(bracket)
    interpolated=base.front_iv+(base.back_iv-base.front_iv)*(base.atm_dte-base.front_dte)/span
    base['term_bend']=(base.atm_iv-interpolated).where(bracket)
    return base.replace([np.inf,-np.inf],np.nan).reset_index()


def build(job):
    folder=OUTPUT/str(job['year']);folder.mkdir(exist_ok=True)
    destination=folder/Path(job['path']).name;meta=destination.with_suffix('.json')
    if meta.exists():return json.loads(meta.read_text())
    source=ROOT/job['path'];selected_path=Path('data/tech_etf/atm_30d')/str(job['year'])/source.name
    selected=pd.read_parquet(selected_path)
    if selected.duplicated(['date','secid']).any():raise ValueError('Duplicate ATM selection')
    stocks=pd.read_parquet(ROOT/'underlying'/f"year={job['year']}.parquet",columns=['date','secid','close'])
    stocks['date']=pd.to_datetime(stocks.date);stocks=stocks[stocks.secid.isin(job['secids'])]
    points=pd.DataFrame();scanned=0
    for batch in pq.ParquetFile(source).iter_batches(batch_size=65536,use_threads=False):
        raw=batch.to_pandas();scanned+=len(raw)
        new=candidates(raw,selected,stocks)
        if len(new):points=reduce_points(pd.concat([points,new],ignore_index=True))
        if len(points)>100000:raise MemoryError('Surface candidate bound exceeded')
    if len(points):
        zero=pd.read_parquet(ROOT/'zero_curve.parquet');divs=pd.read_parquet(ROOT/'distributions.parquet')
        points=pd.concat([fit_quotes(points.iloc[i:i+25000],stocks,zero,divs) for i in range(0,len(points),25000)],ignore_index=True)
    else:points=pd.DataFrame(columns=['date','secid','slot','bs_iv','bs_delta','expiry','strike'])
    result=surface_values(selected,points)
    result.to_parquet(destination,index=False)
    points.to_parquet(destination.with_name(destination.stem+'_points.parquet'),index=False)
    record=dict(source=str(source),output=str(destination),rows_scanned=scanned,points=len(points),asset_dates=len(result),
                valid={v:int(result[v].notna().sum()) for v in VALUES},peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    meta.write_text(json.dumps(record,indent=2)+'\n');return record


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    jobs=sorted([r for r in json.loads((ROOT/'requests.json').read_text()) if r['kind']=='quotes'],key=lambda r:(r['year'],r['path']))
    missing=[j for j in jobs if not (Path('data/tech_etf/atm_30d')/str(j['year'])/Path(j['path']).name).exists()]
    jobs=[j for j in jobs if j not in missing]
    sources=[ROOT/j['path'] for j in jobs]+[Path('data/tech_etf/atm_30d')/str(j['year'])/Path(j['path']).name for j in jobs]
    sources += [ROOT/'underlying'/f'year={y}.parquet' for y in sorted({j['year'] for j in jobs})]
    sources += [ROOT/'zero_curve.parquet',ROOT/'distributions.parquet']
    fingerprint=dict(code={p:digest(p) for p in [__file__,'scripts/black_scholes.py','scripts/run_contract_pilot.py']},sources={str(p):digest(p) for p in sources},jobs=jobs,no_atm_selection_partitions=missing)
    path=OUTPUT/'inputs.json'
    if path.exists() and json.loads(path.read_text())!=fingerprint:raise ValueError('Surface inputs changed')
    path.write_text(json.dumps(fingerprint,indent=2)+'\n');started=time.monotonic();records=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for i,r in enumerate(pool.map(build,jobs),1):
            records.append(r)
            print(f"{i}/{len(jobs)} {r['source']}: {r['points']:,} points",flush=True)
    # Small daily feature panels only; the raw chain never enters this concat.
    panels=pd.concat([pd.read_parquet(r['output']) for r in records],ignore_index=True)
    if panels.duplicated(['date','secid']).any():raise ValueError('Duplicate surface asset dates')
    asset_dir=OUTPUT/'assets';asset_dir.mkdir(exist_ok=True)
    for sid,frame in panels.groupby('secid'):frame.set_index('date').sort_index().to_parquet(asset_dir/f'{int(sid)}.parquet')
    manifest=dict(partitions=records,elapsed_seconds=time.monotonic()-started,inputs_sha256=digest(path),
                  max_worker_peak_rss_bytes=max(r['peak_rss_bytes'] for r in records))
    (OUTPUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
