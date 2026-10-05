"""Build shared selected-ID quote histories, one source partition at a time."""
import argparse
import json
from pathlib import Path
import resource
import time

import pandas as pd
import pyarrow.parquet as pq

from scripts.black_scholes import fit_quotes
from scripts.run_contract_pilot import normalize_quotes
from scripts.run_contract_walkforward import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--years',nargs='+',type=int,default=list(range(2018,2026)))
    parser.add_argument('--root',type=Path,default=Path('data/option_contracts'))
    parser.add_argument('--selected',type=Path,default=Path('data/atm_30d'))
    parser.add_argument('--output',type=Path,default=Path('data/contract_history_shared'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    jobs=sorted([r for r in json.loads((args.root/'requests.json').read_text()) if r['kind']=='quotes' and r['year'] in args.years],key=lambda r:(r['year'],r['path']))
    zero=pd.read_parquet(args.root/'zero_curve.parquet');divs=pd.read_parquet(args.root/'distributions.parquet')
    common={str(p):digest(p) for p in [args.root/'zero_curve.parquet',args.root/'distributions.parquet',Path('scripts/black_scholes.py'),Path('scripts/run_contract_pilot.py')]}
    results=[];start=time.monotonic();stocks=None;last_year=None
    for n,job in enumerate(jobs,1):
        year=job['year'];source=args.root/job['path'];selected_path=args.selected/str(year)/source.name
        meta_path=args.output/str(year)/(source.stem+'.json');meta_path.parent.mkdir(exist_ok=True)
        prior_paths=sorted((args.selected/str(year-1)).glob('*.parquet'))
        fingerprint=dict(common=common,quote_sha256=digest(source),selection_sha256=digest(selected_path) if selected_path.exists() else None,
                         prior_selection_sha256={str(p):digest(p) for p in prior_paths},stock_sha256=digest(args.root/'underlying'/f'year={year}.parquet'))
        if meta_path.exists():
            saved=json.loads(meta_path.read_text())
            if saved['fingerprint']==fingerprint and all(Path(p).exists() for p in saved['files']):
                results.append(saved);continue
        if last_year!=year:
            stocks=pd.read_parquet(args.root/'underlying'/f'year={year}.parquet');stocks['date']=pd.to_datetime(stocks.date);last_year=year
        selected=pd.read_parquet(selected_path) if selected_path.exists() else pd.DataFrame()
        previous=[pd.read_parquet(p,columns=['optionid'],filters=[('secid','in',job['secids'])]) for p in prior_paths]
        wanted=set(selected.optionid) if len(selected) else set()
        for frame in previous:wanted.update(frame.optionid)
        chunks=[];count=0
        if wanted:
            for batch in pq.ParquetFile(source).iter_batches(batch_size=5000,use_threads=False):
                frame=batch.to_pandas();frame=frame[frame.optionid.isin(wanted)]
                if len(frame):chunks.append(frame);count+=len(frame)
                if count>500000:raise MemoryError(f'Selected history exceeds partition bound: {source}')
        files=[]
        if chunks:
            q=normalize_quotes(pd.concat(chunks,ignore_index=True));del chunks
            parts=[fit_quotes(q.iloc[i:i+25000],stocks,zero,divs) for i in range(0,len(q),25000)]
            q=pd.concat(parts,ignore_index=True);del parts
            for field in ['delta','vega','impl_volatility']:q['vendor_'+field]=q[field]
            q['delta']=q.bs_delta;q['vega']=q.bs_vega;q['impl_volatility']=q.bs_iv
            for sid,frame in q.groupby('secid'):
                folder=args.output/str(year);path=folder/f'{int(sid)}.parquet'
                if path.exists() and not meta_path.exists():
                    # Each active source partition owns disjoint underlying IDs.
                    existing=pd.read_parquet(path,columns=['optionid','date'])
                    if len(existing)!=len(frame):raise ValueError(f'Conflicting cache ownership: {path}')
                frame.to_parquet(path,index=False);files.append(str(path))
            del q
        if len(selected):
            for sid,frame in selected.groupby('secid'):
                path=args.output/str(year)/f'{int(sid)}_selection.parquet';frame.to_parquet(path,index=False);files.append(str(path))
        saved=dict(source=str(source),year=year,rows=count,files=files,fingerprint=fingerprint)
        meta_path.write_text(json.dumps(saved,indent=2)+'\n');results.append(saved)
        progress=dict(completed=n,total=len(jobs),rows=sum(r['rows'] for r in results),elapsed_seconds=time.monotonic()-start,
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (args.output/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
        print(f"{n}/{len(jobs)} {year}: {count:,} marks; elapsed {(time.monotonic()-start)/60:.1f}m",flush=True)
    (args.output/'manifest.json').write_text(json.dumps(dict(years=args.years,partitions=results,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),indent=2)+'\n')


if __name__=='__main__':main()
