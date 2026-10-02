"""Stream puts matching cached selected-call strikes/expiries; retain full held-ID histories."""
import argparse
import getpass
import json
import os
from pathlib import Path
import time
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from psycopg2.extras import execute_values
from scripts.download_pair_contract_history import connection,QUOTE_SCHEMA,file_hash

ROOT=Path('data/call_put')


def definitions():
    wanted=set(json.loads(Path('reports/expiry_vix_momentum/matched_pairs.json').read_text()))
    return [d for d in json.loads(Path('artifacts/contracts/expiry_trained_beta_085_115/inputs.json').read_text())['definitions'] if d['pair'] in wanted]


def plan():
    ids=sorted({s for d in definitions() for s in d['secids']});jobs=[]
    for year in range(2010,2026):
        for i in range(0,len(ids),12):
            group=ids[i:i+12];pieces=[]
            for y in [year-1,year]:
                for sid in group:
                    path=Path('data/contract_history_shared')/str(y)/f'{sid}_selection.parquet'
                    if path.exists():pieces.append(pd.read_parquet(path,columns=['secid','expiry','strike']))
            if not pieces:continue
            keys=pd.concat(pieces).drop_duplicates();keys=keys[pd.to_datetime(keys.expiry)>=pd.Timestamp(year,1,1)]
            if keys.empty:continue
            path=ROOT/'keys'/f'{year}_{i:03d}.parquet';path.parent.mkdir(parents=True,exist_ok=True)
            keys.to_parquet(path,index=False)
            jobs.append(dict(year=year,secids=group,keys=str(path),keys_sha256=file_hash(path),path=f'quotes/{year}_{i:03d}.parquet'))
    (ROOT/'requests.json').write_text(json.dumps(jobs,indent=2)+'\n')
    return jobs


def run_job(task):
    i,job,total=task
    db=connection(os.getenv('WRDS_USERNAME',getpass.getuser()))
    path=ROOT/job['path'];meta=path.with_suffix('.json');path.parent.mkdir(exist_ok=True)
    if meta.exists() and path.exists():
        saved=json.loads(meta.read_text())
        assert saved['job']==job and file_hash(path)==saved['sha256'];return
    keys=pd.read_parquet(job['keys']);assert file_hash(Path(job['keys']))==job['keys_sha256']
    values=[(int(r.secid),r.expiry.date(),round(float(r.strike)*1000)) for r in keys.itertuples()]
    table=f"optionm.opprcd{job['year']}";start=time.monotonic()
    with db.cursor() as c:c.execute("SET statement_timeout='15min'")
    fields=','.join('p.'+c for c in QUOTE_SCHEMA.names)
    ids=','.join(str(int(s)) for s in job['secids'])
    query=f"""WITH keys(secid,exdate,strike_price) AS (VALUES %s),
        ids AS (SELECT DISTINCT p.optionid FROM {table} p JOIN keys k USING(secid,exdate,strike_price)
            WHERE p.secid IN ({ids}) AND p.cp_flag='P' AND p.exdate BETWEEN p.date AND p.date+60
            AND p.date<='2025-09-30')
        SELECT {fields} FROM {table} p JOIN ids i USING(optionid)
        WHERE p.secid IN ({ids}) AND p.cp_flag='P' AND p.exdate BETWEEN p.date AND p.date+60
        AND p.date<='2025-09-30' """
    temporary=path.with_suffix('.partial');rows=0
    with pq.ParquetWriter(temporary,QUOTE_SCHEMA,compression='zstd') as writer:
        with db.cursor(name='matched_puts') as c:
            c.itersize=5000;execute_values(c,query,values,page_size=len(values))
            while True:
                values=c.fetchmany(5000)
                if not values:break
                frame=pd.DataFrame(values,columns=QUOTE_SCHEMA.names)
                writer.write_table(pa.Table.from_pandas(frame,schema=QUOTE_SCHEMA,preserve_index=False));rows+=len(frame)
    db.commit();temporary.replace(path)
    meta.write_text(json.dumps(dict(job=job,rows=rows,sha256=file_hash(path),seconds=time.monotonic()-start,
        license='Licensed WRDS/OptionMetrics; no redistribution'),indent=2)+'\n')
    print(f'{i}/{total} {path.name}: {rows:,} rows, {time.monotonic()-start:.1f}s',flush=True)


def main():
    from concurrent.futures import ThreadPoolExecutor
    parser=argparse.ArgumentParser();parser.add_argument('--plan-only',action='store_true');parser.add_argument('--limit',type=int)
    parser.add_argument('--workers',type=int,choices=[1,2],default=2);args=parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    jobs=json.loads((ROOT/'requests.json').read_text()) if (ROOT/'requests.json').exists() else plan()
    print(f'{len(jobs)} bounded put-history partitions; {args.workers} workers',flush=True)
    if args.plan_only:return
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for _ in pool.map(run_job,[(i,j,len(jobs)) for i,j in enumerate(jobs[:args.limit],1)]):pass


if __name__=='__main__':main()
