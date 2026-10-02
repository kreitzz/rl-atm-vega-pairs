"""Read-only incremental calls and puts, 61–110 DTE; one partition per process."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import getpass
import json
import os
from pathlib import Path
import shutil
import time
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scripts.download_pair_contract_history import connection,QUOTE_SCHEMA,file_hash,THREAD
from scripts.download_matched_puts import definitions

OUTPUT=Path('data/curve90/raw')
STREAM_BATCH_ROWS=16000

def sql_for(job):
    fields=', '.join(QUOTE_SCHEMA.names)
    return (f"select {fields} from {job['table']} where secid = ANY(%s) and date between %s and %s "
            "and cp_flag IN ('C','P') and exdate between date + 61 and date + 110",(job['secids'],job['start'],job['end']))

def download(job, output, username):
    path=output/job['path']; manifest=path.with_suffix('.json')
    expected=dict(job,query_version=1,min_dte=61,max_dte=110,cp_flag='CP')
    if path.exists() and manifest.exists():
        saved=json.loads(manifest.read_text())
        if saved.get('request')==expected and path.stat().st_size==saved.get('bytes'):
            if file_hash(path)==saved.get('sha256'):
                return dict(saved,status='cached')
    if shutil.disk_usage(output).free < 10*1024**3:
        raise RuntimeError('Less than 10 GiB free; paused before the next partition')
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.parquet.partial')
    query,params=sql_for(job)
    started=time.monotonic()
    for attempt in range(3):
        writer=None; rows=0; coverage={}
        try:
            db=connection(username)
            with db.cursor() as setup:
                setup.execute("SET statement_timeout = '15min'")
            with db.cursor(name='contract_download') as cur:
                cur.itersize=STREAM_BATCH_ROWS
                cur.execute(query,params)
                # Named cursor metadata is populated on the first fetch.
                while True:
                    values=cur.fetchmany(STREAM_BATCH_ROWS)
                    if not values:break
                    frame=pd.DataFrame.from_records(values,columns=[x.name for x in cur.description])
                    schema=QUOTE_SCHEMA if job['kind']=='quotes' else None
                    table=pa.Table.from_pandas(frame,schema=schema,preserve_index=False)
                    if writer is None:
                        writer=pq.ParquetWriter(temp,table.schema,compression='zstd')
                    elif table.schema!=writer.schema:
                        table=table.cast(writer.schema)
                    writer.write_table(table);rows+=len(frame)
                    if 'secid' in frame and 'date' in frame:
                        for sid,g in frame.groupby('secid'):
                            c=coverage.setdefault(str(int(sid)),dict(rows=0,first=str(g.date.min()),last=str(g.date.max())))
                            c['rows']+=len(g);c['first']=min(c['first'],str(g.date.min()));c['last']=max(c['last'],str(g.date.max()))
                    del values,frame,table
            if writer is None:
                # Empty quote partitions are valid for not-yet-listed/delisted names.
                if job['kind']=='quotes':
                    pq.write_table(pa.Table.from_batches([],schema=QUOTE_SCHEMA),temp,compression='zstd')
                else:
                    with db.cursor() as cur:
                        cur.execute('select * from ('+query+') t limit 0',params)
                        pd.DataFrame(columns=[x.name for x in cur.description]).to_parquet(temp,index=False)
            else:
                writer.close();writer=None
            db.commit()
            temp.replace(path)
            result=dict(request=expected,rows=rows,bytes=path.stat().st_size,
                        sha256=file_hash(path),coverage=coverage,
                        seconds=round(time.monotonic()-started,2),
                        downloaded_at=datetime.now(timezone.utc).isoformat(),
                        license='Licensed WRDS/OptionMetrics; no redistribution',query=query)
            manifest.write_text(json.dumps(result,indent=2)+'\n')
            return dict(result,status='downloaded')
        except Exception:
            if writer is not None:writer.close()
            if getattr(THREAD,'db',None):THREAD.db.close()
            if attempt==2:raise
            time.sleep(2*(attempt+1))


def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);args=p.parse_args()
    jobs=json.loads(Path('data/curve_trading/puts_raw/requests.json').read_text())
    OUTPUT.mkdir(parents=True,exist_ok=True)
    job=jobs[args.index]
    r=download(job,OUTPUT,os.getenv('WRDS_USERNAME',getpass.getuser()))
    print(f"{args.index+1}/{len(jobs)} {r['status']}: {r['rows']:,} rows",flush=True)

if __name__=='__main__':main()
