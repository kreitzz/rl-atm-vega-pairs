"""Resumable actual-contract cache for every previously screened/tested pair.

Calls, all strikes, 0--60 calendar days to expiry; no quote/Greek/liquidity
filter. All previously explored assets request history from January 2010.
Raw licensed records are ignored under data/option_contracts, never published.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


END = '2025-09-30'
THREAD = threading.local()


def file_hash(path):
    """Bounded-memory checksum, even for multi-GB partitions."""
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def registry():
    assets = {}
    pairs = {}

    def add(label, names, start, source):
        ids = [int(n['secid']) for n in names.values()]
        key = '__'.join(map(str, sorted(ids)))
        p = pairs.setdefault(key, dict(pair=key, secid_a=ids[0], secid_b=ids[1],
                                       labels=set(), sources=set(), start=start, end=END))
        p['labels'].add(label); p['sources'].add(source); p['start'] = min(p['start'], start)
        for ticker, n in names.items():
            sid = int(n['secid'])
            a = assets.setdefault(sid, dict(secid=sid, tickers=set(), permnos=set(),
                                           sources=set(), start=start, end=END))
            a['tickers'].add(ticker); a['permnos'].add(str(int(n['permno'])))
            a['sources'].add(source); a['start'] = min(a['start'], start)

    for p in json.loads(Path('config/universe_gap_pairs.json').read_text()):
        add(p['pair'], p['names'], '2010-01-01', 'legacy_stock')
    etf = json.loads(Path('config/etf_pairs.json').read_text())
    for p in etf['pairs']:
        add(p['pair'], {name: etf['assets'][name] for name in p['pair'].split('_')},
            '2010-01-01', 'legacy_etf')
    lookup = pd.read_csv('reports/broad_pairs/assets.csv').set_index('asset')
    screens = pd.read_csv('reports/broad_pairs/candidate_screens.csv')
    for r in screens[screens.year.ge(2018)].itertuples():
        a,b = lookup.loc[r.asset_a],lookup.loc[r.asset_b]
        add(r.label, {str(a.ticker): a.to_dict(), str(b.ticker): b.to_dict()},
            '2010-01-01', 'broad_screen')
    def clean(rows):
        return pd.DataFrame([{k: '|'.join(sorted(v)) if isinstance(v,set) else v
                              for k,v in x.items()} for x in rows])
    return clean(assets.values()).sort_values('secid'), clean(pairs.values()).sort_values('pair')


def quote_jobs(assets, batch_size):
    jobs=[]
    for year in range(2025,2009,-1):
        ids = assets.loc[assets.start.le(f'{year}-12-31'),'secid'].astype(int).tolist()
        end = min(END, f'{year}-12-31')
        for i in range(0,len(ids),batch_size):
            subset=ids[i:i+batch_size]
            fingerprint=hashlib.sha256(json.dumps(subset).encode()).hexdigest()[:12]
            jobs.append(dict(kind='quotes', table=f'optionm.opprcd{year}', secids=subset,
                             start=f'{year}-01-01',end=end,year=year,
                             path=f'quotes/year={year}/batch={fingerprint}.parquet'))
        jobs.append(dict(kind='underlying', table=f'optionm.secprd{year}', secids=ids,
                         start=f'{year}-01-01',end=end,year=year,
                         path=f'underlying/year={year}.parquet'))
    all_ids=assets.secid.astype(int).tolist()
    jobs += [dict(kind='rates',table='optionm.zerocd',start='2010-01-01',end=END,
                  path='zero_curve.parquet'),
             dict(kind='distributions',table='optionm.distrd',secids=all_ids,
                  start='2010-01-01',end=END,path='distributions.parquet')]
    return jobs


QUOTE_SCHEMA=pa.schema([
    ('secid',pa.int64()),('date',pa.date32()),('optionid',pa.int64()),('exdate',pa.date32()),
    ('cp_flag',pa.string()),('strike_price',pa.float64()),('best_bid',pa.float64()),
    ('best_offer',pa.float64()),('volume',pa.float64()),('open_interest',pa.float64()),
    ('impl_volatility',pa.float64()),('delta',pa.float64()),('gamma',pa.float64()),
    ('vega',pa.float64()),('theta',pa.float64()),('ss_flag',pa.string()),
    ('cfadj',pa.float64()),('contract_size',pa.float64())])


def sql_for(job):
    if job['kind']=='quotes':
        fields=', '.join(QUOTE_SCHEMA.names)
        query=f'''select {fields} from {job['table']}
          where secid = ANY(%s) and date between %s and %s
          and cp_flag = 'C' and exdate between date and date + 60'''
        params=(job['secids'],job['start'],job['end'])
    elif job['kind']=='underlying':
        query=f'''select secid, date, close, open, high, low, volume, return,
                  cfadj, cfret, shrout from {job['table']}
                  where secid = ANY(%s) and date between %s and %s'''
        params=(job['secids'],job['start'],job['end'])
    elif job['kind']=='rates':
        query=f"select date, days, rate from {job['table']} where date between %s and %s"
        params=(job['start'],job['end'])
    else:
        query=f'''select secid, record_date, seq_num, ex_date, amount, adj_factor,
                  declare_date, payment_date, link_secid, distr_type, frequency,
                  currency, approx_flag, cancel_flag, liquid_flag from {job['table']}
                  where secid = ANY(%s) and ex_date between %s and %s'''
        params=(job['secids'],job['start'],job['end'])
    return query,params


def connection(username):
    import psycopg2
    if not getattr(THREAD,'db',None) or THREAD.db.closed:
        THREAD.db=psycopg2.connect(host='wrds-pgdata.wharton.upenn.edu',port=9737,
                                  dbname='wrds',user=username,sslmode='require',connect_timeout=20)
        THREAD.db.set_session(readonly=True)
    return THREAD.db


def download(job, output, username):
    path=output/job['path']; manifest=path.with_suffix('.json')
    expected=dict(job,query_version=1,max_dte=60,cp_flag='C')
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
                cur.itersize=5000
                cur.execute(query,params)
                # Named cursor metadata is populated on the first fetch.
                while True:
                    values=cur.fetchmany(5000)
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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('data/option_contracts'))
    parser.add_argument('--username',default=os.getenv('WRDS_USERNAME',getpass.getuser()))
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--batch-size',type=int,default=20)
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args()
    if not 1<=args.workers<=3 or not 1<=args.batch_size<=50:
        raise ValueError('Use 1–3 workers and batches of 1–50 names')
    assets,pairs=registry();jobs=quote_jobs(assets,args.batch_size)
    report=Path('reports/contract_data');report.mkdir(exist_ok=True)
    assets.to_csv(report/'underlyings.csv',index=False);pairs.to_csv(report/'pairs.csv',index=False)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'requests.json').write_text(json.dumps(jobs,indent=2)+'\n')
    print(f'{len(pairs)} pairs; {len(assets)} underlyings; {len(jobs)} resumable partitions',flush=True)
    if args.plan_only:return
    completed=[];failures=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(download,j,args.output,args.username):j for j in jobs}
        for future in as_completed(pending):
            job=pending[future]
            try:
                result=future.result();completed.append(result)
                print(f"[{len(completed)+len(failures)}/{len(jobs)}] {result['status']} {job['path']}: {result['rows']:,} rows",flush=True)
            except Exception as exc:
                failures.append(dict(request=job,error=str(exc)))
                print(f"FAILED {job['path']}: {exc}",flush=True)
            (args.output/'progress.json').write_text(json.dumps(dict(
                completed=len(completed),total=len(jobs),rows=sum(r['rows'] for r in completed),
                bytes=sum(r['bytes'] for r in completed),failures=failures),indent=2)+'\n')
    if failures:raise SystemExit(f'{len(failures)} partitions failed; rerun to resume')
    print('All requested partitions complete.',flush=True)


if __name__=='__main__':main()
