"""Verify download completeness/checksums and report actual name/date coverage."""
import argparse
import json
from pathlib import Path
import sqlite3
import tempfile
import time
import resource
import sys

import pandas as pd
import pyarrow.parquet as pq
from scripts.download_pair_contract_history import file_hash

EPOCH=pd.Timestamp('2010-01-01')


def mask_bounds(mask):
    if not mask:return pd.NaT,pd.NaT
    return (EPOCH+pd.Timedelta(days=(mask&-mask).bit_length()-1),
            EPOCH+pd.Timedelta(days=mask.bit_length()-1))


def iter_contract_batches(root, secids, start, end):
    """Yield at most 5,000 rows; never materialize a full partition."""
    start,end=pd.Timestamp(start),pd.Timestamp(end)
    paths=[]
    requests_path=Path(root)/'requests.json'
    if requests_path.exists():
        requests=json.loads(requests_path.read_text())
        paths=[Path(root)/r['path'] for r in requests if r['kind']=='quotes'
               and start.year<=r['year']<=end.year and set(secids).intersection(r['secids'])]
    else:
        for year in range(start.year,end.year+1):
            paths.extend(sorted((Path(root)/'quotes'/f'year={year}').glob('*.parquet')))
    if not paths:
        raise FileNotFoundError('No downloaded quote partitions in the requested years')
    for path in paths:
        manifest=json.loads(path.with_suffix('.json').read_text())
        if not set(secids).intersection(manifest['request']['secids']):
            continue
        for batch in pq.ParquetFile(path).iter_batches(batch_size=5000,use_threads=False):
            frame=batch.to_pandas()
            frame['date']=pd.to_datetime(frame.date);frame['exdate']=pd.to_datetime(frame.exdate)
            frame=frame[frame.secid.isin(secids)&frame.date.between(start,end)]
            if len(frame):yield frame.reset_index(drop=True)


def read_contracts(root, secids, start, end, max_rows=250000):
    """Bounded convenience reader; use iter_contract_batches for long histories."""
    frames=[];count=0
    for frame in iter_contract_batches(root,secids,start,end):
        count+=len(frame)
        if count>max_rows:
            raise MemoryError('Requested history exceeds the in-memory row limit; use iter_contract_batches')
        frames.append(frame)
    if not frames:
        raise FileNotFoundError('None of the requested underlyings are in the downloaded partitions')
    return pd.concat(frames,ignore_index=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('data/option_contracts'))
    parser.add_argument('--wait-for-download',action='store_true')
    args=parser.parse_args()
    root=args.root;report=Path('reports/contract_data')
    requests=json.loads((root/'requests.json').read_text())
    assets=pd.read_csv(report/'underlyings.csv').set_index('secid')
    pairs=pd.read_csv(report/'pairs.csv')
    parts=[];coverage=[];quote_dates={};stock_dates={};seen=set()
    temp=tempfile.TemporaryDirectory(prefix='contract_audit_')
    keys_db=sqlite3.connect(str(Path(temp.name)/'keys.sqlite'))
    keys_db.execute('PRAGMA cache_size=-4096')
    keys_db.execute('PRAGMA journal_mode=OFF')
    keys_db.execute('PRAGMA temp_store=FILE')
    keys_db.execute('CREATE TABLE keys (day INTEGER, id INTEGER, PRIMARY KEY(day,id)) WITHOUT ROWID')
    for j,job in enumerate(requests,1):
        path=root/job['path'];manifest=path.with_suffix('.json')
        if args.wait_for_download:
            def ready():
                if not path.exists() or not manifest.exists():return False
                try:m=json.loads(manifest.read_text())
                except json.JSONDecodeError:return False
                return m.get('request')=={**job,'query_version':1,'max_dte':60,'cp_flag':'C'}
            while not ready():
                progress=root/'progress.json'
                if progress.exists():
                    try:state=json.loads(progress.read_text())
                    except json.JSONDecodeError:state={}
                    if any(x['request']['path']==job['path'] for x in state.get('failures',[])):
                        raise RuntimeError(f'Download failed: {job["path"]}')
                time.sleep(5)
        if not path.exists() or not manifest.exists():
            raise FileNotFoundError(f'Incomplete request: {job["path"]}')
        m=json.loads(manifest.read_text())
        if m['request']!={**job,'query_version':1,'max_dte':60,'cp_flag':'C'}:
            raise AssertionError(f'Request/manifest mismatch: {path}')
        if file_hash(path)!=m['sha256']:
            raise AssertionError(f'Checksum mismatch: {path}')
        file=pq.ParquetFile(path)
        if file.metadata.num_rows!=m['rows']:
            raise AssertionError(f'Row count mismatch: {path}')
        parts.append(dict(kind=job['kind'],path=job['path'],rows=m['rows'],bytes=m['bytes'],sha256=m['sha256']))
        if job['kind'] not in ['quotes','underlying']:continue
        cols=['secid','date']+(['optionid','exdate','cp_flag','best_bid','best_offer'] if job['kind']=='quotes' else [])
        counts={sid:dict(rows=0,mask=0,crossed=0) for sid in job['secids']}
        keys_db.execute('DELETE FROM keys')
        for batch in file.iter_batches(batch_size=5000,columns=cols,use_threads=False):
            d=batch.to_pandas();d['date']=pd.to_datetime(d.date)
            if not d.secid.isin(job['secids']).all() or not d.date.between(job['start'],job['end']).all():
                raise AssertionError(f'Out-of-request records: {path}')
            d['day']=(d.date-EPOCH).dt.days
            id_col='optionid' if job['kind']=='quotes' else 'secid'
            try:
                keys_db.executemany('INSERT INTO keys VALUES (?,?)',
                                    ((int(day),int(oid)) for day,oid in zip(d.day,d[id_col])))
            except sqlite3.IntegrityError as exc:
                raise AssertionError(f'Duplicate keys: {path}') from exc
            if job['kind']=='quotes':
                d['exdate']=pd.to_datetime(d.exdate)
                if not d.cp_flag.eq('C').all() or not (d.exdate-d.date).dt.days.between(0,60).all():
                    raise AssertionError(f'Invalid requested option type/maturity: {path}')
            for sid,g in d.groupby('secid'):
                c=counts[int(sid)];c['rows']+=len(g)
                c['mask']|=sum(1<<int(day) for day in g.day.unique())
                if job['kind']=='quotes':c['crossed']+=int(g.best_bid.gt(g.best_offer).sum())
        keys_db.commit()
        for sid in job['secids']:
            key=(job['kind'],job['year'],sid)
            if key in seen:raise AssertionError('Overlapping cache partitions')
            seen.add(key)
            count=counts[sid];mask=count['mask'];first,last=mask_bounds(mask)
            (quote_dates if job['kind']=='quotes' else stock_dates)[(sid,job['year'])]=mask
            coverage.append(dict(kind=job['kind'],year=job['year'],secid=sid,
                                 tickers=assets.loc[sid,'tickers'],rows=count['rows'],dates=mask.bit_count(),
                                 first=first,last=last,crossed_quotes=count['crossed']))
        print(f'Audited {j}/{len(requests)}: {job["path"]}',flush=True)
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024)
        (root/'audit_progress.json').write_text(json.dumps(dict(completed=j,total=len(requests),peak_rss_mib=round(peak,1)))+'\n')
    keys_db.close();temp.cleanup()
    c=pd.DataFrame(coverage);parts=pd.DataFrame(parts)
    c['underlying_dates']=c.apply(lambda r:stock_dates.get((r.secid,r.year),0).bit_count(),axis=1)
    c['underlying_dates_without_calls']=c.apply(lambda r:(stock_dates.get((r.secid,r.year),0)&~quote_dates.get((r.secid,r.year),0)).bit_count(),axis=1)
    c.to_csv(report/'coverage.csv',index=False);parts.to_csv(report/'partitions.csv',index=False)
    pair_rows=[]
    for r in pairs.itertuples():
        a=b=0
        for (sid,_),v in quote_dates.items():
            if sid==r.secid_a:a|=v
            if sid==r.secid_b:b|=v
        lo,hi=(pd.Timestamp(r.start)-EPOCH).days,(pd.Timestamp(r.end)-EPOCH).days
        common=(a&b)&(((1<<(hi+1))-1)^((1<<lo)-1))
        first,last=mask_bounds(common)
        pair_rows.append(dict(pair=r.pair,labels=r.labels,common_quote_dates=common.bit_count(),
                              first=first,last=last,
                              history_span_years=(last-first).days/365.25 if common else 0.))
    pc=pd.DataFrame(pair_rows);pc.to_csv(report/'pair_coverage.csv',index=False)
    q=c[c.kind.eq('quotes')];n_rows=int(parts.loc[parts.kind.eq('quotes'),'rows'].sum())
    with_quotes=q.groupby('secid').rows.sum().gt(0).sum()
    missing=q[q.rows.eq(0)&q.underlying_dates.gt(0)]
    summary=dict(pairs=len(pairs),underlyings=len(assets),underlyings_with_quotes=int(with_quotes),
                 pairs_with_overlapping_quote_dates=int(pc.common_quote_dates.gt(0).sum()),
                 quote_rows=n_rows,total_bytes=int(parts.bytes.sum()),partitions=len(parts),
                 crossed_quote_rows=int(q.crossed_quotes.sum()),
                 name_years_with_stock_but_no_calls=len(missing),
                 pairs_with_ten_year_span=int(pc.history_span_years.ge(10).sum()),
                 pairs_with_ten_year_span_and_2520_dates=int((pc.history_span_years.ge(10)&pc.common_quote_dates.ge(2520)).sum()))
    (root/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Historical contract data for previously explored pairs','',
           f"Downloaded and checksum-verified **{n_rows:,} actual call-option quote rows** for **{int(with_quotes)} of {len(assets)} requested underlyings**, covering {len(pairs)} unique pairs. {int(pc.common_quote_dates.gt(0).sum())} pairs have at least one overlapping quote date. This is a data-coverage statement, not a claim that every pair/date is tradable.",'',
           '- All 317 underlyings request January 2010–September 2025: up to 15.75 years of history. Actual coverage depends on listing, delisting and option availability. September 2025 supplies runoff for August entries.',
           '- This extends a fixed retrospective research roster; it does not make the earlier pair-selection process point-in-time or establish an untouched test sample.',
           f"- {summary['pairs_with_ten_year_span']} pairs span at least ten calendar years of overlapping quote history; {summary['pairs_with_ten_year_span_and_2520_dates']} also have at least 2,520 common quote dates. A long first-to-last span alone does not establish continuous coverage. Short histories remain visible rather than being silently excluded.",
           '- Calls only, all strikes, 0–60 calendar days to expiry. This covers the current simulator’s 21–45-day entry rule and subsequent held-contract marks. Longer maturities and puts are not part of this cache.',
           '- Raw quote records retain crossed markets, missing Greeks, zero bids, nonstandard settlements and corporate-action fields. No future-outcome, liquidity or profitable-pair filter is applied to the download.',
           '- Includes underlying OHLC, adjustment factors and returns; the interest-rate term structure; and distributions with declaration, ex- and payment dates. This does not provide an as-of earnings-calendar feed, stock-borrow quotes, intraday option depth or broker margin.',
           f"- {len(parts)} verified Parquet partitions, {parts.bytes.sum()/1024**3:.2f} GiB. Checks cover hashes, row counts, keys, request bounds and partition overlap.",
           f"- {int(q.crossed_quotes.sum()):,} crossed quote rows remain in the raw data. {len(missing)} requested name-years have underlying records but no calls inside the maturity window; these gaps are explicit in the coverage table.",'',
           '## Files','',
           'Licensed raw files are local in `data/option_contracts/`: `quotes/year=YYYY/batch=ID.parquet`, `underlying/year=YYYY.parquet`, `zero_curve.parquet`, and `distributions.parquet`. Each file has a request/checksum manifest. `requests.json` identifies the active partitions; older superseded batches remain on disk but are excluded by the reader and audit to prevent duplicates. Raw data are git-ignored; collaborators need their own authorized WRDS access to reproduce the download.','',
           '[Pair roster](contract_data/pairs.csv) · [Underlying roster](contract_data/underlyings.csv) · [Name/year coverage](contract_data/coverage.csv) · [Pair overlap coverage](contract_data/pair_coverage.csv) · [Partition manifest](contract_data/partitions.csv)','',
           '```sh',
           'python -m scripts.download_pair_contract_history --workers 1',
           'python -m scripts.audit_contract_history',
           '```','',
           'The downloader resumes verified completed partitions and retries failed requests. One worker and 5,000-row batches are the default; checksums stream in 1 MiB buffers. The audit also streams batches and checks duplicate keys on disk with a 4 MiB SQLite cache. `iter_contract_batches(root, secids, start, end)` streams a chosen pair and interval; `read_contracts` is capped at 250,000 rows by default. Names are keyed by OptionMetrics SECID; ticker labels can be historical. The roster merges the broad 2018+ screens with the earlier stock and ETF configurations. Candidates with insufficient prior model training are retained. Pre-2010 contract training history is not included.']
    Path('reports/contract_data.md').write_text('\n'.join(lines)+'\n')
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024)
    (root/'audit_progress.json').write_text(json.dumps(dict(completed=len(requests),total=len(requests),
        peak_rss_mib=round(peak,1),status='complete'))+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
