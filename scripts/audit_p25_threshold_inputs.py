"""Audit and restore bounded quotes for the fixed-size IV-rank threshold test.

Audit runs in the pinned environment. Download runs in the documented mfe_wrds
environment and uses the existing matching libpq account without displaying or
copying credentials. Quote requests are fixed before outcomes are inspected.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','ARROW_NUM_THREADS']:
    os.environ[key] = '1'
import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import time
import numpy as np
import pandas as pd
from scripts.p25_opportunity_dataset import SOURCE, CALENDAR, evaluation_mask
from scripts.attribute_p25_opportunity_filter import Budget
from scripts.download_p25_call_transfer import normalize, RAW_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/contracts/p25_threshold_search'
REPORT = ROOT/'reports/p25_threshold_search'
OLD_DB = ROOT/'artifacts/contracts/p25_opportunity_filter/quotes.sqlite'
MARKET = ROOT/'data/submission/artifacts/contracts/p25_signal_transfer/market'
ZERO = ROOT/'data/q_contracts/AMD_NVDA/zero_curve.parquet'
GRID = [.05,.10,.15,.20,.25,.30]


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def selected(meta, tail):
    x=meta.copy()
    x['direction']=np.where(x.iv_rank.le(tail),1,np.where(x.iv_rank.gt(1-tail),-1,0))
    positive=x.direction.gt(0)
    mask=x.direction.ne(0)
    for days,minimum in [(91,21),(365,63)]:
        mask &= np.where(positive,x[f'count_long_{days}d'],x[f'count_short_{days}d'])>=minimum
        mask &= np.where(positive,x[f'profit_long_{days}d'],x[f'profit_short_{days}d'])>0
    return x.loc[mask].copy()


def audit(budget):
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    meta=pd.read_parquet(SOURCE)
    calendar=pd.read_csv(CALENDAR,parse_dates=['date'])
    meta=meta.loc[evaluation_mask(meta,calendar)].copy()
    with sqlite3.connect(f'file:{OLD_DB.as_posix()}?mode=ro',uri=True) as db:
        known={x[0] for x in db.execute('SELECT DISTINCT optionid FROM quotes')}
    rows=[]
    for tail in GRID:
        g=selected(meta,tail)
        coverage=g.optionid_a.isin(known)&g.optionid_b.isin(known)
        missing=(set(g.optionid_a.astype(int))|set(g.optionid_b.astype(int)))-known
        rows.append(dict(tail=tail,proposals=len(g),pairs=int(g.pair.nunique()),
                         both_contract_ids_available=int(coverage.sum()),missing_proposals=int((~coverage).sum()),
                         missing_contract_ids=len(missing)))
        if tail==.20:
            expected=meta.loc[meta.signal_eligible]
            assert set(zip(g.date,g.pair,g.direction))==set(zip(expected.date,expected.pair,expected.direction))
        g.to_parquet(OUT/f'proposals_{round(tail*100):02d}.parquet',index=False)
        budget.rest()
    pd.DataFrame(rows).to_csv(REPORT/'input_coverage.csv',index=False)
    g=selected(meta,.30)
    legs=[]
    for suffix,sign in [('a',1),('b',-1)]:
        own=g[['date',f'optionid_{suffix}',f'secid_{suffix}']].rename(columns={f'optionid_{suffix}':'optionid',f'secid_{suffix}':'secid'})
        dte=g.dte_mean+sign*g.dte_gap/2
        assert np.isfinite(dte).all() and np.allclose(dte,np.rint(dte))
        own['expiry']=g.date+pd.to_timedelta(dte,unit='D')
        legs.append(own)
    legs=pd.concat(legs,ignore_index=True)
    request=legs.groupby('optionid').agg(secid=('secid','first'),start=('date','min'),end=('expiry','max')).reset_index()
    request[['optionid','secid']]=request[['optionid','secid']].astype(int)
    request['missing']=~request.optionid.isin(known)
    # Two cached controls per calendar year independently test normalization,
    # rates/dividend conventions, and vendor revisions before adding new data.
    control_ids=set(request.loc[~request.missing].groupby(request.start.dt.year).head(2).optionid)
    request=request.loc[request.missing|request.optionid.isin(control_ids)]
    request.to_csv(OUT/'quote_requests.csv',index=False)
    request.loc[request.missing].to_csv(REPORT/'missing_contracts.csv',index=False)
    # These local quote caches are call-only, as specified by their downloaders.
    # The yearly cheap-vega cache was built from this complete signal-ID list.
    names=pd.read_parquet(ROOT/'data/submission/artifacts/contracts/year_cheap_vega/name_signals.parquet',columns=['optionid'])
    found=set(names.optionid.dropna().astype(int))&set(request.loc[request.missing,'optionid'])
    provenance=dict(grid=GRID,coverage=rows,missing_ids=int(request.missing.sum()),control_ids=len(control_ids),
        possible_matches_year_cheap_vega=len(found),
        other_local_caches='AMD_NVDA contract cache and recent call-transfer cache are calls only; existing P25 archives share the already audited quote DB.',
        rate_source=str(ZERO.relative_to(ROOT)),
        label_gates='Directional 91/365-day positive profit, counts >=21/63; unchanged inherited beta/context/calendar.',
        note='Coverage measures contract IDs, not future tradability. Missingness never removes proposals.')
    write(REPORT/'input_audit.json',provenance)
    print(json.dumps(provenance),flush=True)


def connect():
    """Use only an existing account matching this exact WRDS service."""
    import psycopg2
    host='wrds-pgdata.wharton.upenn.edu';paths=[]
    if os.environ.get('PGPASSFILE'):paths.append(Path(os.environ['PGPASSFILE']))
    if os.environ.get('APPDATA'):paths.append(Path(os.environ['APPDATA'])/'postgresql/pgpass.conf')
    paths.append(Path.home()/'.pgpass');matching=[]
    for path in dict.fromkeys(paths):
        if not path.is_file():continue
        with path.open() as stream:
            for line in stream:
                if line.startswith('#'):continue
                fields=next(csv.reader([line.rstrip('\n')],delimiter=':',escapechar='\\'))
                if len(fields)==5 and fields[0] in [host,'*'] and fields[1] in ['9737','*'] and fields[2] in ['wrds','*'] and fields[3] not in ['', '*']:
                    matching.append((path,fields[3]))
                del fields
    if len(set(user for _,user in matching))!=1:raise RuntimeError('No unique configured WRDS account')
    path,user=matching[0]
    return psycopg2.connect(host=host,port=9737,dbname='wrds',user=user,passfile=str(path),sslmode='require',
        connect_timeout=15,options='-c default_transaction_read_only=on -c statement_timeout=60000')


def download(budget):
    request=pd.read_csv(OUT/'quote_requests.csv',parse_dates=['start','end'])
    folder=OUT/'raw';folder.mkdir(exist_ok=True)
    db=connect();counts=[]
    try:
        for year in range(request.start.dt.year.min(),request.end.dt.year.max()+1):
            g=request.loc[request.start.le(f'{year}-12-31')&request.end.ge(f'{year}-01-01')]
            if g.empty:continue
            path=folder/f'{year}_quotes.parquet'
            if path.exists():continue
            query=f'''SELECT {', '.join('q.'+c for c in RAW_COLUMNS)} FROM optionm.opprcd{year} q
                JOIN unnest(%s::bigint[],%s::date[],%s::date[]) wanted(optionid,start_date,end_date)
                  ON q.optionid=wanted.optionid AND q.date BETWEEN wanted.start_date AND wanted.end_date
                WHERE q.cp_flag='P' ORDER BY q.date,q.optionid'''
            with db.cursor() as cur:
                cur.execute(query,(g.optionid.tolist(),g.start.dt.date.tolist(),g.end.dt.date.tolist()))
                raw=pd.DataFrame(cur.fetchall(),columns=[c.name for c in cur.description])
            if len(raw)>20000:raise RuntimeError('Bounded quote query unexpectedly large')
            q=normalize(raw)
            assert set(q.optionid).issubset(set(g.optionid)) and not q.duplicated(['date','optionid']).any()
            q.to_parquet(path,index=False)
            counts.append(dict(year=year,rows=len(q)))
            print(f'Downloaded {year}: {len(q)} bounded put quotes',flush=True)
            budget.rest()
        path=folder/'distributions.parquet'
        if not path.exists():
            query='''SELECT secid,record_date,seq_num,ex_date,amount,adj_factor,declare_date,payment_date,link_secid,
                distr_type,frequency,currency,approx_flag,cancel_flag,liquid_flag FROM optionm.distrd
                WHERE secid=ANY(%s) AND ex_date BETWEEN %s AND %s'''
            with db.cursor() as cur:
                cur.execute(query,(request.secid.unique().tolist(),request.start.min().date(),request.end.max().date()))
                div=pd.DataFrame(cur.fetchall(),columns=[c.name for c in cur.description])
            if len(div)>15000:raise RuntimeError('Bounded distribution query unexpectedly large')
            div.to_parquet(path,index=False)
    finally:db.close()
    write(REPORT/'download_status.json',dict(status='downloaded',quotes=counts,authentication='Existing configured libpq store; no credentials saved'))


def build(budget):
    from scripts.option_flex.black_scholes import fit_quotes
    request=pd.read_csv(OUT/'quote_requests.csv',parse_dates=['start','end'])
    q=pd.concat([pd.read_parquet(p) for p in sorted((OUT/'raw').glob('*_quotes.parquet'))],ignore_index=True)
    assert not q.duplicated(['date','optionid']).any()
    requested=set(request.optionid);returned=set(q.optionid)
    if requested-returned:raise ValueError(f'{len(requested-returned)} requested contracts absent from vendor response')
    stocks=pd.read_parquet(MARKET/'stocks.parquet',filters=[('secid','in',request.secid.unique().tolist())])
    zero=pd.read_parquet(ZERO)
    dist=pd.read_parquet(OUT/'raw/distributions.parquet')
    # Existing repaired cash events are the historical experiment's convention.
    # Replace matching raw distribution events with their archived repaired form.
    known=pd.read_parquet(MARKET/'dividends.parquet')
    for col in ['ex_date','declare_date']:dist[col]=pd.to_datetime(dist[col]);known[col]=pd.to_datetime(known[col])
    keys=['secid','ex_date','declare_date']
    patched=known.loc[known.secid.isin(request.secid),dist.columns]
    # Append only genuinely additional/changed archived cash events; otherwise
    # retain the complete raw list so unsupported distributions stay flagged.
    for event in patched.to_dict('records'):
        mask=dist.secid.eq(event['secid'])&dist.ex_date.eq(event['ex_date'])&dist.declare_date.eq(event['declare_date'])
        if mask.any():
            for key,value in event.items():dist.loc[mask,key]=value
        else:dist=pd.concat([dist,pd.DataFrame([event])],ignore_index=True)
    fitted=fit_quotes(q,stocks,zero,dist)
    fitted['delta']=fitted.bs_delta;fitted['vega']=fitted.bs_vega
    control=set(request.loc[~request.missing,'optionid'])
    with sqlite3.connect(f'file:{OLD_DB.as_posix()}?mode=ro',uri=True) as old:
        stored=[]
        for oid in control:
            for (payload,) in old.execute('SELECT payload FROM quotes WHERE optionid=?',(int(oid),)):
                stored.append(json.loads(payload))
    original=pd.DataFrame(stored);original['date']=pd.to_datetime(original.date)
    comparison=fitted.loc[fitted.optionid.isin(control)].merge(original,on=['date','optionid'],suffixes=('_new','_old'),validate='one_to_one')
    errors={}
    for column in ['bid','ask','strike','bs_rate','known_dividend_pv','bs_iv','delta','vega']:
        a=pd.to_numeric(comparison[column+'_new'],errors='coerce').to_numpy(float)
        b=pd.to_numeric(comparison[column+'_old'],errors='coerce').to_numpy(float)
        if not np.array_equal(np.isfinite(a),np.isfinite(b)):raise ValueError(f'Control availability mismatch: {column}')
        good=np.isfinite(a);error=float(np.max(abs(a[good]-b[good]),initial=0.));errors[column]=error
        np.testing.assert_allclose(a,b,atol=1e-8,rtol=1e-8,equal_nan=True,err_msg=column)
    added=fitted.loc[fitted.optionid.isin(set(request.loc[request.missing,'optionid']))].copy()
    added.to_parquet(OUT/'supplemental_quotes.parquet',index=False)
    if not (OUT/'quotes.sqlite').exists():shutil.copyfile(OLD_DB,OUT/'quotes.sqlite')
    with sqlite3.connect(OUT/'quotes.sqlite') as db:
        records=[]
        for row in added.to_dict('records'):
            for key,value in row.items():
                if isinstance(value,pd.Timestamp):row[key]=value.isoformat()
                elif pd.isna(value):row[key]=None
            records.append((str(pd.Timestamp(row['date']).date()),int(row['optionid']),json.dumps(row)))
        db.executemany('INSERT OR REPLACE INTO quotes VALUES(?,?,?)',records);db.commit()
    result=dict(status='complete',new_contracts=int(added.optionid.nunique()),new_quote_rows=len(added),
        controls=len(control),control_quote_comparisons=len(comparison),control_max_errors=errors,
        invalid_status_counts=added.bs_status.value_counts().to_dict(),existing_quotes_preserved=True)
    write(REPORT/'quote_restoration.json',result);print(json.dumps(result),flush=True)
    budget.rest()


def verify(budget):
    """Audit requested day keys independently from subsequent trade admission."""
    cal=pd.read_parquet(MARKET/'rates.parquet').index
    with sqlite3.connect(f'file:{(OUT/"quotes.sqlite").as_posix()}?mode=ro',uri=True) as db:
        keys={(pd.Timestamp(date),int(oid)) for date,oid in db.execute('SELECT date,optionid FROM quotes')}
        ids={oid for _,oid in keys}
        db.execute('ATTACH DATABASE ? AS original',(str(OLD_DB),))
        changed=db.execute('SELECT COUNT(*) FROM original.quotes o LEFT JOIN quotes n ON o.date=n.date AND o.optionid=n.optionid WHERE n.payload IS NULL OR n.payload<>o.payload').fetchone()[0]
        assert changed==0
    requests=pd.read_csv(OUT/'quote_requests.csv',parse_dates=['start','end'])
    new_ids=set(requests.loc[requests.missing,'optionid'])
    records=[];absent=set()
    for tail in GRID:
        g=pd.read_parquet(OUT/f'proposals_{round(tail*100):02d}.parquet')
        signal=set();entry=set();history=set()
        for row in g.itertuples():
            next_date=cal[cal.get_loc(row.date)+1]
            for suffix,sign in [('a',1),('b',-1)]:
                oid=int(getattr(row,'optionid_'+suffix));assert oid in ids
                end=row.date+pd.Timedelta(days=int(round(row.dte_mean+sign*row.dte_gap/2)))
                signal.add((row.date,oid));entry.add((next_date,oid))
                history.update((date,oid) for date in cal[(cal>=row.date)&(cal<end)])
        missing=history-keys;absent.update(missing)
        records.append(dict(tail=tail,proposals=len(g),both_contract_ids_available=len(g),
            required_signal_keys=len(signal),missing_signal_keys=len(signal-keys),
            required_next_session_keys=len(entry),missing_next_session_keys=len(entry-keys),
            required_signal_through_pre_expiry_keys=len(history),missing_history_keys=len(missing),
            missing_history_keys_new_ids=sum(oid in new_ids for _,oid in missing)))
        budget.rest()
    pd.DataFrame(records).to_csv(REPORT/'restored_day_coverage.csv',index=False)
    pd.DataFrame(sorted(absent),columns=['date','optionid']).to_csv(REPORT/'absent_vendor_day_keys.csv',index=False)
    write(REPORT/'restored_day_audit.json',dict(status='verified',original_payloads_changed=changed,
        total_quote_keys=len(keys),total_contracts=len(ids),coverage=records,
        original_query_scope='run_p25_memory_ablation.quotes streams full yearly native history filtered only by optionid; uniform_smile_data.compact also retains yearly contract history, without proposal-date clipping.',
        supplemental_query_scope='Every requested missing contract queried inclusively from first proposal signal to contract expiry, unfiltered by quote quality.',
        missing_keys='Absent vendor observations remain absent. No forward-fill, synthetic quote, or proposal exclusion is applied. Funded engine decides entry validity and must resolve actual held positions.'))
    print(json.dumps(records),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['audit','download','build','verify'],required=True)
    args=parser.parse_args();budget=Budget()
    globals()[args.phase](budget)


if __name__=='__main__':main()
