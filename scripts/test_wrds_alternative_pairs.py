"""Other WRDS families on fixed pair outcomes; bounded, serial and resumable."""
import argparse
from collections import deque
from contextlib import closing
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import time

import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import spearmanr
from scripts.test_wrds_pair_signals import (availability, interval_ids, match_states,
    statistics, timestamp_columns, write_json)

BASE=Path('artifacts/contracts/wrds_pair_signals')
OUT=Path('artifacts/contracts/wrds_alternative_pairs')
REPORT=Path('reports/wrds_alternative_pairs')
STATUS={}

# Small, fixed and transparent lexical proxies, not an optimized sentiment model.
NEGATIVE=set('adverse challenged challenges challenging decline declined declines declining deteriorate deteriorated deterioration difficult difficulties disruption disruptions headwind headwinds loss losses negative pressure pressures risk risks weak weaker weakness'.split())
UNCERTAIN=set('uncertain uncertainty uncertainties unpredictable volatility volatile risk risks may might could perhaps possible possibly potential anticipate anticipated expect expected expectations estimate estimates approximately'.split())
TOKEN=re.compile(r'[a-z]+')


def load_status():
    global STATUS
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    STATUS=json.loads((OUT/'fetch_status.json').read_text()) if (OUT/'fetch_status.json').exists() else {}


def bounded(db,name,statement,params=(),cap=150000):
    path=OUT/(name+'.parquet')
    if path.exists():return pd.read_parquet(path)
    started=time.monotonic()
    try:
        with db.cursor() as cursor:
            cursor.execute(statement,params)
            values=cursor.fetchmany(cap+1)
            if len(values)>cap:raise MemoryError(f'Row cap {cap} exceeded')
            frame=pd.DataFrame(values,columns=[c.name for c in cursor.description])
        frame.to_parquet(path,index=False)
        STATUS[name]=dict(status='complete',rows=len(frame),seconds=round(time.monotonic()-started,2))
        print(f'{name}: {len(frame):,} rows',flush=True)
    except (psycopg2.Error,MemoryError) as exc:
        STATUS[name]=dict(status='unavailable',sqlstate=getattr(exc,'pgcode',None),
                         message=getattr(getattr(exc,'diag',None),'message_primary',None) or str(exc))
        frame=None;print(f'{name}: {STATUS[name]}',flush=True)
    write_json(OUT/'fetch_status.json',STATUS)
    return frame


def fetch():
    load_status()
    companies=pd.read_parquet(BASE/'company_links.parquet')
    links=pd.read_parquet(BASE/'option_links.parquet')
    ciks=sorted({str(x).strip().zfill(10) for x in companies.cik.dropna()})
    cik_forms=sorted(set(ciks+[str(int(x)) for x in ciks]))
    gvkeys=sorted({str(x).strip().zfill(6) for x in companies.gvkey.dropna()})
    gvkey_forms=sorted(set(gvkeys+[str(int(x)) for x in gvkeys]))
    permnos=sorted(map(int,links.permno.dropna().unique()))
    with closing(psycopg2.connect(host='wrds-pgdata.wharton.upenn.edu',port=9737,
                 dbname='wrds',user=os.getenv('WRDS_USERNAME',getpass.getuser()),
                 sslmode='require',connect_timeout=20,
                 application_name='bounded_alternative_pair_screen')) as db:
        db.set_session(readonly=True,autocommit=True)
        with db.cursor() as c:
            c.execute("SET statement_timeout='20s'");c.execute("SET lock_timeout='2s'")
            c.execute("SELECT current_setting('transaction_read_only')");assert c.fetchone()[0]=='on'
        mapping=bounded(db,'revelio_mapping',"""
          SELECT rcid,cik,gvkey,company,ticker,ultimate_parent_rcid
          FROM revelio_common.company_mapping WHERE cik=ANY(%s)
        """,(cik_forms,),cap=10000)
        if mapping is not None and len(mapping):
            rcids=sorted(map(int,mapping.rcid.unique()))
            bounded(db,'review_months',"""
              SELECT rcid,date_trunc('month',review_date)::date AS month,
                COUNT(DISTINCT review_id) AS reviews,
                COUNT(rating_overall) AS overall_n,SUM(rating_overall) AS overall_sum,
                COUNT(rating_senior_leadership) AS leadership_n,SUM(rating_senior_leadership) AS leadership_sum,
                COUNT(CASE WHEN lower(rating_business_outlook) IN ('positive','neutral','negative') THEN 1 END) AS outlook_n,
                SUM(CASE lower(rating_business_outlook) WHEN 'positive' THEN 1.0 WHEN 'neutral' THEN 0.0 WHEN 'negative' THEN -1.0 END) AS outlook_sum
              FROM revelio_sentiment.sentiment_individual_reviews
              WHERE rcid=ANY(%s) AND review_date BETWEEN '2016-01-01' AND '2025-07-10'
              GROUP BY rcid,date_trunc('month',review_date)::date
            """,(rcids,),cap=20000)
            bounded(db,'layoff_notices',"""
              SELECT rcid,notice_date,SUM(num_employees) AS employees,COUNT(*) AS notices
              FROM revelio_layoffs.layoffs WHERE rcid=ANY(%s)
                AND notice_date BETWEEN '2016-01-01' AND '2025-07-10'
                AND num_employees>=0 GROUP BY rcid,notice_date
            """,(rcids,),cap=20000)
            bounded(db,'workforce_months',"""
              SELECT rcid,datemonth AS month,SUM(scaled_count) AS employees,
                     SUM(scaled_external_inflow) AS inflow,SUM(scaled_external_outflow) AS outflow
              FROM revelio_workforce_dynamics.workforce_dynamics_geo
              WHERE rcid=ANY(%s) AND country='United States'
                AND datemonth BETWEEN '2016-01-01' AND '2025-07-10'
              GROUP BY rcid,datemonth
            """,(rcids,),cap=20000)
            bounded(db,'posting_months',"""
              SELECT rcid,date_trunc('month',post_date)::date AS month,
                     COUNT(*) AS postings
              FROM revelio_job_postings.postings_cosmos WHERE rcid=ANY(%s)
                AND country='United States' AND post_date BETWEEN '2016-01-01' AND '2025-07-10'
              GROUP BY rcid,date_trunc('month',post_date)::date
            """,(rcids,),cap=20000)
        bounded(db,'insider_transactions',"""
          SELECT n.issuercik AS cik,n.accession,n.transactiondate AS transaction_date,
                 n.transactioncode AS code,n.transactionshares AS shares,
                 n.transactionpricepershare AS price,n.securitytitle,
                 d.fdate,d.secadate,d.secpdate,d.aff10b5one
          FROM wrdssec_insiders.nonderivatives n JOIN wrdssec_insiders.document_info d
            ON n.accession=d.accession AND n.issuercik=d.issuercik
          WHERE n.issuercik=ANY(%s) AND d.documenttype='4'
            AND n.transactioncode IN ('P','S') AND n.transactionshares>0
            AND n.transactionpricepershare>0
            AND (d.aff10b5one IS NULL OR d.aff10b5one<>1)
            AND d.fdate BETWEEN '2016-01-01' AND '2025-07-10'
        """,(cik_forms,),cap=150000)
        ciq=bounded(db,'ciq_links',"""
          SELECT gvkey,relatedcompanyid,symbolstartdate,symbolenddate
          FROM ciq_common.ciqgvkeyiid WHERE gvkey=ANY(%s)
        """,(gvkey_forms,),cap=10000)
        if ciq is not None and len(ciq):
            ids=sorted(map(int,ciq.relatedcompanyid.dropna().unique()))
            calls=bounded(db,'transcript_metadata',"""
              SELECT companyid,keydevid,transcriptid,keydeveventtypename,
                companyname,mostimportantdateutc AS event_date,
                transcriptcollectiontypename AS collection,
                transcriptcreationdate_utc AS creation_date,
                transcriptcreationtime_utc AS creation_time
              FROM ciq_transcripts.wrds_transcript_detail WHERE companyid=ANY(%s)
                AND mostimportantdateutc BETWEEN '2016-01-01' AND '2025-07-10'
                AND keydeveventtypename ILIKE '%%Earnings%%'
            """,(ids,),cap=30000)
            speaker=bounded(db,'speaker_types','SELECT * FROM ciq_transcripts.ciqtranscriptspeakertype',cap=50)
            component=bounded(db,'component_types','SELECT * FROM ciq_transcripts.ciqtranscriptcomponenttype',cap=50)
            if calls is not None:
                print('Transcript collections:',calls.collection.value_counts().to_dict(),flush=True)
            if speaker is not None:print('Speaker types:',speaker.to_dict('records'),flush=True)
            if component is not None:print('Component types:',component.to_dict('records'),flush=True)
        bounded(db,'liquidity_months',"""
          SELECT permno,year_month_end_date AS month,bbd,wbbd
          FROM contrib_liquidity_taq.bbd WHERE permno=ANY(%s)
            AND year_month_end_date BETWEEN '2016-01-01' AND '2025-07-10'
        """,(permnos,),cap=20000)
        bounded(db,'institutional_liquidity',"""
          SELECT permno,year_month_end_date AS month,ilmt_1m
          FROM contrib_liquidity_taq.ilc WHERE permno=ANY(%s)
            AND year_month_end_date BETWEEN '2016-01-01' AND '2025-07-10'
        """,(permnos,),cap=20000)
        bounded(db,'credit_months',"""
          SELECT permno,date AS month,COUNT(*) AS bonds,
            SUM(yield_spread*mv)/NULLIF(SUM(mv),0) AS credit_spread,
            SUM(ret_texc*mv)/NULLIF(SUM(mv),0) AS credit_return
          FROM contrib_corporate_bond_returns.bonds WHERE permno=ANY(%s)
            AND date BETWEEN '2016-01-01' AND '2025-07-10'
            AND mv>0 AND yield_spread IS NOT NULL AND ret_texc IS NOT NULL
          GROUP BY permno,date
        """,(permnos,),cap=20000)
    write_json(OUT/'fetch_manifest.json',dict(fetched_at=datetime.now(timezone.utc).isoformat(),
       ciks=len(ciks),permnos=len(permnos),read_only=True,statement_timeout=20))


def text_counts(text):
    tokens=TOKEN.findall((text or '').lower())
    return len(tokens),sum(t in NEGATIVE for t in tokens),sum(t in UNCERTAIN for t in tokens)


def batched_labor():
    load_status();mapping=pd.read_parquet(OUT/'revelio_mapping.parquet')
    mapping.cik=mapping.cik.map(lambda x:str(x).strip().zfill(10))
    counts=mapping.groupby('cik').rcid.transform('nunique')
    ids=sorted(map(int,mapping[counts.eq(1)].rcid.unique()))
    coverage=[]
    with closing(psycopg2.connect(host='wrds-pgdata.wharton.upenn.edu',port=9737,
                 dbname='wrds',user=os.getenv('WRDS_USERNAME',getpass.getuser()),
                 sslmode='require',connect_timeout=20,
                 application_name='bounded_labor_pair_batches')) as db:
        db.set_session(readonly=True,autocommit=True)
        with db.cursor() as c:c.execute("SET statement_timeout='20s'");c.execute("SET lock_timeout='2s'")
        for family,size in [('postings',8),('workforce',4)]:
            pieces=[]
            for start in range(0,len(ids),size):
                group=ids[start:start+size];name=family+'_batch_'+hashlib.sha256(json.dumps(group).encode()).hexdigest()[:12]
                if family=='workforce':
                    query="""SELECT rcid,datemonth AS month,SUM(scaled_count) AS employees,
                       SUM(scaled_external_inflow) AS inflow,SUM(scaled_external_outflow) AS outflow
                       FROM revelio_workforce_dynamics.workforce_dynamics_geo
                       WHERE rcid=ANY(%s) AND country='United States'
                         AND datemonth BETWEEN '2016-01-01' AND '2025-07-10'
                       GROUP BY rcid,datemonth"""
                else:
                    query="""SELECT rcid,date_trunc('month',post_date)::date AS month,COUNT(*) AS postings
                       FROM revelio_job_postings.postings_cosmos WHERE rcid=ANY(%s)
                         AND country='United States' AND post_date BETWEEN '2016-01-01' AND '2025-07-10'
                       GROUP BY rcid,date_trunc('month',post_date)::date"""
                result=bounded(db,name,query,(group,),cap=1000)
                coverage.append(dict(family=family,batch=name,requested_rcids=group,
                     status='complete' if result is not None else 'unavailable',
                     observed_rcids=sorted(map(int,result.rcid.unique())) if result is not None and len(result) else []))
                if result is not None and len(result):pieces.append(result)
                print(f'{family}: batch {start//size+1}/{(len(ids)+size-1)//size}',flush=True)
                write_json(OUT/'labor_batch_coverage.json',coverage)
            # Preserve the successful initial pilot if its new batch timed out.
            pilot=OUT/('workforce_pilot.parquet' if family=='workforce' else 'posting_pilot.parquet')
            if pilot.exists():pieces.append(pd.read_parquet(pilot))
            if pieces:
                combined=pd.concat(pieces,ignore_index=True)
                combined.rcid=combined.rcid.map(int)
                combined=combined.drop_duplicates(['rcid','month'],keep='first')
                combined.to_parquet(OUT/('workforce_months.parquet' if family=='workforce' else 'posting_months.parquet'),index=False)


def assemble_labor():
    """Finalize saved batches after interruption, without further network reads."""
    for family,output,pilot in [('workforce','workforce_months','workforce_pilot'),
                                ('postings','posting_months','posting_pilot')]:
        paths=sorted(OUT.glob(family+'_batch_*.parquet'))
        if (OUT/(pilot+'.parquet')).exists():paths.append(OUT/(pilot+'.parquet'))
        if not paths:continue
        frame=pd.concat([pd.read_parquet(path) for path in paths],ignore_index=True)
        frame.rcid=frame.rcid.map(int)
        frame=frame.drop_duplicates(['rcid','month'],keep='first')
        frame.to_parquet(OUT/(output+'.parquet'),index=False)
        print(f'Assembled {family}: {frame.rcid.nunique()} companies, {len(frame):,} month rows',flush=True)


def transcripts_and_pilot():
    load_status()
    calls=pd.read_parquet(OUT/'transcript_metadata.parquet')
    calls=calls[calls.creation_date.notna()].sort_values(['companyid','keydevid','creation_date','creation_time','transcriptid'])
    calls=calls.drop_duplicates(['companyid','keydevid'],keep='first')
    calls['available']=availability(calls,['event_date','creation_date'])
    calls=calls[calls.available<='2025-07-10'].copy()
    calls.to_parquet(OUT/'selected_calls.parquet',index=False)
    ids=sorted(map(int,calls.transcriptid.unique()));pieces=[]
    print(f'First available transcript versions: {len(ids):,}',flush=True)
    with closing(psycopg2.connect(host='wrds-pgdata.wharton.upenn.edu',port=9737,
                 dbname='wrds',user=os.getenv('WRDS_USERNAME',getpass.getuser()),
                 sslmode='require',connect_timeout=20,
                 application_name='bounded_transcript_pair_screen')) as db:
        db.set_session(readonly=True,autocommit=True)
        with db.cursor() as c:c.execute("SET statement_timeout='20s'");c.execute("SET lock_timeout='2s'")
        mapping=pd.read_parquet(OUT/'revelio_mapping.parquet')
        pilots=sorted(map(int,mapping[mapping.cik.map(lambda x:str(x).strip().zfill(10)).isin(['0000002488','0001045810'])].rcid.unique()))
        if pilots:
            if not (OUT/'workforce_months.parquet').exists():
                bounded(db,'workforce_pilot',"""
                  SELECT rcid,datemonth AS month,SUM(scaled_count) AS employees,
                    SUM(scaled_external_inflow) AS inflow,SUM(scaled_external_outflow) AS outflow
                  FROM revelio_workforce_dynamics.workforce_dynamics_geo
                  WHERE rcid=ANY(%s) AND country='United States'
                    AND datemonth BETWEEN '2016-01-01' AND '2025-07-10'
                  GROUP BY rcid,datemonth
                """,(pilots,),cap=1000)
            if not (OUT/'posting_months.parquet').exists():
                bounded(db,'posting_pilot',"""
                  SELECT rcid,date_trunc('month',post_date)::date AS month,COUNT(*) AS postings
                  FROM revelio_job_postings.postings_cosmos WHERE rcid=ANY(%s)
                    AND country='United States' AND post_date BETWEEN '2016-01-01' AND '2025-07-10'
                  GROUP BY rcid,date_trunc('month',post_date)::date
                """,(pilots,),cap=1000)
        for start in range(0,len(ids),100):
            chunk=ids[start:start+100]
            name='call_counts_'+hashlib.sha256(json.dumps(chunk).encode()).hexdigest()[:12]
            path=OUT/(name+'.parquet')
            if path.exists():pieces.append(pd.read_parquet(path));continue
            records={};totalchars=totalrows=0;failed=None;began=time.monotonic()
            try:
                with db.cursor() as c:
                    c.execute("""
                      SELECT t.transcriptid,t.transcriptcomponenttypeid,
                             p.speakertypeid,t.componenttext
                      FROM ciq_transcripts.ciqtranscriptcomponent t
                      LEFT JOIN ciq_transcripts.ciqtranscriptperson p
                        ON t.transcriptpersonid=p.transcriptpersonid
                      WHERE t.transcriptid=ANY(%s)
                    """,(chunk,))
                    while True:
                        values=c.fetchmany(300)
                        if not values:break
                        for tid,component,speaker,body in values:
                            totalrows+=1;totalchars+=len(body or '')
                            if totalrows>25000 or totalchars>12000000:
                                raise MemoryError('Transcript batch exceeds row/text cap; batch excluded')
                            record=records.setdefault(tid,dict(transcriptid=tid,exec_words=0,exec_negative=0,
                              exec_uncertain=0,prepared_words=0,prepared_uncertain=0,
                              answer_words=0,answer_uncertain=0,analyst_words=0,analyst_negative=0))
                            n,negative,uncertain=text_counts(body)
                            if speaker==2:
                                record['exec_words']+=n;record['exec_negative']+=negative;record['exec_uncertain']+=uncertain
                                if component==2:record['prepared_words']+=n;record['prepared_uncertain']+=uncertain
                                if component==4:record['answer_words']+=n;record['answer_uncertain']+=uncertain
                            if speaker==3 and component==3:
                                record['analyst_words']+=n;record['analyst_negative']+=negative
                frame=pd.DataFrame(list(records.values()))
                frame.to_parquet(path,index=False);pieces.append(frame)
                STATUS[name]=dict(status='complete',transcripts=len(frame),components=totalrows,characters=totalchars,
                                  seconds=round(time.monotonic()-began,2))
                print(f'Transcript batch {start//100+1}/{(len(ids)+99)//100}: {len(frame)} scored',flush=True)
            except (psycopg2.Error,MemoryError) as exc:
                failed=str(exc);STATUS[name]=dict(status='unavailable',message=failed)
                print(f'Transcript batch {start//100+1}: {failed}',flush=True)
            write_json(OUT/'fetch_status.json',STATUS)
    if pieces:pd.concat(pieces,ignore_index=True).to_parquet(OUT/'transcript_counts.parquet',index=False)


def monthly_date(month,lag):
    return pd.to_datetime(month)+pd.offsets.MonthEnd(0)+pd.Timedelta(days=lag)+pd.offsets.BDay(1)


def numeric_ids(values):
    # Decimal IDs from psycopg2 compare differently with NumPy integer scalars.
    return np.asarray([int(x) if pd.notna(x) else None for x in values],dtype=object)


def monthly_grid(frame,identifier,fields):
    frames=[]
    for key,g in frame.groupby(identifier):
        g=g.copy();g['month']=pd.to_datetime(g.month)
        assert not g.month.duplicated().any()
        months=pd.date_range(g.month.min(),g.month.max(),freq='MS')
        h=g.set_index('month')[fields].reindex(months).fillna(0)
        h[identifier]=key;h.index.name='month';frames.append(h.reset_index())
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()


def build_review_states(raw):
    fields=['overall_n','overall_sum','leadership_n','leadership_sum','outlook_n','outlook_sum']
    frames=[]
    for rcid,g in monthly_grid(raw,'rcid',fields).groupby('rcid'):
        g=g.sort_values('month').copy();g['available']=monthly_date(g.month,15)
        values={}
        for kind in ['overall','leadership','outlook']:
            count=g[kind+'_n'].rolling(3,min_periods=3).sum()
            values[kind]=(g[kind+'_sum'].rolling(3,min_periods=3).sum()/count).where(count>=20)
        g['review_rating_change']=values['overall']-values['overall'].shift(3)
        g['review_outlook_change']=values['outlook']-values['outlook'].shift(3)
        g['review_leadership']=values['leadership']
        frames.append(g[['rcid','available','review_rating_change','review_outlook_change','review_leadership']])
    return pd.concat(frames,ignore_index=True)


def review_states():
    return build_review_states(pd.read_parquet(OUT/'review_months.parquet'))


def rolling_events(events,keys,dates,identifier,fields,days,end):
    """Cumulative event sums; no future events or post-coverage zeros."""
    frames=[]
    for key in keys:
        g=events[events[identifier].eq(key)].sort_values('available')
        idx=pd.DatetimeIndex(g.available)
        right=idx.searchsorted(dates,side='right')
        left=idx.searchsorted(dates-pd.Timedelta(days=days),side='right')
        h=pd.DataFrame({identifier:key,'available':dates})
        for col in fields:
            cumulative=np.r_[0,np.cumsum(pd.to_numeric(g[col],errors='coerce').fillna(0).to_numpy(float))]
            h[col]=cumulative[right]-cumulative[left]
            h.loc[h.available>end,col]=np.nan
        frames.append(h)
    return pd.concat(frames,ignore_index=True)


def transcript_states():
    calls=pd.read_parquet(OUT/'selected_calls.parquet')
    counts=pd.read_parquet(OUT/'transcript_counts.parquet')
    assert not counts.transcriptid.duplicated().any()
    f=calls.merge(counts,on='transcriptid',how='inner',validate='many_to_one')
    f=f[f.exec_words>=1000].sort_values(['companyid','available','transcriptid']).copy()
    f['call_uncertainty']=1000*f.exec_uncertain/f.exec_words
    negative=1000*f.exec_negative/f.exec_words
    f['call_negative_change']=negative.groupby(f.companyid).diff()
    lag=f.groupby('companyid').available.diff().dt.days
    f['call_negative_change']=f.call_negative_change.where(lag.between(30,200))
    qa=1000*f.answer_uncertain/f.answer_words-1000*f.prepared_uncertain/f.prepared_words
    f['call_qa_uncertainty']=qa.where((f.answer_words>=100)&(f.prepared_words>=300))
    f['call_analyst_negative']=(1000*f.analyst_negative/f.analyst_words).where(f.analyst_words>=100)
    return f[['companyid','available','call_uncertainty','call_negative_change','call_qa_uncertainty','call_analyst_negative']]


def other_monthly():
    result=[]
    for filename in ['workforce_months','workforce_pilot']:
        if not (OUT/(filename+'.parquet')).exists():continue
        w=pd.read_parquet(OUT/(filename+'.parquet'));frames=[]
        for rcid,g in w.groupby('rcid'):
            g=g.sort_values('month').copy();g['available']=monthly_date(g.month,15)
            g['month']=pd.to_datetime(g.month)
            consecutive=(g.month.dt.year*12+g.month.dt.month).diff().rolling(2,min_periods=2).sum().eq(2)
            denominator=g.employees.rolling(3,min_periods=3).mean().where(lambda x:x>0)
            g['workforce_net_hiring']=((g.inflow-g.outflow).rolling(3,min_periods=3).sum()/denominator).where(consecutive)
            g['workforce_attrition']=(g.outflow.rolling(3,min_periods=3).sum()/denominator).where(consecutive)
            frames.append(g[['rcid','available','workforce_net_hiring','workforce_attrition']])
        result.append(('workforce','rcid',pd.concat(frames,ignore_index=True),['workforce_net_hiring','workforce_attrition'],60,'reconstructed_history'))
        break
    for filename in ['posting_months','posting_pilot']:
        if not (OUT/(filename+'.parquet')).exists():continue
        p=pd.read_parquet(OUT/(filename+'.parquet'));frames=[]
        for rcid,g in monthly_grid(p,'rcid',['postings']).groupby('rcid'):
            g=g.sort_values('month').copy();g['available']=monthly_date(g.month,15)
            total=g.postings.rolling(3,min_periods=3).sum();prior=total.shift(3)
            g['posting_growth']=(np.log1p(total)-np.log1p(prior)).where((total>=20)&(prior>=20))
            frames.append(g[['rcid','available','posting_growth']])
        result.append(('postings','rcid',pd.concat(frames,ignore_index=True),['posting_growth'],60,'reconstructed_history'))
        break
    for filename,cols,family in [('institutional_liquidity',['ilmt_1m'],'liquidity'),
                                ('liquidity_months',['bbd'],'price_impact'),
                                ('credit_months',['credit_spread','credit_return'],'credit')]:
        if not (OUT/(filename+'.parquet')).exists():continue
        f=pd.read_parquet(OUT/(filename+'.parquet'));frames=[]
        for permno,g in f.groupby('permno'):
            g=g.sort_values('month').copy();g['available']=monthly_date(g.month,45)
            if family=='credit':
                month=pd.to_datetime(g.month);consecutive=(month.dt.year*12+month.dt.month).diff().eq(1)
                g['credit_spread_change']=g.credit_spread.diff().where(consecutive)
                use=['credit_spread_change','credit_return']
            else:use=cols
            frames.append(g[['permno','available',*use]])
        result.append((family,'permno',pd.concat(frames,ignore_index=True),use,60,'historical_publication_unverified'))
    return result


def prepare():
    load_status();data=pd.read_parquet(BASE/'aligned.parquet')
    dates=pd.DatetimeIndex(sorted(data.date.unique()))
    weekly=pd.date_range('2017-01-06','2025-07-10',freq='W-FRI')
    mapping=pd.read_parquet(OUT/'revelio_mapping.parquet');mapping.cik=mapping.cik.map(lambda x:str(x).strip().zfill(10))
    # Current Revelio mappings cannot prove historical corporate boundaries.
    mapping.rcid=mapping.rcid.map(int);rcids=sorted(mapping.rcid.unique())
    datasets=[('reviews','rcid',review_states(),['review_rating_change','review_outlook_change','review_leadership'],60,'dated_reviews_no_archived_vintages')]
    l=pd.read_parquet(OUT/'layoff_notices.parquet');l.rcid=l.rcid.map(int)
    l['available']=availability(l,['notice_date'])
    l=rolling_events(l,rcids,weekly,'rcid',['employees'],90,pd.Timestamp('2025-07-10'))
    l['layoff_intensity']=np.log1p(l.employees)
    datasets.append(('layoffs','rcid',l[['rcid','available','layoff_intensity']],['layoff_intensity'],14,'notice_date_proxy'))
    insider=pd.read_parquet(OUT/'insider_transactions.parquet')
    insider=insider[insider.securitytitle.fillna('').str.contains('common|ordinary',case=False)&
                    ~insider.securitytitle.fillna('').str.contains('preferred',case=False)].copy()
    insider['available']=availability(insider,['fdate','secadate','secpdate'])
    insider=insider[pd.to_datetime(insider.transaction_date)<=insider.available].copy()
    insider.cik=insider.cik.map(lambda x:str(x).strip().zfill(10))
    dollars=insider.shares*insider.price
    insider['buy_dollars']=dollars.where(insider.code.eq('P'),0.)
    insider['sell_dollars']=dollars.where(insider.code.eq('S'),0.)
    insider['buy_filing']=insider.code.eq('P').astype(float)
    event=insider.groupby(['cik','accession','available'],as_index=False).agg(
         buy_dollars=('buy_dollars','sum'),sell_dollars=('sell_dollars','sum'),buy_filing=('buy_filing','max'))
    end=insider.available.max()
    company=pd.read_parquet(BASE/'company_links.parquet')
    allciks=sorted({str(x).strip().zfill(10) for x in company.cik.dropna()})
    e=rolling_events(event,allciks,weekly,'cik',['buy_dollars','sell_dollars','buy_filing'],30,end)
    total=e.buy_dollars+e.sell_dollars
    e['insider_imbalance']=((e.buy_dollars-e.sell_dollars)/total.where(total>0)).where(total.ne(0),0.)
    e['insider_buy_activity']=np.log1p(e.buy_filing)
    datasets.append(('insiders','cik',e[['cik','available','insider_imbalance','insider_buy_activity']],
                     ['insider_imbalance','insider_buy_activity'],14,'SEC_acceptance_proxy'))
    datasets.append(('transcripts','companyid',transcript_states(),
       ['call_uncertainty','call_negative_change','call_qa_uncertainty','call_analyst_negative'],180,'first_version_creation_proxy'))
    datasets+=other_monthly()
    normalized=[]
    for family,identifier,states,cols,age,timing in datasets:
        states=states.copy()
        if identifier in ['rcid','companyid','permno']:
            states[identifier]=pd.Series(numeric_ids(states[identifier]),index=states.index).astype(np.int64)
        normalized.append((family,identifier,states,cols,age,timing))
    datasets=normalized
    o=timestamp_columns(pd.read_parquet(BASE/'option_links.parquet'),['sdate','edate']);o=o[o.score<=1].copy()
    o.edate=o.edate.fillna(pd.Timestamp('2026-01-01'))
    c=timestamp_columns(company,['linkdt','linkenddt']);c.linkdt=c.linkdt.fillna(pd.Timestamp('1900-01-01'))
    c.linkenddt=c.linkenddt.fillna(pd.Timestamp('2026-01-01'));c.cik=c.cik.map(lambda x:str(x).strip().zfill(10))
    c.gvkey=c.gvkey.map(lambda x:str(x).strip().zfill(6))
    ciq=timestamp_columns(pd.read_parquet(OUT/'ciq_links.parquet'),['symbolstartdate','symbolenddate'])
    ciq.gvkey=ciq.gvkey.map(lambda x:str(x).strip().zfill(6))
    ciq.symbolstartdate=ciq.symbolstartdate.fillna(pd.Timestamp('1900-01-01'))
    ciq.symbolenddate=ciq.symbolenddate.fillna(pd.Timestamp('2026-01-01'))
    definitions={x['pair']:x for x in json.loads(Path('artifacts/contracts/option_flex/inputs.json').read_text())['definitions']}
    assets=sorted({sid for pair in data.pair.unique() for sid in definitions[pair]['secids']})
    fields=[col for _,_,_,cols,_,_ in datasets for col in cols];legs={};coverage=[]
    for sid in assets:
        perm=interval_ids(dates,o[o.secid.eq(sid)],None,'sdate','edate','permno')
        cik=np.full(len(dates),None,object);gvkey=cik.copy();rcid=cik.copy();companyid=cik.copy()
        for value in pd.Series(perm).dropna().unique():
            mask=perm==value;own=c[c.permno.eq(value)]
            cik[mask]=interval_ids(dates[mask],own,None,'linkdt','linkenddt','cik')
            gvkey[mask]=interval_ids(dates[mask],own,None,'linkdt','linkenddt','gvkey')
        for value in pd.Series(cik).dropna().unique():
            hit=mapping[mapping.cik.eq(value)].rcid.unique()
            if len(hit)==1:rcid[cik==value]=hit[0]
        for value in pd.Series(gvkey).dropna().unique():
            mask=gvkey==value
            companyid[mask]=interval_ids(dates[mask],ciq[ciq.gvkey.eq(value)],None,
                                        'symbolstartdate','symbolenddate','relatedcompanyid')
        ids={'permno':numeric_ids(perm),'cik':cik,'rcid':numeric_ids(rcid),
             'companyid':numeric_ids(companyid)};parts=[]
        for family,identifier,states,cols,age,timing in datasets:
            part=match_states(dates,ids[identifier],states,identifier,cols,age);parts.append(part)
            coverage.append(dict(secid=sid,family=family,mapped_dates=int(pd.notna(ids[identifier]).sum()),
                              observed_dates=int(part.notna().any(axis=1).sum())))
        legs[sid]=pd.concat(parts,axis=1)
    frames=[]
    for pair,g in data.groupby('pair',sort=True):
        g=g.copy();a,b=definitions[pair]['secids']
        x=legs[a].reindex(pd.DatetimeIndex(g.date));y=legs[b].reindex(pd.DatetimeIndex(g.date))
        for field in fields:
            g[field+'_gap']=(x[field]-y[field]).to_numpy()
            g[field+'_mean']=((x[field]+y[field])/2).to_numpy()
        frames.append(g)
    aligned=pd.concat(frames,ignore_index=True).sort_values(['date','pair'])
    pd.testing.assert_frame_equal(aligned[data.columns].reset_index(drop=True),data.sort_values(['date','pair']).reset_index(drop=True))
    aligned.to_parquet(OUT/'aligned.parquet',index=False)
    pd.DataFrame(coverage).to_csv(REPORT/'identifier_coverage.csv',index=False)
    for family,identifier,states,cols,age,timing in datasets:states.to_parquet(OUT/(family+'_states.parquet'),index=False)
    write_json(OUT/'features.json',dict(families={family:dict(identifier=identifier,features=cols,timing=timing,max_age_days=age)
               for family,identifier,states,cols,age,timing in datasets},
               pair_features=[field+ending for field in fields for ending in ['_gap','_mean']],
               insider_observed_end=str(end.date()),lexicon_negative=sorted(NEGATIVE),lexicon_uncertain=sorted(UNCERTAIN)))
    print(f'Aligned {len(aligned):,} opportunities, {len(datasets)} families, {len(fields)*2} features',flush=True)


def analyze():
    load_status();data=pd.read_parquet(OUT/'aligned.parquet')
    spec=json.loads((OUT/'features.json').read_text());summary=[];annual=[];pairs=[];periods=[];coverage=[]
    lookup={col+ending:family for family,d in spec['families'].items() for col in d['features'] for ending in ['_gap','_mean']}
    for feature in spec['pair_features']:
        target='directional_edge' if feature.endswith('_gap') else 'best_net';family=lookup[feature]
        for sample,g in [('full',data),('early_2018_2021',data[data.end<'2022-01-01']),
                         ('later_2022_2025',data[data.date>='2022-01-01']),('nonoverlap',data[data.nonoverlap])]:
            summary.append(dict(family=family,feature=feature,target=target,sample=sample,
                                **statistics(g,feature,target,bootstrap=sample=='full')))
        for year,g in data.groupby(data.date.dt.year):
            annual.append(dict(family=family,feature=feature,target=target,year=year,**statistics(g,feature,target)))
        for pair,g in data.groupby('pair'):
            h=g[np.isfinite(g[feature])&np.isfinite(g[target])]
            if len(h)>=20:
                rho=float(spearmanr(h[feature],h[target]).statistic) if h[feature].nunique()>1 else np.nan
                pairs.append(dict(family=family,feature=feature,target=target,pair=pair,label=g.label.iloc[0],
                                  observations=len(h),nonoverlap=int(h.nonoverlap.sum()),spearman=rho))
            for sample,v in [('early_2018_2021',g[g.end<'2022-01-01']),
                             ('later_2022_2025',g[g.date>='2022-01-01']),('nonoverlap',g[g.nonoverlap])]:
                v=v[np.isfinite(v[feature])&np.isfinite(v[target])]
                rho=float(spearmanr(v[feature],v[target]).statistic) if len(v)>=8 and v[feature].nunique()>1 else np.nan
                periods.append(dict(family=family,feature=feature,target=target,pair=pair,label=g.label.iloc[0],
                                    sample=sample,observations=len(v),spearman=rho))
        h=data[data[feature].notna()]
        coverage.append(dict(family=family,feature=feature,observations=len(h),pairs=h.pair.nunique(),
                             first=h.date.min(),last=h.date.max(),nonoverlap=int(h.nonoverlap.sum())))
    summary=pd.DataFrame(summary);summary.to_csv(REPORT/'correlations.csv',index=False)
    pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False)
    pd.DataFrame(pairs).to_csv(REPORT/'pairs.csv',index=False)
    pd.DataFrame(periods).to_csv(REPORT/'pair_periods.csv',index=False)
    pd.DataFrame(coverage).to_csv(REPORT/'coverage.csv',index=False)
    previous=json.loads((OUT/'analysis.json').read_text()) if (OUT/'analysis.json').exists() else {}
    write_json(OUT/'analysis.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),
      source_sha256=hashlib.sha256((BASE/'aligned.parquet').read_bytes()).hexdigest(),
      aligned_sha256=hashlib.sha256((OUT/'aligned.parquet').read_bytes()).hexdigest(),
      code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      features=spec,reference_promoted=False,policy_retrained=False,opportunities=len(data),pairs=data.pair.nunique(),
      feature_snapshot_hashes={family:hashlib.sha256((OUT/(family+'_states.parquet')).read_bytes()).hexdigest()
                              for family in spec['families']},
      peak_rss_bytes=max(previous.get('peak_rss_bytes',0),resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)))
    show=summary[summary['sample'].eq('full')]
    print(show[['family','feature','observations','pairs','within_pair_equal_weight_rank_corr',
                'partial_value_rank_corr','ci90_low','ci90_high']].to_string(index=False),flush=True)


def report():
    spec=json.loads((OUT/'features.json').read_text())
    manifest=json.loads((OUT/'analysis.json').read_text())
    s=pd.read_csv(REPORT/'correlations.csv');coverage=pd.read_csv(REPORT/'coverage.csv')
    batches=json.loads((OUT/'labor_batch_coverage.json').read_text()) if (OUT/'labor_batch_coverage.json').exists() else []
    lines=['# Other WRDS alternatives versus option-pair profitability','',
      'October 1, 2026. Extended the first correlation screen using the same 26,334 cached resolved ATM call/call opportunities over 128 pairs, January 2018–July 2025. Signal coverage varies by source. The option outcomes and existing SEC/I/B/E/S features were held fixed; no NN or RL policy was retrained.','',
      'These are exploratory fixed-feature diagnostics. Workforce and job-posting histories are reconstructed current-vintage data, not verified historical trading information. Dated reviews, insider filings and transcript versions also require publication/revision and identifier audits before a strategy backtest. No Sharpe or return improvement is inferred.','',
      '## Compact comparison','',
      'Representative directional features below are differences between the two legs. Negative correlation favors the reverse association; it does not imply either directional trade makes money. The employee-leadership and transcript-negative-change rows are promising-looking examples chosen after examining the full feature set, not independently validated selections.','',
      '| Source | Feature difference | Dates / pairs | Full correlation | Early | Later | Nonoverlap |',
      '|---|---|---:|---:|---:|---:|---:|']
    chosen=[('Reviews','review_leadership_gap'),('Layoffs','layoff_intensity_gap'),
            ('Insiders','insider_imbalance_gap'),('Transcripts','call_negative_change_gap'),
            ('Workforce, provisional','workforce_net_hiring_gap'),('Postings, provisional','posting_growth_gap'),
            ('Institutional liquidity','ilmt_1m_gap'),('Credit','credit_spread_change_gap')]
    def fmt(value):return '—' if pd.isna(value) else f'{value:+.3f}'
    for name,feature in chosen:
        d=s[s.feature.eq(feature)].set_index('sample');f=d.loc['full']
        values=[fmt(d.loc[sample,'within_pair_equal_weight_rank_corr']) for sample in ['full','early_2018_2021','later_2022_2025','nonoverlap']]
        lines.append(f'| {name} | {feature} | {int(f.observations):,} / {int(f.pairs)} | '+ ' | '.join(values)+' |')
    lines+=['','The table uses ranks within each pair and equal total weight per eligible pair. At least 20 observed rows are required per pair in each sample; consequently the nonoverlap results can use a much smaller and different pair set. Early outcomes must finish before January 2022, while later entries begin from January 2022. Later-year data are repeatedly explored research history, not an untouched holdout.','',
      '## Every tested feature','',
      '| Feature | Target | Dates / pairs | Full within-pair rank correlation | Partial IV-value control | 90% quarter-block interval |',
      '|---|---|---:|---:|---:|---|']
    for row in s[s['sample'].eq('full')].itertuples():
        target='Direction' if row.target=='directional_edge' else 'Best-direction payoff'
        interval='—' if pd.isna(row.ci90_low) else f'[{row.ci90_low:+.3f}, {row.ci90_high:+.3f}]'
        lines.append(f'| {row.feature} | {target} | {row.observations:,} / {row.pairs} | {fmt(row.within_pair_equal_weight_rank_corr)} | {fmt(row.partial_value_rank_corr)} | {interval} |')
    lines+=['','A direction target is `(long-first/short-second net payoff − reverse-direction net payoff)/2`. A best-direction target is the maximum of both realized payoffs and assumes hindsight about direction. It measures the size of a possible opportunity, not executable strategy profit. All dollar outcomes are normalized by initial mean-leg vega.','',
      '## Features and timing','',
      '- Reviews: three-month employee overall, business-outlook and senior-leadership ratings. Overall/outlook changes compare the latest three months with the preceding three. Require at least 20 applicable reviews in each three-month mean. Monthly scores become eligible one business day after month-end plus 15 days. Reviews include all countries. No personal reviewer data or review text was downloaded. Historical edits and current company mappings remain unresolved vintage risks.',
      '- Layoffs: log of one plus employees affected by notices in the trailing 90 days. Use notice date plus one business day, never the future effective layoff date. Weekly snapshots, with at most 14 days of carry. Zero means no observed notice in this feed, not proof of no layoffs anywhere. WARN notices have state/geographic coverage differences; notice date is a publication proxy, not a verified vendor receipt timestamp.',
      '- Insiders: trailing 30-day open-market purchase/sale dollar imbalance and log count of purchase filings. Non-derivative common/ordinary stock only; original Form 4, excluding amendments and filings explicitly flagged as 10b5-1. Missing historical flags do not exclude all preplanned trades. Anchor to the latest filing/acceptance/publication date plus one business day. Weekly snapshots. Stop constructing zero-event observations after the last observed disclosure date; later dates remain missing.',
      '- Transcripts: fixed small lexicons count negative and uncertainty words per 1,000 executive words, negative-rate change from the previous call, Q&A versus prepared-remarks uncertainty, and analyst-question negative words. These are crude lexical proxies, not a trained sentiment model: negation and context are not modeled. Select the earliest creation-dated version for each company/event; wait one business day after the later call or creation date. Require at least 1,000 executive words; Q&A and analyst measures have additional word-count requirements. Use at most 180 days of carry. No raw transcript text was retained.',
      '- Workforce: three-month net external hiring and external attrition divided by average reconstructed U.S. headcount. Sum joint role/seniority/geography cells. Month-end plus 15 days, then one business day. Scaled employee estimates and reconstructed resumes can use information learned later; this fixed lag does not remove that hindsight.',
      '- Postings: log growth of total U.S. job postings over the latest three months versus the preceding three, requiring at least 20 postings in each block. Same monthly lag as workforce. Never use future removal dates. Historical backfills, coverage/source changes and current issuer boundaries are unresolved; source mix can create apparent growth.',
      '- Liquidity: monthly institutional trading-cost estimates, lagged 45 calendar days plus one business day. The BBD price-impact series available here stops in 2016 and has no usable 2018–2025 overlap. It is not treated as a zero signal.',
      '- Credit: monthly market-value-weighted issuer credit-spread change and Treasury-adjusted bond returns, lagged 45 days plus one business day. Bond selection and weight changes can affect the aggregate. Historical publication timing and issuer links are unverified.',
      '',
      'Revelio links use direct, unambiguous current CIK-to-RCID mappings, with historical OptionMetrics→CRSP→CCM issuer intervals on the equity side. Subsidiaries are not automatically rolled into current parents. Transcripts use historical GVKEY→Capital IQ company links. Numeric WRDS identifiers are normalized before joins. Ambiguous or missing links produce missing features. These checks do not establish historically correct vendor company boundaries.','',
      '## Coverage and bounded extraction','',
      '| Feature | Any observed pairs | First eligible date | Last eligible date | Nonoverlap outcomes before minimum-row filter |',
      '|---|---:|---|---|---:|']
    for row in coverage[coverage.feature.str.endswith('_gap')].itertuples():
        lines.append(f'| {row.feature} | {row.pairs} | {str(row.first)[:10] if pd.notna(row.first) else "—"} | {str(row.last)[:10] if pd.notna(row.last) else "—"} | {row.nonoverlap} |')
    if batches:
        lines+=['','Broad workforce/posting queries initially hit the 20-second limit. Smaller serial batches were then attempted for unambiguous mapped companies. Failed batches stay unavailable, with the initial AMD/NVDA pilot retained where available.']
        for family in ['workforce','postings']:
            rows=[r for r in batches if r['family']==family]
            observed={x for r in rows for x in r['observed_rcids']}
            failed=sum(r['status']!='complete' for r in rows)
            attempted={x for r in rows for x in r['requested_rcids']}
            universe={x for r in batches if r['family']=='postings' for x in r['requested_rcids']}
            unattempted=len(universe-attempted)
            lines.append(f'- {family}: {len(rows)} attempted batches, {failed} failed, {len(observed)} distinct RCIDs with returned rows; {unattempted} mapped companies not attempted in this product.')
        lines.append('The workforce extraction stalled and the supervisor ended that run at its overall deadline. Saved batches were assembled locally, without restarting the remaining downloads. Its 45-company extract supports ten pairs after the observation filter. Posting extracts contain 120 companies and support 44 pairs; lack of U.S. observations and mapping ambiguity reduce coverage further. Missing companies are not assigned zero hiring or zero postings.')
    lines+=['',
      'Underlying input history is shorter for some sources: the fetched insider transactions end in October 2023; issuer bond data end in December 2023; institutional liquidity ends in February 2020. Derived features can remain eligible for the stated carry period after the final available observation. Later empty periods are not forward-filled indefinitely.','',
      '## Limits and next experiment','',
      'The outcomes retain the older $0.65 per-contract commission, 0.10-vol-point option entry penalty, daily stock hedging costs and financing/borrow assumptions from the first screen. They are not repriced to the current $0.03 reference commission. They are standalone initially vega-matched trajectories, with no daily option-vega rebalance or overlapping-book capacity simulation. Source membership, beta/correlation filters and future outcome resolution restrict the diagnostic sample.','',
      'The partial rank correlations control IV spread and RV-adjusted relative value, not every existing state. Approximate 90% intervals resample entire calendar quarters across all pairs jointly 500 times, with fixed ranks. They are not adjusted for the 34 candidate features or subsequent pair inspection. Nonoverlap thinning reduces expiry-window repetition but does not remove shared-name dependence across pairs. Neither same-sign early/later correlations nor an unadjusted interval excluding zero proves a tradable edge.','',
      'Use these results to predeclare a small hypothesis set for a separate chronological NN/RL comparison on matched costs and opportunities. Give employee sentiment and selected transcript changes a focused test if their later and annual diagnostics support it. Verify historical vintages before treating reconstructed labor data as a backtest input. A signal screen alone cannot say whether the current policy or portfolio Sharpe improves.','',
      '## Validation','',
      f'One guarded worker at 50% of one core with a 3 GiB memory stop; one read-only WRDS connection, 20-second statement timeouts and bounded company aggregates/transcript batches. Peak correlation-worker RSS observed: {manifest["peak_rss_bytes"]/1024**2:.0f} MiB. Existing baseline outcome and feature columns were checked exactly unchanged after alignment. Licensed extracts and derived row data remain in ignored artifacts; CSV reports are aggregates.','',
      'Eleven targeted tests passed, covering numeric identifier normalization, delayed monthly availability, exclusion of future events and post-coverage zeros, future-review independence, sparse review counts and fixed lexical counts, plus the original timing/link/correlation checks.','',
      '- [All correlations](wrds_alternative_pairs/correlations.csv)',
      '- [Annual diagnostics](wrds_alternative_pairs/annual.csv)',
      '- [Pair correlations](wrds_alternative_pairs/pairs.csv)',
      '- [Early/later and nonoverlap pair diagnostics](wrds_alternative_pairs/pair_periods.csv)',
      '- [Feature coverage](wrds_alternative_pairs/coverage.csv)',
      '- [Identifier coverage](wrds_alternative_pairs/identifier_coverage.csv)',
      '- [Original SEC/I/B/E/S screen](wrds_pair_signals.md)',
      '- [Research script](../scripts/test_wrds_alternative_pairs.py)','']
    REPORT.with_suffix('.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser();p.add_argument('--fetch',action='store_true')
    p.add_argument('--transcripts',action='store_true')
    p.add_argument('--prepare',action='store_true');p.add_argument('--analyze',action='store_true')
    p.add_argument('--labor',action='store_true')
    p.add_argument('--report',action='store_true')
    p.add_argument('--assemble',action='store_true')
    args=p.parse_args()
    if args.fetch:fetch()
    if args.transcripts:transcripts_and_pilot()
    if args.labor:batched_labor()
    if args.assemble:assemble_labor()
    if args.prepare:prepare()
    if args.analyze:analyze()
    if args.report:report()
    print(json.dumps(dict(peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)),flush=True)


if __name__=='__main__':main()
