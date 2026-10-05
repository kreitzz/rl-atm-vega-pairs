"""Bounded WRDS information-signal diagnostic on cached ATM pair outcomes.

Run with the one-worker research supervisor. No simulator or policy refits.
"""
import argparse
from collections import deque
from contextlib import closing
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import resource

import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import spearmanr

SOURCE = Path('artifacts/contracts/option_signal_diagnostics/outcomes.parquet')
OUT = Path('artifacts/contracts/wrds_pair_signals')
REPORT = Path('reports/wrds_pair_signals')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix('.tmp').write_text(json.dumps(value, indent=2, default=str)+'\n')
    path.with_suffix('.tmp').replace(path)


def query(db, name, statement, params=(), cap=100000):
    path = OUT / (name+'.parquet')
    if path.exists():
        return pd.read_parquet(path)
    with db.cursor() as cursor:
        cursor.execute(statement, params)
        rows = cursor.fetchmany(cap+1)
        if len(rows)>cap:
            raise MemoryError(f'{name}: row cap {cap} exceeded; no partial sample saved')
        columns = [c.name for c in cursor.description]
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_parquet(path, index=False)
    print(f'{name}: {len(frame):,} rows', flush=True)
    return frame


def fetch():
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(SOURCE)
    ids = sorted({int(x) for p in data.pair for x in p.split('__')})
    with closing(psycopg2.connect(host='wrds-pgdata.wharton.upenn.edu', port=9737,
                 dbname='wrds', user=os.getenv('WRDS_USERNAME', getpass.getuser()),
                 sslmode='require', connect_timeout=20,
                 application_name='bounded_pair_information_signals')) as db:
        db.set_session(readonly=True, autocommit=True)
        with db.cursor() as c:
            c.execute("SET statement_timeout='20s'")
            c.execute("SET lock_timeout='2s'")
            c.execute("SELECT current_setting('transaction_read_only')")
            assert c.fetchone()[0]=='on'
        # Metadata for the exact identifier tables, before requesting rows.
        meta=query(db, 'identifier_columns', """
          SELECT table_schema,table_name,column_name,data_type
          FROM information_schema.columns
          WHERE (table_schema,table_name) IN
            (('wrdsapps_link_crsp_optionm','opcrsphist'),
             ('crsp','ccmxpf_linktable'),('wrdsapps_link_crsp_ibes','ibcrsphist'))
          ORDER BY table_schema,table_name,ordinal_position
        """, cap=200)
        print(meta.to_string(index=False), flush=True)
        option_links=query(db,'option_links',"""
          SELECT * FROM wrdsapps_link_crsp_optionm.opcrsphist
          WHERE secid=ANY(%s)
        """,(ids,),cap=10000)
        permnos=sorted(map(int,option_links.permno.dropna().unique()))
        company_links=query(db,'company_links',"""
          SELECT l.lpermno AS permno,l.gvkey,l.linkdt,l.linkenddt,
                 l.linktype,l.linkprim,c.cik,c.conm
          FROM crsp.ccmxpf_linktable l JOIN comp.company c ON c.gvkey=l.gvkey
          WHERE l.lpermno=ANY(%s) AND l.linktype IN ('LC','LU')
            AND l.linkprim IN ('P','C') AND c.cik IS NOT NULL
        """,(permnos,),cap=10000)
        analyst_links=query(db,'analyst_links',"""
          SELECT ticker,permno,ncusip,sdate,edate,score
          FROM wrdsapps_link_crsp_ibes.ibcrsphist
          WHERE permno=ANY(%s) AND score<=1
        """,(permnos,),cap=10000)
        ciks=sorted({str(x).strip().zfill(10) for x in company_links.cik.dropna() if str(x).strip()})
        tickers=sorted(analyst_links.ticker.dropna().unique())
        sample=query(db,'analyst_sample',"""
          SELECT ticker,actdats,measure,curr,fpi,value
          FROM tr_ibes.det_epsus WHERE ticker='AMD' AND actdats>='2018-01-01'
          LIMIT 12
        """,cap=12)
        print(sample.to_string(index=False),flush=True)
        filings=query(db,'filings',"""
          SELECT b.cik,b.accession,b.form,b.fdate,b.rdate,b.secadate,b.secpdate,
                 b.secatime,n.lm_negative,n.lm_uncertainty,n.word_count
          FROM wrdssec_secsa.bow_filingsummary b
          JOIN wrdssec_secsa.wrds_nlpsa n ON b.accession=n.accession AND b.cik=n.cik
          WHERE b.cik=ANY(%s) AND b.form IN ('10-K','10-Q')
            AND b.fdate BETWEEN '2014-01-01' AND '2025-07-10'
        """,(ciks,),cap=30000)
        similarity=query(db,'similarity',"""
          SELECT cik,accession,accession2,form,fdate,lookback,cosine
          FROM wrdssec_secsa.bow_cossim
          WHERE cik=ANY(%s) AND form IN ('10-K','10-Q') AND lookback=1
            AND fdate BETWEEN '2014-01-01' AND '2025-07-10'
        """,(ciks,),cap=30000)
        for year in range(2017,2026):
            query(db,f'annual_eps_{year}',"""
              SELECT ticker,fpedats,estimator,analys,actdats,acttims,
                     anndats,anntims,value,fpi
              FROM tr_ibes.det_epsus
              WHERE ticker=ANY(%s) AND measure='EPS' AND usfirm=1
                AND (curr IS NULL OR curr='USD') AND fpi='1'
                AND actdats>=%s AND actdats<%s
            """,(tickers,f'{year}-01-01',f'{year+1}-01-01'),cap=150000)
        write_json(OUT/'fetch.json',dict(fetched_at=datetime.now(timezone.utc).isoformat(),
          outcome_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          secids=len(ids),permnos=len(permnos),ciks=len(ciks),ibes_tickers=len(tickers),
          read_only=True,timeout_seconds=20,row_cap_per_query=150000))


def interval_ids(dates, links, key, begin, end, value):
    """Return an ID only when all simultaneous eligible links agree."""
    result=np.full(len(dates),None,dtype=object)
    ambiguous=np.zeros(len(dates),bool)
    for row in links[[begin,end,value]].itertuples(index=False,name=None):
        start,finish,identifier=row
        if pd.isna(identifier):continue
        hit=np.asarray((dates>=start)&(dates<=finish))
        assigned=pd.notna(result)
        ambiguous |= hit & assigned & (result!=identifier)
        result[hit & ~assigned]=identifier
    result[ambiguous]=None
    return result


def timestamp_columns(frame, columns):
    for c in columns:frame[c]=pd.to_datetime(frame[c],errors='coerce')
    return frame


def availability(frame, date_columns):
    """Ignore intra-day timing by waiting a full business day after all dates."""
    dates=pd.concat([pd.to_datetime(frame[c],errors='coerce') for c in date_columns],axis=1)
    return dates.max(axis=1)+pd.offsets.BDay(1)


def sec_states():
    f=pd.read_parquet(OUT/'filings.parquet')
    f['available']=availability(f,['fdate','secadate','secpdate'])
    f['rdate']=pd.to_datetime(f.rdate)
    f['quarter']=f.rdate.dt.quarter
    # Multiple copies with inconsistent text scores are not silently selected.
    cols=['lm_negative','lm_uncertainty','word_count']
    bad=f.groupby(['cik','accession'])[cols].nunique().max(axis=1)>1
    assert not bad.any(),'Inconsistent duplicate filing text scores'
    f=f.drop_duplicates(['cik','accession']).sort_values(['cik','available','accession'])
    same=f.groupby(['cik','form','quarter'],dropna=False)
    prev_date=same.available.shift()
    comparable=(f.available-prev_date).dt.days.between(250,500)
    for field in ['lm_negative','lm_uncertainty']:
        f[field+'_change']=same[field].diff().where(comparable)
    s=pd.read_parquet(OUT/'similarity.parquet')
    assert not s.duplicated(['cik','accession']).any(),'Ambiguous previous filing comparison'
    f=f.merge(s[['cik','accession','cosine']],on=['cik','accession'],how='left',validate='one_to_one')
    f['text_change']=1-pd.to_numeric(f.cosine,errors='coerce')
    features=['lm_negative','lm_uncertainty','lm_negative_change','lm_uncertainty_change','text_change']
    return f[['cik','available',*features]],features


def build_analyst_states(e, dates):
    e=e.copy()
    e['available']=availability(e,['actdats','anndats'])
    e['fpedats']=pd.to_datetime(e.fpedats)
    e['value']=pd.to_numeric(e.value,errors='coerce')
    # Date/time ordering defines the previous estimate; never use future actuals.
    e=e.sort_values(['ticker','fpedats','estimator','analys','available','actdats','acttims','anndats','anntims'])
    e=e.drop_duplicates(['ticker','fpedats','estimator','analys','actdats','acttims'],keep='last')
    prior=e.groupby(['ticker','fpedats','estimator','analys'],dropna=False).value.shift()
    e['revision']=(e.value-prior)/prior.abs().where(prior.abs()>=.1)
    e['revision']=e.revision.clip(-1,1)
    frames=[]
    for ticker,g in e.groupby('ticker',sort=True):
        g=g.sort_values(['available','actdats','acttims','anndats','anntims'])
        events=list(g.itertuples(index=False));pos=0;latest={};recent=deque();records=[]
        for date in dates:
            while pos<len(events) and events[pos].available<=date:
                row=events[pos];pos+=1
                key=(row.fpedats,None if pd.isna(row.estimator) else row.estimator,
                     None if pd.isna(row.analys) else row.analys)
                latest[key]=(row.available,row.value)
                if np.isfinite(row.revision):recent.append((row.available,row.fpedats,row.revision))
            lower=date-pd.Timedelta(days=180)
            latest={k:v for k,v in latest.items() if v[0]>=lower and k[0]>=date}
            while recent and recent[0][0]<=date-pd.Timedelta(days=30):recent.popleft()
            if not latest:continue
            period=min(k[0] for k in latest)
            values=np.array([v[1] for k,v in latest.items() if k[0]==period and np.isfinite(v[1])])
            if len(values)<3:continue
            revisions=np.array([v for _,fy,v in recent if fy==period])
            records.append(dict(ticker=ticker,available=date,
              revision_breadth=float(np.sign(revisions).mean()) if len(revisions) else 0.,
              revision_magnitude=float(revisions.mean()) if len(revisions) else 0.,
              revision_activity=float(np.log1p(len(revisions))),
              forecast_dispersion=float(np.clip(values.std(ddof=1)/max(abs(np.median(values)),.1),0,10))))
        frames.extend(records)
    result=pd.DataFrame(frames)
    print(f'Analyst weekly snapshots: {len(result):,} across {result.ticker.nunique()} tickers',flush=True)
    return result,['revision_breadth','revision_magnitude','revision_activity','forecast_dispersion']


def analyst_states():
    e=pd.concat([pd.read_parquet(OUT/f'annual_eps_{y}.parquet') for y in range(2017,2026)],ignore_index=True)
    return build_analyst_states(e,pd.date_range('2017-01-06','2025-07-10',freq='W-FRI'))


def match_states(dates, ids, states, id_column, fields, max_age):
    out=pd.DataFrame(np.nan,index=pd.DatetimeIndex(dates),columns=fields)
    for key in pd.Series(ids).dropna().unique():
        mask=np.asarray(ids)==key;g=states[states[id_column].eq(key)].sort_values('available')
        # Filing collisions on one day are ambiguous: leave those days unavailable.
        g=g.drop_duplicates('available',keep=False)
        if g.empty:continue
        left=pd.DataFrame({'date':pd.DatetimeIndex(dates)[mask]})
        joined=pd.merge_asof(left,g,on=None,left_on='date',right_on='available',
             direction='backward',tolerance=pd.Timedelta(days=max_age))
        assert (joined.available.dropna()<=joined.loc[joined.available.notna(),'date']).all()
        out.loc[left.date,fields]=joined[fields].to_numpy(float)
    return out


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    ff=['lm_negative','lm_uncertainty','lm_negative_change','lm_uncertainty_change','text_change']
    ef=['revision_breadth','revision_magnitude','revision_activity','forecast_dispersion']
    if (OUT/'sec_states.parquet').exists() and (OUT/'analyst_states.parquet').exists():
        f=pd.read_parquet(OUT/'sec_states.parquet');e=pd.read_parquet(OUT/'analyst_states.parquet')
        if not (OUT/'snapshot_ordering_v2.json').exists():
            e,ef=analyst_states();e.to_parquet(OUT/'analyst_states.parquet',index=False)
            write_json(OUT/'snapshot_ordering_v2.json',dict(chronological_same_day_updates=True))
    else:
        f,ff=sec_states();e,ef=analyst_states()
        f.to_parquet(OUT/'sec_states.parquet',index=False);e.to_parquet(OUT/'analyst_states.parquet',index=False)
        write_json(OUT/'snapshot_ordering_v2.json',dict(chronological_same_day_updates=True))
    o=timestamp_columns(pd.read_parquet(OUT/'option_links.parquet'),['sdate','edate'])
    o=o[o.score<=1].copy();o.edate=o.edate.fillna(pd.Timestamp('2026-01-01'))
    c=timestamp_columns(pd.read_parquet(OUT/'company_links.parquet'),['linkdt','linkenddt'])
    c.linkdt=c.linkdt.fillna(pd.Timestamp('1900-01-01'));c.linkenddt=c.linkenddt.fillna(pd.Timestamp('2026-01-01'))
    c.cik=c.cik.map(lambda x:str(x).strip().zfill(10))
    a=timestamp_columns(pd.read_parquet(OUT/'analyst_links.parquet'),['sdate','edate'])
    a.edate=a.edate.fillna(pd.Timestamp('2026-01-01'))
    data=pd.read_parquet(SOURCE);data=data[data.combination.eq('CC')].copy()
    definitions={x['pair']:x for x in json.loads(Path('artifacts/contracts/option_flex/inputs.json').read_text())['definitions']}
    # Build one causal asset panel and reuse it across pairs sharing that name.
    all_dates=pd.DatetimeIndex(sorted(data.date.unique()));legs={};coverage=[]
    ids=sorted({sid for pair in data.pair.unique() for sid in definitions[pair]['secids']})
    for sid in ids:
        own=o[o.secid.eq(sid)]
        perm=interval_ids(all_dates,own,'secid','sdate','edate','permno')
        cik=np.full(len(all_dates),None,object);ticker=cik.copy()
        for p in pd.Series(perm).dropna().unique():
            mask=perm==p
            cik[mask]=interval_ids(all_dates[mask],c[c.permno.eq(p)],'permno','linkdt','linkenddt','cik')
            ticker[mask]=interval_ids(all_dates[mask],a[a.permno.eq(p)],'permno','sdate','edate','ticker')
        sp=match_states(all_dates,cik,f,'cik',ff,180)
        ap=match_states(all_dates,ticker,e,'ticker',ef,14)
        legs[sid]=pd.concat([sp,ap],axis=1)
        coverage.append(dict(secid=sid,option_crsp_dates=int(pd.notna(perm).sum()),
          company_link_dates=int(pd.notna(cik).sum()),analyst_link_dates=int(pd.notna(ticker).sum()),
          sec_observed_dates=int(sp.lm_negative.notna().sum()),analyst_observed_dates=int(ap.revision_breadth.notna().sum())))
        if len(coverage)%25==0:print(f'Historical links aligned: {len(coverage)}/{len(ids)} assets',flush=True)
    frames=[]
    for pair,g in data.groupby('pair',sort=True):
        g=g.copy();pa,pb=definitions[pair]['secids']
        xa=legs[pa].reindex(pd.DatetimeIndex(g.date));xb=legs[pb].reindex(pd.DatetimeIndex(g.date))
        for col in ff+ef:
            g[col+'_gap']=(xa[col]-xb[col]).to_numpy()
            g[col+'_mean']=((xa[col]+xb[col])/2).to_numpy()
        # Existing value inputs are controls, not new candidate mining variables.
        p=pd.read_parquet(Path('artifacts/contracts/option_flex')/pair/'states.parquet',
                          columns=['spread','rv_adjusted_value']).reindex(pd.DatetimeIndex(g.date))
        g['spread']=p.spread.to_numpy();g['rv_adjusted_value']=p.rv_adjusted_value.to_numpy()
        g['best_net']=g[['long_net','short_net']].max(axis=1)
        frames.append(g)
    data=pd.concat(frames,ignore_index=True).sort_values(['date','pair'])
    assert not data.duplicated(['date','pair']).any()
    data.to_parquet(OUT/'aligned.parquet',index=False)
    pd.DataFrame(coverage).to_csv(REPORT/'identifier_coverage.csv',index=False)
    write_json(OUT/'features.json',dict(stock_features=ff+ef,
          pair_features=[x+s for x in ff+ef for s in ['_gap','_mean']],
          direction='long_net: long vega in definition.secids[0], short in definition.secids[1]',
          company_mapping='OptionMetrics→historical CRSP→historical CCM→Compustat CIK; ambiguous active IDs excluded',
          analyst_mapping='historical CRSP→historical I/B/E/S ticker; score≤1; ambiguous active IDs excluded'))
    print(f'Aligned {len(data):,} opportunities in {data.pair.nunique()} pairs',flush=True)


def within_ranks(frame, feature, target):
    h=frame.loc[np.isfinite(frame[feature])&np.isfinite(frame[target])].copy()
    counts=h.groupby('pair').pair.transform('size')
    h=h[counts>=20].copy()
    if h.empty:return h
    h['x']=h.groupby('pair')[feature].rank(pct=True)
    h['y']=h.groupby('pair')[target].rank(pct=True)
    h['x']-=h.groupby('pair').x.transform('mean');h['y']-=h.groupby('pair').y.transform('mean')
    h['w']=1/h.groupby('pair').pair.transform('size')
    return h


def weighted_corr(x,y,w):
    if len(x)<20 or w.sum()<=0:return np.nan
    x=x-np.average(x,weights=w);y=y-np.average(y,weights=w)
    den=np.sqrt(np.sum(w*x*x)*np.sum(w*y*y))
    return float(np.sum(w*x*y)/den) if den>1e-12 else np.nan


def statistics(frame, feature, target, bootstrap=False):
    h=within_ranks(frame,feature,target)
    row=dict(observations=len(h),pairs=h.pair.nunique(),spearman=np.nan,
             within_pair_equal_weight_rank_corr=np.nan,partial_value_rank_corr=np.nan,
             ci90_low=np.nan,ci90_high=np.nan)
    if len(h)<40:return row
    x=h.x.to_numpy();y=h.y.to_numpy();w=h.w.to_numpy()
    row['spearman']=float(spearmanr(h[feature],h[target]).statistic) if h[feature].nunique()>1 else np.nan
    row['within_pair_equal_weight_rank_corr']=weighted_corr(x,y,w)
    controls=[]
    for col in ['spread','rv_adjusted_value']:
        rank=h.groupby('pair')[col].rank(pct=True)
        controls.append((rank-rank.groupby(h.pair).transform('mean')).fillna(0).to_numpy())
    z=np.column_stack([np.ones(len(h)),*controls]);zw=z*np.sqrt(w[:,None])
    bx=np.linalg.lstsq(zw,x*np.sqrt(w),rcond=None)[0];by=np.linalg.lstsq(zw,y*np.sqrt(w),rcond=None)[0]
    row['partial_value_rank_corr']=weighted_corr(x-z@bx,y-z@by,w)
    if bootstrap:
        # Resample whole quarters across all pairs jointly, preserving shared-name
        # dependence and most expiry-window overlap. Descriptive, fixed-rank CI.
        h['block']=h.date.dt.to_period('Q')
        h['wx']=w*x;h['wy']=w*y;h['wxx']=w*x*x;h['wyy']=w*y*y;h['wxy']=w*x*y
        blocks=h.groupby('block')[['w','wx','wy','wxx','wyy','wxy']].sum().to_numpy()
        rng=np.random.default_rng(230);chosen=rng.integers(len(blocks),size=(500,len(blocks)))
        s=blocks[chosen].sum(axis=1);sw,sx,sy,sxx,syy,sxy=s.T
        den=np.sqrt(np.maximum(sxx-sx*sx/sw,0)*np.maximum(syy-sy*sy/sw,0))
        bs=np.divide(sxy-sx*sy/sw,den,out=np.full(len(den),np.nan),where=den>1e-12)
        row['ci90_low'],row['ci90_high']=np.nanquantile(bs,[.05,.95])
    return row


def analyze():
    data=pd.read_parquet(OUT/'aligned.parquet');features=json.loads((OUT/'features.json').read_text())['pair_features']
    summary=[];yearly=[];pairs=[];pair_periods=[]
    for feature in features:
        target='directional_edge' if feature.endswith('_gap') else 'best_net'
        for sample,h in [('full',data),('early_2018_2021',data[data.end<'2022-01-01']),
                         ('later_2022_2025',data[data.date>='2022-01-01']),
                         ('nonoverlap',data[data.nonoverlap])]:
            summary.append(dict(feature=feature,target=target,sample=sample,
              **statistics(h,feature,target,bootstrap=sample=='full')))
        for year,g in data.groupby(data.date.dt.year):
            yearly.append(dict(feature=feature,target=target,year=year,**statistics(g,feature,target)))
        for pair,g in data.groupby('pair'):
            h=g[np.isfinite(g[feature])&np.isfinite(g[target])]
            if len(h)<20:continue
            rho=float(spearmanr(h[feature],h[target]).statistic) if h[feature].nunique()>1 else np.nan
            pairs.append(dict(pair=pair,label=g.label.iloc[0],feature=feature,target=target,
                             observations=len(h),spearman=rho))
            for sample,v in [('early_2018_2021',g[g.end<'2022-01-01']),
                             ('later_2022_2025',g[g.date>='2022-01-01']),
                             ('nonoverlap',g[g.nonoverlap])]:
                v=v[np.isfinite(v[feature])&np.isfinite(v[target])]
                rho=float(spearmanr(v[feature],v[target]).statistic) if len(v)>=8 and v[feature].nunique()>1 else np.nan
                pair_periods.append(dict(pair=pair,label=g.label.iloc[0],feature=feature,target=target,
                  sample=sample,observations=len(v),unique_signal_values=v[feature].nunique(),spearman=rho))
    result=pd.DataFrame(summary);result.to_csv(REPORT/'correlations.csv',index=False)
    pd.DataFrame(yearly).to_csv(REPORT/'annual.csv',index=False)
    pd.DataFrame(pairs).to_csv(REPORT/'pairs.csv',index=False)
    pd.DataFrame(pair_periods).to_csv(REPORT/'pair_periods.csv',index=False)
    coverage=[]
    for pair,g in data.groupby('pair'):
        coverage.append(dict(pair=pair,label=g.label.iloc[0],opportunities=len(g),
          sec_complete=int(g[['lm_negative_gap','text_change_gap']].notna().all(axis=1).sum()),
          analyst_complete=int(g[['revision_breadth_gap','forecast_dispersion_gap']].notna().all(axis=1).sum())))
    pd.DataFrame(coverage).to_csv(REPORT/'pair_coverage.csv',index=False)
    previous=json.loads((OUT/'analysis.json').read_text()) if (OUT/'analysis.json').exists() else {}
    write_json(OUT/'analysis.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),
      source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
      code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      opportunities=len(data),pairs=data.pair.nunique(),features=len(features),
      bootstrap='500 joint calendar-quarter resamples, fixed within-pair ranks, 90% interval; no multiple-testing correction',
      reference_promoted=False,policy_retrained=False,
      peak_rss_bytes=max(previous.get('peak_rss_bytes',0),resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
      aligned_sha256=hashlib.sha256((OUT/'aligned.parquet').read_bytes()).hexdigest(),
      feature_snapshot_hashes={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest()
          for name in ['sec_states.parquet','analyst_states.parquet']},
      label_commission_per_contract=.65,availability='one business day after latest public/vendor date; analyst snapshots weekly'))
    print(result[result['sample'].eq('full')][['feature','target','observations','pairs',
      'within_pair_equal_weight_rank_corr','partial_value_rank_corr','ci90_low','ci90_high']].to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--fetch',action='store_true')
    p.add_argument('--prepare',action='store_true');p.add_argument('--analyze',action='store_true')
    args=p.parse_args()
    if args.fetch:fetch()
    if args.prepare:prepare()
    if args.analyze:analyze()
    print(json.dumps(dict(peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)),flush=True)


if __name__=='__main__':main()
