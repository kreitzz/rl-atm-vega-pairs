"""New volatility-signal screen: bounded downloads, lagged states, no RL fits."""
import argparse
from contextlib import closing
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import resource
import urllib.request
import re
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import spearmanr
from scripts.test_wrds_pair_signals import (interval_ids, match_states, statistics,
    timestamp_columns, write_json, weighted_corr)
from scripts.run_contract_walkforward import digest

BASE=Path('artifacts/contracts/wrds_pair_signals')
OUT=Path('artifacts/contracts/new_vol_signals')
REPORT=Path('reports/new_vol_signals')
SOURCE=BASE/'aligned.parquet'
FIELDS=['option_activity','put_call_volume','volume_oi','oi_change','borrow_level','borrow_change',
        'late_filing_90d','restatement_90d','litigation_90d']
GLOBAL=['vvix','vvix_change','vix9d_ratio','equity_news_30d','trade_news_monthly','monetary_news_monthly']


def bounded(db,name,query,parameters=(),cap=150000):
    path=OUT/(name+'.parquet')
    if path.exists():return pd.read_parquet(path)
    with db.cursor() as c:
        c.execute(query,parameters); data=c.fetchmany(cap+1)
        if len(data)>cap:raise MemoryError(f'{name}: exceeds {cap} rows; no partial data saved')
        frame=pd.DataFrame(data,columns=[d.name for d in c.description])
    frame.to_parquet(path,index=False)
    print(f'{name}: {len(frame):,} rows',flush=True)
    return frame


def download(name,url):
    path=OUT/name
    if not path.exists():
        request=urllib.request.Request(url,headers={'User-Agent':'Academic volatility research'})
        with urllib.request.urlopen(request,timeout=25) as response:
            data=response.read(8*1024*1024+1)
        if len(data)>8*1024*1024:raise MemoryError('Public series exceeds 8 MiB cap')
        path.write_bytes(data)
    return path


def fetch():
    status=json.loads((OUT/'fetch.json').read_text()) if (OUT/'fetch.json').exists() else {}
    data=pd.read_parquet(SOURCE);ids=sorted({int(x) for p in data.pair for x in p.split('__')})
    links=pd.read_parquet(BASE/'company_links.parquet')
    ciks={str(int(str(c).strip())) for c in links.cik.dropna()}
    keys=sorted(ciks|{c.zfill(10) for c in ciks})
    with closing(psycopg2.connect(host='wrds-pgdata.wharton.upenn.edu',port=9737,dbname='wrds',
                 user=os.getenv('WRDS_USERNAME',getpass.getuser()),sslmode='require',connect_timeout=20,
                 application_name='bounded_new_vol_signal_screen')) as db:
        db.set_session(readonly=True,autocommit=True)
        with db.cursor() as c:c.execute("SET statement_timeout='20s'");c.execute("SET lock_timeout='2s'")
        for year in range(2017,2026):
            begin=f'{year}-01-01';end=f'{year}-12-31' if year<2025 else '2025-08-29'
            for family,query in [
                ('volume',"SELECT secid,date,cp_flag,volume,open_interest FROM optionm.opvold "
                 "WHERE secid=ANY(%s) AND date BETWEEN %s AND %s AND cp_flag IN ('C','P')"),
                ('borrow',f'SELECT DISTINCT ON (securityid,date) securityid AS secid,date,days,borrowrate '
                 f'FROM optionm.borrate{year} WHERE securityid=ANY(%s) AND date BETWEEN %s AND %s '
                 'AND days BETWEEN 7 AND 60 AND borrowrate IS NOT NULL '
                 'ORDER BY securityid,date,abs(days-30),days,expirationdate')]:
                name=f'{family}_{year}'
                try:
                    frame=bounded(db,name,query,(ids,begin,end));status[name]=dict(status='complete',rows=len(frame))
                except (psycopg2.Error,MemoryError) as exc:
                    status[name]=dict(status='unavailable',error=str(exc));print(f'{name}: unavailable ({exc})',flush=True)
                write_json(OUT/'fetch.json',status)
        try:
            frame=bounded(db,'litigation', '''SELECT p.company_fkey,c.legal_case_key,c.case_start_date,
                    c.create_date,p.create_date AS link_create_date
                  FROM audit_corp_legal.f14_lit_legal_party p
                  JOIN audit_corp_legal.f14_lit_legal_case c ON p.legal_case_fkey=c.legal_case_key
                 WHERE p.company_fkey=ANY(%s) AND c.case_start_date BETWEEN '2010-01-01' AND '2025-08-29'
                 ORDER BY p.company_fkey,c.legal_case_key''',(keys,),cap=20000)
            status['litigation']=dict(status='complete',rows=len(frame))
        except (psycopg2.Error,MemoryError) as exc:status['litigation']=dict(status='unavailable',error=str(exc))
        write_json(OUT/'fetch.json',status)
    public={f'{s}.csv':f'https://cdn-api.cboe.com/api/global/us_indices/daily_prices/{s}_History.csv'
            for s in ['VIX','VVIX','VIX9D']}
    public.update({'equity_news.csv':'https://www.policyuncertainty.com/media/All_Daily_Equity_Data.csv',
                   'emv.xlsx':'https://www.policyuncertainty.com/media/EMV_Data.xlsx'})
    for name,url in public.items():
        try:
            path=download(name,url);status[name]=dict(status='complete',url=url,bytes=path.stat().st_size)
            print(f'{name}: downloaded',flush=True)
        except Exception as exc:status[name]=dict(status='unavailable',url=url,error=str(exc));print(f'{name}: {exc}',flush=True)
        write_json(OUT/'fetch.json',status)


def daily_features(raw,dates):
    raw=raw.copy();raw['date']=pd.to_datetime(raw.date);raw.secid=raw.secid.map(int)
    rows=[]
    for sid,g in raw.groupby('secid'):
        assert not g.duplicated(['date','cp_flag']).any()
        v=g.pivot(index='date',columns='cp_flag',values='volume').reindex(columns=['C','P'])
        oi=g.pivot(index='date',columns='cp_flag',values='open_interest').reindex(columns=['C','P'])
        v=v.where(v>=0);oi=oi.where(oi>=0)
        total=v.sum(axis=1,min_count=2); interest=oi.sum(axis=1,min_count=2)
        h=pd.DataFrame(index=v.index)
        h['option_activity']=np.log1p(total)-np.log1p(total.rolling(63,min_periods=42).median().shift(1))
        h['put_call_volume']=np.log1p(v.P)-np.log1p(v.C)
        h['volume_oi']=total/interest.where(interest>0)
        h['oi_change']=np.log1p(interest)-np.log1p(interest.shift(21))
        h['available']=h.index+pd.offsets.BDay(1);h['secid']=sid
        rows.append(h.reset_index(drop=True))
    return pd.concat(rows,ignore_index=True)


def dated_events(raw,columns):
    f=raw.copy()
    for c in columns:
        text=f[c].astype(str).str.strip()
        parsed=pd.to_datetime(text,format='mixed',errors='coerce')
        compact=text.str.fullmatch(r'\d{14}')
        parsed.loc[compact]=pd.to_datetime(text[compact],format='%Y%m%d%H%M%S',errors='coerce')
        f[c]=parsed
    f['available']=f[columns].max(axis=1).dt.normalize()+pd.offsets.BDay(1)
    f['cik']=f.company_fkey.map(lambda x:str(int(str(x).strip())).zfill(10))
    return f


def borrow_features(raw):
    g=raw.copy();g['date']=pd.to_datetime(g.date);g.secid=g.secid.map(int)
    g.borrowrate=pd.to_numeric(g.borrowrate,errors='coerce')
    # Repeated -99.99 is an apparent missing-data sentinel, not an economic fee.
    g.borrowrate=g.borrowrate.mask(np.isclose(g.borrowrate,-99.99,atol=1e-6,rtol=0))
    g=g.sort_values(['secid','date'])
    assert not g.duplicated(['secid','date']).any()
    g['borrow_level']=g.borrowrate
    g['borrow_change']=g.borrowrate-g.groupby('secid').borrowrate.shift(21)
    g['available']=g.date+pd.offsets.BDay(1)
    return g


def event_panel(events,ciks,dates,name):
    rows=[]
    for cik in ciks:
        # Vendor absence is provisionally a zero event count; coverage is audited separately.
        available=pd.DatetimeIndex(events.loc[events.cik.eq(cik),'available'].dropna().sort_values())
        count=available.searchsorted(dates,side='right')-available.searchsorted(dates-pd.Timedelta(days=90),side='right')
        rows.append(pd.DataFrame({'cik':cik,'available':dates,name:np.log1p(count)}))
    return pd.concat(rows,ignore_index=True)


def causal_global(series,dates,lag_days=0,max_age=7):
    g=pd.DataFrame({'source_date':series.index,'value':series.to_numpy()})
    g['available']=g.source_date+pd.Timedelta(days=lag_days)+pd.offsets.BDay(1)
    joined=pd.merge_asof(pd.DataFrame({'date':dates}),g.sort_values('available'),
                         left_on='date',right_on='available',direction='backward',tolerance=pd.Timedelta(days=max_age))
    assert (joined.available.dropna()<=joined.loc[joined.available.notna(),'date']).all()
    return pd.Series(joined.value.to_numpy(),index=dates)


def read_news_workbook(path):
    """Read the small public numeric workbook without adding an Excel dependency."""
    ns={'x':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(path) as archive:
        strings=[]
        if 'xl/sharedStrings.xml' in archive.namelist():
            strings=[''.join(item.itertext()) for item in ET.fromstring(archive.read('xl/sharedStrings.xml'))]
        root=ET.fromstring(archive.read('xl/worksheets/sheet1.xml'))
    rows=[]
    for row in root.findall('x:sheetData/x:row',ns):
        values={}
        for cell in row:
            letters=re.match(r'[A-Z]+',cell.attrib['r']).group()
            index=0
            for letter in letters:index=index*26+ord(letter)-64
            kind=cell.attrib.get('t');value=cell.find('x:v',ns)
            if kind=='inlineStr':value=''.join(cell.find('x:is',ns).itertext())
            elif value is None:value=None
            elif kind=='s':value=strings[int(value.text)]
            else:value=float(value.text)
            values[index-1]=value
        rows.append([values.get(i) for i in range(max(values)+1)])
    return pd.DataFrame(rows[1:],columns=rows[0])


def global_features(dates):
    result=pd.DataFrame(np.nan,index=dates,columns=GLOBAL)
    series={}
    for name in ['VIX','VVIX','VIX9D']:
        path=OUT/(name+'.csv')
        if not path.exists():continue
        raw=pd.read_csv(path);raw.columns=raw.columns.str.upper().str.strip()
        date=pd.to_datetime(raw.DATE);column='CLOSE' if 'CLOSE' in raw else name
        series[name]=pd.Series(pd.to_numeric(raw[column],errors='coerce').to_numpy(),index=date).sort_index()
    if 'VVIX' in series:
        v=series['VVIX'];result['vvix']=causal_global(v,dates)
        result['vvix_change']=causal_global(v-v.shift(21),dates)
    if {'VIX','VIX9D'}<=series.keys():
        ratio=series['VIX9D']/series['VIX'].where(series['VIX']>0)
        result['vix9d_ratio']=causal_global(ratio.dropna(),dates)
    if (OUT/'equity_news.csv').exists():
        raw=pd.read_csv(OUT/'equity_news.csv');write_json(REPORT/'equity_news_columns.json',list(raw))
        lookup={c.lower().strip():c for c in raw}
        date=pd.to_datetime(dict(year=raw[lookup['year']],month=raw[lookup['month']],day=raw[lookup['day']]))
        value=next(c for c in raw if 'equity' in c.lower() or 'index' in c.lower())
        s=pd.Series(pd.to_numeric(raw[value],errors='coerce').to_numpy(),index=date).sort_index()
        # Current-vintage diagnostic only. Extra 30 days does not certify vintage availability.
        result['equity_news_30d']=causal_global(s.rolling(30,min_periods=20).mean(),dates,lag_days=30)
    if (OUT/'emv.xlsx').exists():
        raw=read_news_workbook(OUT/'emv.xlsx');write_json(REPORT/'emv_columns.json',list(raw.columns))
        date=pd.to_datetime(dict(year=pd.to_numeric(raw['Year'],errors='coerce'),
                                month=pd.to_numeric(raw['Month'],errors='coerce'),day=1))+pd.offsets.MonthEnd(0)
        for field,col in [('trade_news_monthly','Trade Policy EMV Tracker'),
                          ('monetary_news_monthly','Monetary Policy EMV Tracker')]:
            s=pd.Series(pd.to_numeric(raw[col],errors='coerce').to_numpy(),index=date).dropna().sort_index()
            result[field]=causal_global(s,dates,lag_days=45,max_age=45)
    return result


def prepare():
    data=pd.read_parquet(SOURCE);dates=pd.DatetimeIndex(sorted(data.date.unique()))
    definitions={x['pair']:x for x in json.loads(Path('artifacts/contracts/option_flex/inputs.json').read_text())['definitions']}
    volume=[pd.read_parquet(p) for p in sorted(OUT.glob('volume_*.parquet'))]
    activity=daily_features(pd.concat(volume,ignore_index=True),dates) if volume else pd.DataFrame()
    borrowed=[]
    for p in sorted(OUT.glob('borrow_*.parquet')):
        borrowed.append(pd.read_parquet(p))
    borrow=borrow_features(pd.concat(borrowed,ignore_index=True)) if borrowed else pd.DataFrame()
    company=timestamp_columns(pd.read_parquet(BASE/'company_links.parquet'),['linkdt','linkenddt'])
    company.linkdt=company.linkdt.fillna(pd.Timestamp('1900-01-01'));company.linkenddt=company.linkenddt.fillna(pd.Timestamp('2026-01-01'))
    company.cik=company.cik.map(lambda x:str(x).strip().zfill(10));ciks=sorted(company.cik.dropna().unique())
    option=timestamp_columns(pd.read_parquet(BASE/'option_links.parquet'),['sdate','edate']);option=option[option.score<=1].copy()
    option.edate=option.edate.fillna(pd.Timestamp('2026-01-01'))
    events=[]
    for field,file,cols in [('late_filing_90d','late_filings',['file_date','file_accepted']),
                            ('restatement_90d','restatements',['event_date','file_date','file_accepted','date_of_8k_402']),
                            ('litigation_90d','litigation',['case_start_date','create_date','link_create_date'])]:
        p=OUT/(file+'.parquet') if file=='litigation' else Path('artifacts/contracts/vol_alternative_data')/(file+'.parquet')
        if p.exists():
            e=dated_events(pd.read_parquet(p),cols)
            if file=='litigation':e=e.drop_duplicates(['cik','legal_case_key'])
            events.append((field,event_panel(e,ciks,dates,field)))
    assets=sorted({sid for p in data.pair for sid in definitions[p]['secids']});legs={};coverage=[]
    for sid in assets:
        panel=pd.DataFrame(np.nan,index=dates,columns=FIELDS)
        for frame,fields in [(activity,FIELDS[:4]),(borrow,FIELDS[4:6])]:
            if len(frame):panel[fields]=match_states(dates,np.full(len(dates),sid),frame[['secid','available',*fields]],'secid',fields,7)
        perm=interval_ids(dates,option[option.secid.eq(sid)],None,'sdate','edate','permno')
        cik=np.full(len(dates),None,object)
        for value in pd.Series(perm).dropna().unique():
            mask=perm==value;cik[mask]=interval_ids(dates[mask],company[company.permno.eq(value)],None,'linkdt','linkenddt','cik')
        for field,frame in events:panel[field]=match_states(dates,cik,frame,'cik',[field],7)
        legs[sid]=panel
        coverage.append(dict(secid=sid,**{f:int(panel[f].notna().sum()) for f in FIELDS}))
    market=global_features(dates);frames=[]
    for pair,g in data.groupby('pair',sort=True):
        g=g.copy();a,b=definitions[pair]['secids'];idx=pd.DatetimeIndex(g.date)
        x=legs[a].reindex(idx);y=legs[b].reindex(idx)
        for field in FIELDS:
            g[field+'_gap']=(x[field]-y[field]).to_numpy();g[field+'_mean']=((x[field]+y[field])/2).to_numpy()
        g[GLOBAL]=market.reindex(idx).to_numpy()
        p=pd.read_parquet(Path('artifacts/contracts/option_flex')/pair/'states.parquet',columns=['vix_previous_close']).reindex(idx)
        g['vix_previous_close']=p.vix_previous_close.to_numpy()
        g['value_rule_net']=np.where(g.rv_adjusted_value>=0,g.long_net,g.short_net)
        g['value_rule_loss']=(g.value_rule_net<0).astype(float)
        frames.append(g)
    aligned=pd.concat(frames,ignore_index=True).sort_values(['date','pair']).reset_index(drop=True)
    pd.testing.assert_frame_equal(aligned[data.columns],data.sort_values(['date','pair']).reset_index(drop=True))
    aligned.to_parquet(OUT/'aligned.parquet',index=False);pd.DataFrame(coverage).to_csv(REPORT/'asset_coverage.csv',index=False)
    write_json(OUT/'features.json',dict(pair=[f+s for f in FIELDS for s in ['_gap','_mean']],global_features=GLOBAL))
    print(f'Aligned {len(aligned):,} outcomes, {aligned.pair.nunique()} pairs.',flush=True)


def residual_statistics(frame,feature,target):
    """Remove within-pair ranks of IV, RV value and VIX from both quantities."""
    h=frame.loc[np.isfinite(frame[feature])&np.isfinite(frame[target])].copy()
    h=h[h.groupby('pair').pair.transform('size')>=20]
    if len(h)<40 or h[feature].nunique()<2:return np.nan
    ranked=h.groupby('pair')[[feature,target,'spread','rv_adjusted_value','vix_previous_close']].rank(pct=True)
    ranked-=ranked.groupby(h.pair).transform('mean')
    w=1/h.groupby('pair').pair.transform('size').to_numpy(float)
    z=np.column_stack([np.ones(len(h)),ranked[['spread','rv_adjusted_value','vix_previous_close']].fillna(0)])
    zw=z*np.sqrt(w[:,None]);x=ranked[feature].to_numpy();y=ranked[target].to_numpy()
    bx=np.linalg.lstsq(zw,x*np.sqrt(w),rcond=None)[0];by=np.linalg.lstsq(zw,y*np.sqrt(w),rcond=None)[0]
    return weighted_corr(x-z@bx,y-z@by,w)


def analyze():
    data=pd.read_parquet(OUT/'aligned.parquet');settings=json.loads((OUT/'features.json').read_text())
    features=settings['pair']+GLOBAL;rows=[];pairs=[];annual=[];events=[]
    for feature in features:
        targets=['directional_edge'] if feature.endswith('_gap') else ['value_rule_net','value_rule_loss']
        for target in targets:
            for sample,g in [('full',data),('early',data[data.end<'2022-01-01']),
                             ('later',data[data.date>='2022-01-01']),('nonoverlap',data[data.nonoverlap])]:
                rows.append(dict(feature=feature,target=target,sample=sample,
                    partial_iv_rv_vix=residual_statistics(g,feature,target),**statistics(g,feature,target,bootstrap=sample=='full')))
                for pair,v in g.groupby('pair'):
                    h=v[np.isfinite(v[feature])&np.isfinite(v[target])]
                    if len(h)<12 or h[feature].nunique()<2 or h[target].nunique()<2:continue
                    rho=float(spearmanr(h[feature],h[target]).statistic)
                    pairs.append(dict(pair=pair,label=h.label.iloc[0],feature=feature,target=target,sample=sample,
                        rows=len(h),unique_values=h[feature].nunique(),rho=rho))
            for year,g in data.groupby(data.date.dt.year):
                annual.append(dict(feature=feature,target=target,year=int(year),**statistics(g,feature,target)))
            if '90d' in feature:
                h=data[np.isfinite(data[feature])]
                for sample,g in [('early',h[h.end<'2022-01-01']),('later',h[h.date>='2022-01-01'])]:
                    for flag,v in g.groupby(g[feature].ne(0)):
                        events.append(dict(feature=feature,target=target,sample=sample,nonzero=bool(flag),rows=len(v),
                            pairs=v.pair.nunique(),dates=v.date.nunique(),nonoverlap=int(v.nonoverlap.sum()),
                            mean_payoff=float(v.value_rule_net.mean()),loss_fraction=float(v.value_rule_loss.mean())))
    pd.DataFrame(rows).to_csv(REPORT/'correlations.csv',index=False)
    pd.DataFrame(pairs).to_csv(REPORT/'pair_periods.csv',index=False)
    pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False)
    pd.DataFrame(events).to_csv(REPORT/'event_slices.csv',index=False)
    write_json(REPORT/'manifest.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),
        source_sha256=digest(SOURCE),aligned_sha256=digest(OUT/'aligned.parquet'),code_sha256=digest(Path(__file__)),
        rows=len(data),pairs=data.pair.nunique(),features=features,policy_retrained=False,
        label_commission=.65,current_checkpoint_commission=.03,
        label_note='Existing standalone labels at the old commission, not a replay of the current portfolio.',
        direction='First-minus-second fields; long_net is long first leg / short second.',
        abstention_target='Net payoff or loss of the pre-existing RV-adjusted-value sign rule, not hindsight best-direction profit.',
        news_timing='Current-vintage diagnostic: daily news delayed 30 calendar days + 1 business day; monthly news month-end + 45 days + 1 business day.',
        event_timing='Latest relevant record/filing date plus next business day; historical initial versions remain unverified.',
        market_timing='Previous published observation plus one business day; up to seven calendar days carry.',
        borrow_treatment='Nearest 30D observation; repeated -99.99 treated as missing; option-implied signal, not actual stock-loan charges.',
        inference='Descriptive correlations, overlapping outcomes and shared-name dependence; 500 joint quarter resamples; no multiple-testing adjustment.',
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    result=pd.DataFrame(rows);print(result[(result['sample']=='full')&(result.target!='value_rule_loss')][
        ['feature','target','observations','pairs','within_pair_equal_weight_rank_corr','partial_iv_rv_vix','ci90_low','ci90_high']].to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fetch-only',action='store_true');parser.add_argument('--cached-only',action='store_true')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    before=digest(SOURCE)
    if not args.cached_only:fetch()
    if not args.fetch_only:prepare();analyze()
    assert digest(SOURCE)==before


if __name__=='__main__':main()
