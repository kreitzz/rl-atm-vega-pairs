"""Previous-close aggregate option turnover, in the pair's existing leg order."""
from contextlib import closing
import getpass
import os
from pathlib import Path
import numpy as np
import pandas as pd
import psycopg2
from scripts.test_wrds_pair_signals import match_states

RAW=Path('artifacts/contracts/atm65_volume_oi_raw')
ADDED=['volume_oi_gap','volume_oi_available']


def fetch(assets):
    RAW.mkdir(parents=True,exist_ok=True)
    with closing(psycopg2.connect(host='wrds-pgdata.wharton.upenn.edu',port=9737,dbname='wrds',
        user=os.getenv('WRDS_USERNAME',getpass.getuser()),sslmode='require',connect_timeout=20,
        application_name='bounded_volume_oi_rl')) as db:
        db.set_session(readonly=True,autocommit=True)
        with db.cursor() as cursor:
            cursor.execute("SET statement_timeout='20s'");cursor.execute("SET lock_timeout='2s'")
        for year in range(2010,2026):
            path=RAW/f'volume_{year}.parquet'
            if path.exists():continue
            end=f'{year}-12-31' if year<2025 else '2025-08-29'
            with db.cursor() as cursor:
                cursor.execute("SELECT secid,date,cp_flag,volume,open_interest FROM optionm.opvold "
                    "WHERE secid=ANY(%s) AND date BETWEEN %s AND %s AND cp_flag IN ('C','P')",
                    (sorted(assets),f'{year}-01-01',end))
                rows=cursor.fetchmany(220001)
                if len(rows)>220000:raise MemoryError('Annual volume query exceeds 220,000-row cap')
                data=pd.DataFrame(rows,columns=[d.name for d in cursor.description])
            temp=path.with_suffix('.tmp');data.to_parquet(temp,index=False);temp.replace(path)
            print(f'Volume/OI {year}: {len(data):,} rows',flush=True)


def turnover(raw):
    raw=raw.copy();raw['date']=pd.to_datetime(raw.date);raw.secid=raw.secid.map(int)
    assert not raw.duplicated(['secid','date','cp_flag']).any()
    rows=[]
    for sid,g in raw.groupby('secid'):
        v=g.pivot(index='date',columns='cp_flag',values='volume').reindex(columns=['C','P'])
        oi=g.pivot(index='date',columns='cp_flag',values='open_interest').reindex(columns=['C','P'])
        v=v.where(v>=0).sum(axis=1,min_count=2)
        oi=oi.where(oi>=0).sum(axis=1,min_count=2)
        rows.append(pd.DataFrame(dict(secid=sid,available=v.index+pd.offsets.BDay(1),
                                     volume_oi=(v/oi.where(oi>0)).to_numpy())))
    return pd.concat(rows,ignore_index=True)


def leg_states(dates,assets):
    raw=pd.concat([pd.read_parquet(RAW/f'volume_{year}.parquet') for year in range(2010,2026)],ignore_index=True)
    states=turnover(raw)
    return {sid:match_states(dates,np.full(len(dates),sid),states,'secid',['volume_oi'],7).volume_oi
            for sid in assets}


def add_volume_oi(panel,first,second):
    result=panel.copy();gap=first.reindex(panel.index)-second.reindex(panel.index)
    observed=np.isfinite(gap)
    result[ADDED[0]]=gap.where(observed,0.).astype(float)
    result[ADDED[1]]=observed.astype(float)
    pd.testing.assert_frame_equal(result[panel.columns],panel)
    return result
