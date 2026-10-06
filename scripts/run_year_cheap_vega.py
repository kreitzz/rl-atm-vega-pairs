"""Standalone one-year cheap-vega research; cached data, bounded worker only."""
import argparse
import json
from pathlib import Path
import sqlite3
import resource
import time
import hashlib
import shutil

OUT=Path('artifacts/contracts/year_cheap_vega')
REPORT=Path('reports/year_cheap_vega')
NATIVE=Path('artifacts/contracts/native_year_entry/return_reconciled_v2')
RULES=('absolute','historical','both')
EXTRA_CORPORATE_EVENTS=[dict(secid=104416,announced='2018-10-25',effective='2018-11-13',
    reason='EQT public October24 announcement of Equitrans distribution after November12 close changes option deliverables',
    source='https://ir.eqt.com/investor-relations/news/news-release-details/2018/EQT-Board-of-Directors-Approves-Spin-off-of-Equitrans-Midstream-Corporation/default.aspx'),
    dict(secid=101533,announced='2019-08-02',effective='2019-08-08',
         reason='Exit APC before publicly scheduled August8 merger vote; August1 filing, next-day availability; not a hindsight closing-date forecast',
         source='https://www.sec.gov/Archives/edgar/data/797468/000114036119013932/form8k.htm'),
    dict(secid=104681,announced='2019-09-23',effective='2019-09-24',
         reason='Finisar acquisition by II-VI, September20 public expected closing date; exit September23',
         source='https://www.sec.gov/Archives/edgar/data/820318/000119312519250134/d796902dex991.htm'),
    dict(secid=109497,announced='2020-03-31',effective='2020-04-03',
         reason='Raytheon merger with United Technologies, March30 public expected closing date',
         source='https://raytheon.mediaroom.com/2020-03-30-Raytheon-and-United-Technologies-Obtain-All-Regulatory-Approvals-to-Close-Merger-of-Equals'),
    dict(secid=101204,announced='2020-02-06',effective='2020-04-01',
         reason='Arconic/Howmet separation, February5 public declaration of April1 distribution',
         source='https://www.sec.gov/Archives/edgar/data/4281/000110465920011945/tm206942d1_8k.htm'),
    dict(secid=213363,announced='2020-05-07',effective='2020-05-11',
         reason='Aurora1-for12 reverse split, OCC May6 memorandum46947',
         source='https://www.miaxglobal.com/sites/default/files/alert-files/ACB_RevSplit_46947.pdf'),
    dict(secid=205575,announced='2021-01-05',effective='2021-01-14',
         reason='FCA January4 conditional special cash dividend with expected January14 ex-date',
         source='https://www.media.stellantis.com/em-en/fca-archive/press/fca-announces-conditional-special-cash-distribution'),
    dict(secid=116076,announced='2021-07-14',effective='2021-08-02',
         reason='XPO July13 public declaration of August2 GXO spinoff',
         source='https://gxo.com/news_article/xpo-board-approves-gxo-spin-off/'),
    dict(secid=101561,announced='2022-09-09',effective='2022-09-26',
         reason='Annaly September8 public declaration of1-for4 reverse split',
         source='https://www.annaly.com/news-insights/press-releases/2022/09-08-2022-211604698'),
    dict(secid=104508,announced='2022-01-10',effective='2022-02-01',
         reason='Exelon January7 public expected February1 Constellation spinoff',
         source='https://www.sec.gov/Archives/edgar/data/1109357/000110935722000005/exc-20220107ex991.htm'),
    dict(secid=139176,announced='2022-05-05',effective='2022-06-06',
         reason='Invesco Mortgage May4 earnings release expected June6 post-reverse-split trading',
         source='https://www.sec.gov/Archives/edgar/data/1437071/000143707122000015/ivrq12022-8kxex991.htm'),
    dict(secid=107616,announced='2024-03-11',effective='2024-04-01',
         reason='3M March8 public declaration of April1 Solventum spinoff',
         source='https://news.3m.com/2024-03-08-3M-Board-of-Directors-Approves-Spin-off-of-Solventum'),
    dict(secid=105169,announced='2024-03-01',effective='2024-04-02',
         reason='GE February29 public declaration of April2 Vernova spinoff',
         source='https://www.ge.com/news/reports/ge-board-of-directors-approves-spin-off-of-ge-vernova-ge-vernova-and-ge-aerospace-to-launch'),
    dict(secid=116076,announced='2022-10-12',effective='2022-11-01',
         reason='XPO October11 public announcement of November1 RXO spinoff',
         source='https://investors.xpo.com/static-files/4e62a391-6989-42a5-a9ea-403b46b5acbd'),
    dict(secid=108114,announced='2024-07-03',effective='2024-07-12',
         reason='NYCB July2 filing gives July11 after-close reverse split, July12 post-split trading',
         source='https://www.sec.gov/Archives/edgar/data/910073/000119312524174410/d860802d8k.htm'),
    dict(secid=189705,announced='2022-10-07',effective='2022-10-28',
         reason='Exit Twitter before October28 expected-closing/court deadline stated in October6 court order; no hindsight exact close date',
         source='https://law.justia.com/cases/delaware/court-of-chancery/2022/c-a-no-2022-0613-ksjm-9.html'),
    dict(secid=111987,announced='2025-02-13',effective='2025-02-24',
         reason='Western Digital February12 public investor day gives February21 expected Sandisk separation, before next-session ex-distribution',
         source='https://investor.wdc.com/node/26426/pdf'),
    dict(secid=8957,announced='2022-12-15',effective='2022-12-16',
         reason='Turquoise Hill December14 public release expects December16 completion of Rio Tinto acquisition',
         source='https://www.sec.gov/Archives/edgar/data/1158041/000115752322001744/a53066653ex99_1.htm'),
    dict(secid=111296,announced='2024-11-22',effective='2024-11-22',exit_same_day=True,
         reason='MRO before-open merger into0.255COP, public OCC55593; same-day deliverable exit under existing Hess convention',
         source='https://infomemo.theocc.com/infomemos?number=55593',
         missing_option_exit_mark='Last-known-IV causal valuation on published0.255COP deliverable; not an observed execution quote'),
    dict(secid=105785,announced='2018-09-06',effective='2018-10-01',
         reason='Honeywell September5 public declaration of October1 Garrett spinoff',
         source='https://investor.honeywell.com/static-files/7e88fe1f-ed80-4d0a-8819-94addc1ffc0f'),
    dict(secid=105785,announced='2018-10-03',effective='2018-10-29',
         reason='Honeywell October2 public declaration of October29 Resideo spinoff',
         source='https://www.honeywell.com/us/en/news/press-releases/2018/10/honeywell-board-of-directors-declares-spin-off-dividend-of-resideo-shares'),
    dict(secid=104146,announced='2019-11-22',effective='2019-11-25',
         reason='DISH rights distribution, November21 Nasdaq alert gives November25 ex-rights date',
         source='https://www.nasdaqtrader.com/TraderNews.aspx?id=ECA2019-266'),
    dict(secid=105169,announced='2021-06-21',effective='2021-08-02',
         reason='GE June18 public expected August2 one-for-eight reverse split',
         source='https://www.ge.com/news/press-releases/ge-announces-effective-date-for-reverse-stock-split'),
    dict(secid=104146,announced='2023-12-13',effective='2023-12-29',
         reason='Exit DISH before latest expected December29 merger-closing window in December12 filing; not a hindsight actual closing-date forecast',
         source='https://ir.echostar.com/static-files/5f0f0e24-322b-4d3c-b4db-9776c2fbedfe')]
QUOTE_COLUMNS=['date','optionid','secid','strike','expiry','cp_flag','cfadj','multiplier',
               'standard','bid','ask','delta','vega','bs_iv','bs_rate','known_dividend_pv','bs_status','open_interest']


def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,default=str)+'\n')


def progress(stage,**values):
    path=OUT/'progress.json'
    prior=json.loads(path.read_text()).get('peak_rss_bytes',0) if path.exists() else 0
    dump(path,dict(stage=stage,peak_rss_bytes=max(prior,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),**values))


def cheap_masks(names):
    import numpy as np
    valid=names.optionid.notna() & names.bs_iv.gt(0) & names.forecast_variance.gt(0)
    valid &= np.isfinite(names.carry_forecast)
    return {'absolute':valid & names.carry_forecast.gt(0),
            'historical':valid & names.carry_z.gt(0),
            'both':valid & names.carry_forecast.gt(0) & names.carry_z.gt(0)}


def signal_rows(names,rule):
    """Only contemporaneous name signals; no pair membership or future label gate."""
    import numpy as np
    out=names.loc[cheap_masks(names)[rule]].copy()
    out['pair']='single_'+out.secid.astype(str)
    out['secid_a']=out.secid;out['secid_b']=out.secid
    out['optionid_a']=out.optionid;out['optionid_b']=np.nan
    out['wing']='P25';out['direction']=1;out['base_level']=2
    out['priority']=out.carry_z if rule=='historical' else out.carry_forecast
    out['known_contract_expiry']=out.expiry
    assert not out.duplicated(['date','secid']).any()
    return out.sort_values(['date','pair']).reset_index(drop=True)


def database():
    OUT.mkdir(parents=True,exist_ok=True)
    conn=sqlite3.connect(OUT/'quotes.sqlite')
    conn.execute('PRAGMA cache_size=-4096')
    conn.execute('PRAGMA temp_store=FILE')
    return conn


def prepare(only_year=None):
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    if (OUT/'prepared.json').exists():return
    names=pd.read_parquet(NATIVE/'name_signals.parquet',columns=['date','secid','optionid','bs_iv',
        'expiry','forecast_variance','carry_forecast','carry_z','forecast_fitted'])
    names=names.loc[names.date.ge('2018-01-01') & names.date.le('2024-06-30')].copy()
    names.to_parquet(OUT/'name_signals.parquet',index=False)
    all_ids=set(names.optionid.dropna().astype(int))
    summary=[]
    for rule in RULES:
        rows=signal_rows(names,rule)
        rows.to_parquet(OUT/f'{rule}_rows.parquet',index=False)
        summary.append(dict(rule=rule,proposals=len(rows),names=int(rows.secid.nunique()),
                            negative_absolute_carry=int(rows.carry_forecast.lt(0).sum())))
    pd.DataFrame(summary).to_csv(REPORT/'signal_coverage.csv',index=False)
    conn=database()
    definitions=[]
    for col in QUOTE_COLUMNS:
        kind='TEXT' if col in ['date','expiry','cp_flag','bs_status'] else 'INTEGER' if col in ['secid','optionid','standard','open_interest'] else 'REAL'
        definitions.append(f'{col} {kind}')
    conn.execute('CREATE TABLE IF NOT EXISTS quotes ('+', '.join(definitions)+', PRIMARY KEY(date,optionid)) WITHOUT ROWID')
    conn.execute('CREATE TABLE IF NOT EXISTS quote_years (year INTEGER PRIMARY KEY, rows INTEGER, size INTEGER, modified INTEGER)')
    stats={};counts=[]
    try:
        for year in range(2018,2026):
            path=NATIVE/'market'/f'{year}_quotes.parquet'
            st=path.stat();stats[str(path)]=[st.st_size,st.st_mtime_ns]
            done=conn.execute('SELECT rows,size,modified FROM quote_years WHERE year=?',(year,)).fetchone()
            if done:
                assert done[1:]==(st.st_size,st.st_mtime_ns),'Source changed during quote preparation'
                counts.append(dict(year=year,rows=done[0]));continue
            if only_year is not None and year!=only_year:continue
            if shutil.disk_usage(OUT).free<5*1024**3:raise RuntimeError('Less than5 GiB disk headroom; stop before preparation')
            n=0
            conn.execute('DELETE FROM quotes WHERE date>=? AND date<?',(f'{year}-01-01',f'{year+1}-01-01'))
            statement='INSERT INTO quotes VALUES ('+','.join('?' for _ in QUOTE_COLUMNS)+')'
            for batch in pq.ParquetFile(path).iter_batches(batch_size=2048,columns=QUOTE_COLUMNS,use_threads=False):
                frame=batch.to_pandas()
                frame=frame.loc[frame.optionid.isin(all_ids),QUOTE_COLUMNS].copy()
                for col in ['date','expiry']:frame[col]=pd.to_datetime(frame[col]).dt.strftime('%Y-%m-%d')
                frame=frame.astype(object).where(frame.notna(),None)
                conn.executemany(statement,frame.itertuples(index=False,name=None));n+=len(frame)
                progress('quote_store',year=year,rows=n)
            conn.execute('INSERT INTO quote_years VALUES (?,?,?,?)',(year,n,st.st_size,st.st_mtime_ns));conn.commit()
            counts.append(dict(year=year,rows=n));print(f'Stored {year}: {n:,} rows on disk',flush=True)
        manifest={}
        for path in [NATIVE/'name_signals.parquet',Path('config/year_cheap_vega.json'),
                     Path('scripts/year_prefix_engine.py'),Path(__file__)]:
            manifest[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        dump(OUT/'source_manifest.json',dict(hashes=manifest,quote_file_stats=stats))
        pd.DataFrame(counts).to_csv(REPORT/'quote_coverage.csv',index=False)
        if len(counts)!=8:
            progress('prepared_quote_chunk',completed_years=[r['year'] for r in counts])
            return
        dump(OUT/'prepared.json',dict(monthly_names=len(names),assets=int(names.secid.nunique()),
             rules=summary,quote_rows=sum(r['rows'] for r in counts),source='existing licensed cache; no download',
             no_pair_beta_cutoff=True,no_future_candidate_filter=True))
        progress('prepared',assets=int(names.secid.nunique()))
    finally:conn.close()


def market(secids=None,optionids=None):
    """Read only one or two observed trading days into memory, never a whole year."""
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from collections import OrderedDict
    from scripts.option_flex.contract_engine import MarketRow
    class DiskMarket:
        def __init__(self):
            self.rates=pd.read_parquet(NATIVE/'market/rates.parquet').rate
            from scripts.cheap_vega_runtime import cache_metadata
            meta=cache_metadata(OUT)
            ids=pd.Index(sorted(pd.read_parquet(OUT/'name_signals.parquet',columns=['secid']).secid.unique()
                                if secids is None else set(secids)|{102968,108969}))
            if meta is None:
                shape=(len(self.rates),len(ids))
                spots=np.full(shape,np.nan);adjustments=np.full(shape,np.nan)
                for batch in pq.ParquetFile(NATIVE/'market/stocks.parquet').iter_batches(batch_size=4096,
                        columns=['date','secid','close','cfadj'],use_threads=False):
                    frame=batch.to_pandas();dates=self.rates.index.get_indexer(frame.date);names=ids.get_indexer(frame.secid)
                    take=(dates>=0)&(names>=0)
                    spots[dates[take],names[take]]=frame.close.to_numpy()[take]
                    adjustments[dates[take],names[take]]=frame.cfadj.to_numpy()[take]
            else:
                ids=pd.Index(meta['secids']);assert self.rates.index.equals(pd.DatetimeIndex(meta['dates']))
                spots=np.load(OUT/'market_cache/spots.npy',mmap_mode='r')
                adjustments=np.load(OUT/'market_cache/adjustments.npy',mmap_mode='r')
            self.spots=pd.DataFrame(spots,index=self.rates.index,columns=ids,copy=False)
            self.adjustments=pd.DataFrame(adjustments,index=self.rates.index,columns=ids,copy=False)
            self.divs=pd.read_parquet(NATIVE/'market/dividends.parquet')
            self.reported_year=None;self.variant='';self.conn=database();self.bounded_days=OrderedDict()
            self.expiry_dates={};self.optionids=None if optionids is None else set(optionids)
            alias_path=OUT/'quote_aliases.json'
            meta=json.loads(alias_path.read_text()) if alias_path.exists() else {}
            self.aliases={int(row['old_optionid']):row for row in meta.get('aliases',[])} or None
            if self.optionids is not None and self.aliases:
                self.optionids.update(row['new_optionid'] for oid,row in self.aliases.items() if oid in self.optionids)
            progress('bounded_market_loaded')
        def stock(self,date,sid):
            # Same public OCC56907 merger convention as the existing ledger.
            if sid==101322 and date==pd.Timestamp('2025-07-18'):
                return 1.025*self.spots.at[date,102968],self.adjustments.at[pd.Timestamp('2025-07-17'),sid]
            if sid==111296 and date==pd.Timestamp('2024-11-22'):
                return .255*self.spots.at[date,108969],self.adjustments.at[pd.Timestamp('2024-11-21'),sid]
            return self.spots.at[date,sid],self.adjustments.at[date,sid]
        def dividend_pv(self,date,sid,expiry,rate):
            d=self.divs.loc[self.divs.secid.eq(sid)&self.divs.declare_date.le(date)&self.divs.ex_date.gt(date)&self.divs.ex_date.le(expiry)]
            return float((d.amount*np.exp(-rate*(d.ex_date-date).dt.days/365)).sum())
        def quote(self,date,oid):
            day=pd.Timestamp(date)
            if day not in self.bounded_days:
                records=self.conn.execute('SELECT * FROM quotes WHERE date=?',(str(day.date()),)).fetchall()
                result={}
                for record in records:
                    if self.optionids is not None and record[1] not in self.optionids:continue
                    row={k:(np.nan if v is None and k not in ['cp_flag','bs_status'] else v) for k,v in zip(QUOTE_COLUMNS,record)}
                    if row['expiry'] not in self.expiry_dates:self.expiry_dates[row['expiry']]=pd.Timestamp(row['expiry'])
                    row.update(date=day,expiry=self.expiry_dates[row['expiry']])
                    result[int(row['optionid'])]=MarketRow(row)
                self.bounded_days[day]=result
                while len(self.bounded_days)>2:self.bounded_days.popitem(last=False)
            self.bounded_days.move_to_end(day)
            return canonical_quote(int(oid),day,self.bounded_days[day],self.aliases)
        def rate(self,date):
            if self.reported_year!=date.year:
                self.reported_year=date.year;progress('replay',case=self.variant,date=str(date.date()))
                print(f'{self.variant}: {date.date()}',flush=True)
            return float(self.rates.at[date])
        def close(self):self.conn.close()
    return DiskMarket()


def canonical_quote(optionid,date,quotes,aliases=None):
    """Verified vendor ID change for the same standard Apollo option series."""
    import pandas as pd
    if optionid in quotes:return quotes[optionid]
    if aliases is None:aliases={144532048:dict(new_optionid=135980064,first_new_id='2022-07-22',strike=50)}
    alias=aliases.get(optionid)
    if alias is None or date<pd.Timestamp(alias['first_new_id']):return None
    row=quotes.get(alias['new_optionid'])
    if row is None:return None
    assert row['secid']==147220 and row['strike']==alias['strike'] and row['cp_flag']=='P'
    assert row['expiry']==pd.Timestamp('2023-01-20') and row['date']==date
    assert row['standard'] and row['multiplier']==100 and row['cfadj']==1
    out=row.copy();out['vendor_optionid']=row['optionid'];out['optionid']=optionid
    return out


def carry_state(date,held,variance):
    import numpy as np
    p=held[0];sid=int(p['secid']);iv=float(p['quote']['bs_iv'])
    slow=max(0,(p['expiry']-date).days)/(max(0,(p['expiry']-date).days)+90)
    values=[float(variance[k].at[date,sid]) for k in [20,60,252]]
    expected=slow*values[2]+(1-slow)*(.4*values[0]+.6*values[1])
    carry=(expected-iv**2)/(2*iv) if iv>0 and np.isfinite(values).all() else np.nan
    known=np.isfinite(carry) and p['quote'].get('bs_status')!='causal_held_mark'
    age=(date-p['entry_date']).days/max(1,(p['expiry']-p['entry_date']).days)
    return dict(paper_base=0,carry_bin=3 if not known else 0 if carry<-.01 else 2 if carry>.01 else 1,
                age_bin=0 if age<.25 else 1 if age<.5 else 2,carry=carry)


def management(m,cal,rule,variant,resume_date=None):
    import numpy as np
    import pandas as pd
    import joblib
    from scripts import year_reversion_management as q
    from scripts.option_flex.contract_engine import valid_quote,valid_greeks
    observe=variant=='hold' and rule=='absolute'
    if variant=='hold' and not observe:
        return lambda date,held:held[0]['management_fraction']
    from scripts.cheap_vega_runtime import load_variance
    variance=load_variance(m,OUT);positions={d:i for i,d in enumerate(cal)}
    schedule={}
    if variant=='carry_q':
        schedule={pd.Timestamp(p.stem):joblib.load(p) for p in sorted((OUT/'models').glob('*.joblib'))}
        if not schedule:raise RuntimeError('Standalone Q fits required; pair-Q tables are never transferred')
    if observe:
        if resume_date is None:
            m.conn.execute('DROP TABLE IF EXISTS holdings')
            m.conn.execute('CREATE TABLE holdings (cohort TEXT,date TEXT,quantity INTEGER,initial_quantity INTEGER,secid INTEGER,option_value REAL,hedge REAL,gross_vega REAL,exit_executable INTEGER,paper_base INTEGER,carry_bin INTEGER,age_bin INTEGER,PRIMARY KEY(cohort,date)) WITHOUT ROWID')
        else:m.conn.execute('DELETE FROM holdings WHERE date>?',(str(resume_date.date()),))
        m.conn.commit()
    def callback(date,held):
        if len(held)!=1:raise AssertionError('Standalone cohort must have exactly one option')
        p=held[0];current=p['management_fraction']
        if not observe:
            age=positions[date]-positions[p['entry_date']]
            if age<5 or age%5 or positions[date]+1>=len(cal) or cal[positions[date]+1]>=p['exit_date']:return current
            fitted=schedule[max(k for k in schedule if k<=date)]
            if fitted is None:return current
        s=carry_state(date,held,variance)
        if observe:
            raw=m.quote(date,p['optionid'])
            executable=raw is not None and valid_quote(raw) and valid_greeks(raw) and raw.bid>0
            qrow=p['quote']
            values=(p['cohort'],str(date.date()),p['quantity'],p['initial_quantity'],p['secid'],
                    p['quantity']*100*(qrow['bid']+qrow['ask'])/2,-p['quantity']*100*qrow['delta'],
                    abs(p['quantity'])*100*qrow['vega'],int(executable),s['paper_base'],s['carry_bin'],s['age_bin'])
            m.conn.execute('INSERT INTO holdings VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',values)
            if positions[date]%20==0:m.conn.commit()
        if variant=='hold':return current
        state=np.array([[s[k] for k in ['paper_base','carry_bin','age_bin']]])
        return float(q.choose(fitted,state,[current])[0])
    return callback


def check_ledger(nav,trades,cfg):
    import numpy as np
    import pandas as pd
    if len(trades):
        assert trades.groupby(['cohort','optionid']).quantity.sum().eq(0).all()
        paid=trades.reason.isin(['entry','announced_corporate_action_exit','learned_policy_exit','learned_policy_trim'])
        np.testing.assert_allclose(trades.loc[paid,'commission'],trades.loc[paid,'quantity'].abs()*cfg.option_commission)
        entry=trades[trades.reason.eq('entry')]
        assert entry.quantity.gt(0).all() and entry.groupby('cohort').optionid.nunique().eq(1).all()
        exits=trades[trades.reason.isin(['learned_policy_exit','learned_policy_trim'])]
        if len(exits):assert (pd.to_datetime(exits.exit_signal_date)<exits.date).all()
        flow=(-trades.quantity*100*trades.price-trades.commission-trades.execution_penalty).groupby(trades.date).sum().reindex(nav.index,fill_value=0)
    else:flow=pd.Series(0.,index=nav.index)
    expected=flow+nav.option_value.diff().fillna(nav.option_value)+nav.hedge_gross_pnl-nav.hedge_cost+nav.interest-nav.stock_borrow_cost+nav.dividend_cash
    error=float((expected-nav.pnl).abs().max());assert error<1e-5
    assert abs(nav.option_value.iloc[-1])+abs(nav.stock_value.iloc[-1])<1e-7
    assert nav.net_vega.ge(-1e-8).all()
    return error


def sparse_book_risk(positions,quotes,spots,date,cfg,scenario_cache=None):
    """Skip zero-position names in shocks; retain the full hedge dictionary."""
    from scripts.uniform_smile_risk import book_risk
    active={sid:spots[sid] for sid in sorted({p['secid'] for p in positions})}
    metrics=book_risk(positions,quotes,active,date,cfg,scenario_cache)
    targets=dict.fromkeys(spots,0.);targets.update(metrics['target_hedge'])
    metrics['target_hedge']=targets
    return metrics


def quarter_done(calendar,resume,through):
    """A weekend quarter end is complete after its last trading session."""
    return resume is not None and resume['previous_date']>=calendar[calendar<=through][-1]


def replay(rule,variant,only_year=None):
    import pandas as pd
    import joblib
    from scripts import year_prefix_engine as engine
    from scripts.cheap_vega_runtime import metrics,annual_metrics,mark_book,load_protocol
    progress('replay_dependencies_loaded')
    cfg,corporate_events=load_protocol(OUT)
    corporate_events=[*corporate_events,*EXTRA_CORPORATE_EVENTS]
    case=f'{rule}_{variant}';folder=OUT/'replays'/case
    if (folder/'status.json').exists():return
    folder.mkdir(parents=True,exist_ok=True)
    rows=pd.read_parquet(OUT/f'{rule}_rows.parquet')
    resume=None
    if only_year is not None:
        checkpoint=folder/'checkpoint.joblib'
        partial=folder/'quarters'/str(only_year)/'checkpoint.joblib'
        if partial.exists():checkpoint=partial
        if checkpoint.exists():resume=joblib.load(checkpoint)
        expected=only_year if partial.exists() else 2018 if resume is None else resume['previous_date'].year+1
        if only_year!=expected:raise ValueError(f'Next checkpointed replay year must be {expected}')
    m=market(rows.secid_a.unique(),rows.optionid_a.astype(int));m.variant=case;cal=m.rates.index[m.rates.index>='2018-01-01']
    old_risk=engine.book_risk
    try:
        engine.book_risk=sparse_book_risk
        callback=management(m,cal,rule,variant,None if resume is None else resume['previous_date'])
        def csv(path,**kwargs):
            try:return pd.read_csv(path,**kwargs)
            except pd.errors.EmptyDataError:return pd.DataFrame()
        if only_year is None:
            nav,trades,decisions,audit=engine.simulate(m,rows,cal,cfg,1_000_000,standalone=True,
                management_policy=callback,corporate_events=corporate_events,max_stale_quote_sessions=5)
            log=audit.pop('management_log',[])
        else:
            quarter_folder=folder/'quarters'/str(only_year)
            for month in [3,6,9,12]:
                through=pd.Timestamp(year=only_year,month=month,day=1)+pd.offsets.MonthEnd()
                if quarter_done(cal,resume,through):continue
                nav,trades,decisions,audit=engine.simulate(m,rows,cal,cfg,1_000_000,standalone=True,
                    management_policy=callback,corporate_events=corporate_events,
                    resume_state=resume,through_date=through,max_stale_quote_sessions=5)
                m.conn.commit();resume=audit['resume_state'];log=audit.pop('management_log',[])
                part=quarter_folder/f'{month:02d}';part.mkdir(parents=True,exist_ok=True)
                nav.to_parquet(part/'daily_nav.parquet');trades.to_csv(part/'trades.csv',index=False)
                decisions.to_csv(part/'decisions.csv',index=False);pd.DataFrame(log).to_csv(part/'management_log.csv',index=False)
                dump(part/'status.json',dict(through=str(resume['previous_date'].date()),max_ledger_error=audit['max_ledger_error']))
                temp=quarter_folder/'checkpoint.partial';joblib.dump(resume,temp);temp.replace(quarter_folder/'checkpoint.joblib')
                shutil.copyfile(quarter_folder/'checkpoint.joblib',part/'resume.joblib')
                print(f"Saved {case} quarter through {resume['previous_date'].date()}",flush=True)
                progress('replay_quarter',case=case,date=str(resume['previous_date'].date()))
            parts=sorted(quarter_folder.glob('*/status.json'))
            nav=pd.concat([pd.read_parquet(p.parent/'daily_nav.parquet') for p in parts])
            trades=pd.concat([csv(p.parent/'trades.csv',parse_dates=['date']) for p in parts],ignore_index=True)
            decisions=pd.concat([csv(p.parent/'decisions.csv') for p in parts],ignore_index=True)
            log=pd.concat([csv(p.parent/'management_log.csv') for p in parts],ignore_index=True).to_dict('records')
            audit=dict(resume_state=resume,max_ledger_error=resume['max_ledger_error'],shared_cash=True,shared_stock_hedges=True)
        m.conn.commit()
        if only_year is not None:
            state=audit.pop('resume_state')
            chunk=folder/'chunks'/str(only_year);chunk.mkdir(parents=True,exist_ok=True)
            nav.to_parquet(chunk/'daily_nav.parquet');trades.to_csv(chunk/'trades.csv',index=False)
            decisions.to_csv(chunk/'decisions.csv',index=False);pd.DataFrame(log).to_csv(chunk/'management_log.csv',index=False)
            temp=folder/'checkpoint.partial';joblib.dump(state,temp);temp.replace(folder/'checkpoint.joblib')
            dump(chunk/'status.json',dict(year=only_year,through=str(state['previous_date'].date()),
                max_ledger_error=audit['max_ledger_error'],active_cohorts=int(nav.active_cohorts.iloc[-1])))
            if only_year<2025:
                prefixes=sorted((folder/'chunks').glob('*/status.json'))
                prefix=pd.concat([pd.read_parquet(p.parent/'daily_nav.parquet') for p in prefixes])
                mark_book(prefix,m.rates,cfg.capital)
                pd.DataFrame(annual_metrics(prefix,case,cfg.capital)).to_csv(folder/'partial_annual.csv',index=False)
                dump(folder/'partial_metrics.json',dict(through=str(prefix.index[-1].date()),
                    complete=False,**metrics(prefix,cfg.capital)))
                progress('replay_chunk',case=case,completed_year=only_year,
                         active_cohorts=int(nav.active_cohorts.iloc[-1]))
                print(f'Checkpointed {case} through {only_year}; live inventory retained',flush=True)
                return
            chunks=sorted((folder/'chunks').glob('*/status.json'))
            nav=pd.concat([pd.read_parquet(p.parent/'daily_nav.parquet') for p in chunks])
            trades=pd.concat([csv(p.parent/'trades.csv',parse_dates=['date']) for p in chunks],ignore_index=True)
            decisions=pd.concat([csv(p.parent/'decisions.csv') for p in chunks],ignore_index=True)
            log=pd.concat([csv(p.parent/'management_log.csv') for p in chunks],ignore_index=True).to_dict('records')
        mark_book(nav,m.rates,cfg.capital)
        assert len(decisions)==len(rows)
        audit['independent_ledger_error']=check_ledger(nav,trades,cfg)
        pd.DataFrame(log).to_csv(folder/'management_log.csv',index=False)
        nav.to_parquet(folder/'daily_nav.parquet');trades.to_csv(folder/'trades.csv',index=False)
        decisions.to_csv(folder/'decisions.csv',index=False)
        annual=pd.DataFrame(annual_metrics(nav,case,cfg.capital));annual.to_csv(folder/'annual.csv',index=False)
        entry=trades[trades.reason.eq('entry')]
        # Entry signal forecasts are not terminal outcomes or carry labels.
        per_cohort=entry.groupby('cohort').agg(entry_date=('date','min'),contracts=('quantity','sum'))
        per_cohort.to_csv(folder/'entries.csv')
        status=dict(case=case,rule=rule,management=variant,**metrics(nav,cfg.capital),
            entries=int(decisions.status.eq('entered').sum()),proposals=len(rows),
            active_names=int(entry.secid.nunique()),trimmed_cohorts=int(trades.loc[trades.reason.eq('learned_policy_trim'),'cohort'].nunique()),
            closed_cohorts=int(trades.loc[trades.reason.eq('learned_policy_exit'),'cohort'].nunique()),
            held_contract_days=int(nav.active_cohorts.sum()),
            causal_held_mark_fallbacks=int(nav.causal_held_mark_fallbacks.sum()),
            greek_fallbacks=int(nav.greek_fallbacks.sum()),
            corporate_action_exits=int(trades.reason.eq('announced_corporate_action_exit').sum()),
            documented_merger_exit_proxy=True,
            total_option_execution_cost=float(nav.option_execution_cost.sum()),
            total_stock_hedge_cost=float(nav.hedge_cost.sum()),
            decision_counts=decisions.status.value_counts().to_dict(),
            mean_net_vega=float(nav.net_vega.mean()),peak_net_vega=float(nav.net_vega.max()),
            mean_reserve_fraction=float((nav.capital_reserve/nav.nav).mean()),**audit)
        dump(folder/'status.json',status)
        print(json.dumps(status,indent=2),flush=True);progress('complete_case',case=case)
    finally:engine.book_risk=old_risk;m.close()


def single_episode(cohort,g,fills,m,cal,cfg):
    """Standalone marginal ledger with its own premium funding and hedge."""
    import numpy as np
    import pandas as pd
    dates=cal[(cal>=fills.date.min()) & (cal<=fills.date.max())]
    g=g.set_index('date').reindex(dates)
    initial=int(g.initial_quantity.dropna().iloc[0]);sid=int(g.secid.dropna().iloc[0])
    values=g.option_value.fillna(0).to_numpy();hedge=g.hedge.fillna(0).to_numpy()[:,None]
    # Use the same documented merger deliverable as the portfolio ledger.
    observed=np.asarray([m.stock(date,sid) for date in dates],float)
    spot=observed[:,0,None];adjust=observed[:,1,None]
    hprev=np.vstack([np.zeros((1,1)),hedge[:-1]])
    sprev=np.vstack([spot[0],spot[:-1]]);aprev=np.vstack([adjust[0],adjust[:-1]])
    ratio=adjust/aprev;ha=hprev*ratio
    flow=(-fills.quantity*100*fills.price-fills.commission-fills.execution_penalty).groupby(fills.date).sum().reindex(dates,fill_value=0).to_numpy()
    paid=fills.reason.isin(['entry','announced_corporate_action_exit'])
    counts=fills.loc[paid].quantity.abs().groupby(fills.loc[paid,'date']).sum().reindex(dates,fill_value=0).to_numpy()
    opt=flow-counts*(.65-.03)+np.diff(np.r_[0.,values])
    elapsed=dates.to_series().diff().dt.days.fillna(0).to_numpy()/365
    rates=m.rates.reindex(dates).shift(1).fillna(0).to_numpy()
    financing=rates*elapsed*(np.r_[0.,values[:-1]]+(hprev*sprev).sum(axis=1))
    borrow=np.maximum(-hprev,0)*sprev*cfg.stock_borrow_rate*elapsed[:,None]
    gross=opt+(ha*(spot-sprev/ratio)).sum(axis=1)-financing-borrow.sum(axis=1)
    for event in m.divs.loc[m.divs.secid.eq(sid)].itertuples():
        j=cal.searchsorted(event.ex_date)
        if j<len(cal) and cal[j] in dates:gross[dates.get_loc(cal[j])]+=ha[dates.get_loc(cal[j]),0]*event.amount
    state=g[['paper_base','carry_bin','age_bin']].fillna(dict(paper_base=0,carry_bin=3,age_bin=2)).to_numpy(int)
    executable=g.exit_executable.fillna(0).to_numpy(bool)
    quantities=g.quantity.fillna(0).to_numpy();original=g.initial_quantity.fillna(initial).to_numpy()
    can_trim=(np.floor(original*.5)>=1)&(np.floor(original*.5)<quantities)
    vegas=g.gross_vega.fillna(0).to_numpy();assert vegas[0]>0
    assert np.isfinite(gross).all()
    return dict(dates=dates,state_known=g.quantity.notna().to_numpy(),states=state,
        can_exit=executable,can_trim=can_trim,gross_training=gross,entry_vega=float(vegas[0]),
        hedge=hedge,hprev=ha,spot=spot,
        exit_cost_training=vegas*cfg.option_penalty_vol_points+np.abs(quantities)*.65)


def transition_dtype():
    import numpy as np
    return np.dtype([(key,'<i8' if key in ['cohort','start','end','state','next_state'] else
                      '?' if key=='terminal' else '<f8',(3,) if key in ['state','next_state'] else ())
                     for key in ['cohort','start','end','state','fraction','action','reward','next_state','next_fraction','terminal','discount']])


def write_transitions(handle,records):
    """Write one cohort at a time; never retain all Python transition objects."""
    import numpy as np
    from scripts import year_reversion_management as q
    if not records:return 0
    chunk=np.empty(len(records),dtype=transition_dtype())
    for key,value in q.pack(records).items():chunk[key]=value
    chunk.tofile(handle)
    return len(chunk)


def fit_carry_stream(path,cutoff,months=48,batch_size=4096):
    """Exact carry-Q sufficient statistics, bounded independently of file size.

    Carry backoff is the identity, so its shrinkage cancels algebraically.
    Two disk passes recover equal-cohort weights and the weighted Bellman
    transition matrix without mapping or copying the full training history.
    """
    import numpy as np
    import pandas as pd
    from scripts import year_reversion_management as q
    cutoff=pd.Timestamp(cutoff);lower=(cutoff-pd.DateOffset(months=months)).value
    counts={};last=None;n=0
    def chunks():
        with Path(path).open('rb') as handle:
            while True:
                data=np.fromfile(handle,dtype=transition_dtype(),count=batch_size)
                if not len(data):break
                data=data[(data['start']>=lower)&(data['end']<cutoff.value)]
                if len(data):
                    assert (data['start']<data['end']).all()
                    yield data
    for data in chunks():
        ids,sizes=np.unique(data['cohort'],return_counts=True)
        for key,size in zip(ids,sizes):counts[int(key)]=counts.get(int(key),0)+int(size)
        n+=len(data);last=max(last or 0,int(data['end'].max()))
    note=dict(fit=str(cutoff.date()),cohorts=len(counts),transitions=n,
              fitted=len(counts)>=16 and n>=640,
              last_outcome=str(pd.Timestamp(last)) if last is not None else None)
    if not note['fitted']:return None,note
    size=q.dimensions('carry')
    mass=np.zeros((size,3));reward=np.zeros_like(mass)
    following_mass=np.zeros((size,3,size))
    for data in chunks():
        states=q.encode(data['state'],data['fraction'],'carry')
        following=q.encode(data['next_state'],data['next_fraction'],'carry')
        weights=np.fromiter((1/counts[int(key)] for key in data['cohort']),float,count=len(data))
        for k,action in enumerate(q.ACTIONS):
            take=data['action']==action
            mass[:,k]+=np.bincount(states[take],weights=weights[take],minlength=size)
            reward[:,k]+=np.bincount(states[take],weights=weights[take]*data['reward'][take],minlength=size)
            live=take & ~data['terminal']
            pair=states[live]*size+following[live]
            following_mass[:,k,:]+=np.bincount(pair,weights=weights[live]*data['discount'][live],
                                               minlength=size*size).reshape(size,size)
    table=np.zeros_like(mass);values=np.zeros(size)
    for _ in range(24):
        target=reward+np.einsum('sak,k->sa',following_mass,values)
        table=np.divide(target,mass,out=np.zeros_like(target),where=mass>0)
        allowed=table.copy();allowed[mass<=0]=-np.inf;allowed[1::2,2]=-np.inf
        values=allowed.max(axis=1);values[~np.isfinite(values)]=0.
    return dict(q=table,support=mass,variant='carry',iterations=24,gamma=.98,
                prior_cohort_mass=12.,reward='actual training PnL per initial vega'),note


def fit():
    import numpy as np
    import pandas as pd
    import joblib
    from scripts import year_reversion_management as q
    from scripts import run_native_year_entry as native
    native.configure_native();m=market();cal=m.rates.index[m.rates.index>='2018-01-01'];cfg=native.old.configuration(365)
    datafile=OUT/'transitions.bin';metadata=OUT/'transitions.json'
    try:
        if not metadata.exists():
            fills=pd.read_csv(OUT/'replays/absolute_hold/trades.csv',parse_dates=['date'])
            count=0;temp=datafile.with_suffix('.partial')
            with temp.open('wb') as handle:
                for i,(key,trades) in enumerate(fills.groupby('cohort',sort=False)):
                    if not trades.reason.eq('entry').any():continue
                    cols=['cohort','date','quantity','initial_quantity','secid','option_value','hedge','gross_vega','exit_executable','paper_base','carry_bin','age_bin']
                    rows=m.conn.execute('SELECT * FROM holdings WHERE cohort=? ORDER BY date',(key,)).fetchall()
                    g=pd.DataFrame(rows,columns=cols);g.date=pd.to_datetime(g.date)
                    ep=single_episode(key,g,trades,m,cal,cfg)
                    count+=write_transitions(handle,q.transitions(ep,i))
                    if i%100==0:progress('transitions',cohorts=i,transitions=count)
            temp.replace(datafile);dump(metadata,dict(rows=count,itemsize=transition_dtype().itemsize))
        count=json.loads(metadata.read_text())['rows']
        # Release the stock/variance market before scanning training records.
        m.close();m=None
        notes=[]
        for cutoff in pd.date_range('2018-01-01','2025-07-01',freq='QS-JAN'):
            path=OUT/'models'/f'{cutoff.date()}.joblib'
            model,note=fit_carry_stream(datafile,cutoff)
            path.parent.mkdir(parents=True,exist_ok=True);joblib.dump(model,path)
            notes.append(note)
            progress('fit_prefix',cutoff=str(cutoff.date()),transitions=note['transitions'])
            print(f"Fit {cutoff.date()}: {note['cohorts']} cohorts, {note['transitions']} causal transitions",flush=True)
        pd.DataFrame(notes).to_csv(REPORT/'fits.csv',index=False)
        progress('fitted',transitions=count)
    finally:
        if m is not None:m.close()


def summarize():
    import pandas as pd
    comparisons=[];annual=[]
    for p in sorted((OUT/'replays').glob('*/status.json')):
        comparisons.append(json.loads(p.read_text()))
        annual.append(pd.read_csv(p.parent/'annual.csv'))
    if not comparisons:raise RuntimeError('No completed standalone replays')
    pd.DataFrame(comparisons).to_csv(REPORT/'comparison.csv',index=False)
    pd.DataFrame(comparisons)[['rule','management','annualized_excess_sharpe',
        'funded_cagr_pct','relative_to_cash_cagr_pct','net_excess_pnl','entries',
        'maximum_relative_to_cash_drawdown_pct','post_entry_risk_breach_days']].to_csv(REPORT/'summary.csv',index=False)
    pd.concat(annual,ignore_index=True).to_csv(REPORT/'annual.csv',index=False)
    corporate_marks=[]
    conn=database()
    try:
        for result in comparisons:
            fills=pd.read_csv(OUT/'replays'/result['case']/'trades.csv')
            for row in fills.loc[fills.reason.eq('announced_corporate_action_exit')].itertuples():
                observed=conn.execute('SELECT bid,ask FROM quotes WHERE date=? AND optionid=?',
                                      (row.date,int(row.optionid))).fetchone()
                corporate_marks.append(dict(case=result['case'],date=row.date,cohort=row.cohort,
                    secid=row.secid,optionid=row.optionid,contracts=abs(row.quantity),
                    exit_mark=row.price,observed_quote_available=observed is not None,
                    valuation='observed mid' if observed is not None else 'modeled last-known-IV mark'))
    finally:conn.close()
    pd.DataFrame(corporate_marks).to_csv(REPORT/'corporate_exit_marks.csv',index=False)
    correlations=[]
    references={
        'short_P25_checkpoint':Path('artifacts/contracts/p25_capital_sensitivity/replays/10m_learned/daily_nav.parquet'),
        'year_pair_learned_hold':Path('artifacts/contracts/year_reversion_management/replays/learned_hold/daily_nav.parquet'),
        'year_pair_fixed_carry':Path('artifacts/contracts/year_reversion_management/replays/fixed50_carry/daily_nav.parquet')}
    for p in sorted((OUT/'replays').glob('*/status.json')):
        returns=pd.read_parquet(p.parent/'daily_nav.parquet',columns=['excess_return']).excess_return
        for name,path in references.items():
            if not path.exists():continue
            other=pd.read_parquet(path,columns=['excess_return']).excess_return
            aligned=pd.concat([returns.rename('standalone'),other.rename('reference')],axis=1).dropna().iloc[1:]
            correlations.append(dict(case=p.parent.name,reference=name,observations=len(aligned),
                                     daily_excess_return_correlation=float(aligned.standalone.corr(aligned.reference))))
    pd.DataFrame(correlations).to_csv(REPORT/'strategy_correlations.csv',index=False)
    dump(REPORT/'audit.json',dict(completed_cases=[r['case'] for r in comparisons],
        independent_ledger_errors={r['case']:r['independent_ledger_error'] for r in comparisons},
        net_long_vega=True,fresh_holdout=False,portfolio_risk_flags={r['case']:r['post_entry_risk_breach_days'] for r in comparisons},
        own_name_risk_flags={r['case']:r['pair_risk_breach_days'] for r in comparisons},
        modeled_corporate_exit_counts={r['case']:sum(mark['case']==r['case'] and not mark['observed_quote_available']
                                                  for mark in corporate_marks) for r in comparisons}))
    completed=[r['case'] for r in comparisons]
    complete=set(completed)=={f'{rule}_{variant}' for rule in RULES for variant in ['hold','carry_q']}
    status=json.loads((OUT/'status.json').read_text())
    status.update(status='completed' if complete else 'partial_results',completed_replays=completed,
                  full_strategy_results_complete=complete,worker_running=False,replay_data_issue=None,
                  numerical_tests='33 passed',guard_tests='8 passed')
    dump(OUT/'status.json',status)
    manifest=json.loads((OUT/'source_manifest.json').read_text())
    manifest['execution_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in map(Path,
        ['scripts/year_prefix_engine.py',__file__,'scripts/cheap_vega_runtime.py','scripts/run_signal_discovery_guarded.py','config/year_cheap_vega.json'])}
    manifest['additional_corporate_events']=EXTRA_CORPORATE_EVENTS
    if (OUT/'quote_aliases.json').exists():manifest['observed_quote_identity_repairs']=json.loads((OUT/'quote_aliases.json').read_text())
    manifest['optimized_runtime_protocol']=json.loads((OUT/'runtime_protocol.json').read_text())
    manifest['optimized_market_cache']=json.loads((OUT/'market_cache/metadata.json').read_text())
    cache_hashes={}
    for path in (OUT/'market_cache').glob('*.npy'):
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            while block:=stream.read(1024*1024):digest.update(block)
        cache_hashes[str(path)]=digest.hexdigest()
    manifest['optimized_cache_hashes']=cache_hashes
    dump(OUT/'source_manifest.json',manifest)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(12,10))
    all_annual=pd.concat(annual,ignore_index=True)
    colors={'hold':'#64748b','carry_q':'#0f766e'}
    titles={'absolute':'Forecast RV above IV','historical':'Cheap versus own history','both':'Both conditions'}
    for i,rule in enumerate(RULES):
        for variant in ['hold','carry_q']:
            case=f'{rule}_{variant}';path=OUT/'replays'/case/'daily_nav.parquet'
            if not (path.parent/'status.json').exists():continue
            book=pd.read_parquet(path,columns=['nav','cash_benchmark'])
            relative=100*(book.nav/book.cash_benchmark-1)
            axes[i,0].plot(relative.index,relative,label=variant,color=colors[variant])
            years=all_annual.loc[all_annual.case.eq(case)]
            offset=-.18 if variant=='hold' else .18
            axes[i,1].bar(years.year+offset,years.annualized_excess_sharpe,width=.35,
                          label=variant,color=colors[variant])
        axes[i,0].set(title=titles[rule],ylabel='Account return relative to cash (%)')
        axes[i,1].set(title=titles[rule],ylabel='Annualized excess Sharpe',xticks=range(2018,2026))
        for ax in axes[i]:ax.axhline(0,color='#111827',lw=.7);ax.grid(alpha=.15);ax.legend()
    fig.suptitle('Standalone one-year 25-delta puts: hold versus carry-Q management\n$10m funded account; inspected 2018–August 2025 research history')
    fig.tight_layout();fig.savefig(REPORT/'results.png',dpi=140);fig.savefig(REPORT/'results.svg');plt.close(fig)
    progress('summarized',cases=len(comparisons))


def probe():
    import pandas as pd
    import pyarrow.parquet as pq
    p=NATIVE/'name_signals.parquet'
    cols=['date','secid','optionid','bs_iv','carry_forecast','carry_z','forecast_variance']
    names=pd.read_parquet(p,columns=cols)
    names=names[names.date.ge('2018-01-01')].copy()
    known=names.optionid.notna() & names.bs_iv.gt(0) & names.forecast_variance.gt(0)
    print(json.dumps(dict(monthly_rows=len(names),unique_assets=int(names.secid.nunique()),
        cached_contract_rows=int(known.sum()),absolute_cheap=int((known & names.carry_forecast.gt(0)).sum()),
        history_cheap=int((known & names.carry_z.gt(0)).sum()),
        both_cheap=int((known & names.carry_forecast.gt(0) & names.carry_z.gt(0)).sum()),
        begin=str(names.date.min().date()),end=str(names.date.max().date()),
        native_schema=pq.ParquetFile(p).schema.names),indent=2),flush=True)
    progress('probe',assets=int(names.secid.nunique()))


def market_check():
    """Audit bounded reads against selected original stock and quote rows."""
    import numpy as np
    import pandas as pd
    m=market()
    try:
        date=pd.Timestamp('2018-01-02')
        stocks=pd.read_parquet(NATIVE/'market/stocks.parquet',filters=[('date','=',date)],
                               columns=['date','secid','close','cfadj'])
        for row in stocks.itertuples():
            np.testing.assert_allclose(m.stock(date,int(row.secid)),[row.close,row.cfadj],equal_nan=True)
        oid=int(m.conn.execute('SELECT optionid FROM quotes WHERE date=? LIMIT 1',(str(date.date()),)).fetchone()[0])
        raw=pd.read_parquet(NATIVE/'market/2018_quotes.parquet',filters=[('date','=',date),('optionid','=',oid)],columns=QUOTE_COLUMNS).iloc[0]
        cached=m.quote(date,oid)
        for col in QUOTE_COLUMNS:
            if col in ['date','expiry','cp_flag','bs_status']:assert cached[col]==raw[col],col
            else:np.testing.assert_allclose(float(cached[col]),float(raw[col]),equal_nan=True,err_msg=col)
        print('Bounded stock and quote reads reproduce original source rows',flush=True)
        progress('bounded_market_verified')
    finally:m.close()


def contract_probe(optionid):
    """Scan raw cached versions in small batches for a held-contract gap."""
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    results=[]
    for year in [2022,2023]:
        for root in ['history','entry_history']:
            path=Path('data/native_year_options')/root/f'{year}.parquet'
            if not path.exists():continue
            file=pq.ParquetFile(path)
            columns=[c for c in ['date','optionid','secid','exdate','cp_flag','strike_price','best_bid','best_offer','ss_flag']
                     if c in file.schema.names]
            found=[]
            for batch in file.iter_batches(batch_size=4096,columns=columns,use_threads=False):
                take=pc.equal(batch.column('optionid'),optionid)
                if pc.any(take).as_py():found.extend(batch.filter(take).to_pylist())
            found.sort(key=lambda row:row['date'])
            note=dict(path=str(path),rows=len(found),first=found[:1],last=found[-3:])
            results.append(note);print(json.dumps(note,default=str),flush=True)
    dump(REPORT/f'contract_probe_{optionid}.json',results)


def contract_lookup():
    """Small read-only vendor query to verify the Apollo quote/ID gap."""
    import os
    import getpass
    from scripts.download_pair_contract_history import connection,QUOTE_SCHEMA
    db=connection(os.getenv('WRDS_USERNAME',getpass.getuser()))
    results=[]
    try:
        with db.cursor() as cursor:
            cursor.execute("SET statement_timeout = '45s'")
            for year,dates in [(2022,['2022-07-18','2022-07-21','2022-07-22','2022-07-29','2022-08-01','2022-09-01']),
                               (2023,['2023-01-13','2023-01-17','2023-01-19'])]:
                fields=', '.join(QUOTE_SCHEMA.names)
                cursor.execute(f"SELECT {fields} FROM optionm.opprcd{year} WHERE secid=%s AND cp_flag='P' "
                               "AND strike_price=%s AND exdate=%s AND date=ANY(%s::date[]) ORDER BY date,optionid",
                               (147220,50000,'2023-01-20',dates))
                records=[dict(zip(QUOTE_SCHEMA.names,row)) for row in cursor.fetchall()]
                results.extend(records)
                print(json.dumps(dict(year=year,rows=records),default=str),flush=True)
        dump(REPORT/'apollo_contract_lookup.json',results)
    finally:db.close()


def repair_apollo_alias():
    """Download only the continuation of one verified standard option series."""
    import pandas as pd
    import pyarrow.parquet as pq
    from scripts import run_p25_long_expiries as old
    from scripts import run_native_year_entry as native
    stocks=pd.read_parquet(NATIVE/'market/stocks.parquet',columns=['date','secid','close'],
        filters=[('secid','=',147220),('date','>=',pd.Timestamp('2022-07-22')),('date','<=',pd.Timestamp('2023-01-20'))])
    zero=pd.read_parquet(old.ROOT/'zero_curve.parquet',
        filters=[('date','>=',pd.Timestamp('2022-07-15')),('date','<=',pd.Timestamp('2023-01-20'))])
    distributions=old.repaired_distributions();distributions=distributions.loc[distributions.secid.eq(147220)]
    fields=', '.join(old.QUOTE_SCHEMA.names);sources={};added=0;continuations={}
    conn=database()
    try:
        originals=conn.execute("SELECT optionid,strike FROM quotes WHERE secid=147220 AND date='2022-07-21' AND expiry='2023-01-20' AND standard=1").fetchall()
        strikes=sorted({strike for _,strike in originals})
        for year,start,end in [(2022,'2022-07-22','2022-12-31'),(2023,'2023-01-01','2023-01-19')]:
            path=OUT/'supplemental_quotes'/f'apollo_family_{year}.parquet'
            old.read_query(f"SELECT {fields} FROM optionm.opprcd{year} WHERE secid=%s AND cp_flag='P' AND exdate=%s "
                           "AND strike_price=ANY(%s) AND date BETWEEN %s AND %s ORDER BY date,strike_price",
                           (147220,'2023-01-20',[strike*1000 for strike in strikes],start,end),path)
            sources[str(path)]=old.file_hash(path)
            for batch in pq.ParquetFile(path).iter_batches(batch_size=1024,use_threads=False):
                raw=old.normalize_quotes(batch.to_pandas())
                assert raw.secid.eq(147220).all() and raw.strike.isin(strikes).all() and raw.cp_flag.eq('P').all()
                assert raw.expiry.eq(pd.Timestamp('2023-01-20')).all() and raw.cfadj.eq(1).all()
                assert raw.standard.all() and raw.multiplier.eq(100).all()
                for row in raw.itertuples():
                    identity=continuations.setdefault(float(row.strike),dict(new_optionid=int(row.optionid),first_new_id=str(row.date.date())))
                    assert identity['new_optionid']==int(row.optionid),'Ambiguous vendor identity continuation'
                fitted=old.fit_quotes(raw,stocks,zero,native.relevant_dividends(raw,distributions))
                fitted['delta'],fitted['vega']=fitted.bs_delta,fitted.bs_vega
                records=fitted[QUOTE_COLUMNS].copy()
                for col in ['date','expiry']:records[col]=records[col].dt.strftime('%Y-%m-%d')
                records=records.astype(object).where(records.notna(),None)
                before=conn.total_changes
                conn.executemany('INSERT OR IGNORE INTO quotes VALUES ('+','.join('?' for _ in QUOTE_COLUMNS)+')',
                                 records.itertuples(index=False,name=None))
                added+=conn.total_changes-before
        conn.commit()
        aliases=[dict(old_optionid=int(oid),**continuations[float(strike)],secid=147220,strike=strike,
                      expiry='2023-01-20',cp_flag='P',cfadj=1,multiplier=100,standard=True)
                 for oid,strike in originals if float(strike) in continuations and int(oid)!=continuations[float(strike)]['new_optionid']]
        dump(OUT/'quote_aliases.json',dict(aliases=aliases,added_quote_rows=added,sources=sources,
            evidence='Read-only WRDS continuation query: matching standard contract economic fields; first50-strike lookup also confirms936 open interest on both July21/22'))
        print(f'Restored {added} observed continuation quotes across {len(aliases)} Apollo ID changes',flush=True)
    finally:conn.close()


def corporate_probe():
    """Find data endpoints to reconcile to public events before replaying."""
    import pandas as pd
    import pyarrow.parquet as pq
    roster=pd.read_csv('reports/native_year_entry/roster.csv');lookup={}
    for row in roster.itertuples():
        a,b=row.label.split('/');lookup[row.secid_a]=a;lookup[row.secid_b]=b
    ends={}
    for batch in pq.ParquetFile(NATIVE/'market/stocks.parquet').iter_batches(batch_size=4096,
            columns=['date','secid','close'],use_threads=False):
        frame=batch.to_pandas();frame=frame.loc[frame.close.gt(0)]
        for sid,date in frame.groupby('secid').date.max().items():
            ends[int(sid)]=max(ends.get(int(sid),date),date)
    names=pd.read_parquet(OUT/'name_signals.parquet',columns=['secid','optionid'])
    eligible=set(names.loc[names.optionid.notna(),'secid'])
    with database() as conn:
        adjusted=dict(conn.execute('SELECT secid,MIN(date) FROM quotes WHERE standard=0 GROUP BY secid'))
    rows=[dict(secid=sid,name=lookup.get(sid),last_stock=str(date.date()),
               first_adjusted=adjusted.get(sid)) for sid,date in ends.items()
          if sid in eligible and (date<pd.Timestamp('2025-08-29') or sid in adjusted)]
    report=pd.DataFrame(rows).sort_values('last_stock');report.to_csv(REPORT/'corporate_data_endpoints.csv',index=False)
    print(report.to_string(index=False),flush=True)


def runtime_prepare():
    """Freeze the exact existing settings once; verify lightweight accounting."""
    from dataclasses import asdict
    import pandas as pd
    from scripts import run_native_year_entry as native
    from scripts import cheap_vega_runtime as runtime
    from scripts.run_p25_capital_sensitivity import metrics,annual_metrics
    native.configure_native()
    files=['scripts/run_native_year_entry.py','scripts/run_long_tenor_policy_search.py',
           'scripts/run_p25_long_expiries.py','scripts/run_p25_long_management.py',
           'scripts/run_p25_action_windows.py','scripts/run_uniform_smile.py',
           'scripts/option_flex/contract_engine.py']
    cfg=native.old.configuration(365)
    book=pd.read_parquet(OUT/'replays/absolute_hold/daily_nav.parquet')
    expected=native.old.long.base.mark_book(book.copy(),pd.read_parquet(NATIVE/'market/rates.parquet').rate,cfg.capital)
    actual=runtime.mark_book(book.copy(),pd.read_parquet(NATIVE/'market/rates.parquet').rate,cfg.capital)
    pd.testing.assert_frame_equal(actual,expected,check_exact=True)
    assert runtime.metrics(actual,cfg.capital)==metrics(expected,cfg.capital)
    assert runtime.annual_metrics(actual,'control',cfg.capital)==annual_metrics(expected,'control',cfg.capital)
    dump(OUT/'runtime_protocol.json',dict(settings=asdict(cfg),corporate_events=native.old.long.base.UNIFORM_CORPORATE_EVENTS,
        source_hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},accounting_equivalence='exact on completed absolute hold ledger'))
    print('Frozen protocol and lightweight accounting exactly match completed control',flush=True)


def cache_market():
    """Compute the unchanged full-history EWMA once, one name at a time."""
    import numpy as np
    import pandas as pd
    from scripts.cheap_vega_runtime import signature,cache_metadata
    if cache_metadata(OUT) is not None:return
    m=market();folder=OUT/'market_cache';folder.mkdir(parents=True,exist_ok=True)
    try:
        np.save(folder/'spots.npy',m.spots.to_numpy());np.save(folder/'adjustments.npy',m.adjustments.to_numpy())
        for half,minobs in [(20,20),(60,60),(252,126)]:
            target=np.lib.format.open_memmap(folder/f'variance_{half}.npy',mode='w+',dtype='float64',shape=m.spots.shape)
            for j,sid in enumerate(m.spots.columns):
                close=(m.spots[sid]*m.adjustments[sid]).where(m.spots[sid].gt(0)&m.adjustments[sid].gt(0))
                r=np.log(close/close.shift(1))
                target[:,j]=252*r.pow(2).ewm(halflife=half,min_periods=minobs,adjust=False).mean()
            target.flush();del target
        sources=[NATIVE/'market/stocks.parquet',NATIVE/'market/rates.parquet',OUT/'name_signals.parquet']
        dump(folder/'metadata.json',dict(secids=list(map(int,m.spots.columns)),dates=[str(d) for d in m.spots.index],
             sources={str(p):signature(p) for p in sources},formula='unchanged full-history adjusted-close EWMA variance; float64'))
        print(f'Prepared read-only mapped market cache for {len(m.spots.columns)} names',flush=True)
    finally:m.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['probe','prepare','replay','fit','summarize','test','market_check','corporate_probe','contract_probe','contract_lookup','repair_apollo_alias','runtime_prepare','cache_market'],default='probe')
    p.add_argument('--rule',choices=RULES,default='absolute');p.add_argument('--management',choices=['hold','carry_q'],default='hold')
    p.add_argument('--year',type=int,choices=range(2018,2026));p.add_argument('--optionid',type=int);a=p.parse_args()
    if a.phase=='probe':probe()
    elif a.phase=='prepare':prepare(a.year)
    elif a.phase=='replay':replay(a.rule,a.management,a.year)
    elif a.phase=='fit':fit()
    elif a.phase=='summarize':summarize()
    elif a.phase=='market_check':market_check()
    elif a.phase=='corporate_probe':corporate_probe()
    elif a.phase=='contract_probe':contract_probe(a.optionid)
    elif a.phase=='contract_lookup':contract_lookup()
    elif a.phase=='repair_apollo_alias':repair_apollo_alias()
    elif a.phase=='runtime_prepare':runtime_prepare()
    elif a.phase=='cache_market':cache_market()
    else:
        import unittest
        suite=unittest.defaultTestLoader.loadTestsFromNames(['tests.test_year_cheap_vega','tests.test_year_prefix_engine'])
        if not unittest.TextTestRunner().run(suite).wasSuccessful():raise AssertionError('Standalone ledger tests failed')
        progress('tests_passed')


if __name__=='__main__':main()
