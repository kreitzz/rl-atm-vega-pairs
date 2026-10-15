"""Train the inherited sizing bandit on actual straddle cash flows.

Single-worker phases checkpoint individual labels and funded replay months.
Sampling uses fixed calendar sessions and current ranks only. All daily live
proposals remain available, including names whose training labels are missing.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import research_atm_rv_persistence as base
from scripts import research_atm_rv_bandit as bandit
from scripts import atm_rv_persistence_engine as engine
from scripts.option_flex.contract_engine import Settings, MarketRow

CFG_PATH=Path('config/atm_rv_straddle_bandit.json')
CFG=json.loads(CFG_PATH.read_text())
OUT=Path(CFG['output_root']);REPORT=Path(CFG['report_root'])
ROOT=Path(CFG['quote_root'])


def check_sources():
    paths=[Path(__file__),CFG_PATH,Path(base.__file__),Path(bandit.__file__),Path(engine.__file__),
           Path('paper_code/execution.json'),Path('config/atm_rv_persistence.json'),
           Path('config/p25_submission.json'),Path(CFG['source']),ROOT/'source_manifest.json']
    record={'hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            'quote_signature':[(ROOT/'quotes.sqlite').stat().st_size,(ROOT/'quotes.sqlite').stat().st_mtime_ns]}
    path=OUT/'source_manifest.json'
    if path.exists():assert json.loads(path.read_text())==record,'Use a new output directory after changing sources'
    else:base.dump(path,record)


def sample_rows(rows,cal):
    """Four predetermined sessions, three current ranks each; no future labels."""
    sessions=[]
    for _,g in pd.Series(cal,index=cal).groupby(cal.to_period('M')):
        sessions.extend(g.iloc[[i for i in [0,5,10,15] if i<len(g)]].tolist())
    pieces=[]
    for _,g in rows.loc[rows.date.isin(sessions)].groupby('date',sort=True):
        g=g.sort_values(['residual_memory','secid'],kind='stable')
        slots=np.unique(np.linspace(0,len(g)-1,min(3,len(g))).round().astype(int))
        pieces.append(g.iloc[slots])
    result=pd.concat(pieces,ignore_index=True).sort_values(['secid','date']).reset_index(drop=True)
    result['sample_id']=np.arange(len(result))
    return result


def prepare():
    if (OUT/'sample.parquet').exists():return
    rows=pd.read_parquet(CFG['source'],use_threads=False)
    rows['wing']='ATM_STRADDLE'
    rows.to_parquet(OUT/'proposals.parquet',index=False)
    sample=sample_rows(rows,base.calendar())
    sample.to_parquet(OUT/'sample.parquet',index=False)
    base.dump(REPORT/'protocol.json',dict(configuration=CFG,proposals=len(rows),names=int(rows.secid.nunique()),
        training_samples=len(sample),execution=base.execution_settings(),
        unchanged_proposal_keys=True,entry_vega='450 dollars per vol point across call plus put at 100%, not per leg'))
    print(json.dumps(dict(phase='sample',samples=len(sample),proposals=len(rows))),flush=True)


class UnitMarket(base.DiskMarket):
    """Indexed quote lookup avoids scanning a whole date for one contract."""
    def quote(self,date,oid):
        record=self.conn.execute('SELECT * FROM quotes WHERE date=? AND optionid=?',(str(date.date()),int(oid))).fetchone()
        if record is None:return None
        row={k:(np.nan if v is None and k not in ['cp_flag','bs_status'] else v) for k,v in zip(base.QUOTE_COLUMNS,record)}
        row['date']=date;row['expiry']=pd.Timestamp(row['expiry'])
        return MarketRow(row)
    def rate(self,date):return float(self.rates.at[date])


def open_market(rows,unit=False):
    previous=base.OUT;base.OUT=ROOT
    try:return (UnitMarket if unit else base.DiskMarket)(rows.secid_a.unique(),pd.concat([rows.optionid_a,rows.optionid_b]))
    finally:base.OUT=previous


def label_one(row,market,cal,cfg):
    record=dict(sample_id=int(row.sample_id),secid=int(row.secid),date=row.date,expiry=row.expiry,
        residual_memory=row.residual_memory,iv=row.iv,vix=row.vix,finite_label=False,
        end=pd.NaT,entry_position=-1,full_profit=np.nan)
    paths=np.zeros((3,bandit.PATH_LENGTH))
    try:
        start=cal.get_loc(row.date);stop=cal.searchsorted(row.expiry,side='right')
        if stop>=len(cal):raise ValueError('Insufficient runoff')
        proposal=pd.DataFrame([{k:getattr(row,k) for k in bandit.EXECUTION}])
        cohort=f'{row.date.date()}|{row.pair}|ATM_STRADDLE'
        nav,trades,decisions,audit=engine.simulate(market,proposal,cal[start:stop+1],cfg,
            base.CFG['account_capital'],straddle=True,corporate_events=bandit.events(),
            max_stale_quote_sessions=2,locked_entry_counts={cohort:(1,1)})
        if not len(decisions) or decisions.status.iloc[0]!='entered':
            record['reason']='no_executable_entry'
        else:
            first=decisions.entry_date.iloc[0];end=trades.date.max()
            total_vega=100*sum(float(market.quote(first,oid).vega) for oid in [row.optionid_a,row.optionid_b])
            counts=np.floor(base.CFG['entry_vega']*np.array(CFG['actions'])/4/total_vega).astype(int)
            reward=bandit.scaled_cash_paths(nav,counts,market.rates,
                capital=base.CFG['account_capital'],borrowing_spread=cfg.borrowing_spread)
            reward=reward[:,(nav.index>=first)&(nav.index<=end)]
            if reward.shape[1]>bandit.PATH_LENGTH:raise ValueError('Path exceeds storage bound')
            paths[:,:reward.shape[1]]=reward
            record.update(finite_label=True,end=end,entry_position=int(cal.get_loc(first)),
                full_profit=float(reward[1].sum()),counts=counts.tolist(),length=reward.shape[1],reason='ok')
            assert audit['max_ledger_error']<1e-5
    except (ValueError,KeyError,IndexError,RuntimeError) as exc:
        record['reason']=f'{type(exc).__name__}: {exc}'
    return record,paths


def labels(limit):
    rows=pd.read_parquet(OUT/'sample.parquet',use_threads=False)
    folder=OUT/'unit_labels';folder.mkdir(exist_ok=True)
    pending=rows.loc[~rows.sample_id.map(lambda i:(folder/f'{i}.parquet').exists())].iloc[:limit]
    if not len(pending):print(json.dumps(dict(phase='labels',remaining=0)),flush=True);return
    market=open_market(pending,unit=True);all_divs=market.divs
    cfg=Settings(**base.execution_settings());cal=base.calendar();engine.book_risk=base.sparse_book_risk
    try:
        for row in pending.itertuples():
            market.divs=all_divs.loc[all_divs.secid.eq(row.secid)]
            record,paths=label_one(row,market,cal,cfg)
            # Write the metadata marker last so interrupted arrays are regenerated.
            np.savez(folder/f'{row.sample_id}.npz',paths=paths)
            pd.DataFrame([record]).to_parquet(folder/f'{row.sample_id}.parquet',index=False)
        print(json.dumps(dict(phase='labels',new=len(pending),done=len(list(folder.glob('*.parquet'))),total=len(rows))),flush=True)
    finally:market.conn.close()


def assemble():
    rows=pd.read_parquet(OUT/'sample.parquet',use_threads=False);folder=OUT/'unit_labels'
    assert len(list(folder.glob('*.parquet')))==len(rows),'Labels incomplete'
    target=OUT/'labels';target.mkdir(exist_ok=True);parts=[]
    for sid,g in rows.groupby('secid',sort=True):
        metadata=[];paths=[]
        for local,i in enumerate(g.sample_id):
            r=pd.read_parquet(folder/f'{i}.parquet',use_threads=False);r['label_row']=local;metadata.append(r)
            with np.load(folder/f'{i}.npz') as z:paths.append(z['paths'])
        meta=pd.concat(metadata,ignore_index=True);meta.to_parquet(target/f'{sid}.parquet',index=False)
        np.savez(target/f'{sid}.npz',paths=np.stack(paths));parts.append(meta)
    meta=pd.concat(parts,ignore_index=True);meta.to_parquet(OUT/'training.parquet',index=False)
    meta.groupby(['finite_label','reason']).size().rename('count').to_csv(REPORT/'label_coverage.csv')
    print(json.dumps(dict(phase='assembled',total=len(meta),valid=int(meta.finite_label.sum()))),flush=True)


def allocations():
    bandit.configure(CFG_PATH)
    (OUT/'allocations').mkdir(exist_ok=True)
    bandit.allocations('tree18','test')


def replay(limit):
    """Identical actual-contract execution; restart only at saved month ends."""
    import joblib
    from scripts.cheap_vega_runtime import mark_book,metrics,annual_metrics
    rows=pd.read_parquet(OUT/'allocations/tree18_test.parquet',columns=bandit.EXECUTION+['learned_level'],use_threads=False)
    for c in ['pair','wing']:rows[c]=rows[c].astype('category')
    market_rows=rows.rename(columns={'secid_a':'secid'})
    previous=base.OUT;base.OUT=ROOT
    try:market=base.DiskMarket(market_rows.secid.unique(),pd.concat([rows.optionid_a,rows.optionid_b]))
    finally:base.OUT=previous
    cfg=Settings(**base.execution_settings());cal=market.rates.index
    engine.book_risk=base.sparse_book_risk;state=None;new=0
    folder=OUT/'replays/tree18';folder.mkdir(parents=True,exist_ok=True)
    try:
        for month in pd.period_range(cal[0],cal[-1],freq='M'):
            dest=folder/'months'/str(month);marker=dest/'checkpoint.joblib'
            if marker.exists():state=joblib.load(marker);continue
            if new>=limit:print(json.dumps(dict(phase='checkpoint',new_months=new)),flush=True);return
            dest.mkdir(parents=True,exist_ok=True);finish=min(month.end_time,cal[-1])
            schedule=bandit.month_schedule(rows,cal[0] if state is None else state['previous_date'],finish)
            nav,trades,decisions,audit=engine.simulate(market,rows,cal,cfg,base.CFG['per_name_reference_capital'],
                straddle=True,corporate_events=bandit.events(),max_stale_quote_sessions=2,
                policy=lambda r,c:4,scale_policy=lambda r,c:float(r['learned_level'])/4,scale_pair_budget=False,
                resume_state=state,through_date=finish,prepared_schedule=schedule)
            state=audit.pop('resume_state');nav.to_parquet(dest/'daily_nav.parquet')
            trades.to_pickle(dest/'trades.pkl');decisions.to_pickle(dest/'decisions.pkl');joblib.dump(state,marker)
            new+=1;print(json.dumps(dict(phase='replay_month',month=str(month))),flush=True)
        months=sorted(p.parent for p in (folder/'months').glob('????-??/checkpoint.joblib'))
        market.days.clear();market.allowed.clear();del rows,market_rows;gc.collect()
        nav=pd.concat([pd.read_parquet(p/'daily_nav.parquet',use_threads=False) for p in months])
        mark_book(nav,market.rates,cfg.capital);nav.to_parquet(folder/'daily_nav.parquet')
        decisions=pd.concat([pd.read_pickle(p/'decisions.pkl') for p in months],ignore_index=True)
        decisions.to_csv(folder/'decisions.csv',index=False)
        trades=pd.concat([pd.read_pickle(p/'trades.pkl') for p in months],ignore_index=True)
        trades.to_csv(folder/'trades.csv',index=False)
        assert not state['positions'] and abs(nav.option_value.iloc[-1])+abs(nav.stock_value.iloc[-1])<1e-7
        assert trades.groupby(['cohort','optionid']).quantity.sum().eq(0).all()
        base.dump(folder/'status.json',dict(status='complete',case='straddle_trained_tree18',
            entries=int(decisions.status.eq('entered').sum()),**metrics(nav,cfg.capital),max_ledger_error=state['max_ledger_error']))
        pd.DataFrame(annual_metrics(nav,'straddle_trained_tree18',cfg.capital)).to_csv(folder/'annual.csv',index=False)
    finally:market.conn.close()


def report():
    from paper_code.metrics import period_metrics
    folder=OUT/'replays/tree18';nav=pd.read_parquet(folder/'daily_nav.parquet',use_threads=False)
    d=pd.read_csv(folder/'decisions.csv',parse_dates=['entry_date','signal_date'])
    records=[];annual=[]
    for period,start,stop in [('validation','2018-01-01','2021-12-31'),('test','2022-01-01','2025-08-29'),('full','2018-01-01','2025-08-29')]:
        records.append(dict(case='straddle_trained_tree18',period=period,**period_metrics(nav,d,start,stop)))
    for y in range(2018,2026):annual.append(dict(year=y,**period_metrics(nav,d,f'{y}-01-01',min(f'{y}-12-31','2025-08-29'))))
    pd.DataFrame(records).to_csv(REPORT/'results.csv',index=False);pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False)
    models=[json.loads(p.read_text()) for p in sorted((OUT/'models/tree18').glob('*.json'))]
    base.dump(REPORT/'training_summary.json',dict(monthly_fits=len(models),
        first_nontrivial_fit=next((m['cutoff'] for m in models if 'tree' in m),None),
        minimum_training_samples=CFG['minimum_training_samples'],latest_training_samples=models[-1]['training_samples']))
    print(pd.DataFrame(records)[['period','annualized_excess_sharpe','annualized_growth_relative_to_cash_pct','entries']].to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',required=True,choices=['test','prepare','labels','assemble','allocations','replay','report'])
    parser.add_argument('--limit',type=int,default=24);args=parser.parse_args()
    if args.phase=='test':
        import unittest
        suite=unittest.defaultTestLoader.loadTestsFromNames(['tests.test_atm_rv_straddle_bandit','tests.test_research_memory_guard'])
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():raise SystemExit(1)
        return
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True);check_sources()
    if args.phase=='labels':labels(min(32,args.limit))
    elif args.phase=='replay':replay(min(8,args.limit))
    else:dict(prepare=prepare,assemble=assemble,allocations=allocations,report=report)[args.phase]()


if __name__=='__main__':main()
