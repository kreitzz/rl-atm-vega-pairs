"""Transfer the put bandit's useful state blocks to short-straddle sizing.

More exact training rewards and completed-profit memory replace reliance on a
tiny RV-minus-IV-only sample. All five candidates receive funded validation,
because sparse shadow Sharpe failed to rank funded portfolios in the last run.
Every numerical phase runs through the existing 256 MiB/20% CPU guard.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

import numpy as np
import pandas as pd

from scripts import straddle_entry_refinement as core
from scripts import atm_rv_straddle_bandit as unit
from scripts.option_flex.contract_engine import Settings
from scripts.run_capped_straddle_book import native_ladder
from paper_code.metrics import period_metrics, paired_interval

CONFIG = Path('config/straddle_feature_transfer.json')
CFG = json.loads(CONFIG.read_text())
OUT, REPORT = Path(CFG['output_root']), Path(CFG['report_root'])
PREVIOUS = Path(CFG['source_root'])
CONTROL = ['residual_memory', 'iv', 'vix']
MEMORY = ['profit_91d', 'count_91d', 'profit_365d', 'count_365d']
RISK = ['rv21', 'rv5', 'ewma_vrp', 'ewma_vrp_z', 'variance_ratio', 'dte',
        'commission_vol_points', 'iv_rank', 'iv_z', 'vrp_rank', 'quoted_vol_width', 'put_call_iv_basis']
EXTRA = MEMORY+RISK


def expanded(fields): return fields+[f+'_available' for f in fields]


def configure():
    core.CONFIG=CONFIG; core.CFG=CFG; core.OUT=OUT; core.REPORT=REPORT; core.DATA=OUT
    core.FEATURES={'control': CONTROL, 'memory': CONTROL+expanded(MEMORY),
                   'risk': CONTROL+expanded(RISK), 'all': CONTROL+expanded(EXTRA)}
    # Match the successful put learner's profit-based partition target. The
    # subsequent exact sizing-map search still maximizes aggregate daily Sharpe.
    core.risk_adjusted_target=lambda paths: paths[:, 2].sum(axis=1)
    unit.OUT=OUT; unit.REPORT=REPORT; unit.CFG=CFG


def check_sources():
    paths=[Path(__file__), CONFIG, Path(core.__file__), Path(unit.__file__),
        Path(unit.engine.__file__), Path(unit.base.__file__), Path(unit.bandit.__file__),
        PREVIOUS/'sample.parquet', PREVIOUS/'proposals.parquet', PREVIOUS/'training.parquet',
        Path('artifacts/contracts/atm_rv_straddle_fixed50/replays/tree18/decisions.csv'),
        Path('paper_code/execution.json'), Path('config/atm_rv_persistence.json')]
    quotes=unit.ROOT/'quotes.sqlite'
    record=dict(hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                quote_signature=[quotes.stat().st_size, quotes.stat().st_mtime_ns])
    marker=OUT/'sources.json'
    if marker.exists(): assert json.loads(marker.read_text())==record, 'Changed sources: preserve this run and use a new root'
    else: core.dump(marker, record)


def prepare():
    if (OUT/'sample.parquet').exists(): return
    rows=pd.read_parquet(PREVIOUS/'proposals.parquet', use_threads=False)
    old=pd.read_parquet(PREVIOUS/'sample.parquet', use_threads=False)
    d=pd.read_csv('artifacts/contracts/atm_rv_straddle_fixed50/replays/tree18/decisions.csv',
                  usecols=['status','signal_date','pair'], parse_dates=['signal_date'])
    keys=d.loc[d.status.eq('entered'), ['signal_date','pair']].rename(columns={'signal_date':'date'})
    extra=rows.merge(keys, on=['date','pair'], how='inner', validate='one_to_one')
    known=pd.MultiIndex.from_frame(old[['date','secid']])
    extra=extra.loc[~pd.MultiIndex.from_frame(extra[['date','secid']]).isin(known)].sort_values(['secid','date']).copy()
    extra['sample_id']=np.arange(len(old),len(old)+len(extra))
    samples=pd.concat([old,extra],ignore_index=True)
    assert not samples.duplicated(['date','secid']).any()
    folder=OUT/'unit_labels'; folder.mkdir(parents=True,exist_ok=True)
    for i in old.sample_id:
        for ext in ['npz','parquet']:
            shutil.copy2(PREVIOUS/'unit_labels'/f'{i}.{ext}', folder/f'{i}.{ext}')
    rows.to_parquet(OUT/'base_proposals.parquet',index=False)
    samples.to_parquet(OUT/'sample.parquet',index=False)
    core.dump(REPORT/'protocol.json',dict(configuration=CFG, proposals=len(rows),
        original_samples=len(old), additional_samples=len(extra), samples=len(samples),
        transferred_features=core.FEATURES,
        profit_memory='Mean completed independent full-size straddle P&L per reference vega, by name; trailing 91/365 calendar days, strict end < decision date, with counts and availability flags. No profitability gate.',
        reward_target='Completed 100% daily profit above cash, summed per episode; portfolio-Sharpe sizing search.',
        sampling_notice='Extra training examples come from entry dates admitted by the causal fixed-50 book; future payoff availability never restricts live proposals.',
        training_examples_before_2018=0, warmup='50% until 128 completed examples',
        unchanged_execution_settings=unit.base.execution_settings()))
    print(json.dumps(dict(phase='prepared', samples=len(samples), additional=len(extra))),flush=True)


def label_batch(limit):
    original=unit.label_one
    def corrected(*args,**kwargs):
        row,paths=original(*args,**kwargs)
        if row['finite_label']: row['full_profit']=float(paths[2].sum())
        return row,paths
    unit.label_one=corrected
    try: unit.labels(limit)
    finally: unit.label_one=original


def past_z(series):
    previous=series.shift(1).rolling(252,min_periods=126)
    return (series-previous.mean())/previous.std().where(lambda x:x>1e-8)


def profit_memory(dates, labels):
    result={}
    valid=labels.loc[labels.finite_label]
    for days in [91,365]:
        mean,count=unit.base.completed_memory(dates,valid.end,
            valid.full_profit.to_numpy()/unit.base.CFG['entry_vega'],days,1)
        result[f'profit_{days}d']=mean;result[f'count_{days}d']=count.astype(float)
    return pd.DataFrame(result,index=pd.DatetimeIndex(dates))


def features(limit):
    folder=OUT/'features'; folder.mkdir(exist_ok=True)
    rows=pd.read_parquet(OUT/'base_proposals.parquet',use_threads=False)
    meta=pd.read_parquet(OUT/'training.parquet',use_threads=False)
    cal=unit.base.calendar(); fee=unit.base.execution_settings()['option_commission']
    conn=sqlite3.connect(f'file:{unit.ROOT/"quotes.sqlite"}?mode=ro',uri=True)
    done=0
    try:
        for sid,g in rows.groupby('secid',sort=True):
            marker=folder/f'{sid}.parquet'
            if marker.exists(): continue
            if done>=limit: break
            panel=pd.read_parquet(unit.base.OUT/'panel'/f'{sid}.parquet',
                                  columns=['date','iv'],use_threads=False).set_index('date').reindex(cal)
            stock=pd.read_parquet(unit.base.SOURCE/f'{sid}_stocks.parquet',
                                  columns=['date','close','cfadj'],use_threads=False).set_index('date').reindex(cal)
            price=(stock.close*stock.cfadj).where(stock.close.gt(0)&stock.cfadj.gt(0))
            returns=np.log(price/price.shift(1)); iv=panel.iv
            variance=252*returns.pow(2).ewm(halflife=20,min_periods=63,adjust=False).mean()
            historical=pd.DataFrame(dict(rv5=np.sqrt(252*returns.pow(2).rolling(5,min_periods=5).mean()),
                ewma_vrp=iv.pow(2)-variance, variance_ratio=variance/iv.pow(2),
                iv_rank=iv.rolling(252,min_periods=63).rank(pct=True),iv_z=past_z(iv)))
            historical['ewma_vrp_z']=past_z(historical.ewma_vrp)
            historical['vrp_rank']=historical.ewma_vrp.rolling(252,min_periods=63).rank(pct=True)
            current=historical.reindex(g.date).reset_index(drop=True)
            current['date']=g.date.to_numpy();current['secid']=int(sid);current['rv21']=g.rv21.to_numpy()
            current['dte']=(g.expiry-g.date).dt.days.to_numpy()
            memory=profit_memory(g.date,meta.loc[meta.secid.eq(sid)])
            for c in MEMORY: current[c]=memory[c].to_numpy()
            quote_features=[]
            for row in g.itertuples():
                c=conn.execute('SELECT vega,bs_iv,bid,ask FROM quotes WHERE date=? AND optionid=?',
                               (str(row.date.date()),int(row.optionid_a))).fetchone()
                p=conn.execute('SELECT vega,bs_iv,bid,ask FROM quotes WHERE date=? AND optionid=?',
                               (str(row.date.date()),int(row.optionid_b))).fetchone()
                if c is None or p is None or any(v is None for v in c+p) or c[0]+p[0]<=0:
                    quote_features.append([np.nan]*3)
                else:
                    vega=c[0]+p[0]
                    quote_features.append([2*fee/(100*vega), (c[3]-c[2]+p[3]-p[2])/vega, p[1]-c[1]])
            current[['commission_vol_points','quoted_vol_width','put_call_iv_basis']]=np.asarray(quote_features)
            for c in EXTRA:
                current[c]=current[c].replace([np.inf,-np.inf],np.nan)
                current[c+'_available']=current[c].notna().astype(float)
                current[c]=current[c].fillna(0.)
            current.to_parquet(marker,index=False);done+=1
        print(json.dumps(dict(phase='features',new_names=done,done=len(list(folder.glob('*.parquet'))),total=int(rows.secid.nunique()))),flush=True)
    finally: conn.close()


def assemble_features():
    pieces=[pd.read_parquet(p,use_threads=False) for p in sorted((OUT/'features').glob('*.parquet'))]
    f=pd.concat(pieces,ignore_index=True)
    rows=pd.read_parquet(OUT/'base_proposals.parquet',use_threads=False)
    assert len(f)==len(rows) and not f.duplicated(['date','secid']).any()
    rows=rows.drop(columns=[c for c in expanded(EXTRA) if c in rows])
    joined=rows.merge(f,on=['date','secid'],how='left',validate='one_to_one')
    assert np.isfinite(joined[core.FEATURES['all']]).all().all()
    joined.to_parquet(OUT/'proposals.parquet',index=False)
    meta=pd.read_parquet(OUT/'training.parquet',use_threads=False)
    meta=meta.drop(columns=[c for c in expanded(EXTRA) if c in meta])
    meta=meta.merge(f,on=['date','secid'],how='left',validate='many_to_one')
    assert np.isfinite(meta.loc[meta.finite_label,core.FEATURES['all']]).all().all()
    meta.to_parquet(OUT/'training.parquet',index=False)
    core.dump(REPORT/'feature_coverage.json',dict(samples=len(meta),valid_labels=int(meta.finite_label.sum()),
        availability={c:float(joined[c+'_available'].mean()) for c in EXTRA}, proposals=len(joined)))
    pd.testing.assert_frame_equal(joined[unit.bandit.EXECUTION],
        pd.read_parquet(PREVIOUS/'proposals.parquet',columns=unit.bandit.EXECUTION,use_threads=False))
    core.dump(OUT/'training_manifest.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [OUT/'training.parquet',OUT/'proposals.parquet']+sorted((OUT/'labels').glob('*.npz'))})
    print(json.dumps(dict(phase='features_assembled',valid=int(meta.finite_label.sum()),inputs=len(core.FEATURES['all']))),flush=True)


def verify_training():
    for path,signature in json.loads((OUT/'training_manifest.json').read_text()).items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==signature,path


def allocations(name,later):
    verify_training()
    marker=OUT/'cases'/name/('evaluation_allocations.json' if later else 'validation_allocations.json')
    if marker.exists(): return
    if later: assert json.loads((REPORT/'selection.json').read_text())['selected']==name
    core.allocations(name,CFG['evaluation_end'] if later else CFG['validation_end'])
    core.dump(marker,dict(complete=True,case=name,later=later))


def book(name,months,later):
    verify_training()
    if later: assert json.loads((REPORT/'selection.json').read_text())['selected']==name
    unit.OUT=OUT/'cases'/name
    if not later:
        done=len(list((unit.OUT/'replays/tree18/months').glob('????-??/checkpoint.joblib')))
        months=min(months,48-done)
        if months<=0: return
    original=unit.engine.simulate
    unit.engine.simulate=lambda *a,**kw:native_ladder(original,*a,**kw)
    try: unit.replay(months)
    finally: unit.engine.simulate=original


def select():
    rows=[]
    for spec in CFG['specifications']:
        nav,d=core.funded_nav(spec['name'],CFG['validation_end'])
        rows.append(dict(case=spec['name'],**period_metrics(nav,d,'2018-01-01',CFG['validation_end'])))
    ranked=sorted(rows,key=lambda r:(-r['annualized_excess_sharpe'],r['case']))
    record=dict(selected=ranked[0]['case'],candidates=ranked,selection_period='2018-2021 actual funded validation',
                evaluation_status=CFG['evaluation_status'])
    path=REPORT/'selection.json'
    if path.exists(): assert json.loads(path.read_text())==record
    else: core.dump(path,record)
    pd.DataFrame(rows).to_csv(REPORT/'validation_candidates.csv',index=False)
    print(json.dumps(record),flush=True)


def report():
    chosen=json.loads((REPORT/'selection.json').read_text())['selected']
    cases={'fixed50':Path('artifacts/contracts/atm_rv_straddle_fixed50/replays/tree18'),
        'fixed75':Path('artifacts/contracts/straddle_capped_books/fixed75/replays/tree18'),
        'fixed100':Path('artifacts/contracts/atm_rv_straddle_full/replays/straddle100'),
        'original_bandit':PREVIOUS/'replays/tree18',
        'previous_refinement':Path('artifacts/contracts/straddle_entry_refinement/cases/economic_tree4_w36/replays/tree18'),
        chosen:OUT/'cases'/chosen/'replays/tree18'}
    rows=[];annual=[];audit={};navs={};intervals=[]
    for name,folder in cases.items():
        nav=pd.read_parquet(folder/'daily_nav.parquet',use_threads=False)
        d=pd.read_csv(folder/'decisions.csv',parse_dates=['entry_date','signal_date'])
        expected=nav.option_pnl+nav.hedge_gross_pnl+nav.dividend_cash+nav.interest-nav.hedge_cost-nav.stock_borrow_cost
        error=float((expected-nav.pnl).abs().max());assert error<1e-5
        audit[name]=dict(ledger_error=error,decisions=len(d));navs[name]=nav.excess_return
        for period,start,stop in core.PERIODS:
            rows.append(dict(case=name,period=period,**period_metrics(nav,d,start,stop)))
        for y in range(2018,2026): annual.append(dict(case=name,year=y,
            **period_metrics(nav,d,f'{y}-01-01',min(f'{y}-12-31',CFG['evaluation_end']))))
    for name in cases:
        if name==chosen: continue
        intervals.append(dict(control=name,**paired_interval(navs[chosen].loc['2022':],navs[name].loc['2022':],block=63,draws=1000)))
    models=[json.loads(p.read_text()) for p in sorted((OUT/'models'/chosen).glob('*.json'))]
    assert all(m['last_completed'] is None or pd.Timestamp(m['last_completed'])<pd.Timestamp(m['cutoff']) for m in models)
    audit['selected_fits']=dict(fits=len(models),last_training_samples=models[-1]['training_samples'],
        last_occupancy=models[-1].get('occupancy'),strict_completed_outcome_timing=True)
    pd.DataFrame(rows).to_csv(REPORT/'results.csv',index=False)
    pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False)
    pd.DataFrame(intervals).to_csv(REPORT/'intervals.csv',index=False);core.dump(REPORT/'audit.json',audit)
    table=pd.DataFrame(rows)[['case','period','annualized_excess_sharpe','annualized_growth_relative_to_cash_pct','entries']]
    text=table.to_string(index=False)+'\n\n'+json.dumps(json.loads((REPORT/'selection.json').read_text()),indent=2)+'\n'
    (REPORT/'FINDINGS.txt').write_text(text);print(text,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',required=True,choices=['test','prepare','labels','assemble','features','assemble_features','allocations','book','select','report'])
    p.add_argument('--case',choices=[s['name'] for s in CFG['specifications']]);p.add_argument('--limit',type=int,default=32)
    p.add_argument('--later',action='store_true');args=p.parse_args();configure()
    if args.phase=='test':
        import unittest
        suite=unittest.defaultTestLoader.loadTestsFromNames(['tests.test_straddle_feature_transfer','tests.test_research_memory_guard'])
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():raise SystemExit(1)
        return
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True);check_sources()
    if args.phase=='labels':label_batch(min(64,args.limit))
    elif args.phase=='features':features(min(16,args.limit))
    elif args.phase in ['allocations','book']:
        if not args.case:p.error('--case required')
        if args.phase=='allocations':allocations(args.case,args.later)
        else:book(args.case,min(8,args.limit),args.later)
    else:dict(prepare=prepare,assemble=unit.assemble,assemble_features=assemble_features,select=select,report=report)[args.phase]()


if __name__=='__main__':main()
