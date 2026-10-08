"""Frozen-policy bridge from training labels to chronological funded performance.

No fitting, searching, or allocation changes. Run with the pinned pca312 Python.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','ARROW_NUM_THREADS']:
    os.environ[key]='1'
import ast
from dataclasses import replace
import gc
import json
import sqlite3
import numpy as np
import pandas as pd
from scripts import run_p25_direct_policy as prior
from scripts.analyze_p25_opportunity_filter import sharpe, paired_sharpe_interval, PERIODS
from scripts.option_flex.black_scholes import option_values

base=prior.base
ROOT=base.ROOT
OUT=ROOT/'artifacts/contracts/p25_generalization_bridge'
REPORT=ROOT/'reports/p25_generalization_bridge'
CASES=['tree_cem','direct_linear','direct_additive']


def legacy_functions():
    # Load the unchanged numerical definitions without importing the old Unix-only
    # training pipeline. Constants are evaluated directly from their source AST.
    path=ROOT/'scripts/curve_trading/config.py'
    names={'TENORS','BUCKETS','SLOTS','COMBOS','LADDER'}
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.Assign)
           and any(isinstance(t,ast.Name) and t.id in names for t in n.targets)]
    env={'np':np,'pd':pd,'option_values':option_values}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    path=ROOT/'scripts/curve_trading/unit_paths.py'
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    return env['UnitPaths'],env['pair_labels']


def aggregate(paths,positions,calendar,mask=None):
    if not np.isfinite(paths).all():raise ValueError('Unresolved paths cannot be zero-imputed')
    result=np.zeros(len(calendar))
    for j,(path,pos) in enumerate(zip(paths,positions)):
        if mask is not None and not mask[j]:continue
        end=min(len(calendar),int(pos)+1+len(path))
        if np.any(path[end-int(pos)-1:]):raise ValueError('Nonzero path outside calendar')
        result[int(pos)+1:end]+=path[:end-int(pos)-1]*450
    return pd.Series(result,index=calendar)


def prepare(signals,budget):
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    files=[base.SOURCE,prior.QUOTES,prior.CONFIG,base.MARKET/'stocks.parquet',
           base.MARKET/'rates.parquet',base.MARKET/'dividends.parquet',
           ROOT/'reports/aligned_contract_learning/manifest.json',
           ROOT/'scripts/curve_trading/config.py',ROOT/'scripts/curve_trading/unit_paths.py',
           ROOT/'scripts/option_flex/black_scholes.py',prior.REPORT/'fits.csv',
           prior.REPORT/'verification.json']
    files += [prior.OUT/'replays'/c/n for c in CASES for n in prior.FILES]
    files += [base.LABELS/p/'P25_labels.npy' for p in sorted(signals.pair.unique())]
    hashes={}
    for p in files:
        hashes[str(p.relative_to(ROOT))]=base.digest(p);budget.rest()
    protocol=dict(question='Where does the additive training advantage disappear?',
        policies=CASES,proposals=608,retraining=False,
        stages=['archived old-cost standalone','reconstructed old-cost standalone',
                'reconstructed current-cost standalone','current-cost standalone restricted to actual funded admissions',
                'actual funded excess returns'],
        reconstruction='Original UnitPaths and pair_labels numerical definitions loaded unchanged; original settings (1m standalone capital,450 reference vega) and saved selected contracts. Change only commission .65 to .03.',
        coverage='Require finite paths for every proposal and chosen action. Fail rather than omit unresolved proposals. Compare reconstruction to archived float32 paths before interpreting costs.',
        aggregation='Sum overlapping daily cash-relative dollar paths on the full calendar, retaining zero days. Divide standalone series by constant10m for storage; scale does not affect Sharpe. Funded stage uses saved cash-excess percentage returns.',
        admissions='Use actual saved entered decisions as a descriptive mask; fixed450vega shadow quantities, not an independently deployable policy or causal attribution.',
        inference='Paired circular blocks63/126,2000 draws,seed2026, within each stage. Already inspected history, no research-selection adjustment. Training summaries are across overlapping windows and not independent observations.',
        limitations='Bridge changes training population to gated proposals, time period, and monthly deployed policies; cannot uniquely separate overfitting from regime shift, gate conditioning or optimization noise. Last step also changes sizing with NAV, netted hedges, financing, execution lifecycle and return denominator; not solely capital constraints.',
        resources=dict(threads=1,duty=.2,rss_limit_mib=384),hashes=hashes)
    p=REPORT/'protocol.json'
    if p.exists() and json.loads(p.read_text())!=protocol:raise ValueError('Frozen inputs changed')
    base.write(p,protocol)
    return protocol


def quote_cache(signals,budget):
    path=OUT/'selected_quotes.sqlite'
    if path.exists():return path
    wanted=set(signals.optionid_a.astype(int))|set(signals.optionid_b.astype(int))
    with sqlite3.connect(f'file:{prior.QUOTES.as_posix()}?mode=ro',uri=True) as src,sqlite3.connect(path) as dst:
        dst.execute('CREATE TABLE quotes(optionid INTEGER, payload TEXT)')
        cursor=src.execute('SELECT optionid,payload FROM quotes')
        while True:
            batch=cursor.fetchmany(2000)
            if not batch:break
            dst.executemany('INSERT INTO quotes VALUES (?,?)',[(oid,p) for oid,p in batch if oid in wanted])
            budget.rest()
        dst.execute('CREATE INDEX by_option ON quotes(optionid)')
    return path


def reconstruct(rows,calendar,budget):
    path=OUT/'reconstructed.npz'
    if path.exists():
        z=np.load(path);return z['old'],z['current']
    UnitPaths,pair_labels=legacy_functions()
    settings=json.loads((ROOT/'reports/aligned_contract_learning/manifest.json').read_text())['settings']
    settings.update(hold_to_expiry=True,min_dte=7,max_dte=55,max_moneyness_distance=1e6)
    cfg=base.Settings(**settings)
    stocks=pd.read_parquet(base.MARKET/'stocks.parquet',use_threads=False)
    rates=pd.read_parquet(base.MARKET/'rates.parquet',use_threads=False).rate
    divs=pd.read_parquet(base.MARKET/'dividends.parquet',use_threads=False)
    cache=quote_cache(rows,budget)
    old=np.full((len(rows),8,cfg.max_dte+2),np.nan);current=old.copy();checks=[]
    with sqlite3.connect(f'file:{cache.as_posix()}?mode=ro',uri=True) as db:
        for number,(pair,g) in enumerate(rows.groupby('pair'),1):
            dates=pd.DatetimeIndex(g.date);units={.65:[],.03:[]}
            for side in ['a','b']:
                sid=int(g['secid_'+side].iloc[0]);ids=g['optionid_'+side].astype(int).unique().tolist()
                query='SELECT payload FROM quotes WHERE optionid IN ('+','.join('?' for _ in ids)+')'
                q=pd.DataFrame([json.loads(p) for p, in db.execute(query,ids)])
                for c in ['date','expiry']:q[c]=pd.to_datetime(q[c])
                s=stocks.loc[stocks.secid.eq(sid)].set_index('date').reindex(calendar)
                s['rate']=rates.reindex(calendar);s.index.name='date'
                sel=g[['date','optionid_'+side]].rename(columns={'optionid_'+side:'optionid'}).copy()
                sel['tenor']=30;sel['cp_flag']='P';sel['bucket']=25
                for fee in units:
                    units[fee].append(UnitPaths(q,s.reset_index(),sel,calendar,dates,divs.loc[divs.secid.eq(sid)],replace(cfg,option_commission=fee)))
                del q,s;budget.rest()
            results=[]
            for fee in units:
                daily,ends,vegas,status=pair_labels(*units[fee],(30,'P',25,25),replace(cfg,option_commission=fee))
                results.append(daily)
                checks.append(dict(pair=pair,commission=fee,proposals=len(g),complete=int((status=='complete').sum()),
                                   no_entry=int((status=='no_entry').sum()),unresolved=int((status=='unresolved').sum())))
                if not np.isfinite(daily).all():raise ValueError(f'Unresolved reconstructed paths: {pair},fee{fee}')
            old[g.index]=results[0];current[g.index]=results[1]
            del units,results;gc.collect();budget.rest()
            if number%10==0:print(f'Reconstructed {number}/46 pairs; peak RSS {budget.peak:.1f}MiB',flush=True)
    np.savez_compressed(path,old=old,current=current)
    pd.DataFrame(checks).to_csv(REPORT/'reconstruction_coverage.csv',index=False)
    return old,current


def main():
    budget=base.Budget()
    meta,signals,_,_,calendar,_=prior.load_inputs()
    rows=signals.merge(meta[['date','pair','position','label_row']],on=['date','pair'],validate='one_to_one').reset_index(drop=True)
    del meta
    protocol=prepare(signals,budget)
    old,current=reconstruct(rows,calendar,budget)
    daily=[];coverage=[];reconciliations=[];cohorts=[]
    for case in CASES:
        folder=prior.OUT/'replays'/case
        choices=pd.read_parquet(folder/'scored_allocations.parquet',use_threads=False)
        own=rows.drop(columns='learned_level').merge(choices[['date','pair','learned_level']],on=['date','pair'],validate='one_to_one',sort=False)
        assert own[['date','pair']].equals(rows[['date','pair']])
        label_idx=np.where(own.direction>0,4+own.learned_level,5-own.learned_level).astype(int)
        raw=np.empty((len(rows),old.shape[-1]))
        for pair,g in own.groupby('pair'):
            source=np.load(base.LABELS/pair/'P25_labels.npy',mmap_mode='r')
            raw[g.index]=source[g.label_row.astype(int).to_numpy(),label_idx[g.index]]
            del source
        if not np.isfinite(raw).all():raise ValueError('Incomplete archived paths')
        rebuilt=old[np.arange(len(rows)),label_idx-1];recost=current[np.arange(len(rows)),label_idx-1]
        # Original labels were serialized as float32. This checks the complete
        # path, not just aggregate returns, before treating the next stage as costs.
        errors=np.max(np.abs(rebuilt-raw),axis=1)*450
        exact32=np.all(rebuilt.astype(np.float32)==raw.astype(np.float32),axis=1)
        reconciliations.append(dict(case=case,proposals=len(rows),float32_exact_paths=int(exact32.sum()),
            max_daily_dollar_error=float(errors.max()),total_dollar_difference=float((rebuilt-raw).sum()*450)))
        if not np.allclose(rebuilt,raw,rtol=2e-6,atol=2e-6):
            raise ValueError(f'Legacy reconstruction mismatch: {reconciliations[-1]}')
        dec=pd.read_csv(folder/'decisions.csv',parse_dates=['signal_date','entry_date'])
        own=own.merge(dec[['signal_date','pair','status','entry_date']],left_on=['date','pair'],right_on=['signal_date','pair'],validate='one_to_one',sort=False)
        admitted=own.status.eq('entered').to_numpy()
        coverage.append(dict(case=case,proposals=len(own),archived_complete=len(own),reconstructed_complete=len(own),funded_entries=int(admitted.sum())))
        for stage,paths,mask in [('archived',raw,None),('reconstructed_old',rebuilt,None),('current_cost',recost,None),('admitted_shadow',recost,admitted)]:
            series=aggregate(paths,rows.position,calendar,mask).loc['2018-01-02':'2025-08-29']/1e7
            daily.append(pd.DataFrame(dict(date=series.index,case=case,stage=stage,value=series.to_numpy())))
        nav=pd.read_parquet(folder/'daily_nav.parquet',use_threads=False)
        trades=pd.read_csv(folder/'trades.csv',parse_dates=['date'])
        verification=base.verify_ledger(nav,trades,dec,choices)
        base.write(REPORT/f'funded_verification_{case}.json',verification)
        daily.append(pd.DataFrame(dict(date=nav.index,case=case,stage='funded',value=nav.excess_return.to_numpy())))
        entrycost=(recost[:,0]-rebuilt[:,0])*450
        lifetime=(recost-rebuilt).sum(axis=1)*450
        if np.any(entrycost < -1e-7) or np.any(lifetime < -1e-6):raise ValueError('Lower commissions increased standalone costs')
        for j,r in own.iterrows():
            cohorts.append(dict(case=case,date=r.date,pair=r.pair,size=r.learned_level/4,status=r.status,
                legacy_profit=float(raw[j].sum()*450),current_profit=float(recost[j].sum()*450),
                entry_commission_saving=float(entrycost[j]),lifetime_cost_saving=float(lifetime[j]),
                inferred_contracts=round(entrycost[j]/.62),reconstruction_max_error=float(errors[j])))
        np.testing.assert_allclose(entrycost/.62,np.round(entrycost/.62),atol=1e-6)
        budget.rest()
    daily=pd.concat(daily,ignore_index=True)
    daily.to_parquet(OUT/'daily_series.parquet',index=False)
    pd.DataFrame(coverage).to_csv(REPORT/'coverage.csv',index=False)
    pd.DataFrame(reconciliations).to_csv(REPORT/'label_reconciliation.csv',index=False)
    pd.DataFrame(cohorts).to_csv(REPORT/'cohorts.csv',index=False)
    metrics=[];intervals=[]
    for stage,group in daily.groupby('stage',sort=False):
        wide=group.pivot(index='date',columns='case',values='value')
        assert not wide.isna().any().any()
        for period,start,stop in PERIODS:
            w=wide.loc[start:stop]
            if w.index[0]==wide.index[0]:w=w.iloc[1:]
            for case in CASES:
                metrics.append(dict(stage=stage,period=period,case=case,sharpe=sharpe(w[case]),sessions=len(w),
                    standalone_profit=float(w[case].sum()*1e7) if stage!='funded' else None))
            if stage=='reconstructed_old':continue
            for case in CASES[1:]:
                for block in [63,126]:
                    intervals.append(dict(stage=stage,period=period,case=case,reference='tree_cem',
                        **paired_sharpe_interval(w[case],w.tree_cem,block=block,rest=budget.rest)))
        print(f'Analyzed {stage}',flush=True)
    pd.DataFrame(metrics).to_csv(REPORT/'stages.csv',index=False)
    pd.DataFrame(intervals).to_csv(REPORT/'intervals.csv',index=False)
    # Verify protected inputs after all calculations.
    for name,sha in protocol['hashes'].items():
        assert base.digest(ROOT/name)==sha;budget.rest()
    base.write(REPORT/'verification.json',dict(status='passed',proposals=608,cases=3,
        finite_coverage=True,legacy_path_reconciliation=reconciliations,
        saved_funded_ledgers_reconciled=True,protected_hashes_unchanged=True,
        peak_rss_mib=budget.peak,code_sha256=base.digest(__file__)))
    print(pd.DataFrame(metrics).query("period=='evaluation'").pivot(index='stage',columns='case',values='sharpe').to_string(),flush=True)


if __name__=='__main__':main()
