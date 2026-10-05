"""Monthly expiry-trained RL/NN with rebuilt memory and validation gates."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import resource
import time
import numpy as np
import pandas as pd
from scripts.contract_engine import Settings, PreparedMarket, simulate, UnresolvedPosition
from scripts.contract_learning_labels import profitability_memory
from scripts.run_aligned_contract_learning import (contract_states,load_labels,train_pair,mark_book,
    asset_paths,EXTRA,MODELS)
from scripts.run_contract_surface_features import augment, FEATURES
from scripts.run_nn_rl_benchmark import NEW_FEATURES
from scripts.run_broad_contract_replay import Market, ROOT, CACHE
from scripts.run_selected_pair_backtest import audit,scores
from scripts.run_contract_walkforward import digest
from scripts.select_atm_contracts import load_calendar

FILTERS={'rl_sized':'memory_both','nn_fixed':'memory_year'}
BETA_TOLERANCE=.15
_market=None


def settings():
    return Settings(**dict(json.loads(Path('reports/aligned_contract_learning/manifest.json').read_text())['settings'],hold_to_expiry=True))


def worker(task):
    global _market
    d,output,sources=task;folder=Path(output)/d['pair'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'status.json').exists():return json.loads((folder/'status.json').read_text())
    cal=load_calendar(ROOT);cal=cal[(cal>='2010-01-01')&(cal<='2025-08-29')]
    cfg=settings();evaluation=cal[cal>='2018-01-01'];stop=cal[-1]-pd.Timedelta(days=cfg.max_dte+5)
    if _market is None:_market=Market(cal)
    q,s,selection,divs=_market.pair(d['secids'])
    panel,screen=contract_states(d,selection,s,cal,cfg,beta_tolerance=BETA_TOLERANCE)
    panel=panel[panel.index<=stop]
    # The old feature charged commissions twice. Expiry has entry only.
    panel['commission_vol_entry']=panel.pop('commission_vol_roundtrip')/2
    panel=augment(d,panel,cal)
    daily,ends,vegas,statuses=load_labels(d,panel,q,s,selection,cal,divs,cfg,folder,sources)
    ends=pd.DatetimeIndex(ends)
    rewards=np.divide(daily[:,[0,-1]].sum(axis=2)*cfg.entry_vega_per_leg,vegas[:,[0,-1]],
        out=np.full((len(panel),2),np.nan),where=vegas[:,[0,-1]]>0)
    panel=panel.join(profitability_memory(panel.index,ends,rewards,vegas[:,[0,-1]]>0))
    panel.to_parquet(folder/'states.parquet');screen.to_csv(folder/'beta_screen.csv')
    pd.Series(statuses).value_counts().rename_axis('status').to_csv(folder/'label_status.csv')
    features=NEW_FEATURES+[c if c!='commission_vol_roundtrip' else 'commission_vol_entry' for c in EXTRA]
    features += [c for c in panel if c.startswith(('profit_','count_'))]+FEATURES
    print(f'{d["label"]}: {len(panel)} labels; {dict(pd.Series(statuses).value_counts())}',flush=True)
    actions,fits=train_pair(d,panel,daily,ends,cal,q,s,selection,divs,cfg,_market.rates,
        pd.Timestamp('2018-01-01'),stop,folder,feature_columns=features,memory_filters=FILTERS,recent_refit=True)
    actions.to_csv(folder/'actions.csv');fits.to_csv(folder/'fits.csv',index=False)
    prepared=PreparedMarket(q,s,selection);records=[]
    for model in MODELS:
        dest=folder/model;dest.mkdir(exist_ok=True)
        a=actions[model];a.to_csv(dest/'actions.csv')
        row=dict(pair=d['pair'],label=d['label'],group=d['group'],model=model,nonzero_signals=int(a.ne(0).sum()))
        try:
            nav,t,dec=simulate(q,s,a,evaluation,d['secids'],cfg,divs,selection,prepared)
            mark_book(nav,_market.rates,cfg.capital)
            checks=audit(nav,t,s,cfg,d['secids'])
            assert t.empty or set(t.reason).issubset({'entry','split_adjustment','expiry_settlement'})
            nav.to_parquet(dest/'daily_nav.parquet');t.to_csv(dest/'trades.csv',index=False);dec.to_csv(dest/'decisions.csv',index=False)
            row.update(status='complete',entries=int(dec.status.eq('entered').sum()) if len(dec) else 0,
                fallbacks=int(nav.greek_fallbacks.sum()),risk_breach_days=int(nav.post_trade_risk_breach.sum()),
                **scores(nav,cfg.capital),**checks)
        except (UnresolvedPosition,ValueError) as exc:row.update(status='incomplete',error=str(exc))
        row['worker_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        records.append(row)
    (folder/'status.json').write_text(json.dumps(records,indent=2)+'\n')
    print(f'Completed {d["label"]}',flush=True)
    return records


def report(output,report_dir,definitions,records):
    status=pd.DataFrame(records);status.to_csv(report_dir/'status.csv',index=False)
    frozen=pd.read_csv('reports/expiry_holding/status.csv')
    failed=set(status.loc[status.status.ne('complete'),'pair'])|set(frozen.loc[frozen.status.ne('complete'),'pair'])
    # Frozen experiment covered the prior 348 complete broad pairs. New results
    # outside that set remain in status; comparison never silently substitutes.
    available=set(frozen.pair)
    members=[d['pair'] for d in definitions if d['group']=='broad' and d['pair'] in available and d['pair'] not in failed]
    rows=[];annual=[]
    for model in MODELS:
        old='surface_filtered_rl' if model=='rl_sized' else 'surface_filtered_nn'
        for variant,root,sub in [('frozen',Path('artifacts/contracts/expiry_holding'),old),('retrained',output,model)]:
            if not members:continue
            book=sum(pd.read_parquet(root/pair/sub/'daily_nav.parquet',columns=['nav','cash_benchmark']) for pair in members)
            book['return_']=book.nav.pct_change().fillna(0)
            book['excess_return']=book.return_-book.cash_benchmark.pct_change().fillna(0)
            book['excess_pnl']=book.nav.diff().fillna(0)-book.cash_benchmark.diff().fillna(0)
            book.to_csv(report_dir/f'{model}_{variant}_portfolio.csv')
            for period,use in [('full',slice(None)),('2018_2021',slice(None,'2021-12-31')),('2022_2025',slice('2022-01-01',None))]:
                g=book.loc[use];ex=g.excess_return.iloc[1:] if period!='2022_2025' else g.excess_return
                rows.append(dict(model=model,variant=variant,period=period,pairs=len(members),
                    total_return=(1+g.return_).prod()-1,excess_pnl=g.excess_pnl.sum(),
                    annualized_excess_sharpe=np.sqrt(252)*ex.mean()/ex.std() if ex.std()>1e-12 else None))
            for year,g in book.groupby(book.index.year):
                ex=g.excess_return
                annual.append(dict(model=model,variant=variant,year=year,return_percent=100*((1+g.return_).prod()-1),
                    annualized_excess_sharpe=np.sqrt(252)*ex.mean()/ex.std() if ex.std()>1e-12 else None))
    comparison=pd.DataFrame(rows);comparison.to_csv(report_dir/'comparison.csv',index=False)
    pd.DataFrame(annual).to_csv(report_dir/'annual.csv',index=False)
    fits=[]
    for d in definitions:
        try:f=pd.read_csv(output/d['pair']/'fits.csv')
        except pd.errors.EmptyDataError:continue
        if 'last_training_outcome' in f:
            mask=f.last_training_outcome.notna()
            assert (pd.to_datetime(f.loc[mask,'last_training_outcome'])<pd.to_datetime(f.loc[mask,'validation_start'])).all()
        if 'last_recent_outcome' in f:
            mask=f.last_recent_outcome.notna()
            assert (pd.to_datetime(f.loc[mask,'last_recent_outcome'])<pd.to_datetime(f.loc[mask,'fit'])).all()
        f['pair']=d['pair'];fits.append(f)
    (pd.concat(fits,ignore_index=True) if fits else pd.DataFrame()).to_csv(report_dir/'fits.csv',index=False)
    lines=['# Retraining for hold-to-expiry','',
        f'Pair eligibility: monthly preceding-126-session first-leg-on-second-leg beta {1-BETA_TOLERANCE:.2f}–{1+BETA_TOLERANCE:.2f}, with return correlation at least 0.5. Applied to training, validation and new entries; existing positions remain held through expiry.','',
        'Both models use regenerated actual-contract daily net excess P&L trajectories through each leg\'s expiry. Entry cost is 0.1 volatility point per leg plus $0.65 per contract; expiry settlement has no option spread or commission. Stock hedges cost 1 bp per executed notional, including liquidation; short stock borrow is 0.3% annually, with cash interest and a 1% borrowing premium on negative cash. These are the same stated costs as the frozen expiry replay, not a reduction selected for performance.','',
        'RL optimizes daily shadow-book excess Sharpe with a −1 to +1 ladder in 0.25 steps. NN predicts net outcomes and selects full-size long/short or flat. Independent pair models use the existing smile/term structure and other inputs. The commission state now describes entry only. Profitability memory is regenerated from completed expiry outcomes, available strictly after the later leg expires; RL retains positive 91/365-day gates and NN the positive 365-day gate.','',
        'Each month: fit five years ending before a separate one-year validation period; validate the filtered policy using the same expiry simulator and entry risk limits; permit trading only after positive validation excess Sharpe. Refit permitted specifications on the latest five years of completed outcomes before deployment. The gate validates the prior fitted proposal, not the refitted weights. Nothing from a still-open cohort enters training or memory. Validation and final entry cutoffs allow 50 calendar days of runoff.','',
        'Hold-to-expiry assumes cash intrinsic settlement and no early assignment. Daily marks remain observed mids; failed fitted Greeks use prior valid IV for that day\'s hedge. Existing risk breaches do not force liquidation; entry is blocked. Standalone training cohorts do not reproduce the overlapping portfolio\'s NAV sizing or risk entry rejections; validation and final evaluation apply those rules.','',
        f'Attempted {len(definitions)} pairs. Comparisons use {len(members)} common complete broad pairs from the prior frozen-expiry roster, retaining cash-only sleeves. Incomplete histories are recorded, not assigned zero P&L. Complete-roster selection can be biased. All periods have already been inspected in research.','',
        '| Model | Policy | Period | Annualized excess Sharpe |','|---|---|---|---:|']
    for r in comparison.itertuples():lines.append(f'| {r.model} | {r.variant} | {r.period} | {r.annualized_excess_sharpe:.3f} |')
    lines+=['','Sharpe uses daily cash-excess returns and √252. 2025 ends in August.','',
        f'[Comparison]({report_dir.name}/comparison.csv) · [Annual returns]({report_dir.name}/annual.csv) · [Run status]({report_dir.name}/status.csv) · [Fit audit]({report_dir.name}/fits.csv)','']
    report_dir.with_suffix('.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pairs',nargs='+')
    p.add_argument('--workers',type=int,default=2);p.add_argument('--output',type=Path,default=Path('artifacts/contracts/expiry_trained_beta_085_115'))
    p.add_argument('--report',type=Path,default=Path('reports/expiry_trained_beta_085_115'));args=p.parse_args()
    if not 1<=args.workers<=2:raise ValueError('Use one or two bounded workers')
    args.output.mkdir(parents=True,exist_ok=True);args.report.mkdir(parents=True,exist_ok=True)
    definitions=json.loads(Path('reports/aligned_contract_learning/universe.json').read_text())
    if args.pairs:definitions=[d for d in definitions if d['pair'] in args.pairs or d['label'] in args.pairs]
    if not definitions:raise ValueError('No matching pairs')
    code=[__file__,'scripts/batched_contract_labels.py','scripts/contract_learning_labels.py','scripts/contract_engine.py',
        'scripts/run_aligned_contract_learning.py','scripts/run_contract_walkforward.py','scripts/contract_risk.py',
        'scripts/run_contract_surface_features.py','scripts/test_contract_entry_gates.py']
    sources=dict(cache=digest(CACHE/'manifest.json'),surface=digest('data/contract_surface/manifest.json'))
    fingerprint=dict(settings=asdict(settings()),definitions=definitions,sources=sources,filters=FILTERS,
        beta_screen=dict(minimum=1-BETA_TOLERANCE,maximum=1+BETA_TOLERANCE,window=126,correlation_floor=.5),
        code={f:digest(f) for f in code},assets={str(p):digest(p) for d in definitions for p in asset_paths(d)})
    meta=args.output/'inputs.json'
    if meta.exists() and json.loads(meta.read_text())!=fingerprint:raise ValueError('Resume inputs changed')
    meta.write_text(json.dumps(fingerprint,indent=2)+'\n')
    started=time.monotonic();records=[]
    tasks=[(d,str(args.output),dict(sources,**{str(p):fingerprint['assets'][str(p)] for p in asset_paths(d)})) for d in definitions]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,result in enumerate(pool.map(worker,tasks),1):
            records.extend(result);pd.DataFrame(records).to_csv(args.report/'status.csv',index=False)
            print(f'Finished {i}/{len(definitions)} pairs',flush=True)
    report(args.output,args.report,definitions,records)
    (args.report/'manifest.json').write_text(json.dumps(dict(settings=asdict(settings()),workers=args.workers,
        elapsed_seconds=time.monotonic()-started,max_worker_peak_rss_bytes=max(r['worker_peak_rss_bytes'] for r in records),
        inputs_sha256=digest(meta)),indent=2)+'\n')


if __name__=='__main__':main()
