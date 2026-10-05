"""Refit validated model specifications through the latest completed outcomes.

Keep the original monthly historical validation decision, but refit the model
weights on five years ending at deployment. This is not a validation of those
refitted weights. Pair decisions remain independent; two bounded workers.
"""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd

from scripts.contract_engine import Settings, PreparedMarket, UnresolvedPosition, simulate
from scripts.run_aligned_contract_learning import EXTRA, mark_book
from scripts.run_broad_contract_replay import Market, ROOT
from scripts.run_contract_walkforward import fit_rl, fit_nn, predict, digest
from scripts.run_nn_rl_benchmark import NEW_FEATURES
from scripts.run_selected_pair_backtest import audit
from scripts.select_atm_contracts import load_calendar
from scripts.test_contract_entry_gates import gate_actions, metrics

SOURCE=Path('artifacts/contracts/aligned_learning')
OUTPUT=Path('artifacts/contracts/recent_refit')
REPORT=Path('reports/recent_contract_refit')
MODELS=['rl_sized','nn_fixed']
FILTERS={'rl_sized':'memory_both','nn_fixed':'memory_year'}
VARIANTS=['recent','recent_memory']
_market=None


def recent_training_mask(dates, ends, daily, cutoff):
    return ((dates>=cutoff-pd.DateOffset(years=5)) & (dates<cutoff) &
            (ends<cutoff) & np.isfinite(daily).all(axis=(1,2)))


def train(panel,daily,ends,calendar,old_actions,old_fits,extra_features=None):
    actions=pd.DataFrame(0.,index=old_actions.index,columns=MODELS)
    features=NEW_FEATURES+EXTRA+[c for c in panel if c.startswith(('profit_','count_'))]+list(extra_features or [])
    logs=[]
    for row in old_fits[old_fits.status.eq('enabled')].itertuples():
        cutoff=pd.Timestamp(row.fit); neural=row.model=='nn_fixed'
        dates=actions.index[(actions.index>=cutoff)&(actions.index<cutoff+pd.offsets.MonthBegin(1))]
        if not len(dates):continue
        use=recent_training_mask(panel.index,ends,daily,cutoff)
        if use.sum()<504:
            logs.append(dict(fit=str(cutoff.date()),model=row.model,status='insufficient_recent_history',
                training_rows=int(use.sum()),last_outcome=None,original_validation_sharpe=row.validation_sharpe))
            continue
        labels=daily[use].copy();labels[~panel.loc[use,'eligible'].to_numpy(bool)]=0
        x=panel.loc[use,features]
        model,details=fit_nn(x,labels) if neural else fit_rl(x,labels,calendar.get_indexer(panel.index[use]),calendar.get_indexer(ends[use]))
        actions.loc[dates,row.model]=predict(model,panel.loc[dates,features],neural)*panel.loc[dates,'eligible'].to_numpy()
        logs.append(dict(fit=str(cutoff.date()),model=row.model,status='refitted',training_rows=int(use.sum()),
            last_outcome=str(ends[use].max().date()),original_validation_sharpe=row.validation_sharpe,**details))
    for model in MODELS:
        allowed=[-1,0,1] if model=='nn_fixed' else np.arange(-1,1.01,.25)
        assert np.isin(actions[model],allowed).all()
    return actions,pd.DataFrame(logs)


def run_pair(d,state_transform=None,extra_features=None,output=None):
    global _market
    folder=(OUTPUT if output is None else output)/d['pair'];folder.mkdir(exist_ok=True)
    if (folder/'status.json').exists():return json.loads((folder/'status.json').read_text())
    calendar=load_calendar(ROOT);calendar=calendar[(calendar>='2010-01-01')&(calendar<='2025-08-29')]
    evaluation=calendar[calendar>='2018-01-01']
    previous=json.loads(Path('reports/aligned_contract_learning/manifest.json').read_text())
    cfg=Settings(**previous['settings'])
    panel=pd.read_parquet(SOURCE/d['pair']/'states.parquet')
    if state_transform is not None:
        original_index=panel.index.copy()
        panel=state_transform(d,panel,calendar)
        assert panel.index.equals(original_index)
        panel.to_parquet(folder/'states.parquet')
    old_actions=pd.read_csv(SOURCE/d['pair']/'actions.csv',index_col=0,parse_dates=[0])
    try:old_fits=pd.read_csv(SOURCE/d['pair']/'fits.csv')
    except pd.errors.EmptyDataError:old_fits=pd.DataFrame(columns=['status'])
    with np.load(SOURCE/d['pair']/'labels.npz',allow_pickle=False) as z:
        daily=z['daily'];ends=pd.DatetimeIndex(z['ends'])
    assert len(daily)==len(panel)==len(ends)
    actions,fits=train(panel,daily,ends,calendar,old_actions,old_fits,extra_features)
    actions.to_csv(folder/'actions.csv');fits.to_csv(folder/'fits.csv',index=False)
    del daily
    if _market is None:_market=Market(evaluation)
    prepared=None;records=[]
    for model in MODELS:
        for variant in VARIANTS:
            a=actions[model] if variant=='recent' else gate_actions(actions[model],panel,FILTERS[model])
            dest=folder/f'{model}_{variant}';dest.mkdir(exist_ok=True)
            a.to_csv(dest/'actions.csv')
            row=dict(pair=d['pair'],label=d['label'],group=d['group'],model=model,variant=variant,
                     nonzero_signals=int(a.ne(0).sum()))
            try:
                if a.ne(0).any():
                    if prepared is None:
                        q,stocks,selection,divs=_market.pair(d['secids']);prepared=PreparedMarket(q,stocks,selection)
                    nav,trades,decisions=simulate(q,stocks,a,evaluation,d['secids'],cfg,divs,selection,prepared)
                    mark_book(nav,_market.rates,cfg.capital)
                    checks=audit(nav,trades,stocks,cfg,d['secids'])
                    trades.to_csv(dest/'trades.csv',index=False);decisions.to_csv(dest/'decisions.csv',index=False)
                    entries=int(decisions.status.eq('entered').sum()) if len(decisions) else 0
                else:
                    nav=pd.DataFrame(index=evaluation)
                    nav['return_']=_market.rates.shift(1).fillna(0)*evaluation.to_series().diff().dt.days.fillna(0)/365
                    nav['nav']=cfg.capital*(1+nav.return_).cumprod();mark_book(nav,_market.rates,cfg.capital)
                    checks={'ledger_error':0.};entries=0
                nav.to_parquet(dest/'daily_nav.parquet')
                row.update(status='complete',entries=entries,**checks)
            except (UnresolvedPosition,ValueError) as exc:row.update(status='incomplete',error=str(exc))
            row['worker_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            records.append(row)
    (folder/'status.json').write_text(json.dumps(records,indent=2)+'\n')
    return records


def make_report(definitions,records):
    status=pd.DataFrame(records);status.to_csv(REPORT/'status.csv',index=False)
    old=pd.read_csv('reports/aligned_contract_learning/status.csv')
    gated=pd.read_csv('reports/contract_entry_gates/status.csv')
    gated=gated[gated.apply(lambda r:r.variant==FILTERS[r.model],axis=1)]
    failed=set(status.loc[status.status.ne('complete'),'pair'])|set(old.loc[old.status.ne('complete'),'pair'])|set(gated.loc[gated.status.ne('complete'),'pair'])
    status[status.status.ne('complete')].to_csv(REPORT/'unresolved.csv',index=False)
    rows=[];annual=[]
    for group in ['broad','legacy_extra']:
        members=[d['pair'] for d in definitions if d['group']==group and d['pair'] not in failed]
        for model in MODELS:
            configs=[('baseline',SOURCE,model),('previous_memory',Path('artifacts/contracts/entry_gates'),f'{model}_{FILTERS[model]}')]
            configs += [(v,OUTPUT,f'{model}_{v}') for v in VARIANTS]
            for variant,root,subdir in configs:
                book=None
                for pair in members:
                    nav=pd.read_parquet(root/pair/subdir/'daily_nav.parquet',columns=['nav','cash_benchmark'])
                    if book is None:book=nav.copy()
                    else:
                        assert book.index.equals(nav.index)
                        book+=nav
                if book is None:continue
                book.to_csv(REPORT/f'{group}_{model}_{variant}_portfolio.csv')
                for result in metrics(book):rows.append(dict(group=group,model=model,variant=variant,pairs=len(members),**result))
                returns=book.nav.pct_change().fillna(0);excess=returns-book.cash_benchmark.pct_change().fillna(0)
                for year,g in excess.groupby(excess.index.year):
                    annual.append(dict(group=group,model=model,variant=variant,year=year,total_return=(1+returns.loc[g.index]).prod()-1,
                        annualized_excess_sharpe=np.sqrt(252)*g.mean()/g.std() if g.std()>1e-12 else None))
    comparison=pd.DataFrame(rows);comparison.to_csv(REPORT/'comparison.csv',index=False)
    pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False)
    logs=[]
    for d in definitions:
        try:f=pd.read_csv(OUTPUT/d['pair']/'fits.csv')
        except pd.errors.EmptyDataError:continue
        if len(f):
            known=f.last_outcome.notna()
            assert (pd.to_datetime(f.loc[known,'last_outcome'])<pd.to_datetime(f.loc[known,'fit'])).all()
            f['pair']=d['pair'];logs.append(f)
    pd.concat(logs,ignore_index=True).to_csv(REPORT/'fits.csv',index=False)
    lines=['# Recent-outcome refit experiment','',
        'All 358 pairs attempted. Same contract marks, costs, sizing ladders, beta screens, historical pair membership and risk limits as the aligned learning experiment. NN remains full-size/flat.', '',
        'The original model uses five years ending before a separate one-year validation window. This experiment retains exactly the original monthly enable/disable decision, then refits enabled models on five years of outcomes completed strictly before the deployment month. Thus the historical gate validates the model specification/proposal, not the newly refitted weights. No new full-book validation of those weights is claimed.', '',
        'At least 504 usable completed outcomes are required in the new window. If shifting the window forward leaves fewer, that model-month stays flat and the insufficient-history status is logged.', '',
        'Two predefined changes: recent refit alone, and recent refit plus the previous development-selected profitability gate (RL: positive 91/365-day directional net profit; NN: positive 365-day profit). No new gate tuning. Earlier 2018–2021 and later 2022–August 2025 results are separate; all dates have already been inspected in prior research, so this is not an untouched holdout.', '',
        f'Comparisons use the same complete roster across all compared policies, retaining cash-only sleeves. {len(failed)} pairs have an unresolved evaluation in at least one compared policy. Ex-post complete-roster selection can be biased.', '',
        '| Model | Variant | Period | Pairs | Return | Excess vs cash | Annualized excess Sharpe |',
        '|---|---|---|---:|---:|---:|---:|']
    for r in comparison[comparison.group.eq('broad')].itertuples():
        lines.append(f'| {r.model} | {r.variant} | {r.period} | {r.pairs} | {r.total_return:.3%} | {r.excess_return:.3%} | {r.annualized_excess_sharpe:.3f} |')
    lines+=['','Sharpe uses daily excess returns and √252. Total return includes cash interest. 2025 is partial.', '',
            '[Comparison CSV](recent_contract_refit/comparison.csv) · [Annual results](recent_contract_refit/annual.csv) · [Fit audit](recent_contract_refit/fits.csv) · [Unresolved](recent_contract_refit/unresolved.csv)','']
    Path('reports/recent_contract_refit.md').write_text('\n'.join(lines))


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    definitions=json.loads(Path('reports/aligned_contract_learning/universe.json').read_text())
    inputs=dict(code={p:digest(p) for p in [__file__,'scripts/run_contract_walkforward.py','scripts/contract_engine.py','scripts/contract_risk.py','scripts/test_contract_entry_gates.py']},
        cache=digest('data/contract_history_shared/manifest.json'),source={d['pair']:{f:digest(SOURCE/d['pair']/f) for f in ['states.parquet','actions.csv','fits.csv','labels.npz']} for d in definitions},
        source_manifest=digest('reports/aligned_contract_learning/manifest.json'),filters=FILTERS,workers=2)
    path=REPORT/'inputs.json'
    if path.exists() and json.loads(path.read_text())!=inputs:raise ValueError('Resume inputs changed')
    path.write_text(json.dumps(inputs,indent=2)+'\n')
    started=time.monotonic();records=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for i,result in enumerate(pool.map(run_pair,definitions),1):
            records.extend(result)
            print(f"{i}/358 {result[0]['label']} "+', '.join(f"{r['model']}/{r['variant']}: {r.get('entries',r.get('error'))}" for r in result),flush=True)
    make_report(definitions,records)
    (REPORT/'manifest.json').write_text(json.dumps(dict(elapsed_seconds=time.monotonic()-started,
        max_worker_peak_rss_bytes=max(r['worker_peak_rss_bytes'] for r in records),workers=2,
        inputs_sha256=digest(path)),indent=2)+'\n')


if __name__=='__main__':main()
