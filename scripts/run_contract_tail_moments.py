"""Controlled trailing return skewness/kurtosis feature experiment."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd

from scripts.run_recent_contract_refit import run_pair, SOURCE, MODELS, FILTERS
from scripts.run_aligned_contract_learning import asset_paths
from scripts.run_contract_walkforward import digest
from scripts.test_contract_entry_gates import metrics

OUTPUT=Path('artifacts/contracts/tail_moments')
REPORT=Path('reports/contract_tail_moments')
FEATURES=['skew_a','kurt_a','skew_b','kurt_b','skew_difference','kurt_difference','moments_available_a','moments_available_b']


def trailing_moments(returns,calendar,window=63):
    """Signal-close returns included; require all sessions, never bridge gaps.

    Unbiased sample skewness and Fisher excess kurtosis. Missing/constant
    histories get neutral zero values with an explicit unavailability flag.
    """
    if window<4:raise ValueError('At least four observations required')
    result=pd.DataFrame(0.,index=calendar,columns=['skew','kurt','available'])
    if len(calendar)<window:return result
    r=returns.reindex(calendar).to_numpy(float)
    # Center each window locally. Some rolling implementations center using the
    # whole input series, allowing extreme future values to alter past rounding.
    windows=np.lib.stride_tricks.sliding_window_view(r,window)
    valid=np.isfinite(windows).all(axis=1)
    positions=np.flatnonzero(valid)+window-1
    centered=windows[valid]-windows[valid].mean(axis=1,keepdims=True)
    m2=(centered**2).mean(axis=1)
    nonconstant=np.sqrt(m2*window/(window-1))>1e-12
    centered=centered[nonconstant];m2=m2[nonconstant];positions=positions[nonconstant]
    skew=np.sqrt(window*(window-1))/(window-2)*(centered**3).mean(axis=1)/m2**1.5
    kurt=((window**2-1)*(centered**4).mean(axis=1)/m2**2-3*(window-1)**2)/((window-2)*(window-3))
    result.iloc[positions]=np.column_stack([skew,kurt,np.ones(len(positions))])
    return result


def augment(definition,panel,calendar):
    result=panel.copy()
    for suffix,path in zip(['a','b'],asset_paths(definition)):
        moments=trailing_moments(pd.read_parquet(path,columns=['ret']).ret,calendar).reindex(panel.index)
        result[f'skew_{suffix}']=moments['skew']
        result[f'kurt_{suffix}']=moments['kurt']
        result[f'moments_available_{suffix}']=moments.available
    both=result.moments_available_a.eq(1)&result.moments_available_b.eq(1)
    result['skew_difference']=(result.skew_a-result.skew_b).where(both,0.)
    result['kurt_difference']=(result.kurt_a-result.kurt_b).where(both,0.)
    assert np.isfinite(result[FEATURES]).all().all()
    return result


def worker(definition):
    return run_pair(definition,state_transform=augment,extra_features=FEATURES,output=OUTPUT)


def report(definitions,records):
    status=pd.DataFrame(records);status.to_csv(REPORT/'status.csv',index=False)
    baseline=pd.read_csv('reports/recent_contract_refit/status.csv')
    older=pd.read_csv('reports/contract_entry_gates/status.csv')
    older=older[older.apply(lambda r:r.variant==FILTERS[r.model],axis=1)]
    failed=set(status.loc[status.status.ne('complete'),'pair'])|set(baseline.loc[baseline.status.ne('complete'),'pair'])|set(older.loc[older.status.ne('complete'),'pair'])
    status[status.status.ne('complete')].to_csv(REPORT/'unresolved.csv',index=False)
    rows=[];annual=[];activity=[];availability=[];fits=[]
    for d in definitions:
        states=pd.read_parquet(OUTPUT/d['pair']/'states.parquet')
        evaluation=states[states.index.year>=2018]
        availability.append(dict(pair=d['pair'],rows=len(evaluation),both_available_fraction=float((evaluation.moments_available_a.eq(1)&evaluation.moments_available_b.eq(1)).mean())))
        try:f=pd.read_csv(OUTPUT/d['pair']/'fits.csv')
        except pd.errors.EmptyDataError:continue
        known=f.last_outcome.notna()
        assert (pd.to_datetime(f.loc[known,'last_outcome'])<pd.to_datetime(f.loc[known,'fit'])).all()
        f['pair']=d['pair'];fits.append(f)
    pd.DataFrame(availability).to_csv(REPORT/'availability.csv',index=False)
    pd.concat(fits,ignore_index=True).to_csv(REPORT/'fits.csv',index=False)
    for group in ['broad','legacy_extra']:
        members=[d['pair'] for d in definitions if d['group']==group and d['pair'] not in failed]
        for model in MODELS:
            configs=[('previous_filtered',Path('artifacts/contracts/entry_gates'),f'{model}_{FILTERS[model]}',older,None)]
            for name,root,frame in [('refit',Path('artifacts/contracts/recent_refit'),baseline),('moments',OUTPUT,status)]:
                configs.extend([(name,root,f'{model}_recent',frame,'recent'),(name+'_filtered',root,f'{model}_recent_memory',frame,'recent_memory')])
            for name,root,subdir,frame,variant in configs:
                book=None
                for pair in members:
                    nav=pd.read_parquet(root/pair/subdir/'daily_nav.parquet',columns=['nav','cash_benchmark'])
                    if book is None:book=nav.copy()
                    else:
                        assert book.index.equals(nav.index)
                        book+=nav
                if book is None:continue
                book.to_csv(REPORT/f'{group}_{model}_{name}_portfolio.csv')
                for row in metrics(book):rows.append(dict(group=group,model=model,variant=name,pairs=len(members),**row))
                returns=book.nav.pct_change().fillna(0);excess=returns-book.cash_benchmark.pct_change().fillna(0)
                for year,g in excess.groupby(excess.index.year):
                    annual.append(dict(group=group,model=model,variant=name,year=year,total_return=(1+returns.loc[g.index]).prod()-1,
                        annualized_excess_sharpe=np.sqrt(252)*g.mean()/g.std() if g.std()>1e-12 else None))
                subset=frame[frame.pair.isin(members)&frame.model.eq(model)]
                if variant is not None:subset=subset[subset.variant.eq(variant)]
                activity.append(dict(group=group,model=model,variant=name,pairs=len(members),entries=int(subset.entries.sum()),active_pairs=int(subset.entries.gt(0).sum())))
    comparison=pd.DataFrame(rows);comparison.to_csv(REPORT/'comparison.csv',index=False)
    pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False);pd.DataFrame(activity).to_csv(REPORT/'activity.csv',index=False)
    lines=['# Trailing return skewness and kurtosis', '',
        'Feature-only change relative to the recent-refit experiment: 63-session sample skewness and Fisher excess kurtosis for each leg, their pairwise differences, and two availability flags. A normal distribution has population excess kurtosis zero. These are historical return moments, not option-implied smile/skew or implied risk-neutral moments.', '',
        'Use CRSP retx histories already present in the asset panels, reindexed to the trading calendar. All 63 observations must be available and variance nonzero. Missing values become neutral zero plus an explicit availability flag; pair differences are zero unless both legs are available. The source asset panels inherit the older standardized-IV availability calendar; missing sessions are not compressed together. Signal-close observations are available for next-session execution.', '',
        'Two models: independent RL sizing and fixed-size NN. Both retain their existing features, costs, contract labels, monthly five-year refit window, beta gates and risk limits. The original historical validation enable/disable decisions remain fixed. Thus the gate does not independently validate the added features or refitted weights. Profitability-filtered variants use the previously selected rules, without retuning.', '',
        f'Attempted all 358 pairs. {len(failed)} pairs are unresolved in at least one compared policy; all tables use a matched complete roster and retain cash-only sleeves. Ex-post coverage selection can be biased. 2018–2021 and 2022–August 2025 are separated, but both periods have been inspected before. This is exploratory research, not an untouched test.', '',
        '| Model | Variant | Period | Pairs | Return | Excess vs cash | Annualized excess Sharpe |',
        '|---|---|---|---:|---:|---:|---:|']
    for r in comparison[comparison.group.eq('broad')].itertuples():
        lines.append(f'| {r.model} | {r.variant} | {r.period} | {r.pairs} | {r.total_return:.3%} | {r.excess_return:.3%} | {r.annualized_excess_sharpe:.3f} |')
    lines+=['','Daily cash-excess Sharpe is annualized with √252. Returns include cash interest. 2025 is partial. Skewness and kurtosis are tested together, so this does not identify their individual contributions.', '',
        '[Results](contract_tail_moments/comparison.csv) · [Annual results](contract_tail_moments/annual.csv) · [Activity](contract_tail_moments/activity.csv) · [Feature availability](contract_tail_moments/availability.csv) · [Unresolved](contract_tail_moments/unresolved.csv)','']
    Path('reports/contract_tail_moments.md').write_text('\n'.join(lines))


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    definitions=json.loads(Path('reports/aligned_contract_learning/universe.json').read_text())
    fingerprint=dict(code={p:digest(p) for p in [__file__,'scripts/run_recent_contract_refit.py','scripts/run_contract_walkforward.py','scripts/contract_engine.py','scripts/contract_risk.py','scripts/test_contract_entry_gates.py']},
        cache=digest('data/contract_history_shared/manifest.json'),source={d['pair']:{f:digest(SOURCE/d['pair']/f) for f in ['states.parquet','actions.csv','fits.csv','labels.npz']} for d in definitions},
        assets={str(p):digest(p) for d in definitions for p in asset_paths(d)},
        features=FEATURES,window=63,settings=json.loads(Path('reports/aligned_contract_learning/manifest.json').read_text())['settings'],workers=2)
    path=REPORT/'inputs.json'
    if path.exists() and json.loads(path.read_text())!=fingerprint:raise ValueError('Resume inputs changed')
    path.write_text(json.dumps(fingerprint,indent=2)+'\n');started=time.monotonic();records=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for i,result in enumerate(pool.map(worker,definitions),1):
            records.extend(result)
            print(f"{i}/358 {result[0]['label']}",flush=True)
    report(definitions,records)
    (REPORT/'manifest.json').write_text(json.dumps(dict(elapsed_seconds=time.monotonic()-started,
        max_worker_peak_rss_bytes=max(r['worker_peak_rss_bytes'] for r in records),workers=2,inputs_sha256=digest(path)),indent=2)+'\n')


if __name__=='__main__':main()
