"""Smiles and term structure as states; identical contract rewards and execution."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scripts.build_contract_surface import OUTPUT as SURFACE, VALUES
from scripts.run_recent_contract_refit import run_pair, SOURCE, MODELS, FILTERS
from scripts.run_contract_walkforward import digest
from scripts.test_contract_entry_gates import metrics

OUTPUT=Path('artifacts/contracts/surface_features')
REPORT=Path('reports/contract_surface_features')
FEATURES=[f'surface_{value}_{side}{ending}' for value in VALUES for side in ['a','b','difference'] for ending in ['', '_available']]


@lru_cache(maxsize=4)
def surface_asset(sid):
    path=SURFACE/'assets'/f'{sid}.parquet'
    return pd.read_parquet(path,columns=VALUES) if path.exists() else pd.DataFrame(columns=VALUES,index=pd.DatetimeIndex([]))


def augment(definition,panel,calendar):
    result=panel.copy()
    for side,sid in zip(['a','b'],definition['secids']):
        asset=surface_asset(sid).reindex(panel.index)
        for value in VALUES:
            observed=asset[value].astype(float).replace([np.inf,-np.inf],np.nan)
            result[f'surface_{value}_{side}']=observed.fillna(0.)
            result[f'surface_{value}_{side}_available']=observed.notna().astype(float)
    for value in VALUES:
        both=result[f'surface_{value}_a_available'].eq(1)&result[f'surface_{value}_b_available'].eq(1)
        result[f'surface_{value}_difference']=(result[f'surface_{value}_a']-result[f'surface_{value}_b']).where(both,0.)
        result[f'surface_{value}_difference_available']=both.astype(float)
    assert np.isfinite(result[FEATURES]).all().all()
    return result


def worker(d):
    return run_pair(d,state_transform=augment,extra_features=FEATURES,output=OUTPUT)


def report(definitions,records):
    status=pd.DataFrame(records);status.to_csv(REPORT/'status.csv',index=False)
    baseline=pd.read_csv('reports/recent_contract_refit/status.csv')
    older=pd.read_csv('reports/contract_entry_gates/status.csv')
    older=older[older.apply(lambda r:r.variant==FILTERS[r.model],axis=1)]
    failed=set(status.loc[status.status.ne('complete'),'pair'])|set(baseline.loc[baseline.status.ne('complete'),'pair'])|set(older.loc[older.status.ne('complete'),'pair'])
    status[status.status.ne('complete')].to_csv(REPORT/'unresolved.csv',index=False)
    rows=[];annual=[];activity=[];coverage=[];fit_frames=[]
    for d in definitions:
        panel=pd.read_parquet(OUTPUT/d['pair']/'states.parquet')
        actions=pd.read_csv(OUTPUT/d['pair']/'actions.csv',index_col=0,parse_dates=[0]);actions.index=pd.DatetimeIndex(actions.index)
        eligible=panel.reindex(actions.index);eligible=eligible[eligible.eligible.astype(bool)]
        for value in VALUES:
            coverage.append(dict(pair=d['pair'],label=d['label'],feature=value,eligible_rows=len(eligible),
                both_available_rows=int(eligible[f'surface_{value}_difference_available'].sum())))
        try:f=pd.read_csv(OUTPUT/d['pair']/'fits.csv')
        except pd.errors.EmptyDataError:continue
        known=f.last_outcome.notna()
        assert (pd.to_datetime(f.loc[known,'last_outcome'])<pd.to_datetime(f.loc[known,'fit'])).all()
        f['pair']=d['pair'];fit_frames.append(f)
    pd.DataFrame(coverage).to_csv(REPORT/'coverage.csv',index=False)
    pd.concat(fit_frames,ignore_index=True).to_csv(REPORT/'fits.csv',index=False)
    for group in ['broad','legacy_extra']:
        members=[d['pair'] for d in definitions if d['group']==group and d['pair'] not in failed]
        for model in MODELS:
            configs=[('previous_filtered',Path('artifacts/contracts/entry_gates'),f'{model}_{FILTERS[model]}',older,None)]
            for name,root,frame in [('refit',Path('artifacts/contracts/recent_refit'),baseline),('surface',OUTPUT,status)]:
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
    lines=['# Option smile and term structure states','',
        'Actual option-chain features added to the recent-refit model. The historical return skewness/kurtosis experiment is not included here. Each stock receives four surface attributes, and the pair receives their differences; every value also has an availability flag.', '',
        '- Downside skew: lower-strike call IV near 0.75 call delta minus the traded ATM contract IV, at exactly the same expiry.',
        '- Smile curvature: mean IV of lower/upper strikes near 0.75/0.25 call delta minus that ATM IV.',
        '- Term slope: (longer ATM IV − shorter ATM IV) × 30 / actual calendar-day expiry gap. Shorter expiry is nearest 15 days within 8–22 days; longer is nearest 45 within 35–55 days. They must strictly bracket the traded expiry.',
        '- Term bend: traded ATM IV minus linear interpolation between the two neighboring ATM IVs at its actual maturity. This is local IV curvature, not a fitted arbitrage-free surface or a risk-neutral fourth moment.', '',
        'Quality and timing: standard 100-share calls, positive bid, uncrossed quotes, bid/ask width ≤25% of midpoint, and signal-close open interest ≥100. Wing candidates use vendor delta within 0.10 of the target; after midpoint Black–Scholes inversion they must still pass that delta tolerance and lie on the correct side of the ATM strike. ATM neighbors use the nearest strike within 10% of spot. Ties are deterministic. No wing extrapolation, stale-quote filling or interpolation across dates. Missing surface features become zero plus an explicit unavailable flag.', '',
        'All selected points are refitted from their midpoints using the same declared-dividend escrow, rates and European-call approximation as the existing simulator. The archive contains calls only: lower-strike call IV is a downside proxy, not directly observed put skew. American early-exercise effects and approximate wing deltas remain limitations. Expiries are approximate, not exact constant-maturity 15/30/45-day contracts.', '',
        'Model and accounting controls: same actual-contract rewards, monthly recent five-year training, completed-outcome purge, sizing, transaction costs, beta eligibility and risk limits. Independent RL ladder decisions and fixed-size/flat NN are retained. The original validation enable gates are held fixed; they do not independently validate the added features or newly refitted weights. Filtered variants retain the previously selected profitability rules without retuning.', '',
        f'Attempted all 358 pairs; {len(failed)} have an unresolved evaluation in at least one compared policy. Tables use matched complete rosters and retain cash-only sleeves. Ex-post coverage selection can be biased. The development and later periods have already been inspected; these are exploratory results, not an untouched holdout.', '',
        '| Model | Variant | Period | Pairs | Return | Excess vs cash | Annualized excess Sharpe |',
        '|---|---|---|---:|---:|---:|---:|']
    for r in comparison[comparison.group.eq('broad')].itertuples():
        lines.append(f'| {r.model} | {r.variant} | {r.period} | {r.pairs} | {r.total_return:.3%} | {r.excess_return:.3%} | {r.annualized_excess_sharpe:.3f} |')
    lines+=['','Sharpe uses daily cash-excess returns and √252; total returns include interest. 2025 is partial. Smile and term structure are tested together in this first pass.', '',
        '[Results](contract_surface_features/comparison.csv) · [Annual results](contract_surface_features/annual.csv) · [Activity](contract_surface_features/activity.csv) · [Coverage](contract_surface_features/coverage.csv) · [Unresolved](contract_surface_features/unresolved.csv)','']
    Path('reports/contract_surface_features.md').write_text('\n'.join(lines))


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    definitions=json.loads(Path('reports/aligned_contract_learning/universe.json').read_text())
    fingerprint=dict(code={p:digest(p) for p in [__file__,'scripts/run_recent_contract_refit.py','scripts/run_contract_walkforward.py','scripts/contract_engine.py','scripts/contract_risk.py','scripts/test_contract_entry_gates.py']},
        surface=digest(SURFACE/'manifest.json'),surface_assets={str(p):digest(p) for p in sorted((SURFACE/'assets').glob('*.parquet'))},
        cache=digest('data/contract_history_shared/manifest.json'),source={d['pair']:{f:digest(SOURCE/d['pair']/f) for f in ['states.parquet','actions.csv','fits.csv','labels.npz']} for d in definitions},
        features=FEATURES,settings=json.loads(Path('reports/aligned_contract_learning/manifest.json').read_text())['settings'],workers=2)
    path=REPORT/'inputs.json'
    if path.exists() and json.loads(path.read_text())!=fingerprint:raise ValueError('Resume inputs changed')
    path.write_text(json.dumps(fingerprint,indent=2)+'\n');started=time.monotonic();records=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for i,result in enumerate(pool.map(worker,definitions),1):
            records.extend(result);print(f"{i}/358 {result[0]['label']}",flush=True)
    report(definitions,records)
    (REPORT/'manifest.json').write_text(json.dumps(dict(elapsed_seconds=time.monotonic()-started,
        max_worker_peak_rss_bytes=max(r['worker_peak_rss_bytes'] for r in records),workers=2,inputs_sha256=digest(path)),indent=2)+'\n')


if __name__=='__main__':main()
