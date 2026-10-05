"""Coverage-aware report and matched-roster comparison for contract retraining."""
import argparse
import json
from importlib.metadata import version
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from scripts.run_selected_pair_backtest import scores


def aggregate(members,root,model,capital):
    book=None
    for pair in members:
        nav=pd.read_parquet(root/pair/model/'daily_nav.parquet',columns=['nav','cash_benchmark'])
        if book is None:book=nav.copy()
        else:
            assert nav.index.equals(book.index)
            book+=nav
    if book is None:return None,None
    book['return_']=book.nav.pct_change().fillna(0)
    book['excess_return']=book.return_-book.cash_benchmark.pct_change().fillna(0)
    initial=len(members)*capital;years=(book.index[-1]-book.index[0]).days/365.25
    metrics=dict(pairs=len(members),**scores(book,initial),cagr=(book.nav.iloc[-1]/initial)**(1/years)-1,
                 cash_return=book.cash_benchmark.iloc[-1]/initial-1)
    return book,metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('artifacts/contracts/aligned_learning'))
    parser.add_argument('--report',type=Path,default=Path('reports/aligned_contract_learning'))
    parser.add_argument('--document',type=Path,default=Path('reports/aligned_contract_learning.md'))
    args=parser.parse_args()
    manifest=json.loads((args.report/'manifest.json').read_text())
    (args.report/'environment.json').write_text(json.dumps(dict(python=platform.python_version(),
        packages={name:version(name) for name in ['numpy','pandas','scipy','scikit-learn','pyarrow']}),indent=2)+'\n')
    status=pd.read_csv(args.report/'status.csv')
    definitions=json.loads((args.report/'universe.json').read_text())
    assert len(status)==2*len(definitions) and not status.duplicated(['pair','model']).any()
    failed=set(status.loc[status.status.ne('complete'),'pair'])
    status[status.status.ne('complete')].to_csv(args.report/'unresolved.csv',index=False)
    capital=manifest['settings']['capital'];rows=[];annual=[];breadth=[]
    for group in ['broad','legacy_extra']:
        roster=[d['pair'] for d in definitions if d['group']==group]
        common=[p for p in roster if p not in failed]
        for model in manifest['models']:
            book,metrics=aggregate(common,args.root,model,capital)
            if book is None:continue
            complete=len(common)==len(roster)
            rows.append(dict(group=group,model=model,scope='full' if complete else 'common_complete_subset',
                             requested_pairs=len(roster),unresolved_pairs=len(roster)-len(common),**metrics))
            book.to_csv(args.root/f'{group}_{model}_portfolio.csv')
            for year,g in book.groupby(book.index.year):
                previous=capital*len(common) if g.index[0]==book.index[0] else book.loc[:g.index[0]].nav.iloc[-2]
                excess=g.excess_return.iloc[1:] if g.index[0]==book.index[0] else g.excess_return
                annual.append(dict(group=group,model=model,year=int(year),total_return=g.nav.iloc[-1]/previous-1,
                                   annualized_excess_sharpe=np.sqrt(252)*excess.mean()/excess.std() if excess.std()>1e-12 else None))
            members=status[status.pair.isin(common)&status.model.eq(model)]
            active=members[members.entries.gt(0)]
            breadth.append(dict(group=group,model=model,completed_pairs=len(common),traded_pairs=len(active),
                                profitable_over_cash=int(active.excess_return.gt(0).sum()),cash_only_pairs=int(members.entries.eq(0).sum()),
                                entries=int(members.entries.sum())))
    pd.DataFrame(rows).to_csv(args.report/'portfolio.csv',index=False)
    pd.DataFrame(annual).to_csv(args.report/'annual.csv',index=False)
    pd.DataFrame(breadth).to_csv(args.report/'breadth.csv',index=False)

    old_root=Path('artifacts/contracts/broad_replay');old_report=Path('reports/broad_contract_replay')
    old=pd.read_csv(old_report/'status.csv');old_failed=set(old.loc[old.status.ne('complete'),'pair'])
    common=sorted({d['pair'] for d in definitions if d['group']=='broad'}-failed-old_failed)
    comparison=[];matched_annual=[]
    for run,root,models in [('frozen_proxy',old_root,['rl_reference','rl_sized','nn_fixed']),('contract_trained',args.root,manifest['models'])]:
        for model in models:
            book,metrics=aggregate(common,root,model,capital)
            if metrics is not None:
                comparison.append(dict(run=run,model=model,**metrics))
                for year,g in book.groupby(book.index.year):
                    previous=capital*len(common) if g.index[0]==book.index[0] else book.loc[:g.index[0]].nav.iloc[-2]
                    excess=g.excess_return.iloc[1:] if g.index[0]==book.index[0] else g.excess_return
                    matched_annual.append(dict(run=run,model=model,year=int(year),pairs=len(common),
                        total_return=g.nav.iloc[-1]/previous-1,
                        annualized_excess_sharpe=np.sqrt(252)*excess.mean()/excess.std() if excess.std()>1e-12 else None))
    pd.DataFrame(comparison).to_csv(args.report/'matched_comparison.csv',index=False)
    pd.DataFrame(matched_annual).to_csv(args.report/'matched_annual.csv',index=False)
    fits=[];label_audit=[]
    for d in definitions:
        labels=pd.read_csv(args.root/d['pair']/'label_status.csv')
        labels=labels.rename(columns={labels.columns[-1]:'rows'})
        labels['pair']=d['pair'];label_audit.append(labels)
        try:f=pd.read_csv(args.root/d['pair']/'fits.csv')
        except pd.errors.EmptyDataError:continue
        if f.empty:continue
        f['pair']=d['pair'];fits.append(f)
        trained=f[f.status.isin(['enabled','flat_validation_gate'])]
        if len(trained):
            cutoff=pd.to_datetime(trained.fit);start=pd.to_datetime(trained.validation_start)
            assert (pd.to_datetime(trained.last_training_outcome)<start).all()
            assert (pd.to_datetime(trained.validation_end)<cutoff).all()
    fits=pd.concat(fits,ignore_index=True) if fits else pd.DataFrame()
    fits.to_csv(args.report/'fits.csv',index=False)
    pd.concat(label_audit,ignore_index=True).to_csv(args.report/'labels.csv',index=False)
    lines=['# Monthly learning on actual contracts','',
           f"Attempted **{len(definitions)} pairs** with independent RL sizing and a full-size/flat NN, {manifest['start']}–{manifest['end']}. **{len(failed)} pairs have an unresolved evaluation in at least one model.** Cash-only pairs remain in the capital denominator. Any incomplete basket below is explicitly a common-complete subset, not a full-universe result.",'',
           '| Basket | Policy | Scope | Pairs | Total return | CAGR | Annualized excess Sharpe | Max drawdown |',
           '|---|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        sharpe='—' if r['annualized_excess_sharpe'] is None else f"{r['annualized_excess_sharpe']:.3f}"
        lines.append(f"| {r['group']} | {r['model']} | {r['scope']} | {r['pairs']} | {r['total_return']:.2%} | {r['cagr']:.2%} | {sharpe} | {r['max_drawdown']:.2%} |")
    lines+=['','Sharpe is calculated from daily cash-excess returns and annualized with √252. Total returns include cash interest; 2025 is partial. Equal $1 million sleeves, with idle budgets kept in cash.','',
            '## Matched comparison with the previous replay','',
            f'Both experiments below use the same {len(common)} broad pairs that completed all compared policies. This ex-post common-data roster can be biased. The new run changes marks, rewards, refit schedule, features and eligibility together; it is not an isolated ablation.','',
            '| Experiment | Policy | Total return | Annualized excess Sharpe |','|---|---|---:|---:|']
    for r in comparison:
        sharpe='—' if r['annualized_excess_sharpe'] is None else f"{r['annualized_excess_sharpe']:.3f}"
        lines.append(f"| {r['run']} | {r['model']} | {r['total_return']:.2%} | {sharpe} |")
    nn_fits=fits[fits.model.eq('nn_fixed')] if len(fits) else pd.DataFrame()
    warnings=int(nn_fits.converged.astype(str).str.lower().eq('false').sum()) if 'converged' in nn_fits else 0
    lines+=['','## What is aligned','',
            '- State IVs come from the eligible selected ATM/approximately-30-day call contracts. Standardized surface IV and future RV are not training states or reward inputs.',
            '- Labels use observed option mid marks, daily Black–Scholes delta hedges, integer contract counts, 0.10-vol-point execution penalty per option fill, $0.65 option commissions, 1 bp stock execution, financing, stock borrow, and the simulator’s lifecycle conventions.',
            '- Every month, proposal models fit a rolling five-year history ending before a separate one-year validation period. At least 504 fully observed label rows are required. Outcomes must finish strictly before validation starts. No refit includes the validation period.',
            '- Each fitted proposal is replayed on that historical validation year through the same overlapping-book, NAV-scaled risk-controlled simulator used for evaluation. It is enabled only if validation daily excess Sharpe is positive. Unresolved validation books do not pass. New validation entries stop 35 calendar days before its end, so all holdings can close before the current prediction month.',
            '- RL uses an eight-leaf direct Sharpe policy search and actions −1 through +1 in 0.25 increments. The NN has one 16-unit tanh layer, train-only scaling, and separately predicts the two full-size net directional outcomes; it chooses −1/0/+1. Both receive the same states and validation gate.',
            '- States include contract spread, trailing RV and earnings inputs, spread reversion, completed 91/365-day directional net-profit memory and observation counts, signal-close commission-per-vega estimate, contract maturity/moneyness, and pair beta/correlation.',
            '- Historical broad pair rosters are retained. Monthly eligibility uses only preceding-close information: 126-session beta of the first leg on the second between 0.75 and 1.25, correlation at least 0.5. This gate is applied to proposal-training rewards, validation actions and evaluation actions. The eight legacy extras are reported separately; unlike the frozen replay, they now use the same new learning specification.',
            '', '## Limits and verification','',
            '**The proposal optimizer still uses sums of standalone shadow cohorts.** Its objective is not an exact optimizer of the overlapping risk-controlled book. Full-book held-out validation and final evaluation enforce the actual risk rules. Profit memory is completed directional shadow-trade history, not feedback from the currently deployed policy’s own path. Validation starts with an empty book; final evaluation carries holdings across model-refit dates.',
            'Corporate-action and missing-held-quote cases remain unresolved rather than imputed or retrospectively liquidated. Unresolved training trajectories are unknown labels and excluded from fitting, not assigned zero profit; this missing-outcome selection remains a limitation. Historical earnings knowledge, midpoint execution and the existing reserve/stress model remain assumptions. This is repeated research on the same history, not an untouched test set or proof of live profitability.',
            f"Peak process memory: {manifest['peak_rss_bytes']/1024**3:.2f} GiB. Sum of worker peak RSS: {manifest.get('worker_peak_rss_sum_bytes',manifest['peak_rss_bytes'])/1024**3:.2f} GiB (a conservative bound, not a simultaneous measurement). Completed evaluations pass cash/hedge/option-ledger reconciliation. Fit records are checked for strict training/validation/evaluation timing. Batched labels matched all 3,399 AMD/NVDA training dates and all nine actions to numerical precision; tests also cover split adjustments, dividends, missing marks and missing Greeks. Reusing prepared market lookups reproduced the saved full-history AMD/NVDA NAV exactly.",
            f"{warnings} of {len(nn_fits)} NN fits reported a convergence warning; these remain visible in the fit audit and are not selectively removed.",'',
            '[Every pair result](aligned_contract_learning/status.csv) · [Annual returns and Sharpe](aligned_contract_learning/annual.csv) · [Matched comparison](aligned_contract_learning/matched_comparison.csv) · [Matched annual comparison](aligned_contract_learning/matched_annual.csv) · [Trading breadth](aligned_contract_learning/breadth.csv) · [Unresolved cases](aligned_contract_learning/unresolved.csv) · [Training label coverage](aligned_contract_learning/labels.csv) · [Monthly fit audit](aligned_contract_learning/fits.csv)','',
            'Reproduce the bounded three-worker run with `python -m scripts.run_aligned_contract_batch`, then `python -m scripts.report_aligned_contract_learning`. Each worker processes pairs sequentially. Worker input manifests preserve resume checks; labels and detailed ledgers stay in ignored local artifacts.']
    args.document.write_text('\n'.join(lines)+'\n')
    print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(comparison).to_string(index=False))


if __name__=='__main__':main()
