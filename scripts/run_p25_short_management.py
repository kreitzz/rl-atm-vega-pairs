"""6M/12M management-history extension; frozen entries, costs and action bank.

Reuse the audited shared ledger. Separate output directories preserve the
18M/36M study, including its code and saved model hashes. One guarded worker.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
import unittest

import pandas as pd
import numpy as np

from scripts import run_p25_long_management as long

PREVIOUS=long.OUT
PREVIOUS_REPORT=long.REPORT
OUT=Path('artifacts/contracts/p25_short_management')
REPORT=Path('reports/p25_short_management')
WINDOWS=[6,12]


def protected_sources():
    sources=json.loads((PREVIOUS_REPORT/'source_manifest.json').read_text())
    for name,want in sources.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==want,f'Previous source changed: {name}'
    return sources


def training_episode(rows,cal,cut,window):
    start=cut-pd.DateOffset(months=window)
    use=rows.loc[rows.date.ge(start)&rows.date.lt(cut)&rows.known_contract_expiry.lt(cut)].copy()
    history=cal[(cal>=start)&(cal<cut)]
    # The original 250-session floor would prevent every six-month fit.
    minimum=100 if window==6 else 250
    ready=len(use)>=16 and len(history)>=minimum and start>=cal[0]-pd.Timedelta(days=7)
    assert use.empty or use.known_contract_expiry.max()<cut
    return use,history,ready,minimum


def fit_management(rows,market,tenor,window,rv):
    folder=OUT/'management_models'/f'{tenor}D_{window}M'
    folder.mkdir(parents=True,exist_ok=True)
    expiry=[]
    for row in rows.itertuples():
        quotes=[market.quote(row.date,int(oid)) if np.isfinite(oid) else None for oid in [row.optionid_a,row.optionid_b]]
        expiry.append(max(q.expiry for q in quotes) if all(q is not None for q in quotes) else pd.NaT)
    rows=rows.assign(known_contract_expiry=expiry)
    cal=market.rates.index[market.rates.index>='2018-01-01']
    records=[]
    for cut in pd.date_range('2018-01-01',cal[-1],freq='2QS-JAN'):
        stamp=str(cut.date());path=folder/f'{stamp}.json'
        if path.exists():records.append(json.loads(path.read_text()));continue
        use,history,ready,minimum=training_episode(rows,cal,cut,window)
        record=dict(fit=stamp,tenor=tenor,window_months=window,training_proposals=len(use),history_sessions=len(history),minimum_history_sessions=minimum,
                    training_last_outcome=str(use.known_contract_expiry.max().date()) if len(use) else None,
                    selected='hold',fitted=bool(ready),candidates=[])
        if ready:
            for rule in long.BANK:
                market.variant=f'train{tenor}_{window}_{stamp}_{rule}';market.reported_year=0
                callback=lambda date,held,rule=rule:long.held_rule(date,held,rv,rule)
                nav,trades,decisions,_=long.replay(use,market,history,tenor,callback,True)
                score=long.base.sharpe(nav.excess_return.iloc[1:])
                record['candidates'].append(dict(rule=rule,annualized_training_excess_sharpe=score,
                    entered_cohorts=int(decisions.status.eq('entered').sum()),active_sessions=int(nav.active_cohorts.gt(0).sum())))
                del nav,trades,decisions;gc.collect()
            record['selected']=max(record['candidates'],key=lambda r:r['annualized_training_excess_sharpe'])['rule']
        long.dump(path,record);records.append(record)
        print(f'Management {tenor}D/{window}M {stamp}: {record["selected"]}; {len(use)} completed proposals',flush=True)
        long.progress('management_fits',tenor=tenor,window=window,fit=stamp)
    return records


def run_cases():
    frame=pd.read_parquet(PREVIOUS/'rv_states.parquet')
    rv={int(sid):g.set_index('date')[['rv20','rv60']] for sid,g in frame.groupby('secid')};del frame
    for tenor in [30,90]:
        market=long.configure_market(tenor)
        folder=PREVIOUS/'replays'/('30D_current_hold' if tenor==30 else '90D_hold')
        source=pd.read_parquet(folder/'scored_allocations.parquet')
        accepted=pd.read_csv(folder/'decisions.csv',parse_dates=['signal_date']).query("status=='entered'")
        locked_rows=source.merge(accepted[['signal_date','pair']].rename(columns={'signal_date':'date'}),on=['date','pair'],validate='one_to_one')
        trades=pd.read_csv(folder/'trades.csv').query("reason=='entry'")
        by_cohort={key:g.set_index('secid') for key,g in trades.groupby('cohort')}
        counts={f'{r.date.date()}|{r.pair}|{r.wing}':tuple(by_cohort[f'{r.date.date()}|{r.pair}|{r.wing}'].loc[[r.secid_a,r.secid_b],'quantity'].abs()) for r in locked_rows.itertuples()}
        for window in WINDOWS:
            records=fit_management(source,market,tenor,window,rv)
            schedule={pd.Timestamp(r['fit']):r['selected'] for r in records}
            callback=lambda date,held:long.held_rule(date,held,rv,schedule[max(k for k in schedule if k<=date)])
            long.save_case(f'{tenor}D_manage_{window}M',source,market,tenor,callback)
            long.save_case(f'{tenor}D_locked_manage_{window}M',locked_rows,market,tenor,callback,locked_counts=counts)


def summarize():
    rows=[];annual=[];periods=[]
    previous_cases=['30D_current_hold','90D_hold','30D_manage_18M','30D_manage_36M','90D_manage_36M',
                    '30D_locked_manage_18M','30D_locked_manage_36M','90D_locked_manage_36M']
    folders=[PREVIOUS/'replays'/case for case in previous_cases]+sorted((OUT/'replays').iterdir())
    for folder in folders:
        record=json.loads((folder/'status.json').read_text())
        record['management_training_months']=int(record['case'].rsplit('_',1)[1][:-1]) if '_manage_' in record['case'] else 0
        rows.append(record);annual.append(pd.read_csv(folder/'annual.csv'))
        nav=pd.read_parquet(folder/'daily_nav.parquet')
        for name,start,end in [('2018–2020','2018-01-01','2020-12-31'),('2021–2023','2021-01-01','2023-12-31'),('2024–partial2025','2024-01-01','2025-08-29'),('2021–partial2025','2021-01-01','2025-08-29')]:
            selected=nav.loc[start:end]
            returns=nav.excess_return.iloc[1:].reindex(selected.index).dropna()
            periods.append(dict(case=record['case'],period=name,annualized_excess_sharpe=long.base.sharpe(returns)))
    result=pd.DataFrame(rows);result.to_csv(REPORT/'comparison.csv',index=False)
    years=pd.concat(annual,ignore_index=True);years.to_csv(REPORT/'annual.csv',index=False)
    pd.DataFrame(periods).to_csv(REPORT/'periods.csv',index=False)
    fits=[json.loads(p.read_text()) for p in (OUT/'management_models').glob('*/*.json')]
    assert all(r['training_last_outcome'] is None or pd.Timestamp(r['training_last_outcome'])<pd.Timestamp(r['fit']) for r in fits)
    long.dump(REPORT/'management_fits.json',fits)
    summary=[]
    for tenor in [30,90]:
        for window in ([6,12,18,36] if tenor==30 else [6,12,36]):
            case=f'{tenor}D_manage_{window}M';locked=f'{tenor}D_locked_manage_{window}M'
            a=result.loc[result.case.eq(case)].iloc[0];b=result.loc[result.case.eq(locked)].iloc[0]
            summary.append(dict(tenor=tenor,management_months=window,annualized_excess_sharpe=float(a.annualized_excess_sharpe),entries=int(a.entries),trimmed_cohorts=int(a.trimmed_cohorts),closed_cohorts=int(a.closed_cohorts),net_excess_pnl=float(a.net_excess_pnl),relative_to_cash_cagr_pct=float(a.relative_to_cash_cagr_pct),locked_initial_counts_sharpe=float(b.annualized_excess_sharpe),locked_entries=int(b.entries)))
    table=pd.DataFrame(summary);table.to_csv(REPORT/'windows.csv',index=False)
    fitted=pd.DataFrame([dict(tenor=tenor,window_months=window,fit_count=sum(r['fitted'] for r in fits if r['tenor']==tenor and r['window_months']==window),total_cutoffs=sum(r['tenor']==tenor and r['window_months']==window for r in fits),nonhold_selections=sum(r['selected']!='hold' for r in fits if r['tenor']==tenor and r['window_months']==window)) for tenor in [30,90] for window in WINDOWS])
    fitted.to_csv(REPORT/'training_coverage.csv',index=False)
    note=('6- AND 12-MONTH MANAGEMENT TRAINING\n\n'
          'The windows are management training history, not option maturity. Keep the original 18M monthly entry-size learner, frozen directions and 608 (30D) / 577 (90D) proposals. Same six hold/trim/close policies, semiannual refits, next-session observed-quote execution, financing, costs and risk limits as the prior 18M/36M management study. Actual option commission $0.03 plus 0.1 vol point per leg per entry/elective exit; training commission $0.65. Both legs trim proportionally; no continuous vega rebalance.\n\n'
          'Training outcomes must finish before each fit, purged to original scheduled expiries. Require at least 16 completed proposals. The history minimum is 100 sessions for 6M and 250 for 12M; using the inherited 250-session floor for 6M would prevent all training. Require a full calendar window in the cached 2018+ market. Otherwise hold fallback; all entry proposals remain executable. Sparse training and early warmup differ by window. The 2021–partial2025 comparison reports a common post-warmup interval. Selection uses actual shared-ledger episodes; this remains finite policy search, not a neural sequential learner.\n\n'
          'Locked controls use the original 212 (30D) / 116 (90D) accepted dates and initial integer counts, with risk checks retained. They are counterfactual attribution controls, not live fixed-count forecasts. Dynamic entry admission and NAV-scaled counts can change in the unrestricted strategy.\n\n'
          'Evaluation 2018-01-02 to 2025-08-29 at $10m equity. Sharpe is annualized daily excess over the same-equity cash benchmark, sqrt(252). 2025 partial. All dates previously researched; no fresh holdout. The 90D entry signals/sizes are transfers from 30D; only 213 of 577 dates have both selected contracts cached. Do not infer a pure maturity effect from different coverage. Prior code, data, models and checkpoint results verified unchanged. One worker, 50% of one core, 3GiB RSS stop.\n\n'+table.to_string(index=False)+'\n\nTRAINING COVERAGE\n'+fitted.to_string(index=False)+'\n\nANNUAL EXCESS SHARPE\n'+years.pivot(index='year',columns='case',values='annualized_excess_sharpe').round(3).to_string()+'\n')
    (REPORT/'findings.txt').write_text(note)
    long.dump(REPORT/'audit.json',dict(status='complete',new_replays=8,all_ledgers_verified=True,completed_outcome_purge=True,original_sources_unchanged=True,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    sources=[Path(__file__),Path('tests/test_p25_short_management.py')]+list((OUT/'management_models').glob('*/*.json'))+list((OUT/'replays').glob('*/scored_allocations.parquet'))
    long.dump(REPORT/'source_manifest.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    print(table.to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['test','replay','summarize','all'],default='all');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    protected=protected_sources();long.OUT=OUT;long.REPORT=REPORT
    long.dump(REPORT/'protected_sources.json',protected)
    if args.phase=='test':
        suite=unittest.defaultTestLoader.loadTestsFromNames(['tests.test_p25_short_management'])
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        long.dump(REPORT/'tests.json',dict(tests_run=result.testsRun,successful=result.wasSuccessful()))
        if not result.wasSuccessful():raise SystemExit(1)
        return
    if args.phase in ['replay','all']:run_cases()
    if args.phase in ['summarize','all']:summarize()
    protected_sources()
    long.dump(OUT/'resource_usage.json',dict(observed_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,cpu_core_fraction=.5,workers=1,rss_limit_bytes=3*1024**3,last_completed_phase=args.phase))


if __name__=='__main__':main()
