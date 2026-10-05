"""RL-only transcript experiments on the frozen employee-sentiment best.

Same monthly fits, completed-outcome cutoffs and contract accounting. Single
guarded worker; baseline enabled refits are recovered before candidate replay.
"""
import argparse
from dataclasses import asdict,replace
import gc
import json
from pathlib import Path
import resource
import time
import numpy as np
import pandas as pd
from scripts.run_atm65_ablation import features,read_actions,write_json,NN,CURVE,FROZEN,PERIODS
from scripts.run_atm65_employee_sentiment import OUT as EMPLOYEE,REPORT as EMPLOYEE_REPORT,ADDED as EMPLOYEE_ADDED,CODES
from scripts.atm65_transcript_features import FIELDS,ADDED,add_transcripts,leg_states
from scripts.curve_trading.config import config,COMBOS
from scripts.curve_trading.policy import ATM,Replay,fit,predict,gate,memory_gates
from scripts.run_atm_cost_hurdles import make_units,reward_and_memory
from scripts.replay_frozen_atm_commission import load_pair,cash_book,save_result
from scripts.run_broad_contract_replay import Market,ROOT
from scripts.run_curve_trading import RUNOFF,screen_dates
from scripts.run_contract_walkforward import digest
from scripts.run_selected_pair_backtest import scores,audit
from scripts.run_aligned_contract_learning import mark_book
from scripts.select_atm_contracts import load_calendar
from scripts.option_flex.contract_engine import UnresolvedPosition
from scripts.retrain_atm_commission import check_fits

OUT=Path('artifacts/contracts/atm65_rl_transcripts')
REPORT=Path('reports/atm65_rl_transcripts')
MODELS=['rl_transcripts']
WORKFLOW=Path('config/research_training.json')
CHECKPOINT=Path('config/atm30_fee065_employee_2026-10-01_checkpoint.json')


def configure(feature):
    """Independent outputs keep completed RL/NN experiments immutable."""
    global OUT, REPORT, FIELDS, ADDED
    workflow=json.loads(WORKFLOW.read_text())
    if workflow['retrain_models'] != ['rl'] or workflow['retrain_nn']:
        raise ValueError('This runner supports RL-only retraining')
    from scripts.atm65_transcript_features import FIELDS as all_fields
    FIELDS=list(all_fields) if feature=='all' else [feature]
    ADDED=[f'transcript_{field}_{suffix}' for field in FIELDS for suffix in ('gap','available')]
    name='atm65_rl_transcripts_'+feature
    OUT=Path('artifacts/contracts')/name
    REPORT=Path('reports')/name


def employee_roster():
    c=pd.read_csv(EMPLOYEE_REPORT/'coverage.csv')
    return set(c.loc[c.retrain,'pair'])


def baseline_actions(pair,model,employee):
    assert model.startswith('rl')
    if employee:
        return EMPLOYEE/pair/'rl_sentiment'/'actions.csv'
    return CURVE/pair/'atm_control/actions.csv'


def baseline_results(pair,model,period,employee):
    assert model.startswith('rl')
    if employee: return EMPLOYEE/pair/'rl_sentiment'/period
    return FROZEN/pair/(period+'_003')


def prepare(defs,cal,employees):
    legs=leg_states(cal,sorted({sid for d in defs for sid in d['secids']}));rows=[]
    for d in defs:
        source=EMPLOYEE/d['pair']/'states.parquet';folder=OUT/d['pair'];folder.mkdir(exist_ok=True)
        row=dict(pair=d['pair'],label=d['label'],employee_features=d['pair'] in employees,
                 state_rows=0,observed_test_rows=0,observed_permitted_test_rows=0,retrain=False)
        if source.exists():
            panel=pd.read_parquet(source);a,b=d['secids'];state=add_transcripts(panel,legs[a],legs[b])
            state.to_parquet(folder/'states.parquet')
            test=(panel.index>='2018-01-01')&(panel.index<=cal[-1]-pd.Timedelta(days=RUNOFF))&screen_dates(d,panel.index)
            with np.load(NN/d['pair']/'fee_065/gates.npz') as z:
                permitted=panel.eligible.to_numpy(bool)&z['available']&z['gates'].any(axis=1)
            observed=state[[c for c in ADDED if c.endswith('_available')]].gt(0).any(axis=1).to_numpy()
            row.update(state_rows=len(state),observed_test_rows=int((observed&test).sum()),
                       observed_permitted_test_rows=int((observed&test&permitted).sum()),
                       retrain=bool((observed&test&permitted).any()))
        rows.append(row)
    coverage=pd.DataFrame(rows);coverage.to_csv(REPORT/'coverage.csv',index=False)
    write_json(OUT/'prepared.json',dict(pairs=len(defs),retrained_pairs=int(coverage.retrain.sum()),
        added_features=ADDED,feature_timing='earliest creation-dated call version; max(call,creation) date plus one business day',
        max_age_days=180,baseline='employee sentiment current best; review features active on original 28 sleeves only'))
    print(coverage[coverage.retrain][['label','observed_permitted_test_rows']].to_string(index=False),flush=True)
    print(f'{coverage.retrain.sum()}/{len(defs)} sleeves have transcripts on permitted evaluation dates.',flush=True)


def train_pair(d,panel,daily,finish,gates,available,cal,replay,rates,folder,employee):
    stop=cal[-1]-pd.Timedelta(days=RUNOFF)
    wanted=(panel.index>='2018-01-01')&(panel.index<=stop)&screen_dates(d,panel.index)
    idx=panel.index[wanted];original=features(panel)+(EMPLOYEE_ADDED if employee else [])
    assert len(original)==(49 if employee else 47)
    saved={m:read_actions(baseline_actions(d['pair'],m,employee)) for m in MODELS}
    for a in saved.values(): assert a.index.equals(idx)
    baseline_fits={}
    if employee:
        table=pd.read_csv(EMPLOYEE/d['pair']/'fits.csv').set_index('fit')
        baseline_fits={m:table['rl_sentiment_status'] for m in MODELS}
    else:
        for m in MODELS:
            p=CURVE/d['pair']/'atm_control/fits.csv'
            baseline_fits[m]=pd.read_csv(p).set_index('fit').status
    actions=pd.DataFrame(0,index=idx,columns=MODELS,dtype=int);logs=[]
    checkpoint=folder/'month_checkpoint.json'
    if checkpoint.exists():
        old=json.loads(checkpoint.read_text());assert old['dates']==[str(t) for t in idx]
        actions.iloc[:,:]=np.asarray(old['actions'],dtype=int);logs=old['fits']
    completed={r['fit'] for r in logs};positions=cal.get_indexer(panel.index);completion=cal[finish]
    finite=np.isfinite(daily).all(axis=(1,2));columns=original+ADDED
    for month in pd.period_range('2018-01-01',stop,freq='M'):
        cutoff=month.start_time;fitdate=str(cutoff.date());test=(idx>=cutoff)&(idx<(month+1).start_time)
        if not test.any() or fitdate in completed: continue
        vstart=cutoff-pd.DateOffset(years=1)
        use=(panel.index>=vstart-pd.DateOffset(years=5))&(completion<vstart)&finite
        recent=(panel.index>=cutoff-pd.DateOffset(years=5))&(completion<cutoff)&finite
        vcal=cal[(cal>=vstart)&(cal<cutoff)]
        val=(panel.index>=vcal[0])&(panel.index<=vcal[-1]-pd.Timedelta(days=RUNOFF))
        opportunity=(panel.eligible.to_numpy(bool)&available[ATM]&gates[ATM].any(axis=1)&val).any()
        dates=idx[test]
        row=dict(fit=fitdate,training_rows=int(use.sum()),recent_rows=int(recent.sum()),
                 validation_start=str(vcal[0].date()),validation_end=str(vcal[-1].date()))
        for m in MODELS:
            fitter=fit;predictor=predict
            # Recover the current best's enabled recent refit, preserving its historical validation decisions.
            if baseline_fits[m].loc[fitdate]=='enabled':
                assert recent.sum()>=504
                own=daily[recent].copy();own[~panel.loc[recent,'eligible'].to_numpy(bool)]=0
                model,_=fitter(panel.loc[recent,original],own,positions[recent],finish[recent],CODES)
                recovered=gate(predictor(model,panel.loc[dates,original]),dates,panel,gates,available)
            else: recovered=pd.Series(0,index=dates,dtype=int)
            np.testing.assert_array_equal(recovered.to_numpy(),saved[m].loc[dates].to_numpy(),
                err_msg=f'{d["label"]} {fitdate} {m}: current best was not reproduced')
            row[m+'_baseline_reproduced']=True
            row[m+'_status']='insufficient_training' if use.sum()<504 or len(vcal)<200 else 'flat_no_validation_opportunities'
            if use.sum()<504 or len(vcal)<200 or not opportunity: continue
            labels=daily[use].copy();labels[~panel.loc[use,'eligible'].to_numpy(bool)]=0
            model,details=fitter(panel.loc[use,columns],labels,positions[use],finish[use],CODES)
            row['last_training_outcome']=str(completion[use].max().date())
            row.update({m+'_proposal_'+k:v for k,v in details.items()})
            if m.startswith('rl'):
                row[m+'_proposal_transcript_splits']=sum(columns[int(f)].startswith('transcript_') for f in model[0].tree_.feature if f>=0)
            proposal=gate(predictor(model,panel.loc[val,columns]),panel.index[val],panel,gates,available)
            score=None;error=None
            try:
                nav,trades,_=replay.run(proposal,vcal);mark_book(nav,rates,replay.cfg.capital)
                audit(nav,trades,replay.s,replay.cfg,d['secids']);score=scores(nav,replay.cfg.capital)['annualized_excess_sharpe']
            except (UnresolvedPosition,ValueError) as exc:error=str(exc)
            enabled=score is not None and score>0 and recent.sum()>=504
            row[m+'_status']='enabled' if enabled else 'flat_validation_gate'
            row[m+'_validation_sharpe']=score;row[m+'_validation_error']=error
            if enabled:
                own=daily[recent].copy();own[~panel.loc[recent,'eligible'].to_numpy(bool)]=0
                model,details=fitter(panel.loc[recent,columns],own,positions[recent],finish[recent],CODES)
                actions.loc[dates,m]=gate(predictor(model,panel.loc[dates,columns]),dates,panel,gates,available)
                row['last_recent_outcome']=str(completion[recent].max().date())
                row.update({m+'_refit_'+k:v for k,v in details.items()})
                if m.startswith('rl'):
                    row[m+'_refit_transcript_value_splits']=sum(columns[int(f)].startswith('transcript_') and columns[int(f)].endswith('_gap') for f in model[0].tree_.feature if f>=0)
                    row[m+'_refit_transcript_missing_splits']=sum(columns[int(f)].startswith('transcript_') and columns[int(f)].endswith('_available') for f in model[0].tree_.feature if f>=0)
        logs.append(row);write_json(checkpoint,dict(dates=[str(t) for t in idx],actions=actions.to_numpy().tolist(),fits=logs))
        if month.month in [1,7]:print(f'  {d["label"]} {month}: RL {row["rl_transcripts_status"]}',flush=True)
    table=pd.DataFrame(logs);check_fits(table);table.to_csv(folder/'fits.csv',index=False)
    for m in MODELS:
        dest=folder/m;dest.mkdir(exist_ok=True);a=actions[m].rename('code')
        pd.testing.assert_series_equal(a,gate(a,a.index,panel,gates,available),check_names=False);a.to_csv(dest/'actions.csv')
    return actions


def summary(defs,coverage,market,cfg,employees):
    selected=set(coverage.loc[coverage.retrain,'pair']);models=['rl_baseline','rl_transcripts']
    def source(d,m,p):
        return OUT/d['pair']/m/p if m.endswith('transcripts') and d['pair'] in selected else baseline_results(d['pair'],m,p,d['pair'] in employees)
    statuses={(d['pair'],m,p):json.loads((source(d,m,p)/'status.json').read_text()) for d in defs for m in models for p,_ in PERIODS}
    complete=[d for d in defs if all(statuses[d['pair'],m,p]['status']=='complete' for m in models for p,_ in PERIODS)]
    unresolved=[dict(pair=d['pair'],model=m,period=p,error=statuses[d['pair'],m,p].get('error')) for d in defs for m in models for p,_ in PERIODS if statuses[d['pair'],m,p]['status']!='complete']
    write_json(REPORT/'accounting_coverage.json',dict(requested_pairs=len(defs),common_complete_pairs=len(complete),unresolved=unresolved))
    rows=[];annual=[];pairs=[]
    for universe in ['all','transcript_covered']:
        roster=complete if universe=='all' else [d for d in complete if d['pair'] in selected]
        if not roster:continue
        for m in models:
            for p,start in PERIODS:
                cal=market.calendar[market.calendar>=start];cash=cash_book(cal,market.rates,cfg.capital)
                book=cash[['nav','cash_benchmark']]*len(roster);entries=active=0
                for d in roster:
                    status=statuses[d['pair'],m,p];folder=source(d,m,p)
                    nav=cash if status.get('cash_only') else pd.read_parquet(status.get('nav_source',folder/'daily_nav.parquet'),columns=['nav','cash_benchmark'])
                    assert nav.index.equals(cal);book+=nav[['nav','cash_benchmark']]-cash[['nav','cash_benchmark']]
                    entries+=status['entries'];active+=int(status['entries']>0)
                    pairs.append(dict(universe=universe,model=m,period=p,pair=d['pair'],label=d['label'],entries=status['entries'],net_excess_pnl=float(nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1])))
                book['return_']=book.nav.pct_change().fillna(0);book['excess_return']=book.return_-book.cash_benchmark.pct_change().fillna(0)
                book.to_csv(REPORT/f'{universe}_{m}_{p}_portfolio.csv')
                rows.append(dict(universe=universe,model=m,period=p,pairs=len(roster),entries=entries,active_pairs=active,
                    net_excess_pnl=float(book.nav.iloc[-1]-book.cash_benchmark.iloc[-1]),**scores(book,cfg.capital*len(roster))))
                for year,g in book.groupby(book.index.year):
                    sd=g.excess_return.std()
                    annual.append(dict(universe=universe,model=m,period=p,year=year,return_percent=100*((1+g.return_).prod()-1),annualized_excess_sharpe=float(np.sqrt(252)*g.excess_return.mean()/sd) if sd>1e-12 else None))
    comparison=pd.DataFrame(rows);comparison.to_csv(REPORT/'comparison.csv',index=False)
    pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False);pd.DataFrame(pairs).to_csv(REPORT/'pairs.csv',index=False)
    if len(complete)==len(defs):
        saved=pd.read_csv(EMPLOYEE_REPORT/'comparison.csv')
        for m,old in [('rl_baseline','rl_sentiment')]:
            for p,_ in PERIODS:
                got=comparison[(comparison.universe=='all')&(comparison.model==m)&(comparison.period==p)].iloc[0]
                expected=saved[(saved.universe=='all')&(saved.model==old)&(saved.period==p)].iloc[0]
                np.testing.assert_allclose(got.annualized_excess_sharpe,expected.annualized_excess_sharpe,atol=1e-12);assert got.entries==expected.entries
    fit_tables=[pd.read_csv(OUT/d['pair']/'fits.csv').assign(pair=d['pair']) for d in defs if d['pair'] in selected]
    if fit_tables:pd.concat(fit_tables,ignore_index=True).to_csv(REPORT/'fits.csv',index=False)
    lines=['# RL-only transcript signals on the employee-sentiment reference','',
        'Baseline is the current best employee-sentiment strategy: annualized excess Sharpe 0.368 full history and 0.428 since 2022. Both periods start flat; all 345 original matched sleeves remain in the requested roster. Original employee-sentiment features remain active on the same 28 sleeves.','',
        'Selected transcript gaps are added: executive negative-language change versus the previous call, uncertainty in executive answers versus prepared remarks, and negativity in analyst questions. Each first-minus-second gap has its own availability flag. Missing values are zero-filled and flagged. Latest available transcript scores carry for at most 180 days. Earliest creation-dated version is selected for each company/event; availability is one business day after the later event/creation date. Historical company links and word-count requirements are unchanged.','',
        f'Tested transcript fields: {FIELDS}. No NN fits are performed.', '',
        f'{len(selected)} sleeves have an observed transcript feature on a permitted evaluation date and are refitted across the evaluation period. Other sleeves reuse their own current reference policy. Features and coverage were fixed before these runs, with no sign or threshold selected from evaluation returns.','',
        'Same monthly prior-year validation, recent-five-year refits, completed-outcome purging, profitability memory and beta/correlation gates, 25/50/75/100% RL sizing ladder, ATM30 calls, daily delta hedging and hold to expiry. Training/validation commission is $0.65; evaluation commission is $0.03. Spread, hedge, financing and risk settings are unchanged. Enabled current-reference refits are reconstructed and checked against their saved actions before candidate evaluation.','',
        '| Universe | Model | Full Sharpe | Recent Sharpe | Full entries |','|---|---|---:|---:|---:|']
    for universe in ['all','transcript_covered']:
        for m in models:
            a=comparison[(comparison.universe==universe)&(comparison.model==m)&(comparison.period=='full')]
            b=comparison[(comparison.universe==universe)&(comparison.model==m)&(comparison.period=='recent')]
            if len(a):lines.append(f'| {universe} | {m} | {a.iloc[0].annualized_excess_sharpe:.3f} | {b.iloc[0].annualized_excess_sharpe:.3f} | {int(a.iloc[0].entries)} |')
    lines+=['','Full: 2018–August 2025. Recent: 2022–August 2025, starting flat. Sharpe uses daily returns above cash and √252. Any unresolved contract paths are explicitly excluded from every model on a common roster.','',
        'Current best checkpoint remains frozen. This comparison uses previously examined history. Transcript lexicons are simple fixed word counts; vendor historical edits, publication timing and company mappings are not fully verified against archived data versions. The training labels are rebuilt from actual saved option-contract paths at the unchanged checkpoint costs.','',
        f'[Comparison]({REPORT.name}/comparison.csv) · [Annual]({REPORT.name}/annual.csv) · [Pairs]({REPORT.name}/pairs.csv) · [Coverage]({REPORT.name}/coverage.csv) · [Fit diagnostics]({REPORT.name}/fits.csv)','']
    REPORT.with_suffix('.md').write_text('\n'.join(lines))
    print(comparison[['universe','model','period','pairs','entries','annualized_excess_sharpe']].to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--limit',type=int)
    parser.add_argument('--report-only',action='store_true')
    parser.add_argument('--transcript-feature', choices=['all',*FIELDS], default='all')
    args=parser.parse_args()
    configure(args.transcript_feature)
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    inputs=json.loads((EMPLOYEE/'inputs.json').read_text());defs=inputs['definitions'];employees=employee_roster()
    actual=replace(config(),option_commission=.03);training=replace(actual,option_commission=.65)
    assert asdict(actual)==inputs['actual']
    from scripts.run_atm65_employee_sentiment import LINKS,REVIEWS
    files=[Path(__file__),WORKFLOW,Path('scripts/atm65_transcript_features.py'),CHECKPOINT,
        EMPLOYEE/'inputs.json',EMPLOYEE_REPORT/'coverage.csv',EMPLOYEE_REPORT/'comparison.csv',
        LINKS/'option_links.parquet',LINKS/'company_links.parquet',REVIEWS/'ciq_links.parquet',
        REVIEWS/'transcripts_states.parquet',REVIEWS/'selected_calls.parquet',REVIEWS/'transcript_counts.parquet',
        Path('scripts/curve_trading/policy.py'),Path('scripts/curve_trading/unit_paths.py'),
        Path('scripts/run_atm_cost_hurdles.py'),Path('scripts/run_atm_nn_cost_hurdles.py'),
        Path('scripts/run_contract_walkforward.py'),Path('scripts/option_flex/contract_engine.py')]
    for d in defs:
        files.extend(p for p in [EMPLOYEE/d['pair']/'states.parquet',NN/d['pair']/'fee_065/gates.npz',
            baseline_actions(d['pair'],'rl',d['pair'] in employees)] if p.exists())
    signature=dict(models=MODELS,workflow=json.loads(WORKFLOW.read_text()),definitions=defs,baseline_checkpoint=str(CHECKPOINT),actual=asdict(actual),training=asdict(training),
        transcript_features=FIELDS,code_and_sources={str(p):digest(p) for p in files})
    if (OUT/'inputs.json').exists():assert json.loads((OUT/'inputs.json').read_text())==signature,'Inputs changed'
    else:write_json(OUT/'inputs.json',signature)
    cal=load_calendar(ROOT);cal=cal[(cal>='2010-01-01')&(cal<='2025-08-29')]
    if not (OUT/'prepared.json').exists():prepare(defs,cal,employees)
    if args.prepare:return
    coverage=pd.read_csv(REPORT/'coverage.csv');selected=set(coverage.loc[coverage.retrain,'pair'])
    market=Market(cal);started=time.monotonic();newly=0
    if not args.report_only:
        for n,d in enumerate([x for x in defs if x['pair'] in selected],1):
            folder=OUT/d['pair']
            if all((folder/m/p/'status.json').exists() for m in MODELS for p,_ in PERIODS):continue
            pair_started=time.monotonic();print(f'{n}/{len(selected)} {d["label"]}: transcript family',flush=True)
            panel=pd.read_parquet(folder/'states.parquet');q,s,selections,divs=load_pair(market,d['secids'])
            with np.load(NN/d['pair']/'fee_065/gates.npz') as z:
                gates=np.zeros((len(COMBOS),len(panel),2),bool);gates[ATM]=z['gates']
                available=np.zeros((len(COMBOS),len(panel)),bool);available[ATM]=z['available'];saved_ends=z['ends'].copy()
            units=make_units(panel,q,s,selections,divs,cal,training)
            labels,finish,vegas,statuses,rewards,_=reward_and_memory(units,training)
            np.testing.assert_array_equal(finish,saved_ends)
            np.testing.assert_array_equal(gates[ATM],memory_gates(panel.index,cal[finish],rewards,vegas[:,[0,-1]]>0))
            daily=np.zeros((len(panel),9,labels.shape[-1]),dtype=np.float32);daily[:,1:]=labels
            del units,labels;gc.collect();replay=Replay(q,s,selections,d['secids'],divs,training)
            actions=train_pair(d,panel,daily,finish,gates,available,cal,replay,market.rates,folder,d['pair'] in employees)
            replay.cfg=actual
            for m in MODELS:
                for p,start in PERIODS:
                    dest=folder/m/p;dest.mkdir(parents=True,exist_ok=True)
                    if (dest/'status.json').exists():continue
                    a=actions.loc[actions.index>=start,m]
                    if not a.ne(0).any():result=dict(status='complete',pair=d['pair'],label=d['label'],entries=0,cash_only=True)
                    else:
                        try:
                            nav,trades,decisions=replay.run(a,cal[cal>=start]);mark_book(nav,market.rates,actual.capital)
                            result=save_result(dest,nav,trades,decisions,actual,s,d)
                        except UnresolvedPosition as exc:result=dict(status='incomplete',pair=d['pair'],label=d['label'],error=str(exc))
                    result.update(training_hurdle=.65,execution_commission=.03);write_json(dest/'status.json',result)
            newly+=1
            write_json(OUT/'progress.json',dict(completed_retrained_pairs=n,retrained_pairs=len(selected),last_pair=d['label'],
                last_pair_seconds=time.monotonic()-pair_started,elapsed_seconds=time.monotonic()-started,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            print(f'{d["label"]}: completed in {time.monotonic()-pair_started:.1f}s; peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20:.0f} MiB',flush=True)
            del panel,q,s,selections,divs,replay,daily,actions;gc.collect()
            if args.limit and newly>=args.limit:return
    summary(defs,coverage,market,actual,employees)
    write_json(OUT/'task_status.json',dict(status='complete',models=MODELS,pairs=len(defs),retrained_pairs=len(selected),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__=='__main__':main()
