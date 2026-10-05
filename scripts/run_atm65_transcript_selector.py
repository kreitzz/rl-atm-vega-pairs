"""Monthly selection between saved policies using their past validation scores."""
import gc
import json
from pathlib import Path
from dataclasses import replace
import resource
import numpy as np
import pandas as pd
from scripts.run_atm65_transcripts import (OUT as TRIAL,REPORT as TRIAL_REPORT,EMPLOYEE,EMPLOYEE_REPORT,
    NN,CURVE,PERIODS,employee_roster,baseline_actions,baseline_results,config,Replay,load_pair,
    Market,ROOT,load_calendar,cash_book,save_result,mark_book,scores,UnresolvedPosition)
from scripts.run_atm65_ablation import read_actions,write_json
from scripts.run_contract_walkforward import digest

OUT=Path('artifacts/contracts/atm65_transcript_selector')
REPORT=Path('reports/atm65_transcript_selector')
MODELS=['rl_selector','nn_selector']


def choose_actions(original,candidate,old_fits,new_fits,kind,employee):
    assert original.index.equals(candidate.index)
    old=old_fits.set_index('fit');new=new_fits.set_index('fit')
    original_status=kind+'_sentiment_status' if employee else 'status'
    original_score=kind+'_sentiment_validation_sharpe' if employee else 'validation_sharpe'
    new_status=kind+'_transcripts_status';new_score=kind+'_transcripts_validation_sharpe'
    result=original.copy();records=[]
    for month in sorted(result.index.to_period('M').unique()):
        key=str(month.start_time.date());a=old.loc[key];b=new.loc[key]
        enabled_a=a[original_status]=='enabled';enabled_b=b[new_status]=='enabled'
        score_a=float(a[original_score]) if enabled_a else None
        score_b=float(b[new_score]) if enabled_b else None
        for row in [a,b]:
            if 'validation_end' in row and pd.notna(row.validation_end):assert pd.Timestamp(row.validation_end)<month.start_time
        use_new=enabled_b and (not enabled_a or score_b>score_a)
        mask=result.index.to_period('M')==month
        if use_new:result.loc[mask]=candidate.loc[mask]
        records.append(dict(fit=key,model=kind+'_selector',reference_validation_sharpe=score_a,
            transcript_validation_sharpe=score_b,chosen='transcripts' if use_new else 'reference',
            validation_end=b.get('validation_end'),last_training_outcome=b.get('last_training_outcome')))
    return result,pd.DataFrame(records)


def main():
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    inputs=json.loads((TRIAL/'inputs.json').read_text());defs=inputs['definitions'];employees=employee_roster()
    coverage=pd.read_csv(TRIAL_REPORT/'coverage.csv');selected=set(coverage.loc[coverage.retrain,'pair'])
    files=[Path(__file__),TRIAL/'inputs.json',TRIAL_REPORT/'comparison.csv',TRIAL_REPORT/'audit.json']
    for d in defs:
        if d['pair'] in selected:
            files.extend([TRIAL/d['pair']/'fits.csv',TRIAL/d['pair']/'rl_transcripts/actions.csv',TRIAL/d['pair']/'nn_transcripts/actions.csv'])
    signature=dict(definitions=defs,rule='higher positive prior-year validation Sharpe; ties retain reference',
        sources={str(p):digest(p) for p in files})
    if (OUT/'inputs.json').exists():assert json.loads((OUT/'inputs.json').read_text())==signature
    else:write_json(OUT/'inputs.json',signature)
    cfg=replace(config(),option_commission=.03);cal=load_calendar(ROOT)
    cal=cal[(cal>='2010-01-01')&(cal<='2025-08-29')];market=Market(cal);all_logs=[]
    for n,d in enumerate([x for x in defs if x['pair'] in selected],1):
        folder=OUT/d['pair'];folder.mkdir(exist_ok=True);employee=d['pair'] in employees
        new_fits=pd.read_csv(TRIAL/d['pair']/'fits.csv');actions={};logs=[]
        for kind in ['rl','nn']:
            old_path=EMPLOYEE/d['pair']/'fits.csv' if employee else (CURVE/d['pair']/'atm_control/fits.csv' if kind=='rl' else NN/d['pair']/'fee_065/fits.csv')
            old_fits=pd.read_csv(old_path)
            original=read_actions(baseline_actions(d['pair'],kind,employee));candidate=read_actions(TRIAL/d['pair']/(kind+'_transcripts')/'actions.csv')
            chosen,table=choose_actions(original,candidate,old_fits,new_fits,kind,employee)
            m=kind+'_selector';actions[m]=chosen;logs.append(table.assign(pair=d['pair']))
            dest=folder/m;dest.mkdir(exist_ok=True);chosen.rename('code').to_csv(dest/'actions.csv')
        table=pd.concat(logs,ignore_index=True);table.to_csv(folder/'fits.csv',index=False);all_logs.append(table)
        pending=[(m,p,start) for m in MODELS for p,start in PERIODS if not (folder/m/p/'status.json').exists()]
        if not pending:continue
        changed={m:actions[m].ne(read_actions(baseline_actions(d['pair'],m,employee))).any() for m in MODELS}
        if any(changed.values()):
            q,s,selections,divs=load_pair(market,d['secids']);replay=Replay(q,s,selections,d['secids'],divs,cfg)
        for m,p,start in pending:
            dest=folder/m/p;dest.mkdir(parents=True,exist_ok=True)
            if not changed[m]:
                source=baseline_results(d['pair'],m,p,employee);status=json.loads((source/'status.json').read_text())
                if not status.get('cash_only'):status['nav_source']=status.get('nav_source',str(source/'daily_nav.parquet'))
                status['source_unchanged_reference']=True
            elif not actions[m].loc[actions[m].index>=start].ne(0).any():
                status=dict(status='complete',pair=d['pair'],label=d['label'],entries=0,cash_only=True)
            else:
                try:
                    nav,trades,decisions=replay.run(actions[m].loc[actions[m].index>=start],cal[cal>=start])
                    mark_book(nav,market.rates,cfg.capital);status=save_result(dest,nav,trades,decisions,cfg,s,d)
                except UnresolvedPosition as exc:status=dict(status='incomplete',pair=d['pair'],label=d['label'],error=str(exc))
            write_json(dest/'status.json',status)
        if any(changed.values()):del q,s,selections,divs,replay
        gc.collect()
        if n%10==0:print(f'Selector replay: {n}/{len(selected)} pairs; peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20:.0f} MiB',flush=True)
    pd.concat(all_logs,ignore_index=True).to_csv(REPORT/'fits.csv',index=False)
    def source(d,m,p):return OUT/d['pair']/m/p if m.endswith('selector') and d['pair'] in selected else baseline_results(d['pair'],m,p,d['pair'] in employees)
    models=['rl_baseline','rl_selector','nn_baseline','nn_selector']
    statuses={(d['pair'],m,p):json.loads((source(d,m,p)/'status.json').read_text()) for d in defs for m in models for p,_ in PERIODS}
    complete=[d for d in defs if all(statuses[d['pair'],m,p]['status']=='complete' for m in models for p,_ in PERIODS)]
    unresolved=[dict(pair=d['pair'],model=m,period=p,error=statuses[d['pair'],m,p].get('error')) for d in defs for m in models for p,_ in PERIODS if statuses[d['pair'],m,p]['status']!='complete']
    write_json(REPORT/'coverage.json',dict(requested_pairs=len(defs),common_complete_pairs=len(complete),unresolved=unresolved))
    rows=[];annual=[];pairs=[]
    for m in models:
        for p,start in PERIODS:
            dates=cal[cal>=start];cash=cash_book(dates,market.rates,cfg.capital);book=cash[['nav','cash_benchmark']]*len(complete);entries=0
            for d in complete:
                status=statuses[d['pair'],m,p];folder=source(d,m,p)
                nav=cash if status.get('cash_only') else pd.read_parquet(status.get('nav_source',folder/'daily_nav.parquet'),columns=['nav','cash_benchmark'])
                assert nav.index.equals(dates);book+=nav[['nav','cash_benchmark']]-cash[['nav','cash_benchmark']];entries+=status['entries']
                pairs.append(dict(model=m,period=p,pair=d['pair'],label=d['label'],entries=status['entries'],net_excess_pnl=float(nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1])))
            book['return_']=book.nav.pct_change().fillna(0);book['excess_return']=book.return_-book.cash_benchmark.pct_change().fillna(0)
            book.to_csv(REPORT/f'{m}_{p}_portfolio.csv')
            rows.append(dict(model=m,period=p,pairs=len(complete),entries=entries,net_excess_pnl=float(book.nav.iloc[-1]-book.cash_benchmark.iloc[-1]),**scores(book,cfg.capital*len(complete))))
            for year,g in book.groupby(book.index.year):
                sd=g.excess_return.std();annual.append(dict(model=m,period=p,year=year,return_percent=100*((1+g.return_).prod()-1),annualized_excess_sharpe=float(np.sqrt(252)*g.excess_return.mean()/sd) if sd>1e-12 else None))
    comparison=pd.DataFrame(rows);comparison.to_csv(REPORT/'comparison.csv',index=False)
    pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False);pd.DataFrame(pairs).to_csv(REPORT/'pairs.csv',index=False)
    if len(complete)==345:
        saved=pd.read_csv(EMPLOYEE_REPORT/'comparison.csv')
        for kind in ['rl','nn']:
            for p,_ in PERIODS:
                a=comparison[(comparison.model==kind+'_baseline')&(comparison.period==p)].iloc[0]
                b=saved[(saved.universe=='all')&(saved.model==kind+'_sentiment')&(saved.period==p)].iloc[0]
                np.testing.assert_allclose(a.annualized_excess_sharpe,b.annualized_excess_sharpe,atol=1e-12);assert a.entries==b.entries
    for path,h in signature['sources'].items():assert digest(path)==h
    write_json(OUT/'task_status.json',dict(status='complete',pairs=len(complete),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    lines=['# Monthly reference/transcript policy selection','',
        'Use the saved current-best and transcript policies. Each pair/month chooses the enabled policy with higher prior-year validation Sharpe at the unchanged $0.65 training hurdle. Ties retain the reference; missing or disabled transcript proposals cannot replace an enabled reference. No evaluation-period realized returns are used for monthly selection, and no models are refitted. Original recent-five-year refit actions, costs, risk limits and contract selection remain unchanged. Evaluation uses $0.03 commission.','',
        '| Model | Full Sharpe | Recent Sharpe | Full entries |','|---|---:|---:|---:|']
    for m in models:
        a=comparison[(comparison.model==m)&(comparison.period=='full')].iloc[0];b=comparison[(comparison.model==m)&(comparison.period=='recent')].iloc[0]
        lines.append(f'| {m} | {a.annualized_excess_sharpe:.3f} | {b.annualized_excess_sharpe:.3f} | {int(a.entries)} |')
    lines+=['','Periods start flat in 2018 and 2022 and end in August 2025. Sharpe is above cash and annualized with √252. All comparisons use the common completed roster; failed paths are reported. This selection rule was tried after observing the mixed transcript result on previously examined history. The prior-year scores were already used for proposal validation; selecting between them adds selection uncertainty and is not an independent validation test.','',
        '[Comparison](atm65_transcript_selector/comparison.csv) · [Annual](atm65_transcript_selector/annual.csv) · [Selection logs](atm65_transcript_selector/fits.csv) · [Coverage](atm65_transcript_selector/coverage.json)','']
    REPORT.with_suffix('.md').write_text('\n'.join(lines));print(comparison[['model','period','pairs','entries','annualized_excess_sharpe']].to_string(index=False),flush=True)


if __name__=='__main__':main()
