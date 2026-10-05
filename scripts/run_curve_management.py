"""Matched 2x2x2 expiry, smile and learned-exit comparison; serial/resumable."""
import argparse
import gc
import json
from pathlib import Path
import resource
import shutil
import tempfile
import time
from dataclasses import asdict
import joblib
import numpy as np
import pandas as pd
from scripts.curve_management.config import (COMBOS,ACTIONS,ATM,UNIVERSES,RUNOFF,config,SHORTLIST,
                                             MAX_EXIT_COHORTS_PER_COMBO)
from scripts.curve_management.config import REFIT_MONTHS
from scripts.curve_management.data import load_pair,entry_context,Paths,Replay
from scripts.curve_management.learning import shortlist,fit_entry,predict_entry,fit_exit,gate
from scripts.run_atm65_ablation import features,read_actions,write_json,NN,CURVE,PERIODS
from scripts.run_atm65_signal_mining import fit_masks
from scripts.run_curve_trading import screen_dates
from scripts.run_broad_contract_replay import ROOT,Market
from scripts.select_atm_contracts import load_calendar
from scripts.run_contract_walkforward import digest
from scripts.run_aligned_contract_learning import mark_book
from scripts.replay_frozen_atm_commission import cash_book,save_result
from scripts.run_selected_pair_backtest import scores,audit
from scripts.option_flex.contract_engine import UnresolvedPosition

OUT=Path('artifacts/contracts/curve_management')
REPORT=Path('reports/curve_management')
MODELS=[u+'_'+exit for u in UNIVERSES for exit in ['hold','exit']]

def dump_model(path,model):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp')
    joblib.dump(model,temporary);temporary.replace(path)

def load_models(folder):
    return {p.stem:joblib.load(p) for p in sorted((folder/'exit_models').glob('*.joblib'))}

def train(d,data,cal,replay,rates,folder,columns):
    panel=data.panel;stop=cal[-1]-pd.Timedelta(days=RUNOFF)
    wanted=(panel.index>='2018-01-01')&(panel.index<=stop)&screen_dates(d,panel.index);idx=panel.index[wanted]
    actions=pd.DataFrame(0,index=idx,columns=MODELS,dtype=int)
    years=sorted(set(idx.year));plans={};rankings={}
    for universe,allowed in UNIVERSES.items():
        plans[universe]={};rankings[universe]=[]
        for year in years:
            selected,records=shortlist(year,allowed,panel,data.labels,data.ends,data.gates,cal)
            plans[universe][year]=selected;rankings[universe].extend(records)
    peers=[]
    for universe,allowed in UNIVERSES.items():
        for manage in ['hold','exit']:
            name=universe+'_'+manage;dest=folder/name;dest.mkdir(exist_ok=True)
            if all((dest/p/'status.json').exists() for p,_ in PERIODS):
                actions[name]=read_actions(dest/'actions.csv') if (dest/'actions.csv').exists() else 0
                peers.append((name,universe,manage));continue
            equivalent=next((peer for peer,u,m in peers if m==manage and plans[u]==plans[universe]),None)
            if equivalent is not None:
                source=folder/equivalent
                for path in source.glob('*'):
                    if path.name in ['actions.csv','fits.csv','month_checkpoint.json']:shutil.copy2(path,dest/path.name)
                if (source/'exit_models').exists():shutil.copytree(source/'exit_models',dest/'exit_models',dirs_exist_ok=True)
                pd.DataFrame(rankings[universe] if rankings[universe] else [dict(year=None,combo=ATM,selected=True)]).to_csv(dest/'shortlists.csv',index=False)
                write_json(dest/'reused_policy.json',dict(source=equivalent,reason='Identical annual contract shortlists, feature columns, dates, costs, gates and management mode'))
                actions[name]=read_actions(dest/'actions.csv');peers.append((name,universe,manage))
                print(f'  {d["label"]} {name}: identical to {equivalent}',flush=True);continue
            own=actions[name].rename('code').copy();logs=[];ranks=[];checkpoint=dest/'month_checkpoint.json'
            if checkpoint.exists():
                old=json.loads(checkpoint.read_text());assert old['dates']==[str(t) for t in idx]
                own.iloc[:]=old['actions'];logs=old['fits'];ranks=old['ranks']
            completed={r['fit'] for r in logs};year=None
            for month in pd.period_range('2018-01-01',cal[-1] if manage=='exit' else stop,freq='Q'):
                cutoff=month.start_time;fitdate=str(cutoff.date());test=(idx>=cutoff)&(idx<(month+1).start_time)
                active_previous=own.loc[(own.index<cutoff)&(own.index>=cutoff-pd.Timedelta(days=110))].ne(0).any()
                if (not test.any() and not (manage=='exit' and active_previous)) or fitdate in completed:continue
                if year!=month.year:
                    year=month.year;combos=plans[universe].get(year,[ATM])
                    ranking=[r for r in rankings[universe] if r['year']==year]
                    ranks.extend(ranking);hold,finish,codes=data.subset(combos)
                    finite=np.isfinite(hold).all(axis=(1,2));completion=cal[finish]
                use,recent=fit_masks(panel.index,completion,finite,cutoff)
                vcal=cal[(cal>=cutoff-pd.DateOffset(years=1))&(cal<cutoff)]
                val=(panel.index>=vcal[0])&(panel.index<=vcal[-1]-pd.Timedelta(days=RUNOFF))
                opportunity=any((panel.eligible.to_numpy(bool)&data.available[c]&data.gates[c].any(axis=1)&val).any() for c in combos)
                info=dict(fit=fitdate,combos=combos,training_rows=int(use.sum()),recent_rows=int(recent.sum()),
                    validation_start=str(vcal[0].date()),validation_end=str(vcal[-1].date()))
                row=dict(info,status='flat_no_validation_opportunities');proposal_exit=recent_exit=None
                if use.sum()<504 or len(vcal)<200:row['status']='insufficient_training'
                elif opportunity and test.any():
                    if manage=='exit':
                        proposal_exit,details=fit_exit(data,use,combos,cutoff-pd.DateOffset(years=1))
                        row.update({'proposal_'+k:v for k,v in details.items()})
                    daily=hold if manage=='hold' else data.subset(combos,proposal_exit)[0]
                    labels=daily[use].copy();labels[~panel.loc[use,'eligible'].to_numpy(bool)]=0
                    model,details=fit_entry(panel.loc[use,columns],labels,data.positions[use],finish[use],codes)
                    row.update(details,last_training_outcome=str(completion[use].max().date()))
                    proposal=gate(predict_entry(model,panel.loc[val,columns]),panel.index[val],panel,data.gates,data.available)
                    score=error=None
                    if proposal.ne(0).any():
                        schedule=None
                        # A fixed proposal exit policy must span the whole validation year.
                        if manage=='exit':schedule=validation_schedule(data,proposal,proposal_exit,vcal)
                        try:
                            nav,t,_=replay.run(proposal,vcal,schedule);mark_book(nav,rates,replay.cfg.capital)
                            audit(nav,t,replay.s,replay.cfg,d['secids']);score=scores(nav,replay.cfg.capital)['annualized_excess_sharpe']
                        except UnresolvedPosition as exc:error=str(exc)
                    permitted=score is not None and score>0 and recent.sum()>=504
                    row.update(status='enabled' if permitted else 'flat_validation_gate',validation_sharpe=score,validation_error=error)
                    if permitted:
                        if manage=='exit':
                            recent_exit,details=fit_exit(data,recent,combos,cutoff);row.update({'recent_'+k:v for k,v in details.items()})
                        fresh=hold if manage=='hold' else data.subset(combos,recent_exit)[0]
                        labels=fresh[recent].copy();labels[~panel.loc[recent,'eligible'].to_numpy(bool)]=0
                        model,details=fit_entry(panel.loc[recent,columns],labels,data.positions[recent],finish[recent],codes)
                        dates=idx[test];own.loc[dates]=gate(predict_entry(model,panel.loc[dates,columns]),dates,panel,data.gates,data.available)
                        row.update(last_recent_outcome=str(completion[recent].max().date()),refit_shadow_sharpe=details['training_shadow_daily_sharpe'])
                # Disabled entry months still manage earlier live positions; being
                # flat for new entries is never itself an exit instruction.
                if manage=='exit' and active_previous and recent_exit is None and recent.sum()>=504:
                    recent_exit,details=fit_exit(data,recent,combos,cutoff)
                    row.update({'recent_'+k:v for k,v in details.items()})
                if manage=='exit':dump_model(dest/'exit_models'/f'{fitdate}.joblib',recent_exit)
                logs.append(row)
                write_json(checkpoint,dict(dates=[str(t) for t in idx],actions=own.tolist(),fits=logs,ranks=ranks))
            own.to_csv(dest/'actions.csv');pd.DataFrame(logs if logs else [dict(fit=None,status='no_test_dates')]).to_csv(dest/'fits.csv',index=False)
            pd.DataFrame(ranks if ranks else [dict(year=None,combo=ATM,selected=True)]).to_csv(dest/'shortlists.csv',index=False)
            actions[name]=own
            peers.append((name,universe,manage))
            print(f'  {d["label"]} {name}: {int(own.ne(0).sum())} entry recommendations',flush=True)
    return actions

def validation_schedule(data,actions,model,calendar):
    if model is None:return {}
    models={str(m.start_time.date()):model for m in pd.period_range(calendar[0],calendar[-1],freq='Q')}
    return data.schedule(actions,models)

def save_managed(dest,nav,trades,decisions,cfg,s,d):
    checks=audit(nav,trades,s,cfg,d['secids'])
    assert abs(nav.option_value.iloc[-1])+abs(nav.stock_value.iloc[-1])<1e-7
    if len(trades):
        assert set(trades.reason).issubset({'entry','learned_policy_exit','split_adjustment','expiry_settlement'})
        paid=trades.reason.isin(['entry','learned_policy_exit'])
        np.testing.assert_allclose(trades.loc[paid,'commission'],abs(trades.loc[paid,'quantity'])*cfg.option_commission)
        assert trades.loc[~paid,'commission'].eq(0).all()
        exits=trades.reason.eq('learned_policy_exit')
        if exits.any():assert (pd.to_datetime(trades.loc[exits,'exit_signal_date'])<pd.to_datetime(trades.loc[exits,'date'])).all()
    entered=decisions[decisions.status.eq('entered')] if len(decisions) else decisions
    if len(entered):assert (abs(entered.vega_a-entered.vega_b)/entered[['vega_a','vega_b']].max(axis=1)).le(cfg.max_entry_vega_imbalance+1e-12).all()
    nav.to_parquet(dest/'daily_nav.parquet');trades.to_csv(dest/'trades.csv',index=False);decisions.to_csv(dest/'decisions.csv',index=False)
    return dict(status='complete',pair=d['pair'],label=d['label'],entries=len(entered),
        exited_cohorts=int(nav.policy_exit_cohorts.sum()),commission=float(trades.commission.sum()) if len(trades) else 0.,
        option_cost=float(nav.option_execution_cost.sum()),hedge_cost=float(nav.hedge_cost.sum()),**scores(nav,cfg.capital),**checks)

def summary(defs,market,cfg):
    statuses={(d['pair'],m,p):json.loads((OUT/d['pair']/m/p/'status.json').read_text()) for d in defs for m in MODELS for p,_ in PERIODS}
    complete=[d for d in defs if all(statuses[d['pair'],m,p]['status']=='complete' for m in MODELS for p,_ in PERIODS)]
    unresolved=[dict(pair=d['pair'],model=m,period=p,error=statuses[d['pair'],m,p].get('error')) for d in defs for m in MODELS for p,_ in PERIODS if statuses[d['pair'],m,p]['status']!='complete']
    write_json(REPORT/'coverage.json',dict(requested_pairs=len(defs),common_complete_pairs=len(complete),unresolved=unresolved))
    rows=[];annual=[];pairs=[];mix=[]
    for m in MODELS:
        for p,start in PERIODS:
            cal=market.calendar[market.calendar>=start];cash=cash_book(cal,market.rates,cfg.capital)
            book=cash[['nav','cash_benchmark']]*len(complete);entries=exits=active=0
            for d in complete:
                dest=OUT/d['pair']/m/p;status=statuses[d['pair'],m,p]
                nav=cash if status.get('cash_only') else pd.read_parquet(dest/'daily_nav.parquet',columns=['nav','cash_benchmark'])
                assert nav.index.equals(cal);book+=nav[['nav','cash_benchmark']]-cash[['nav','cash_benchmark']]
                entries+=status['entries'];exits+=status.get('exited_cohorts',0);active+=int(status['entries']>0)
                pairs.append(dict(model=m,period=p,pair=d['pair'],label=d['label'],entries=status['entries'],net_excess_pnl=float(nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1])))
                if status['entries']:
                    dec=pd.read_csv(dest/'decisions.csv');a=read_actions(OUT/d['pair']/m/'actions.csv')
                    for r in dec[dec.status.eq('entered')].itertuples():
                        code=int(a.loc[pd.Timestamp(r.signal_date)]);combo,size=ACTIONS[code];t,cp,ba,bb=COMBOS[combo]
                        mix.append(dict(model=m,period=p,pair=d['pair'],signal_date=r.signal_date,tenor=t,cp=cp,bucket_a=ba,bucket_b=bb,size=size,optionid_a=r.optionid_a,optionid_b=r.optionid_b))
            book['return_']=book.nav.pct_change().fillna(0);book['excess_return']=book.return_-book.cash_benchmark.pct_change().fillna(0)
            book.to_csv(REPORT/f'{m}_{p}_portfolio.csv')
            rows.append(dict(model=m,period=p,pairs=len(complete),entries=entries,active_pairs=active,exited_cohorts=exits,net_excess_pnl=float(book.nav.iloc[-1]-book.cash_benchmark.iloc[-1]),**scores(book,cfg.capital*len(complete))))
            for year,g in book.groupby(book.index.year):annual.append(dict(model=m,period=p,year=year,return_percent=100*((1+g.return_).prod()-1),annualized_excess_sharpe=float(np.sqrt(252)*g.excess_return.mean()/g.excess_return.std()) if g.excess_return.std()>1e-12 else None))
    table=pd.DataFrame(rows);table.to_csv(REPORT/'comparison.csv',index=False)
    pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False);pd.DataFrame(pairs).to_csv(REPORT/'pairs.csv',index=False);pd.DataFrame(mix).to_csv(REPORT/'contract_mix.csv',index=False)
    lines=['# Expiry, smile and learned-exit comparison','',
        'Same 345 requested pairs, initial vega matching, integer quantities, daily aggregate delta hedge, entry risk limits and beta/correlation eligibility. Models refit quarterly with daily entry and exit decisions, using prior-year validation and completed-outcome refits. $0.65 training commission hurdle; $0.03 executed commission; midpoint plus 0.10 vol point per option leg on entry and early exit; 1 bp stock execution and original borrow/financing. All policies use 110 calendar days of validation/final-entry runoff. The saved 30D checkpoint used monthly refits and 60 days and remains unchanged, so it is contextual rather than the matched control.','',
        '| Universe | Hold full Sharpe | Learned-exit full Sharpe | Hold recent | Learned-exit recent |','|---|---:|---:|---:|---:|']
    def value(model,period):
        x=table[table.model.eq(model)&table.period.eq(period)].annualized_excess_sharpe.iloc[0]
        return 'cash' if pd.isna(x) else f'{x:.3f}'
    for u in UNIVERSES:lines.append(f'| {u} | {value(u+"_hold","full")} | {value(u+"_exit","full")} | {value(u+"_hold","recent")} | {value(u+"_exit","recent")} |')
    lines+=['','Full 2018–August 2025; recent 2022–August 2025 starts flat. Cash-excess daily Sharpe annualized by √252.','',
      'ATM30 is the matched baseline. Tenors chooses 14/30/90D ATM calls. Smile chooses independent lower-strike (~75-delta call), ATM or higher-strike (~25-delta call) buckets on each leg at 30D. Both combines those expiry and strike choices. All legs are calls; puts and cross-expiry pairs are outside this pass. A training-only annual screen selects at most four combinations per universe, retaining ATM30; it uses five years ending before the prior validation year. Dates and gates never depend on future execution or label resolution.','',
      'Entry states are the original 47 inputs plus nine quote-context values with availability flags: ATM IV spread, lower-minus-upper-strike IV skew difference and ATM gamma-per-vega gap at each expiry. All eight models see identical states. Entry partitions retain separate directional targets for each shortlisted combination, with the same eight-leaf Sharpe search and 25/50/75/100% sizing ladder.','',
      'The learned exit policy uses the actual fixed contracts: remaining DTE, live-leg flags, IV spread and its change from entry, current RV/IV variance value, gamma per vega, moneyness, relative vega drift, quote width, VIX, beta and correlation. It learns an eight-leaf hold/close policy by evaluating every binary leaf rule against daily marked and hedged shadow P&L, including early-exit costs and subsequent excess cash interest. Deterministic, reward-independent sampling uses at most 32 completed training cohorts per shortlisted combination per direction. Fully completed expiry paths are required before every training cutoff, even if a candidate exits earlier.','',
      'Proposal entry and exit policies train before the prior validation year. Enabled entries refit using completed outcomes from the last five years; exits also refit to manage older live positions when new entry is disabled. Training entry rewards reflect the learned exits. Memory gates continue to use strictly prior completed hold-to-expiry outcomes at the training hurdle for matched entry eligibility. The learned exit is independent of whether the new-entry policy is flat.','',
      'Inventory records original option IDs, strike, expiry, quantity and cohort. Different expiry/strike cohorts can overlap across days, with one new pair cohort permitted per signal day. Quantities stay fixed apart from verified splits; each leg naturally settles at its own expiry. Learned exits close every remaining leg of a cohort together on the next session. Hedge turnover is netted across all open cohorts. Initial vega matching ≤10% is enforced; daily option-vega rebalancing and early assignment remain unmodeled.','',
      'These are exploratory results on previously examined history. Unsupported paths are listed and excluded from every model on the common roster, never assigned cash P&L.','',
      '[Comparison](curve_management/comparison.csv) · [Annual](curve_management/annual.csv) · [Pairs](curve_management/pairs.csv) · [Inventory choices](curve_management/contract_mix.csv) · [Coverage](curve_management/coverage.json)','']
    REPORT.with_suffix('.md').write_text('\n'.join(lines));print(table[['model','period','pairs','entries','exited_cohorts','annualized_excess_sharpe']].to_string(index=False),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int);p.add_argument('--report-only',action='store_true');a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    defs=json.loads(Path('artifacts/contracts/atm65_ablation/inputs.json').read_text())['definitions']
    training=config(.65);actual=config(.03)
    sources=[Path(__file__),Path('scripts/audit_curve_management.py'),Path('artifacts/contracts/atm65_ablation/inputs.json'),Path('config/atm30_fee065_checkpoint.json'),Path('scripts/curve90/labels.py')]
    sources+=list(Path('scripts/curve_management').glob('*.py'))
    sources+=[Path('scripts/run_atm65_ablation.py'),Path('scripts/run_atm65_signal_mining.py'),Path('scripts/option_flex/contract_engine.py'),Path('scripts/option_flex/contract_risk.py')]
    sources+=[NN/d['pair']/'fee_065/states.parquet' for d in defs if (NN/d['pair']/'fee_065/states.parquet').exists()]
    sources+=[p for root in ['data/curve_trading','data/curve90'] for p in Path(root).glob('*manifest.json')]
    sources+=list(Path('data/curve90/metadata').rglob('*.json'))
    signature=dict(definitions=defs,training=asdict(training),actual=asdict(actual),models=MODELS,combos=COMBOS,
       runoff=RUNOFF,refit_months=REFIT_MONTHS,shortlist=SHORTLIST,exit_cohorts_per_combo=MAX_EXIT_COHORTS_PER_COMBO,sources={str(p):digest(p) for p in sources})
    if (OUT/'inputs.json').exists():assert json.loads((OUT/'inputs.json').read_text())==json.loads(json.dumps(signature)),'Inputs changed'
    else:write_json(OUT/'inputs.json',signature)
    cal=load_calendar(ROOT);cal=cal[(cal>='2010-01-01')&(cal<='2025-08-29')];market=Market(cal)
    started=time.monotonic();newly=0
    if not a.report_only:
        for number,d in enumerate(defs,1):
            folder=OUT/d['pair'];folder.mkdir(exist_ok=True)
            if all((folder/m/p/'status.json').exists() for m in MODELS for p,_ in PERIODS):continue
            source=NN/d['pair']/'fee_065';old=json.loads((CURVE/d['pair']/'atm_control/status.json').read_text())
            base=pd.read_parquet(source/'states.parquet') if (source/'states.parquet').exists() else None
            possible=False
            if base is not None and not base.empty:
                wanted=(base.index>='2018-01-01')&(base.index<=cal[-1]-pd.Timedelta(days=RUNOFF))&screen_dates(d,base.index)
                possible=(wanted&base.eligible.to_numpy(bool)).any()
            if old.get('structural_cash') or base is None or base.empty or not possible:
                for m in MODELS:
                    for p,_ in PERIODS:write_json(folder/m/p/'status.json',dict(status='complete',pair=d['pair'],label=d['label'],entries=0,cash_only=True,training_hurdle=.65,execution_commission=.03))
                continue
            t=time.monotonic();print(f'{number}/{len(defs)} {d["label"]}: maturity / smile / management',flush=True)
            try:q,s,selection,divs,inventory=load_pair(market,d['secids'])
            except ValueError as exc:
                for m in MODELS:
                    for p,_ in PERIODS:write_json(folder/m/p/'status.json',dict(status='incomplete',pair=d['pair'],error=str(exc),training_hurdle=.65,execution_commission=.03))
                continue
            panel,new_columns=entry_context(base,q,s,selection,d['secids']);panel.to_parquet(folder/'states.parquet');write_json(folder/'inventory_inputs.json',inventory)
            columns=features(base)+new_columns
            with tempfile.TemporaryDirectory(prefix='labels_',dir=folder) as temp:
                data=Paths(panel,q,s,selection,divs,cal,d['secids'],training,Path(temp)/'labels.dat')
                pd.DataFrame(data.status).to_csv(folder/'label_status.csv',index=False)
                np.savez_compressed(folder/'gates.npz',gates=data.gates,available=data.available,ends=data.ends,dates=panel.index.values)
                replay=Replay(q,s,selection,d['secids'],divs,training)
                actions=train(d,data,cal,replay,market.rates,folder,columns);replay.cfg=actual
                for m in MODELS:
                    models=load_models(folder/m) if m.endswith('_exit') else None
                    for p,start in PERIODS:
                        dest=folder/m/p;dest.mkdir(parents=True,exist_ok=True)
                        if (dest/'status.json').exists():continue
                        reused=folder/m/'reused_policy.json'
                        if reused.exists():
                            peer=json.loads(reused.read_text())['source'];source=folder/peer/p
                            if (source/'status.json').exists():
                                shutil.copytree(source,dest,dirs_exist_ok=True);continue
                        own=actions.loc[actions.index>=start,m].rename('code')
                        if not own.ne(0).any():result=dict(status='complete',pair=d['pair'],label=d['label'],entries=0,cash_only=True)
                        else:
                            schedule=data.schedule(own,models) if models is not None else None
                            if schedule is not None:write_json(dest/'exit_intents.json',{str(t.date()):sorted(v) for t,v in schedule.items()})
                            try:
                                nav,trades,decisions=replay.run(own,cal[cal>=start],schedule);mark_book(nav,market.rates,actual.capital)
                                result=save_managed(dest,nav,trades,decisions,actual,s,d) if models is not None else save_result(dest,nav,trades,decisions,actual,s,d)
                            except UnresolvedPosition as exc:result=dict(status='incomplete',pair=d['pair'],label=d['label'],error=str(exc))
                        result.update(training_hurdle=.65,execution_commission=.03);write_json(dest/'status.json',result)
                del replay,data,actions;gc.collect()
            newly+=1;elapsed=time.monotonic()-t
            write_json(OUT/'progress.json',dict(roster_position=number,pairs=len(defs),last_pair=d['label'],last_pair_seconds=elapsed,elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            print(f'{d["label"]}: {elapsed:.0f}s; peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30:.2f} GiB',flush=True)
            del q,s,selection,divs,panel,base;gc.collect()
            if a.limit and newly>=a.limit:return
    summary(defs,market,actual);write_json(OUT/'task_status.json',dict(status='complete',pairs=len(defs),elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

if __name__=='__main__':main()
