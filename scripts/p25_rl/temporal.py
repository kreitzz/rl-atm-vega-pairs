"""Matched unshaped myopic controls for sequential allocation/management Q.

Only gamma changes from the corresponding unshaped Q arm to zero. These are
scientific attribution controls, never candidates for strategy selection.
Original experiment modules and their frozen artifacts remain untouched.
"""
import os
for _key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS','ARROW_NUM_THREADS']:
    os.environ[_key]='1'

import argparse
import gc
import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd

from scripts.p25_rl import run as entry
from scripts.p25_rl import action_q,management
from scripts.p25_rl.provenance import stable_model_digest
from scripts.p25_rl.policy import Planner as EntryPlanner,transitions as entry_transitions
from scripts.p25_rl.ledger import simulate_with_observer
from scripts.p25_rl.managed_ledger import simulate_with_management_observer

base=entry.base
ROOT=entry.ROOT
OUT=ROOT/'artifacts/contracts/p25_temporal_q'
REPORT=ROOT/'reports/p25_temporal_q'
SPECS={
    'entry_myopic_capacity':dict(action_q.SPECS['action_q_capacity'],gamma=0.),
    'manage_myopic_capacity':dict(management.SPECS['manage_q'],gamma=0.),
}
KINDS={'entry_myopic_capacity':'entry','manage_myopic_capacity':'management'}
CANDIDATES=[]
FILES=['daily_nav.parquet','trades.csv','decisions.csv','scored_allocations.parquet',
       'transitions.parquet','daily_rewards.parquet','actions.parquet','management_log.csv']


class Budget(base.Budget):
    def check(self):
        super().check()
        if (OUT/'STOP').exists():raise InterruptedError('Temporal-control STOP requested')


def source_hashes():
    names=['temporal.py','provenance.py','action_q.py','management.py','critic.py','policy.py',
           'ledger.py','managed_ledger.py','run.py']
    return {str((ROOT/'scripts/p25_rl'/name).relative_to(ROOT)):
            base.digest(ROOT/'scripts/p25_rl'/name) for name in names}


def training_files(case):
    parent=entry if KINDS[case]=='entry' else management
    return {str((parent.OUT/'behavior'/f'{kind}_{seed}'/name).relative_to(ROOT)):
            base.digest(parent.OUT/'behavior'/f'{kind}_{seed}'/name)
            for kind,seed in parent.BEHAVIORS
            for name in ['transitions.parquet','daily_rewards.parquet','status.json']}


def prepare(budget):
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    source_protocols={str((parent.REPORT/'protocol.json').relative_to(ROOT)):
                      base.digest(parent.REPORT/'protocol.json')
                      for parent in [entry,action_q,management]}
    original=json.loads((entry.REPORT/'protocol.json').read_text())
    for name,expected in original['protected_hashes'].items():
        if base.digest(ROOT/name)!=expected:raise ValueError(f'Protected input changed:{name}')
        budget.rest()
    protocol=dict(question='Does Bellman continuation add value beyond the same unshaped immediate-reward critic?',
        controls=SPECS,kinds=KINDS,candidates=[],source_code=source_hashes(),
        source_protocols=source_protocols,protected_hashes=original['protected_hashes'],
        comparison={'entry_myopic_capacity':'action_q_capacity','manage_myopic_capacity':'manage_q'},
        algorithm='Reuse exact per-action critic fitting, features, behavior transitions and extraction. Change only gamma to0; penalty stays0. One immediate-reward fit. No new predictors or action choices.',
        train='Initial endpoints strictly before2020; final endpoints strictly before2022. Reuse original completed transitions and training-only scaling.',
        validation=original['validation'],evaluation=original['evaluation'],
        selection='Attribution controls only; ineligible for joint selection. Parent selection frozen on original nine Q candidates before later replay.',
        reason='Unshaped sequential Q requires a matched unshaped myopic control; existing penalty.5 myopic arms change two factors and cannot isolate continuation.',
        limitations=original['limitations'],resources=original['resources'])
    path=REPORT/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=json.loads(json.dumps(protocol)):
        raise ValueError('Frozen temporal-control protocol changed')
    base.write(path,protocol)
    return protocol


def data(case):
    if KINDS[case]=='management':return management.behavior_data()
    # Verify the reused actual source provenance even in a standalone phase.
    for kind,seed in entry.BEHAVIORS:
        status=json.loads((entry.OUT/'behavior'/f'{kind}_{seed}'/'status.json').read_text())
        if status['status']!='passed' or status['protocol_sha256']!=base.digest(entry.REPORT/'protocol.json'):
            raise ValueError('Entry behavior provenance changed')
        for name,expected in status['source_code'].items():
            if base.digest(ROOT/name)!=expected:raise ValueError(f'Entry behavior source changed:{name}')
    return entry.behavior_data()


def fit(case,cutoff,records,daily,budget):
    folder=OUT/'models'/cutoff;folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{case}.joblib'
    signature=dict(protocol_sha256=base.digest(REPORT/'protocol.json'),sources=source_hashes(),
        training_files=training_files(case),case=case,cutoff=cutoff,spec=SPECS[case])
    if path.exists():
        model=joblib.load(path)
        if model.audit['cache_signature']!=signature:raise ValueError(f'Temporal model provenance changed:{case}')
        return model
    parent=action_q if KINDS[case]=='entry' else management
    model=parent.fit_policy(records,daily,cutoff,SPECS[case],budget.rest)
    model.audit['cache_signature']=signature
    model.audit['matched_control_for']={'entry_myopic_capacity':'action_q_capacity',
                                       'manage_myopic_capacity':'manage_q'}[case]
    joblib.dump(model,path);base.write(folder/f'{case}.json',model.audit)
    print(f'Fitted matched control{case}@{cutoff}',flush=True)
    return model


def replay_signature(signal,cal,kind,model,active_from,seed=2026):
    return dict(protocol_sha256=base.digest(REPORT/'protocol.json'),sources=source_hashes(),
        signals_sha256=hashlib.sha256(pd.util.hash_pandas_object(signal,index=True).to_numpy().tobytes()).hexdigest(),
        calendar_sha256=hashlib.sha256(pd.DatetimeIndex(cal).asi8.tobytes()).hexdigest(),
        kind=kind,model_hash=stable_model_digest(model),active_from=active_from,seed=seed)


def verify_cache(folder,signature=None):
    status=json.loads((folder/'status.json').read_text())
    if status['status']!='passed' or status['protocol_sha256']!=base.digest(REPORT/'protocol.json'):
        raise ValueError(f'Unverified temporal replay:{folder.name}')
    if status['source_code']!=source_hashes() or (signature is not None and status['cache_signature']!=signature):
        raise ValueError(f'Temporal replay provenance changed:{folder.name}')
    for name,expected in status['output_hashes'].items():
        if base.digest(folder/name)!=expected:raise ValueError(f'Temporal output changed:{name}')
    return status


def replay(case,signal,cal,cfg,events,budget,model,active_from,folder):
    folder.mkdir(parents=True,exist_ok=True);kind=KINDS[case]
    signature=replay_signature(signal,cal,kind,model,active_from)
    if (folder/'status.json').exists():verify_cache(folder,signature);return
    started=time.monotonic();market=entry.Market(budget,signal,entry.prior.QUOTES)
    if kind=='entry':
        planner=EntryPlanner(signal,market.rates,cfg,'learned',model,active_from)
    else:
        rv=management.variance_states(market);budget.rest()
        planner=management.Planner(market.rates,cfg,rv,'learned',model,active_from)
    def observe(*args):
        planner.observe(*args);budget.rest()
        if len(planner.daily)%252==0:print(f'  Temporal{case}: completed{args[0].date()}',flush=True)
    print(f'Replay matchedcontrol{case}:{len(cal)}sessions',flush=True)
    kwargs=dict(corporate_events=events,wing_capital=1e6,session_observer=observe)
    if kind=='entry':
        nav,trades,decisions,ledger=simulate_with_observer(
            market,signal,cal,cfg,1e6,policy=planner.entry,**kwargs)
        records,daily=entry_transitions(planner)
    else:
        nav,trades,decisions,ledger=simulate_with_management_observer(
            market,signal,cal,cfg,1e6,policy=lambda row,context:4,
            management_policy=planner.manage,**kwargs)
        records,daily=management.transitions(planner)
    base.mark_book(nav,market.rates,cfg.capital)
    audit,trades=entry.reconcile(nav,trades,decisions,signal)
    paid=trades.reason.isin(['entry','announced_corporate_action_exit','learned_policy_exit','learned_policy_trim'])
    np.testing.assert_allclose(trades.loc[paid,'commission'],
        cfg.option_commission*trades.loc[paid,'quantity'].abs(),atol=1e-10,rtol=0)
    np.testing.assert_allclose(daily.set_index('date').bps/10000,nav.excess_return,atol=1e-12,rtol=1e-8)
    start=records.start.iloc[0]
    np.testing.assert_allclose(records.reward.sum(),daily.loc[daily.date.gt(start),'bps'].sum(),atol=1e-7)
    log=pd.DataFrame(ledger.pop('management_log',[]))
    if log.empty:log=pd.DataFrame(columns=['cohort','signal_date','date','target','status'])
    nav.to_parquet(folder/'daily_nav.parquet');trades.to_csv(folder/'trades.csv',index=False)
    decisions.to_csv(folder/'decisions.csv',index=False)
    if kind=='entry':
        scored=signal.drop(columns='learned_level').merge(
            decisions[['signal_date','pair','level']].rename(columns={'signal_date':'date','level':'learned_level'}),
            on=['date','pair'],validate='one_to_one',sort=False)
    else:
        scored=signal.copy();scored['learned_level']=4
    scored.to_parquet(folder/'scored_allocations.parquet',index=False)
    records.to_parquet(folder/'transitions.parquet',index=False)
    daily.to_parquet(folder/'daily_rewards.parquet',index=False)
    pd.DataFrame(planner.events).to_parquet(folder/'actions.parquet',index=False)
    log.to_csv(folder/'management_log.csv',index=False)
    base.write(folder/'status.json',dict(status='passed',**audit,**ledger,kind=kind,
        protocol_sha256=base.digest(REPORT/'protocol.json'),source_code=source_hashes(),
        cache_signature=signature,output_hashes={name:base.digest(folder/name) for name in FILES},
        elapsed_seconds=time.monotonic()-started,peak_rss_mib=budget.peak,
        elective_commission_verified=True,
        trimmed_cohorts=int(trades.loc[trades.reason.eq('learned_policy_trim'),'cohort'].nunique()),
        closed_cohorts=int(trades.loc[trades.reason.eq('learned_policy_exit'),'cohort'].nunique())))
    print(f'Finished matchedcontrol{case}:{audit["entries"]}entries',flush=True)
    market.db.close();del market,planner,nav,trades,decisions,records,daily;gc.collect();budget.rest()


def metric(folder,start,stop):
    verify_cache(folder)
    nav=pd.read_parquet(folder/'daily_nav.parquet',use_threads=False)
    decisions=pd.read_csv(folder/'decisions.csv',parse_dates=['signal_date','entry_date'])
    return entry.period_metrics(nav,decisions,start,stop)


def validation(signal,cal,cfg,events,budget):
    selected=signal.loc[signal.date.le('2021-10-31')].copy();history=cal[cal<='2021-12-31']
    scores=[pd.read_csv(entry.REPORT/'validation.csv').loc[lambda t:t.case.eq('fixed100')].iloc[0].to_dict()]
    for case in SPECS:
        records,daily=data(case);model=fit(case,'2020-01-01',records,daily,budget)
        folder=OUT/'validation'/case
        replay(case,selected,history,cfg,events,budget,model,'2020-01-01',folder)
        years=[metric(folder,f'{year}-01-01',f'{year}-12-31')['profit_above_cash'] for year in [2020,2021]]
        scores.append(dict(case=case,**metric(folder,'2020-01-01','2021-12-31'),
                           profit_2020=years[0],profit_2021=years[1]))
        del records,daily,model;gc.collect();budget.rest()
    table=pd.DataFrame(scores);table.to_csv(REPORT/'validation.csv',index=False)
    print(table[['case','annualized_excess_sharpe','entries']].to_string(index=False),flush=True)


def evaluate(signal,cal,cfg,events,budget):
    selection=ROOT/'reports/p25_rl_research/selection.json'
    if not selection.exists():raise ValueError('Joint earlier selection must be frozen first')
    old=pd.read_parquet(entry.prior.OUT/'replays/fixed100/daily_nav.parquet',use_threads=False)
    scores=[]
    for case in SPECS:
        records,daily=data(case);model=fit(case,'2022-01-01',records,daily,budget)
        folder=OUT/'replays'/case
        replay(case,signal,cal,cfg,events,budget,model,'2022-01-01',folder)
        nav=pd.read_parquet(folder/'daily_nav.parquet',use_threads=False)
        pd.testing.assert_frame_equal(nav.loc[:'2021-12-31'],old.loc[:'2021-12-31'],
                                      check_exact=False,atol=1e-7,rtol=1e-11)
        scores.append(dict(case=case,period='evaluation',**metric(folder,'2022-01-01','2025-08-29')))
        del records,daily,model,nav;gc.collect();budget.rest()
    pd.DataFrame(scores).to_csv(REPORT/'periods.csv',index=False)
    base.write(REPORT/'handoff_verification.json',dict(status='passed',
        all_control_prior_nav_matches_baseline=True,inventory_continuous=True))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['prepare','validation','evaluation','all'],default='all')
    args=parser.parse_args();budget=Budget();protocol=prepare(budget)
    if args.phase=='prepare':return
    signal,cal,cfg,events=entry.inputs()
    for phase,fn in [('validation',validation),('evaluation',evaluate)]:
        if args.phase in [phase,'all']:fn(signal,cal,cfg,events,budget)
    for name,expected in protocol['protected_hashes'].items():
        if base.digest(ROOT/name)!=expected:raise ValueError(f'Protected input changed:{name}')
        budget.rest()
    base.write(OUT/'progress.json',dict(status='passed',phase=args.phase,peak_rss_mib=budget.peak))


if __name__=='__main__':main()
