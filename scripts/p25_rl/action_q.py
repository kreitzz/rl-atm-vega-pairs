"""Second allocation iteration: distinct Q regressors for each supported action.

Reuse frozen pre-2022 behavior ledgers, with no new market observations. Select
on 2020-2021 before evaluating the final pre-2022 refit on later history.
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
from scripts.p25_rl.policy import FEATURES,QPolicy,Planner,action_inputs,transitions
from scripts.p25_rl.critic import ActionForest
from scripts.p25_rl.provenance import stable_model_digest
from scripts.p25_rl.ledger import simulate_with_observer

base=entry.base
ROOT=entry.ROOT
OUT=ROOT/'artifacts/contracts/p25_action_q'
REPORT=ROOT/'reports/p25_action_q'
SPECS={
    'action_q_capacity':dict(gamma=.99,penalty=0.,actions=[0,1,2],inventory=True,conservative=False),
    'action_q_risk':dict(gamma=.99,penalty=.5,actions=[0,1,2],inventory=True,conservative=False),
    'action_q_conservative':dict(gamma=.99,penalty=.5,actions=[0,1,2],inventory=True,conservative=True),
    'action_myopic_risk':dict(gamma=0.,penalty=.5,actions=[0,1,2],inventory=True,conservative=False),
}
CANDIDATES=list(SPECS)[:3]
FILES=['daily_nav.parquet','trades.csv','decisions.csv','scored_allocations.parquet',
       'transitions.parquet','daily_rewards.parquet','actions.parquet']


class Budget(entry.Budget):
    def check(self):
        super().check()
        if (OUT/'STOP').exists():raise InterruptedError('Separate-action Q STOP requested')


def source_hashes():
    return {str(p.relative_to(ROOT)):base.digest(p) for p in
            [ROOT/'scripts/p25_rl'/f for f in ['action_q.py','critic.py','policy.py','ledger.py','run.py','provenance.py']]}


def training_files():
    return {str(p.relative_to(ROOT)):base.digest(p) for p in
            [entry.OUT/'behavior'/f'{kind}_{seed}'/f for kind,seed in entry.BEHAVIORS
             for f in ['transitions.parquet','daily_rewards.parquet','status.json']]}


def prepare(budget):
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    original=json.loads((entry.REPORT/'protocol.json').read_text())
    protected=original['protected_hashes'].copy()
    for name,digest in protected.items():
        if base.digest(ROOT/name)!=digest:raise ValueError(f'Protected source changed:{name}')
        budget.rest()
    # Verify reused simulations against their actual frozen source provenance.
    for kind,seed in entry.BEHAVIORS:
        p=entry.OUT/'behavior'/f'{kind}_{seed}'/'status.json'
        status=json.loads(p.read_text())
        if status['status']!='passed':raise ValueError('Unverified behavior ledger')
        for name,digest in status['source_code'].items():
            if base.digest(ROOT/name)!=digest:raise ValueError('Behavior source changed')
        if status['protocol_sha256']!=base.digest(entry.REPORT/'protocol.json'):
            raise ValueError('Behavior protocol changed')
        budget.rest()
    protocol=dict(question='Can action-specific critics learn allocations missed by the shared shallow critic?',
        algorithm='Fitted Q iteration with one 24-tree depth3 minleaf30 ExtraTrees forest per observed action; 24 Bellman fits, gamma0 one fit. Equal decision-date weight across shared-history trajectories. Ordinary max-Q backup. Conservative extraction subtracts one ensemble SD of advantage over full, not a confidence interval.',
        reason='Earlier first-iteration diagnostics found the shared critic often ignored its one-hot action columns. Action-specific critics force representation of conditional action values; no later new-Q returns were inspected before this iteration.',
        experiments=SPECS,features=FEATURES,actions={0:'skip',1:'half',2:'full'},
        train=original['train'],validation=original['validation'],evaluation=original['evaluation'],
        reward=original['reward'],timing=original['timing'],limitations=original['limitations'],
        selection='Three Q arms; >=20validation entries, both2020/2021 excess profits>=0, validation Sharpe>fixed100+.01. Myopic ablation ineligible. Authoritative combined selection across all declared families before later evaluation.',
        resources=original['resources'],source_code=source_hashes(),
        reused_training_files=training_files(),protected_hashes=protected)
    path=REPORT/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=json.loads(json.dumps(protocol)):
        raise ValueError('Frozen action-Q protocol changed')
    base.write(path,protocol)
    return protocol


def fit_policy(records,daily,cutoff,spec,rest=lambda:None):
    cutoff=pd.Timestamp(cutoff)
    use=records.loc[records.end.lt(cutoff)&records.start.lt(cutoff)&records.action.isin(spec['actions'])].copy()
    if len(use)<100 or use.start.nunique()<60:raise ValueError('Insufficient historical dates')
    columns=list(range(len(FEATURES)))
    states=np.stack(use.state.map(np.asarray));next_states=np.stack(use.next_state.map(np.asarray))
    x=action_inputs(states,use.action.to_numpy())
    scale=max(float(daily.loc[daily.date.lt(cutoff),'bps'].std(ddof=1)),.01)
    reward=use.reward.to_numpy()-spec['penalty']*use.squared_reward.to_numpy()/(2*scale)
    terminal=use.terminal.to_numpy(bool);discounts=np.power(spec['gamma'],use.sessions.to_numpy())
    weights=1/use.groupby('start').start.transform('size').to_numpy();weights/=weights.mean()
    nxt=action_inputs(np.repeat(next_states,len(spec['actions']),axis=0),np.tile(spec['actions'],len(use)))
    y=reward.copy();trace=[]
    for _ in range(24 if spec['gamma'] else 1):
        model=ActionForest(n_actions=4,n_estimators=24,max_depth=3,min_samples_leaf=30,random_state=2026,n_jobs=1)
        model.fit(x,y,sample_weight=weights)
        continuation=model.predict(nxt).reshape(len(use),len(spec['actions'])).max(axis=1)
        continuation[terminal]=0
        target=reward+discounts*continuation
        if not np.isfinite(target).all():raise ValueError('Nonfinite Bellman targets')
        trace.append(float(np.mean(np.abs(target-y))));y=target;rest()
    support={str(action):int((use.action==action).sum()) for action in spec['actions']}
    support_dates={str(action):int(use.loc[use.action.eq(action),'start'].nunique()) for action in spec['actions']}
    audit=dict(cutoff=str(cutoff.date()),last_transition_end=str(use.end.max().date()),
        transitions=len(use),distinct_market_dates=int(use.start.nunique()),
        action_support=support,action_distinct_dates=support_dates,daily_reward_scale_bps=scale,iterations=len(trace),
        bellman_target_change=trace,terminal_transitions=int(terminal.sum()),spec=spec,
        critic='Independent forests per action',
        caveat='Behavior paths share history; state-action overlap is imperfect. Ensemble dispersion is not a statistical confidence interval.')
    return QPolicy(model,columns,spec['actions'],spec['conservative'],audit)


def fit(name,cutoff,records,daily,budget):
    folder=OUT/'models'/cutoff;folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{name}.joblib'
    signature=dict(case=name,cutoff=str(pd.Timestamp(cutoff).date()),spec=SPECS[name],
                   protocol_sha256=base.digest(REPORT/'protocol.json'),
                   sources=source_hashes(),training_files=training_files())
    if path.exists():
        model=joblib.load(path)
        if model.audit['cache_signature']!=signature:raise ValueError('Model provenance changed')
        if (model.audit['cutoff']!=cutoff or model.audit['spec']!=SPECS[name]
            or pd.Timestamp(model.audit['last_transition_end'])>=pd.Timestamp(cutoff)):
            raise ValueError('Cached model training boundary changed')
        return model
    model=fit_policy(records,daily,cutoff,SPECS[name],budget.rest)
    model.audit['cache_signature']=signature
    joblib.dump(model,path);base.write(folder/f'{name}.json',model.audit)
    print(f'Fitted {name}@{cutoff}: {model.audit["action_support"]}',flush=True)
    return model


def replay_signature(signal,cal,kind,model,active_from,seed=2026):
    return dict(protocol_sha256=base.digest(REPORT/'protocol.json'),sources=source_hashes(),
        signals_sha256=hashlib.sha256(pd.util.hash_pandas_object(signal,index=True).to_numpy().tobytes()).hexdigest(),
        calendar_sha256=hashlib.sha256(pd.DatetimeIndex(cal).asi8.tobytes()).hexdigest(),
        kind=kind,model_hash=stable_model_digest(model),active_from=active_from,seed=seed)


def replay(name,signal,cal,cfg,events,budget,model,active_from,folder):
    folder.mkdir(parents=True,exist_ok=True)
    signature=replay_signature(signal,cal,'learned',model,active_from)
    if (folder/'status.json').exists():
        status=json.loads((folder/'status.json').read_text())
        if status['status']!='passed' or status['cache_signature']!=signature:
            raise ValueError(f'Replay provenance changed:{name}')
        for f,digest in status['output_hashes'].items():
            if base.digest(folder/f)!=digest:raise ValueError(f'Replay output changed:{name}/{f}')
        return
    started=time.monotonic();market=entry.Market(budget,signal,entry.prior.QUOTES)
    planner=Planner(signal,market.rates,cfg,'learned',model,active_from)
    def observe(*args):planner.observe(*args);budget.rest()
    print(f'Replay {name}: {len(cal)} sessions',flush=True)
    nav,trades,decisions,ledger=simulate_with_observer(market,signal,cal,cfg,1e6,
        policy=planner.entry,corporate_events=events,wing_capital=1e6,session_observer=observe)
    base.mark_book(nav,market.rates,cfg.capital)
    audit,trades=entry.reconcile(nav,trades,decisions,signal)
    records,daily=transitions(planner)
    np.testing.assert_allclose(daily.set_index('date').bps/10000,nav.excess_return,atol=1e-12,rtol=1e-8)
    if len(records):np.testing.assert_allclose(records.reward.sum(),daily.loc[daily.date.gt(records.start.iloc[0]),'bps'].sum(),atol=1e-7)
    nav.to_parquet(folder/'daily_nav.parquet');trades.to_csv(folder/'trades.csv',index=False)
    decisions.to_csv(folder/'decisions.csv',index=False)
    scored=signal.drop(columns='learned_level').merge(decisions[['signal_date','pair','level']].rename(
        columns={'signal_date':'date','level':'learned_level'}),on=['date','pair'],validate='one_to_one',sort=False)
    scored.to_parquet(folder/'scored_allocations.parquet',index=False)
    records.to_parquet(folder/'transitions.parquet',index=False);daily.to_parquet(folder/'daily_rewards.parquet',index=False)
    pd.DataFrame(planner.events).to_parquet(folder/'actions.parquet',index=False)
    base.write(folder/'status.json',dict(status='passed',**audit,**ledger,
        elapsed_seconds=time.monotonic()-started,peak_rss_mib=budget.peak,kind='learned',
        protocol_sha256=base.digest(REPORT/'protocol.json'),cache_signature=signature,
        source_code=source_hashes(),output_hashes={f:base.digest(folder/f) for f in FILES}))
    print(f'Finished {name}: {audit["entries"]} entries,{time.monotonic()-started:.1f}s',flush=True)
    market.db.close();del market,planner,nav,trades,decisions,records;gc.collect();budget.rest()


def validation(signal,cal,cfg,events,budget):
    records,daily=entry.behavior_data()
    selected=signal.loc[signal.date.le('2021-10-31')].copy();history=cal[cal<='2021-12-31']
    for name in SPECS:
        model=fit(name,'2020-01-01',records,daily,budget)
        replay(name,selected,history,cfg,events,budget,model,'2020-01-01',OUT/'validation'/name)
    scores=[pd.read_csv(entry.REPORT/'validation.csv').loc[lambda t:t.case.eq('fixed100')].iloc[0].to_dict()]
    for name in SPECS:
        folder=OUT/'validation'/name;nav=pd.read_parquet(folder/'daily_nav.parquet',use_threads=False)
        decisions=pd.read_csv(folder/'decisions.csv',parse_dates=['signal_date','entry_date'])
        metric=entry.period_metrics(nav,decisions,'2020-01-01','2021-12-31')
        years=[entry.period_metrics(nav,decisions,f'{y}-01-01',f'{y}-12-31')['profit_above_cash'] for y in [2020,2021]]
        scores.append(dict(case=name,**metric,profit_2020=years[0],profit_2021=years[1]))
    table=pd.DataFrame(scores);table.to_csv(REPORT/'validation.csv',index=False)
    baseline=float(table.loc[table.case.eq('fixed100'),'annualized_excess_sharpe'].iloc[0])
    eligible=table.loc[table.case.isin(CANDIDATES)&table.entries.ge(20)&table.profit_2020.ge(0)&
        table.profit_2021.ge(0)&table.annualized_excess_sharpe.gt(baseline+.01)]
    chosen='fixed100' if eligible.empty else eligible.sort_values(['annualized_excess_sharpe','case'],ascending=[False,True]).case.iloc[0]
    base.write(REPORT/'selection.json',dict(selected=chosen,validation_only=True,protocol_sha256=base.digest(REPORT/'protocol.json')))
    print(table[['case','annualized_excess_sharpe','entries','profit_2020','profit_2021']].to_string(index=False),flush=True)


def evaluate(signal,cal,cfg,events,budget):
    if not (ROOT/'reports/p25_rl_research/selection.json').exists():
        raise ValueError('Joint earlier selection must be frozen before later evaluation')
    from scripts.p25_rl.summary import check_selection
    check_selection(json.loads((ROOT/'reports/p25_rl_research/selection.json').read_text()),budget)
    records,daily=entry.behavior_data()
    old=pd.read_parquet(entry.prior.OUT/'replays/fixed100/daily_nav.parquet',use_threads=False)
    periods=[]
    for name in SPECS:
        model=fit(name,'2022-01-01',records,daily,budget)
        folder=OUT/'replays'/name;replay(name,signal,cal,cfg,events,budget,model,'2022-01-01',folder)
        nav=pd.read_parquet(folder/'daily_nav.parquet',use_threads=False)
        pd.testing.assert_frame_equal(nav.loc[:'2021-12-31'],old.loc[:'2021-12-31'],check_exact=False,atol=1e-7,rtol=1e-11)
        decisions=pd.read_csv(folder/'decisions.csv',parse_dates=['signal_date','entry_date'])
        periods.append(dict(case=name,period='evaluation',**entry.period_metrics(nav,decisions,'2022-01-01','2025-08-29')))
    pd.DataFrame(periods).to_csv(REPORT/'periods.csv',index=False)
    base.write(REPORT/'handoff_verification.json',dict(status='passed',inventory_continuous=True,all_q_prior_nav_matches_baseline=True))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['prepare','validation','evaluation','all'],default='all')
    args=parser.parse_args();budget=Budget();protocol=prepare(budget)
    if args.phase=='prepare':return
    signal,cal,cfg,events=entry.inputs()
    for phase,fn in [('validation',validation),('evaluation',evaluate)]:
        if args.phase in [phase,'all']:fn(signal,cal,cfg,events,budget)
    for name,digest in protocol['protected_hashes'].items():
        if base.digest(ROOT/name)!=digest:raise ValueError('Protected source changed')
        budget.rest()
    base.write(OUT/'progress.json',dict(phase=args.phase,status='passed',peak_rss_mib=budget.peak))


if __name__=='__main__':main()
