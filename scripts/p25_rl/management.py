"""Daily hold/trim/close fitted Q on the original funded P25 portfolio.

Entry proposals remain full sized. Management decisions use completed-close
holdings and fill on the next session through the unchanged financial engine.
Run separately from the entry-allocation experiment: one numeric worker only.
"""
import os
for _key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS','ARROW_NUM_THREADS']:
    os.environ[_key]='1'

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd

from scripts.p25_rl import run as entry
from scripts.p25_rl.critic import ActionForest
from scripts.p25_rl.provenance import stable_model_digest
from scripts.p25_rl.managed_ledger import (simulate_with_management_observer,
                                         instrumentation_info)
from scripts.option_flex.contract_engine import execution_terms

base=entry.base
ROOT=entry.ROOT
OUT=ROOT/'artifacts/contracts/p25_inventory_q_management'
REPORT=ROOT/'reports/p25_inventory_q_management'
FEATURES=['carry20','carry60','rv_available','iv_displacement','remaining_dte',
          'release_7d','stress','reserve','pair_stress','name_concentration',
          'liquidation_cost','drawdown','gross_vega','cohorts','retained_fraction']
COHORT_COLUMN=FEATURES.index('cohorts')
ACTION_NAMES={0:'hold',1:'retain_half',2:'close_book'}
SPECS={
    'manage_q':dict(gamma=.99,penalty=0.),
    'manage_q_risk':dict(gamma=.99,penalty=.5),
    'manage_myopic_risk':dict(gamma=0.,penalty=.5),
}
CANDIDATES=['manage_q','manage_q_risk']
BEHAVIORS=[('hold',2026),('half_weekly',2026),('negative_carry',2026),('random_weekly',2026)]
REPLAY_FILES=['daily_nav.parquet','trades.csv','decisions.csv',
              'scored_allocations.parquet','transitions.parquet',
              'daily_rewards.parquet','actions.parquet','management_log.csv']


class Budget(base.Budget):
    def check(self):
        super().check()
        if (OUT/'STOP').exists():
            raise InterruptedError('Management Q STOP requested')


def variance_states(market):
    """Causal trailing variance; current completed close is available to Q.

    Adjustment factors are used only in their own session's price. No backward
    fill or future endpoint determines whether a return is observed.
    """
    adjusted=(market.spots*market.adjustments).where(
        market.spots.gt(0)&market.adjustments.gt(0))
    returns=np.log(adjusted/adjusted.shift(1))
    return {half:252*returns.pow(2).ewm(halflife=half,min_periods=63,
                                      adjust=False).mean() for half in [20,60]}


def held_state(date,day,positions,pairs,global_risk,rv,cfg,drawdown):
    """Fifteen known-close portfolio summaries; no future payoff labels."""
    if not positions:
        state=np.zeros(len(FEATURES))
        state[FEATURES.index('drawdown')]=drawdown
        return state
    equity=float(day['nav']);weights=[];names={};maturities=[]
    carry={20:0.,60:0.};available=displacement=retained=cost=0.
    for p in positions:
        q=p['quote'];quantity=float(p['quantity']);sid=int(p['secid'])
        iv=float(q['bs_iv']);v=abs(quantity)*100*float(q['vega'])
        weights.append(v);names[sid]=names.get(sid,0.)+v
        maturities.append(max(0,(pd.Timestamp(p['expiry'])-date).days))
        displacement+=np.sign(quantity)*v*(iv-float(p['entry_iv']))
        retained+=v*abs(quantity)/abs(float(p['initial_quantity']))
        known=[]
        for half in [20,60]:
            variance=float(rv[half].at[date,sid]) if date in rv[half].index and sid in rv[half].columns else np.nan
            valid=np.isfinite(variance) and variance>=0 and iv>0
            known.append(valid)
            if valid:
                # Observable variance-carry proxy per current vega, not a
                # forecast label or an attribution of total option P&L.
                carry[half]+=np.sign(quantity)*v*(variance-iv*iv)/(2*iv)
        if all(known):available+=v
        _,_,slippage=execution_terms(q,-quantity,cfg,p.get('last_vega'))
        cost+=slippage+abs(quantity)*cfg.option_commission
    total=sum(weights);denominator=max(total,1e-12)
    pair_equity=1e6*equity/cfg.capital
    state=np.array([
        carry[20]/denominator,carry[60]/denominator,available/denominator,
        displacement/denominator,
        np.average(maturities,weights=weights)/55 if total else 0.,
        sum(v for v,dte in zip(weights,maturities) if dte<=7)/denominator,
        global_risk.get('stress_loss',0.)/(cfg.max_stress_loss_fraction*equity),
        global_risk.get('capital_reserve',0.)/(cfg.max_reserve_fraction*equity),
        max([p.get('stress_loss',0.)/(cfg.max_stress_loss_fraction*pair_equity)
             for p in pairs.values()],default=0.),
        max(names.values(),default=0.)/denominator,cost/equity*10000,
        drawdown,global_risk.get('gross_vega',0.)/cfg.max_gross_vega,
        len({p['cohort'] for p in positions}),retained/denominator])
    if not np.isfinite(state).all():
        raise ValueError('Nonfinite known-close management state')
    return state


class Planner:
    """One daily global reduction decision; flat book has only legal hold."""
    def __init__(self,rates,cfg,rv,kind='hold',model=None,
                 active_from='2018-01-01',seed=2026):
        self.rates=rates;self.cfg=cfg;self.rv=rv;self.kind=kind;self.model=model
        self.active_from=pd.Timestamp(active_from)
        self.rng=np.random.default_rng(seed);self.acted_weeks=set()
        self.daily=[];self.events=[];self.actions={}
        self.previous=None;self.benchmark=cfg.capital;self.peak=1.

    def observe(self,date,day,positions,pairs,global_risk):
        date=pd.Timestamp(date)
        if self.previous is not None and date<=self.previous:
            raise ValueError('Management closes must be strictly chronological')
        elapsed=0. if self.previous is None else (date-self.previous).days/365
        cash_return=0. if self.previous is None else float(self.rates.loc[self.previous])*elapsed
        bps=(float(day['return_'])-cash_return)*10000
        if abs(bps)<1e-8:bps=0.
        self.benchmark*=1+cash_return
        relative=float(day['nav'])/self.benchmark;self.peak=max(self.peak,relative)
        state=held_state(date,day,positions,pairs,global_risk,self.rv,self.cfg,
                         max(0.,1-relative/self.peak))
        action=0
        if positions and date>=self.active_from:
            if self.model is not None:
                action=int(self.model.choose(state))
            elif self.kind=='negative_carry':
                action=2 if state[0]<0 and state[2]>.99 else 0
            elif self.kind in ['half_weekly','random_weekly']:
                week=str(date.to_period('W'))
                if week not in self.acted_weeks:
                    self.acted_weeks.add(week)
                    action=1 if self.kind=='half_weekly' else int(self.rng.choice(3,p=[.6,.2,.2]))
            elif self.kind!='hold':raise ValueError(self.kind)
        if action not in ACTION_NAMES:
            raise ValueError('Illegal management action')
        self.actions[date]=action
        self.daily.append(dict(date=date,bps=bps));self.previous=date
        self.events.append(dict(date=date,action=action,state=state.tolist(),
                                held_cohorts=int(state[COHORT_COLUMN]),
                                eligible_decision=bool(positions and date>=self.active_from)))

    def manage(self,date,held):
        # Only the observer's preceding completed-close decision is read.
        # Returning1 never re-expands an already reduced original cohort.
        return {0:1.,1:.5,2:0.}[self.actions[pd.Timestamp(date)]]


def transitions(planner):
    """Every daily close, including flat continuation; final runoff alone ends."""
    records=[];daily=pd.DataFrame(planner.daily)
    events=planner.events
    if events and events[-1]['held_cohorts']:
        raise ValueError('Management terminal must be actual flat runoff')
    for j,(current,nxt) in enumerate(zip(events[:-1],events[1:])):
        reward=float(daily.iloc[j+1].bps)
        records.append(dict(start=current['date'],end=nxt['date'],
            action=current['action'],state=current['state'],next_state=nxt['state'],
            held=current['held_cohorts']>0,next_held=nxt['held_cohorts']>0,
            reward=reward,squared_reward=reward*reward,sessions=1,
            terminal=j+1==len(events)-1))
    return pd.DataFrame(records),daily


def action_inputs(states,actions):
    return np.column_stack([states,np.eye(3)[np.asarray(actions,int)]])


@dataclass
class QPolicy:
    model:object
    audit:dict

    def values(self,states):
        states=np.atleast_2d(states)
        x=action_inputs(np.repeat(states,3,axis=0),np.tile(np.arange(3),len(states)))
        return self.model.predict(x).reshape(len(states),3)

    def choose(self,state):
        state=np.asarray(state)
        if state[COHORT_COLUMN]<=0:return 0
        values=self.values(state)[0];best=int(np.argmax(values))
        return best if values[best]>values[0]+1e-8 else 0


def fit_policy(records,daily,cutoff,spec,rest=lambda:None):
    cutoff=pd.Timestamp(cutoff)
    use=records.loc[records.start.lt(cutoff)&records.end.lt(cutoff)].copy()
    if use.start.nunique()<100 or int(use.held.sum())<60:
        raise ValueError('Insufficient completed held-state history')
    if not use.action.isin(ACTION_NAMES).all() or use.loc[~use.held,'action'].ne(0).any():
        raise ValueError('Flat states have only legal hold')
    states=np.stack(use.state.map(np.asarray));nxt=np.stack(use.next_state.map(np.asarray))
    x=action_inputs(states,use.action.to_numpy())
    past=daily.loc[daily.date.lt(cutoff),'bps'].to_numpy(float)
    scale=max(float(past.std(ddof=1)),.01)
    reward=use.reward.to_numpy(float)-spec['penalty']*use.squared_reward.to_numpy(float)/(2*scale)
    weights=1/use.groupby('start').start.transform('size').to_numpy()
    weights/=weights.mean()
    discounts=np.power(spec['gamma'],use.sessions.to_numpy())
    terminal=use.terminal.to_numpy(bool);next_held=use.next_held.to_numpy(bool)
    targets=reward.copy();trace=[]
    next_x=action_inputs(np.repeat(nxt,3,axis=0),np.tile(np.arange(3),len(nxt)))
    for _ in range(24 if spec['gamma'] else 1):
        model=ActionForest(n_actions=3,n_estimators=24,max_depth=3,min_samples_leaf=30,
                                  random_state=2026,n_jobs=1)
        model.fit(x,targets,sample_weight=weights)
        values=model.predict(next_x).reshape(len(use),3)
        # Cash carries the opportunity value of subsequent fixed proposals.
        # It is not terminal, and trim/close are illegal while flat.
        continuation=np.where(next_held,values.max(axis=1),values[:,0])
        continuation[terminal]=0.
        updated=reward+discounts*continuation
        if not np.isfinite(updated).all():raise ValueError('Nonfinite Q targets')
        trace.append(float(np.mean(np.abs(updated-targets))))
        targets=updated;rest()
    audit=dict(cutoff=str(cutoff.date()),last_transition_end=str(use.end.max().date()),
        transitions=len(use),distinct_market_dates=int(use.start.nunique()),
        held_transitions=int(use.held.sum()),distinct_held_market_dates=int(use.loc[use.held,'start'].nunique()),
        terminal_transitions=int(terminal.sum()),daily_reward_scale_bps=scale,
        action_support_counts={int(action):int(count) for action,count in use.action.value_counts().items()},
        iterations=len(trace),bellman_target_change=trace,spec=spec,features=FEATURES,
        caveat='Shared market dates across behaviors; aggregate partially observed book; censored endpoints retain nonterminal continuation.')
    return QPolicy(model,audit)


def source_hashes():
    names=['management.py','managed_ledger.py','critic.py','provenance.py']
    return {str((ROOT/'scripts/p25_rl'/name).relative_to(ROOT)):
            base.digest(ROOT/'scripts/p25_rl'/name) for name in names}


def prepare(budget):
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    paths=[ROOT/'scripts/p25_rl'/name for name in ['run.py','policy.py','ledger.py','provenance.py']]
    paths += [ROOT/'scripts/uniform_smile_engine.py',ROOT/'scripts/uniform_smile_risk.py',
              ROOT/'scripts/option_flex/contract_engine.py',base.SOURCE,
              entry.prior.QUOTES,entry.prior.CONFIG,ROOT/'config/p25_submission.json']
    paths += [base.MARKET/name for name in ['stocks.parquet','rates.parquet','dividends.parquet']]
    paths += [entry.prior.OUT/'replays/fixed100'/name for name in entry.prior.FILES]
    protected={}
    for path in paths:
        protected[str(path.relative_to(ROOT))]=base.digest(path);budget.rest()
    protocol=dict(question='Can daily held-position Q improve the original fixed-full IV/RV proposal portfolio?',
        features=FEATURES,experiments=SPECS,behaviors=BEHAVIORS,source_code=source_hashes(),
        algorithm='Independent forest per action:24 ExtraTrees depth3,minleaf30,seed2026;24 Bellman iterations;one fit for gamma0. One-hot interface, state-only per-action critics. Equal decision-date weights across behavior paths; action support counts saved.',
        actions=ACTION_NAMES,entry='All608 inherited proposals remain fixed100 entries; original risk and execution checks; no new signals,directions,maturities or re-expansion.',
        state='15 current completed-close summaries. RV20/60 are annualized EWMA squared log returns of close*cfadj, min63 prior/current observations; no future fill or payoff class.',
        timing='Observer at completed close BEFORE management requests; store global action; original next-session actual-quote execution with fees. Modeled causal held quotes may inform state, never elective fills.',
        reward='Actual daily funded cash-excess return inbps; optional .5*squared_daily_bps/(2*training-onlyscale), floor .01bps. Utility is not Sharpe.',
        flat='Only legal hold while flat; daily cash states retain value of subsequent fixed proposals. Terminal only actual end-of-history flat runoff; fit cutoff does not create terminals.',
        train='Initial transitions end strictly before2020; final transitions end strictly before2022. Partial known transitions allowed, boundary continuation may extrapolate.',
        validation='Continuous2018-2021 ledger, ordinary full hold before2020. IdenticalOct31 2021 proposal cutoff andDec31 runoff. Selection only2020-2021.',
        evaluation='Continuous original2018-Aug2025 book, ordinary full hold before2022. Carry actual baseline Dec2021inventory/NAV. No later retuning.',
        selection='Two Q arms only; >=20validation entries, nonnegative profit in2020and2021, Sharpe>fixed100+.01; otherwisefixed100. Root combined selection across frozen entry/management arms is authoritative.',
        limitations='Only608 inherited proposals; four behaviors share market history. Aggregate partial state, midpoint/cost/universe/assignment assumptions inherited. All later years previously inspected; exploratory inference.',
        resources=dict(threads=1,duty=.2,rss_limit_mib=384),
        instrumentation=instrumentation_info(),protected_hashes=protected)
    path=REPORT/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=json.loads(json.dumps(protocol)):
        raise ValueError('Frozen management protocol changed')
    base.write(path,protocol)
    return protocol


def replay_signature(signal,cal,kind,model,active_from,seed=2026):
    return dict(protocol_sha256=base.digest(REPORT/'protocol.json'),sources=source_hashes(),
        signals_sha256=hashlib.sha256(pd.util.hash_pandas_object(signal,index=True).to_numpy().tobytes()).hexdigest(),
        calendar_sha256=hashlib.sha256(pd.DatetimeIndex(cal).asi8.tobytes()).hexdigest(),
        kind=kind,model_hash=stable_model_digest(model),active_from=active_from,seed=seed)


def verify_cache(folder,signature=None):
    status=json.loads((folder/'status.json').read_text())
    if status['status']!='passed' or status['protocol_sha256']!=base.digest(REPORT/'protocol.json'):
        raise ValueError(f'Unverified management cache:{folder.name}')
    if status['source_code']!=source_hashes() or (signature is not None and status['cache_signature']!=signature):
        raise ValueError(f'Management cache provenance changed:{folder.name}')
    for name,expected in status['output_hashes'].items():
        if base.digest(folder/name)!=expected:raise ValueError(f'Management output changed:{name}')
    return status


def replay(name,signal,cal,cfg,events,budget,kind='hold',model=None,
           active_from='2018-01-01',seed=2026,folder=None):
    folder=OUT/'replays'/name if folder is None else folder
    folder.mkdir(parents=True,exist_ok=True)
    signature=replay_signature(signal,cal,kind,model,active_from,seed)
    if (folder/'status.json').exists():
        verify_cache(folder,signature);return
    started=time.monotonic();market=entry.Market(budget,signal,entry.prior.QUOTES)
    rv=variance_states(market);budget.rest()
    planner=Planner(market.rates,cfg,rv,kind,model,active_from,seed)
    def observe(*args):
        planner.observe(*args);budget.rest()
        if len(planner.daily)%252==0:
            print(f'  Management {name}: completed{args[0].date()}',flush=True)
    print(f'Management replay{name}:{len(signal)} proposals,{len(cal)}sessions',flush=True)
    nav,trades,decisions,ledger=simulate_with_management_observer(
        market,signal,cal,cfg,1e6,policy=lambda row,context:4,
        corporate_events=events,wing_capital=1e6,
        management_policy=planner.manage,session_observer=observe)
    base.mark_book(nav,market.rates,cfg.capital)
    audit,trades=entry.reconcile(nav,trades,decisions,signal)
    paid=trades.reason.isin(['entry','announced_corporate_action_exit',
                           'learned_policy_exit','learned_policy_trim'])
    np.testing.assert_allclose(trades.loc[paid,'commission'],
        trades.loc[paid,'quantity'].abs()*cfg.option_commission,atol=1e-10,rtol=0)
    records,daily=transitions(planner)
    np.testing.assert_allclose(daily.set_index('date').bps/10000,nav.excess_return,
                               atol=1e-12,rtol=1e-8)
    np.testing.assert_allclose(records.reward.sum(),daily.bps.iloc[1:].sum(),atol=1e-7)
    log=pd.DataFrame(ledger.pop('management_log',[]))
    if log.empty:log=pd.DataFrame(columns=['cohort','signal_date','date','target','status'])
    nav.to_parquet(folder/'daily_nav.parquet');trades.to_csv(folder/'trades.csv',index=False)
    decisions.to_csv(folder/'decisions.csv',index=False)
    scored=signal.copy();scored['learned_level']=4
    scored.to_parquet(folder/'scored_allocations.parquet',index=False)
    records.to_parquet(folder/'transitions.parquet',index=False)
    daily.to_parquet(folder/'daily_rewards.parquet',index=False)
    pd.DataFrame(planner.events).to_parquet(folder/'actions.parquet',index=False)
    log.to_csv(folder/'management_log.csv',index=False)
    status=dict(status='passed',**audit,**ledger,kind=kind,
        protocol_sha256=base.digest(REPORT/'protocol.json'),source_code=source_hashes(),
        cache_signature=signature,output_hashes={name:base.digest(folder/name) for name in REPLAY_FILES},
        elapsed_seconds=time.monotonic()-started,peak_rss_mib=budget.peak,
        management_filled_instructions=int(log.status.eq('filled').sum()),
        trimmed_cohorts=int(trades.loc[trades.reason.eq('learned_policy_trim'),'cohort'].nunique()),
        closed_cohorts=int(trades.loc[trades.reason.eq('learned_policy_exit'),'cohort'].nunique()),
        elective_commission_verified=True)
    base.write(folder/'status.json',status)
    print(f'Finished management{name}:{audit["entries"]}entries,{time.monotonic()-started:.1f}s',flush=True)
    market.db.close();del market,rv,planner,nav,trades,decisions,records,daily;gc.collect();budget.rest()


def behavior(signal,cal,cfg,events,budget):
    selected=signal.loc[signal.date.le('2021-10-31')].copy()
    history=cal[cal<='2021-12-31']
    for kind,seed in BEHAVIORS:
        replay(f'{kind}_{seed}',selected,history,cfg,events,budget,kind=kind,seed=seed,
               folder=OUT/'behavior'/f'{kind}_{seed}')


def behavior_data():
    records=[];daily=[]
    for kind,seed in BEHAVIORS:
        folder=OUT/'behavior'/f'{kind}_{seed}'
        verify_cache(folder)
        r=pd.read_parquet(folder/'transitions.parquet',use_threads=False)
        d=pd.read_parquet(folder/'daily_rewards.parquet',use_threads=False)
        r['behavior']=d['behavior']=f'{kind}_{seed}';records.append(r);daily.append(d)
    return pd.concat(records,ignore_index=True),pd.concat(daily,ignore_index=True)


def fit(name,cutoff,records,daily,budget):
    folder=OUT/'models'/str(pd.Timestamp(cutoff).date());folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{name}.joblib'
    signature=dict(protocol_sha256=base.digest(REPORT/'protocol.json'),sources=source_hashes(),
        cutoff=str(pd.Timestamp(cutoff).date()),case=name,spec=SPECS[name],
        training_files={str((OUT/'behavior'/f'{kind}_{seed}'/filename).relative_to(ROOT)):
                        base.digest(OUT/'behavior'/f'{kind}_{seed}'/filename)
                        for kind,seed in BEHAVIORS for filename in ['transitions.parquet','daily_rewards.parquet']})
    if path.exists():
        model=joblib.load(path)
        if model.audit['cache_signature']!=signature:
            raise ValueError(f'Management model provenance changed:{name}')
        if (model.audit['cutoff']!=str(pd.Timestamp(cutoff).date()) or model.audit['spec']!=SPECS[name]
            or pd.Timestamp(model.audit['last_transition_end'])>=pd.Timestamp(cutoff)):
            raise ValueError(f'Management cached training boundary changed:{name}')
        return model
    model=fit_policy(records,daily,cutoff,SPECS[name],budget.rest)
    model.audit['cache_signature']=signature
    joblib.dump(model,path);base.write(folder/f'{name}.json',model.audit)
    print(f'Fitted{name}@{cutoff}:{model.audit["distinct_market_dates"]}marketdates,{model.audit["held_transitions"]}heldtransitions',flush=True)
    return model


def check_hold_equivalence(folder,old):
    actual=pd.read_parquet(folder/'daily_nav.parquet',use_threads=False)
    expected=pd.read_parquet(old/'daily_nav.parquet',use_threads=False)
    pd.testing.assert_frame_equal(actual,expected,check_exact=False,atol=1e-7,rtol=1e-11)
    for name in ['trades.csv','decisions.csv']:
        pd.testing.assert_frame_equal(pd.read_csv(folder/name),pd.read_csv(old/name),
                                      check_exact=False,atol=1e-7,rtol=1e-11)


def metrics(folder,start,stop):
    verify_cache(folder)
    nav=pd.read_parquet(folder/'daily_nav.parquet',use_threads=False)
    decisions=pd.read_csv(folder/'decisions.csv',parse_dates=['signal_date','entry_date'])
    return entry.period_metrics(nav,decisions,start,stop)


def validation(signal,cal,cfg,events,budget):
    records,daily=behavior_data()
    selected=signal.loc[signal.date.le('2021-10-31')].copy();history=cal[cal<='2021-12-31']
    replay('fixed100',selected,history,cfg,events,budget,
           folder=OUT/'validation/fixed100')
    check_hold_equivalence(OUT/'validation/fixed100',entry.OUT/'validation/fixed100')
    base.write(REPORT/'hold_verification.json',dict(status='passed',
        nav_trades_decisions_match_entry_validation_baseline=True))
    for name in SPECS:
        model=fit(name,'2020-01-01',records,daily,budget)
        replay(name,selected,history,cfg,events,budget,kind='learned',model=model,
               active_from='2020-01-01',folder=OUT/'validation'/name)
    scores=[]
    for name in ['fixed100']+list(SPECS):
        folder=OUT/'validation'/name
        years=[metrics(folder,f'{year}-01-01',f'{year}-12-31')['profit_above_cash']
               for year in [2020,2021]]
        scores.append(dict(case=name,**metrics(folder,'2020-01-01','2021-12-31'),
                           profit_2020=years[0],profit_2021=years[1]))
    table=pd.DataFrame(scores);table.to_csv(REPORT/'validation.csv',index=False)
    reference=float(table.loc[table.case.eq('fixed100'),'annualized_excess_sharpe'].iloc[0])
    eligible=table.loc[table.case.isin(CANDIDATES)&table.entries.ge(20)&
        table.profit_2020.ge(0)&table.profit_2021.ge(0)&table.annualized_excess_sharpe.gt(reference+.01)]
    chosen='fixed100' if eligible.empty else eligible.sort_values(
        ['annualized_excess_sharpe','case'],ascending=[False,True]).case.iloc[0]
    selection=dict(selected=chosen,validation_only=True,
        authoritative_selection='Root combined frozen entry/management selection',
        protocol_sha256=base.digest(REPORT/'protocol.json'))
    path=REPORT/'selection.json'
    if path.exists() and json.loads(path.read_text())!=selection:
        raise ValueError('Management validation selection changed')
    base.write(path,selection)
    print(table[['case','annualized_excess_sharpe','entries','profit_2020','profit_2021']].to_string(index=False),flush=True)


def evaluate(signal,cal,cfg,events,budget):
    if not (REPORT/'selection.json').exists():
        raise ValueError('Management validation must finish before later evaluation')
    records,daily=behavior_data()
    replay('fixed100',signal,cal,cfg,events,budget)
    check_hold_equivalence(OUT/'replays/fixed100',entry.prior.OUT/'replays/fixed100')
    old=pd.read_parquet(entry.prior.OUT/'replays/fixed100/daily_nav.parquet',use_threads=False)
    for name in SPECS:
        model=fit(name,'2022-01-01',records,daily,budget)
        replay(name,signal,cal,cfg,events,budget,kind='learned',model=model,
               active_from='2022-01-01')
        nav=pd.read_parquet(OUT/'replays'/name/'daily_nav.parquet',use_threads=False)
        pd.testing.assert_frame_equal(nav.loc[:'2021-12-31'],old.loc[:'2021-12-31'],
                                      check_exact=False,atol=1e-7,rtol=1e-11)
    base.write(REPORT/'handoff_verification.json',dict(status='passed',
        all_management_prior_nav_matches_baseline=True,inventory_continuous=True))
    periods=[dict(case=name,period='evaluation',
                  **metrics(OUT/'replays'/name,'2022-01-01','2025-08-29'))
             for name in ['fixed100']+list(SPECS)]
    pd.DataFrame(periods).to_csv(REPORT/'periods.csv',index=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['prepare','behavior','validation','evaluation','all'],default='all')
    args=parser.parse_args();budget=Budget();protocol=prepare(budget)
    if args.phase=='prepare':return
    signal,cal,cfg,events=entry.inputs()
    for phase,fn in [('behavior',behavior),('validation',validation),('evaluation',evaluate)]:
        if args.phase in [phase,'all']:fn(signal,cal,cfg,events,budget)
    for name,expected in protocol['protected_hashes'].items():
        if base.digest(ROOT/name)!=expected:raise ValueError(f'Protected source changed:{name}')
        budget.rest()
    base.write(OUT/'progress.json',dict(status='passed',phase=args.phase,peak_rss_mib=budget.peak))


if __name__=='__main__':
    # Keep saved QPolicy classes importable by a separate reporting process.
    from scripts.p25_rl.management import main as canonical_main
    canonical_main()
