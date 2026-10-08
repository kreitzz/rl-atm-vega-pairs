"""Bounded inventory-Q experiments; one serial, resource-guarded worker.

python -m scripts.p25_rl.run --phase all
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','ARROW_NUM_THREADS']:
    os.environ[key]='1'
import argparse
import gc
import hashlib
import json
import time
import joblib
import numpy as np
import pandas as pd
from scripts import run_p25_direct_policy as prior
from scripts.p25_rl.ledger import simulate_with_observer, instrumentation_info
from scripts.p25_rl.policy import FEATURES,SPECS,Planner,fit_policy,transitions
from scripts.p25_rl.provenance import stable_model_digest
from scripts.run_p25_threshold_search import ThresholdMarket
from scripts.analyze_p25_opportunity_filter import period_metrics

base=prior.base
ROOT=base.ROOT
OUT=ROOT/'artifacts/contracts/p25_inventory_q'
REPORT=ROOT/'reports/p25_inventory_q'
BEHAVIORS=[('full',2026),('half',2026),('diversified',2026),
           ('random',2026),('random',2027),('random',2028),('sticky',2026),('congestion',2026)]


class Budget(base.Budget):
    def check(self):
        super().check()
        if (OUT/'STOP').exists():raise InterruptedError('Inventory Q STOP requested')


class Market(ThresholdMarket):
    def rate(self,date):return float(self.rates.loc[date])


def inputs():
    signal=pd.read_parquet(prior.OUT/'replays/tree_cem/scored_allocations.parquet',use_threads=False)
    cols=['date','pair','iv_mean','oriented_iv_gap','rank_strength','oriented_vrp_gap','profit_91d','vix']
    signal=signal.merge(pd.read_parquet(base.SOURCE,columns=cols,use_threads=False),on=['date','pair'],validate='one_to_one')
    assert len(signal)==608
    cal=pd.read_parquet(base.MARKET/'rates.parquet',use_threads=False).index
    cfg=base.Settings(**json.loads(prior.CONFIG.read_text()))
    events=json.loads((ROOT/'config/p25_submission.json').read_text())['corporate_events']
    return signal,cal[cal>='2018-01-01'],cfg,events


def prepare(budget):
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    files=[base.SOURCE,prior.QUOTES,prior.CONFIG,ROOT/'config/p25_submission.json',
        ROOT/'scripts/uniform_smile_engine.py',ROOT/'scripts/uniform_smile_risk.py',
        base.MARKET/'stocks.parquet',base.MARKET/'rates.parquet',base.MARKET/'dividends.parquet']
    files += [prior.OUT/'replays'/c/f for c in ['fixed100','tree_cem'] for f in prior.FILES]
    hashes={}
    for p in files:hashes[str(p.relative_to(ROOT))]=base.digest(p);budget.rest()
    protocol=dict(question='Can sequential Q learning improve the existing IV/RV proposal edge through actual portfolio allocation?',
        features=FEATURES,experiments=SPECS,behaviors=BEHAVIORS,
        algorithm='Fitted Q iteration:24 ExtraTrees depth3,minleaf30,seed2026,24 Bellman updates; gamma0 uses one immediate-reward fit. One-hot actions. Equal weight per decision date across behavior trajectories.',
        reward='Sum daily actual funded cash-excess returns in bps between opportunity closes; utility reward minus penalty*sum(daily_bps²)/(2*training-only daily scale), floor .01bps. Future bootstrap discounted gamma^session_count. Utility is not Sharpe.',
        timing='One action at completed signal close, next-session entry; unheld membership frozen at signal close. No execution quotes or future waiting time in state. All original fills, costs, risk limits, hedges and cash accounting retained.',
        train='Initial fit uses actual transitions ending strictly before2020; nonterminal partial histories retain continuation. Final fits use transitions ending strictly before2022. No artificial flat terminal at fit boundaries.',
        validation='2020-2021 continuous book, fixed100 before2020. Identical proposal cutoff Oct31 2021 for all controls/variants, runoff throughDec31 2021. Select on validation only.',
        evaluation='2022-Aug29 2025, continuous original2018onward book, fixed100 before2022; carry actual Dec2021 inventory and NAV. Fit once before2022. Every frozen arm reported, no later retuning.',
        selection='Select highest validation Sharpe among four Q arms with >=20validation entries, both2020/2021 excess-profit >=0, and Sharpe > fixed100+.01. Otherwise retain fixed100. Ablations never eligible for promotion.',
        conservatism='Candidate action advantage over full is penalized by one ensemble SD. Tree dispersion is model disagreement, not a confidence interval.',
        controls=['fixed100','fixed50','diversified fixed rule','congestion fixed rule','myopic_risk','q_no_inventory','archived tree_cem','archived learned5%threshold'],
        iteration='Bounded capacity Q, risk-shaped Q, conservative Q and diversified portfolio Q families declared before new results. All validation attempts saved. No test-driven search until an apparent winner appears.',
        limitations='Only608inherited proposals,495distinctdates. Counterfactual behavior paths share market data; aggregated book state is approximate. Previously inspected history and inherited universe, no untouched holdout or universal multiple-testing correction.',
        resources=dict(threads=1,duty=.2,rss_limit_mib=384),
        instrumentation=instrumentation_info(),protected_hashes=hashes)
    path=REPORT/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=json.loads(json.dumps(protocol)):raise ValueError('Frozen protocol changed')
    base.write(path,protocol)
    return protocol


def reconcile(nav,trades,decisions,signal):
    # The original verifier requires trade columns even for a permanent-cash arm.
    if trades.empty:
        trades=pd.DataFrame(columns=['reason','quantity','commission','cohort','optionid','price','execution_penalty','date'])
    for column in ['vega_a','vega_b']:
        if column not in decisions:decisions[column]=np.nan
    return base.verify_ledger(nav,trades,decisions,signal),trades


def replay(name,signal,cal,cfg,events,budget,kind='full',model=None,active_from='2018-01-01',seed=2026,folder=None):
    folder=OUT/'replays'/name if folder is None else folder
    folder.mkdir(parents=True,exist_ok=True)
    signature=replay_signature(signal,cal,kind,model,active_from,seed)
    if (folder/'status.json').exists():
        status=json.loads((folder/'status.json').read_text())
        if status['status']!='passed':raise ValueError('Cached replay is not verified')
        if status['cache_signature']!=signature:raise ValueError(f'Cached replay provenance changed:{name}')
        return
    started=time.monotonic()
    market=Market(budget,signal,prior.QUOTES)
    planner=Planner(signal,market.rates,cfg,kind,model,active_from,seed)
    def observe(*args):
        planner.observe(*args);budget.rest()
        if len(planner.daily)%252==0:print(f'  {name}: completed{args[0].date()}',flush=True)
    print(f'Replay {name}: {len(signal)} proposals,{len(cal)} sessions',flush=True)
    nav,trades,decisions,ledger=simulate_with_observer(market,signal,cal,cfg,1e6,policy=planner.entry,
        corporate_events=events,wing_capital=1e6,session_observer=observe)
    base.mark_book(nav,market.rates,cfg.capital)
    audit,trades=reconcile(nav,trades,decisions,signal)
    records,daily=transitions(planner)
    # Every interval reward must reconcile to the same funded cash-excess series.
    check=daily.set_index('date').bps/10000
    np.testing.assert_allclose(check,nav.excess_return,atol=1e-12,rtol=1e-8)
    if len(records):
        begin=records.start.iloc[0]
        np.testing.assert_allclose(records.reward.sum(),daily.loc[daily.date.gt(begin),'bps'].sum(),atol=1e-7)
    nav.to_parquet(folder/'daily_nav.parquet');trades.to_csv(folder/'trades.csv',index=False)
    decisions.to_csv(folder/'decisions.csv',index=False)
    scored=signal.drop(columns='learned_level').merge(decisions[['signal_date','pair','level']].rename(columns={'signal_date':'date','level':'learned_level'}),on=['date','pair'],validate='one_to_one',sort=False)
    scored.to_parquet(folder/'scored_allocations.parquet',index=False)
    records.to_parquet(folder/'transitions.parquet',index=False);daily.to_parquet(folder/'daily_rewards.parquet',index=False)
    pd.DataFrame(planner.events).to_parquet(folder/'actions.parquet',index=False)
    base.write(folder/'status.json',dict(status='passed',**audit,**ledger,elapsed_seconds=time.monotonic()-started,
        peak_rss_mib=budget.peak,protocol_sha256=base.digest(REPORT/'protocol.json'),kind=kind,cache_signature=signature,
        source_code={str(p.relative_to(ROOT)):base.digest(p) for p in [ROOT/'scripts/p25_rl/run.py',ROOT/'scripts/p25_rl/policy.py',ROOT/'scripts/p25_rl/ledger.py',ROOT/'scripts/p25_rl/provenance.py']}))
    print(f'Finished {name}: {audit["entries"]} entries,{time.monotonic()-started:.1f}s,peak{budget.peak:.1f}MiB',flush=True)
    market.db.close();del nav,trades,decisions,records,market,planner;gc.collect();budget.rest()


def replay_signature(signal,cal,kind,model,active_from,seed):
    return dict(protocol_sha256=base.digest(REPORT/'protocol.json'),
        sources={p:base.digest(ROOT/'scripts/p25_rl'/p) for p in ['run.py','policy.py','ledger.py','provenance.py']},
        signals_sha256=hashlib.sha256(pd.util.hash_pandas_object(signal,index=True).to_numpy().tobytes()).hexdigest(),
        calendar_sha256=hashlib.sha256(pd.DatetimeIndex(cal).asi8.tobytes()).hexdigest(),
        kind=kind,model_hash=stable_model_digest(model),active_from=active_from,seed=seed)


def baseline(signal,cal,cfg,events,budget):
    replay('baseline_verification',signal,cal,cfg,events,budget)
    folder=OUT/'replays/baseline_verification';old=prior.OUT/'replays/fixed100'
    actual=pd.read_parquet(folder/'daily_nav.parquet',use_threads=False)
    expected=pd.read_parquet(old/'daily_nav.parquet',use_threads=False)
    pd.testing.assert_frame_equal(actual,expected,check_exact=False,atol=1e-7,rtol=1e-11)
    for filename in ['trades.csv','decisions.csv']:
        pd.testing.assert_frame_equal(pd.read_csv(folder/filename),pd.read_csv(old/filename),check_exact=False,atol=1e-7,rtol=1e-11)
    base.write(REPORT/'baseline_verification.json',dict(status='passed',all_nav_columns_match=True,all_trades_match=True,all_decisions_match=True))
    print('Original fixed100 baseline exactly reproduced within ledger tolerance',flush=True)


def behavior(signal,cal,cfg,events,budget):
    selected=signal.loc[signal.date.le('2021-10-31')].copy();history=cal[cal<='2021-12-31']
    for kind,seed in BEHAVIORS:
        replay(f'{kind}_{seed}',selected,history,cfg,events,budget,kind=kind,seed=seed,folder=OUT/'behavior'/f'{kind}_{seed}')


def behavior_data():
    records=[];daily=[]
    for kind,seed in BEHAVIORS:
        folder=OUT/'behavior'/f'{kind}_{seed}'
        r=pd.read_parquet(folder/'transitions.parquet',use_threads=False);r['behavior']=f'{kind}_{seed}';records.append(r)
        d=pd.read_parquet(folder/'daily_rewards.parquet',use_threads=False);d['behavior']=f'{kind}_{seed}';daily.append(d)
    return pd.concat(records,ignore_index=True),pd.concat(daily,ignore_index=True)


def fit(name,cutoff,records,daily,budget):
    folder=OUT/'models'/str(pd.Timestamp(cutoff).date());folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{name}.joblib'
    signature=dict(case=name,cutoff=str(pd.Timestamp(cutoff).date()),spec=SPECS[name],
        protocol_sha256=base.digest(REPORT/'protocol.json'),policy_sha256=base.digest(ROOT/'scripts/p25_rl/policy.py'),
        provenance_sha256=base.digest(ROOT/'scripts/p25_rl/provenance.py'),
        sources={str((OUT/'behavior'/f'{kind}_{seed}'/filename).relative_to(ROOT)):base.digest(OUT/'behavior'/f'{kind}_{seed}'/filename)
                 for kind,seed in BEHAVIORS for filename in ['transitions.parquet','daily_rewards.parquet']})
    if path.exists():
        model=joblib.load(path)
        if model.audit['cache_signature']!=signature:raise ValueError(f'Cached model provenance changed:{name}')
        if (model.audit['cutoff']!=str(pd.Timestamp(cutoff).date()) or model.audit['spec']!=SPECS[name]
            or pd.Timestamp(model.audit['last_transition_end'])>=pd.Timestamp(cutoff)):
            raise ValueError(f'Cached model training boundary changed:{name}')
        return model
    model=fit_policy(records,daily,cutoff,SPECS[name],budget.rest)
    model.audit['cache_signature']=signature
    joblib.dump(model,path);base.write(folder/f'{name}.json',model.audit)
    print(f'Fitted {name}@{cutoff}: {model.audit["transitions"]} transitions,{model.audit["distinct_market_dates"]} distinct dates',flush=True)
    return model


def validation(signal,cal,cfg,events,budget):
    records,daily=behavior_data()
    selected=signal.loc[signal.date.le('2021-10-31')].copy();history=cal[cal<='2021-12-31']
    for name,kind in [('fixed100','full'),('fixed50','half'),('diversified','diversified'),('congestion','congestion')]:
        replay(name,selected,history,cfg,events,budget,kind=kind,active_from='2020-01-01',folder=OUT/'validation'/name)
    for name in SPECS:
        model=fit(name,'2020-01-01',records,daily,budget)
        replay(name,selected,history,cfg,events,budget,kind='learned',model=model,active_from='2020-01-01',folder=OUT/'validation'/name)
    scores=[]
    for name in ['fixed100','fixed50','diversified','congestion']+list(SPECS):
        folder=OUT/'validation'/name;nav=pd.read_parquet(folder/'daily_nav.parquet',use_threads=False)
        decisions=pd.read_csv(folder/'decisions.csv',parse_dates=['signal_date','entry_date'])
        metric=period_metrics(nav,decisions,'2020-01-01','2021-12-31')
        years=[period_metrics(nav,decisions,f'{year}-01-01',f'{year}-12-31')['profit_above_cash'] for year in [2020,2021]]
        scores.append(dict(case=name,**metric,profit_2020=years[0],profit_2021=years[1]))
    table=pd.DataFrame(scores);table.to_csv(REPORT/'validation.csv',index=False)
    reference=float(table.loc[table.case.eq('fixed100'),'annualized_excess_sharpe'].iloc[0])
    eligible=table.loc[table.case.isin(['q_capacity','q_risk','q_conservative','q_portfolio'])&table.entries.ge(20)&
        table.profit_2020.ge(0)&table.profit_2021.ge(0)&table.annualized_excess_sharpe.gt(reference+.01)]
    selected='fixed100' if eligible.empty else eligible.sort_values(['annualized_excess_sharpe','case'],ascending=[False,True]).case.iloc[0]
    best=table.loc[table.case.isin(['q_capacity','q_risk','q_conservative','q_portfolio'])].sort_values('annualized_excess_sharpe',ascending=False).case.iloc[0]
    selection=dict(selected=selected,best_q_by_validation=best,criterion=prepare_selection_text(),validation_only=True,
        protocol_sha256=base.digest(REPORT/'protocol.json'))
    p=REPORT/'selection.json'
    if p.exists() and json.loads(p.read_text())!=selection:raise ValueError('Frozen validation selection changed')
    base.write(p,selection);print(table[['case','annualized_excess_sharpe','entries','profit_2020','profit_2021']].to_string(index=False),flush=True)
    print(f'Validation selection: {selected}; best Q for diagnostics: {best}',flush=True)


def prepare_selection_text():return 'Four Q candidates only; >=20entries, nonnegative profit in both2020/2021, Sharpe exceeds fixed100 by.01; otherwise baseline.'


def evaluate(signal,cal,cfg,events,budget):
    assert (REPORT/'selection.json').exists(),'Validation selection must be frozen before evaluation'
    records,daily=behavior_data()
    for name in SPECS:
        model=fit(name,'2022-01-01',records,daily,budget)
        replay(name,signal,cal,cfg,events,budget,kind='learned',model=model,active_from='2022-01-01')
    for name,kind in [('fixed50','half'),('diversified','diversified'),('congestion','congestion')]:
        replay(name,signal,cal,cfg,events,budget,kind=kind,active_from='2022-01-01')
    for name in SPECS:
        nav=pd.read_parquet(OUT/'replays'/name/'daily_nav.parquet',use_threads=False)
        old=pd.read_parquet(prior.OUT/'replays/fixed100/daily_nav.parquet',use_threads=False)
        pd.testing.assert_frame_equal(nav.loc[:'2021-12-31'],old.loc[:'2021-12-31'],check_exact=False,atol=1e-7,rtol=1e-11)
    base.write(REPORT/'handoff_verification.json',dict(status='passed',all_q_prior_nav_matches_baseline=True,
        inventory_continuity='Same unbroken simulation; no reset or return splicing'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['prepare','baseline','behavior','validation','evaluation','all'],default='all')
    args=parser.parse_args();budget=Budget();protocol=prepare(budget)
    if args.phase=='prepare':return
    signal,cal,cfg,events=inputs()
    for phase,fn in [('baseline',baseline),('behavior',behavior),('validation',validation),('evaluation',evaluate)]:
        if args.phase in [phase,'all']:fn(signal,cal,cfg,events,budget)
    for name,sha in protocol['protected_hashes'].items():assert base.digest(ROOT/name)==sha;budget.rest()
    base.write(OUT/'progress.json',dict(phase=args.phase,status='passed',peak_rss_mib=budget.peak))


if __name__=='__main__':main()
