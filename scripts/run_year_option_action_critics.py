"""One-year action discrimination audit and bounded fitted-Q follow-up."""
import argparse
from functools import partial
import gc
import json
from pathlib import Path
import resource
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from scripts import run_long_tenor_reversible as previous

OUT = Path('artifacts/contracts/year_option_action_critics/P25')
REPORT = Path('reports/year_option_action_critics')
VARIANTS = {
    'action_outputs_16': dict(iterations=16,discount=.98,risk=False),
    'long_horizon_64': dict(iterations=64,discount=.995,risk=False),
    'long_horizon_risk_64': dict(iterations=64,discount=.995,risk=True),
}
CASES = {name:dict(model=name,penalty=1.) for name in VARIANTS}
CASES.update({name+'_greedy':dict(model=name,penalty=0.) for name in VARIANTS})


def configure():
    previous.configure()
    previous.base.OUT,previous.base.REPORT = OUT,REPORT
    OUT.mkdir(parents=True,exist_ok=True)
    REPORT.mkdir(parents=True,exist_ok=True)
    reference = OUT/'references.json'
    if not reference.exists():
        paths = [previous.OUT/'365D/episodes.joblib',previous.full.OUT/'P25/365D/signal_rows.parquet']
        for root in [previous.OUT/'365D/replays',previous.old.OUT/'365D/replays',
                     Path('artifacts/contracts/long_tenor_policy_search/P25/365D/replays/fixed_50_hold')]:
            paths += [p for p in root.rglob('*') if p.is_file() and p.suffix in ['.parquet','.csv','.json']]
        previous.base.dump(reference,{str(p):previous.base.digest(p) for p in paths})
    assert all(previous.base.digest(p)==want for p,want in json.loads(reference.read_text()).items())
    previous.base.dump(OUT/'design.json',dict(variants=VARIANTS,replay_cases=CASES,
        shared_parameters=dict(trees=20,depth=5,min_leaf=40,training_months=48,refit='quarterly',
            warmup_cohorts=16,warmup_weekly_steps=128,ensemble_advantage_penalty=1.,
            entries='same 529 proposals; fixed50; original allocation is maximum',
            features=previous.FEATURES,costs_and_risk_limits='unchanged'),
        leaf_regularization='40 inventory-state rows in multioutput critic versus40 augmented state-action rows previously; effective regularization differs',
        selection='All variants reported. All historical periods already explored; no fresh holdout.'))


def features(states,fractions):
    # Explicit output per action; action is never a split feature.
    return previous.inputs(states,fractions,np.zeros(len(fractions)))[:,:-1]


def group_transitions(records):
    grouped = {}
    for r in records:
        key = (r['cohort'],r['start'],r['end'],r['fraction'])
        actions = grouped.setdefault(key,{})
        if r['action'] in actions:
            raise AssertionError('Duplicate counterfactual action')
        actions[r['action']] = r
    batches = []
    for choices in grouped.values():
        if set(choices)!=set(previous.ACTIONS):
            raise AssertionError('Each observation needs all three reversible actions')
        batch = [choices[a] for a in previous.ACTIONS]
        assert all(np.array_equal(batch[0]['state'],r['state']) and
                   np.array_equal(batch[0]['next_state'],r['next_state']) for r in batch)
        batches.append(batch)
    return batches


def mask_values(values,fractions):
    values = np.asarray(values).copy()
    values[fractions==0,1:] = -np.inf
    return values


def choose_actions(model,states,fractions,penalty=1.):
    predictions = np.stack([tree.predict(features(states,fractions)) for tree in model.estimators_])
    index = np.where(fractions==1.,2,1)
    advantage = predictions-predictions[:,np.arange(len(states)),index][:,:,None]
    score = mask_values(advantage.mean(axis=0)-penalty*advantage.std(axis=0,ddof=1),fractions)
    best = score.argmax(axis=1)
    best = np.where(score[np.arange(len(states)),best]<=1e-12,index,best)
    best = np.where(fractions==0,0,best)
    return previous.ACTIONS[best]


def bellman_target(utility,continuation,terminal,discount):
    continuation = np.asarray(continuation).copy()
    continuation[terminal] = 0.
    return utility+discount*continuation


def fit_q(records,parameters):
    batches = group_transitions(records)
    states,next_states = [np.stack([b[0][name] for b in batches]) for name in ['state','next_state']]
    fractions = np.array([b[0]['fraction'] for b in batches])
    rewards,next_fractions,terminal = [np.array([[r[name] for r in b] for b in batches])
                                      for name in ['reward','next_fraction','terminal']]
    # The transition builder already discounts by observed trading-step length.
    # Recover that length from its .98 discount, then use the declared gamma.
    weeks = np.array([np.log(b[0]['discount'])/np.log(previous.old.DISCOUNT) for b in batches])
    discount = parameters['discount']**weeks[:,None]
    cohort = pd.Series([b[0]['cohort'] for b in batches])
    weights = (1/cohort.map(cohort.value_counts())).to_numpy()
    weights /= weights.mean()
    full = fractions==1.
    coefficient = previous.risk_coefficient(rewards[full,2],weights[full]) if parameters['risk'] else 0.
    utility = rewards-.5*coefficient*rewards**2
    target = utility.copy()
    x = features(states,fractions)
    following = features(np.repeat(next_states,3,axis=0),next_fractions.ravel())
    for iteration in range(parameters['iterations']):
        model = ExtraTreesRegressor(n_estimators=20,max_depth=5,min_samples_leaf=40,n_jobs=1,random_state=2026)
        model.fit(x,target,sample_weight=weights)
        if iteration+1<parameters['iterations']:
            values = mask_values(model.predict(following),next_fractions.ravel())
            continuation = values.max(axis=1).reshape(len(batches),3)
            target = bellman_target(utility,continuation,terminal,discount)
            assert np.isfinite(target).all()
    return model,coefficient


def fit(variant=None):
    packed = joblib.load(previous.OUT/'365D/episodes.joblib')
    records = [r for ep in packed['episodes'] for r in previous.transitions(ep,True)]
    for name in ([variant] if variant else VARIANTS):
        parameters = VARIANTS[name]
        for cutoff in pd.date_range('2018-01-01',packed['calendar'][-1],freq='QS-JAN'):
            folder = OUT/'365D/models'/name
            path = folder/f'{cutoff.date()}.joblib'
            if path.exists():
                continue
            past = previous.old.causal_transitions(records,cutoff)
            note = dict(fit=str(cutoff.date()),variant=name,transitions=len(past),
                        cohorts=len({r['cohort'] for r in past}),weekly_steps=len(past)//6,
                        last_training_step=str(max(r['end'] for r in past).date()) if past else None,
                        parameters=parameters)
            eligible = previous.enough_history(note,True)
            model,coefficient = fit_q(past,parameters) if eligible else (None,0.)
            note.update(fitted=model is not None,risk_coefficient=coefficient)
            folder.mkdir(parents=True,exist_ok=True)
            joblib.dump(model,path)
            previous.base.dump(path.with_suffix('.json'),note)
            print(f'{name} {cutoff.date()}: {note["weekly_steps"]} past weekly steps, coefficient {coefficient:.4f}',flush=True)
            gc.collect()


def callback(schedule,rv,cal,cfg,penalty=1.):
    positions = {d:i for i,d in enumerate(cal)}
    def policy(date,held):
        current = min(p['management_fraction'] for p in held) if held else 1.
        if len(held)!=2:
            return current
        age = positions[date]-positions[held[0]['entry_date']]
        model = schedule[max(k for k in schedule if k<=date)]
        if model is None or age<5 or age%5:
            return current
        values = previous.full.slow_features(date,held,rv)
        if values is None:
            return current
        exposure = previous.risk_features(date,held,cfg,held[0]['book_context']['reserve_utilization'])
        state = np.r_[[values[f] for f in previous.old.FEATURES],exposure][None,:]
        return float(choose_actions(model,state,np.array([current]),penalty)[0])
    return policy


def replay(variant=None):
    base = previous.base
    rows = pd.read_parquet(previous.full.OUT/'P25/365D/signal_rows.parquet').assign(learned_level=2)
    market = base.market_for(365,'action_critic_calendar')
    cal = market.rates.index[market.rates.index>='2018-01-01']
    rv,cfg = base.load_rv(),base.source.configuration(365)
    old_engine,old_simulate = base.LONG.base.engine,previous.engine.simulate
    try:
        base.LONG.base.engine = previous.engine
        previous.engine.simulate = partial(old_simulate,reversible_management=True)
        base.save_replay(365,'fixed50_hold_control',rows,lambda date,held:min(p['management_fraction'] for p in held))
        actual = pd.read_parquet(OUT/'365D/replays/fixed50_hold_control/daily_nav.parquet')
        expected = pd.read_parquet('artifacts/contracts/long_tenor_policy_search/P25/365D/replays/fixed_50_hold/daily_nav.parquet')
        pd.testing.assert_frame_equal(actual,expected,rtol=1e-12,atol=1e-6)
        for case,spec in CASES.items():
            name = spec['model']
            if variant is not None and name!=variant:
                continue
            schedule = {}
            for path in sorted((OUT/'365D/models'/name).glob('*.joblib')):
                note = json.loads(path.with_suffix('.json').read_text())
                assert note['parameters']==VARIANTS[name]
                schedule[pd.Timestamp(path.stem)] = joblib.load(path) if previous.enough_history(note,True) else None
            base.save_replay(365,case,rows,callback(schedule,rv,cal,cfg,spec['penalty']))
            folder = OUT/'365D/replays'/case
            trades = pd.read_csv(folder/'trades.csv',parse_dates=['date'])
            restored = trades.loc[trades.reason.eq('learned_policy_restore')]
            if len(restored):
                np.testing.assert_allclose(restored.commission,restored.quantity.abs()*cfg.option_commission)
                assert (pd.to_datetime(restored.exit_signal_date)<restored.date).all()
                assert restored.groupby(['cohort','date']).secid.nunique().eq(2).all()
            status = json.loads((folder/'status.json').read_text())
            status.update(restore_fills=len(restored),restored_cohorts=int(restored.cohort.nunique()),
                          critic_model=name,ensemble_advantage_penalty=spec['penalty'])
            base.dump(folder/'status.json',status)
    finally:
        previous.engine.simulate = old_simulate
        base.LONG.base.engine = old_engine


def summarize():
    from scripts.report_long_tenor_search import sharpe,paired_bootstrap
    records,years,returns = [],[],{}
    folders = [OUT/'365D/replays/fixed50_hold_control',
               previous.old.OUT/'365D/replays/fixed50_fitted_q_conservative_aligned_runoff',
               previous.OUT/'365D/replays/enriched_reversible_aligned_warmup']
    folders += [OUT/'365D/replays'/name for name in CASES]
    for folder in folders:
        status = json.loads((folder/'status.json').read_text())
        nav = pd.read_parquet(folder/'daily_nav.parquet')
        r = nav.excess_return.iloc[1:]
        assert np.isclose(sharpe(r),status['annualized_excess_sharpe'])
        returns[status['case']] = r
        status.setdefault('restore_fills',0)
        status.update(source_folder=str(folder),formation_2018_2023_sharpe=sharpe(r.loc[:'2023-12-31']),
                      recent_2024_partial2025_sharpe=sharpe(r.loc['2024-01-01':]))
        records.append(status)
        years.append(pd.read_csv(folder/'annual.csv').assign(case=status['case']))
    frame = pd.DataFrame(records)
    frame.to_csv(REPORT/'comparison.csv',index=False)
    pd.concat(years).to_csv(REPORT/'annual.csv',index=False)
    comparisons = []
    for name in CASES:
        lo,hi = paired_bootstrap(returns[name],returns['fixed50_hold_control'])
        comparisons.append(dict(case=name,sharpe_gain=sharpe(returns[name])-sharpe(returns['fixed50_hold_control']),
                                unadjusted_gain_95_low=lo,unadjusted_gain_95_high=hi))
    pd.DataFrame(comparisons).to_csv(REPORT/'paired_comparison.csv',index=False)
    plot_results(frame,returns)
    notes = [json.loads(p.read_text()) for p in (OUT/'365D/models').glob('*/*.json')]
    assert all(n['last_training_step'] is None or pd.Timestamp(n['last_training_step'])<pd.Timestamp(n['fit']) for n in notes)
    previous.base.dump(REPORT/'audit.json',dict(fixed_control_reproduced=True,past_only_completed_weekly_steps=True,
        original_reference_hashes_unchanged=True,capital_and_cost_limits_unchanged=True,
        one_worker=True,cpu_core_fraction=.5,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        shadow_training='Local fractional cohort hedges and logged reference-book state; evaluation is actual shared-cash integer inventory',
        no_fresh_holdout=True,variants=VARIANTS,replay_cases=CASES))
    paths = [Path(__file__),Path('tests/test_year_option_action_critics.py'),
             Path(previous.__file__),Path(previous.engine.__file__),Path('scripts/run_signal_discovery_guarded.py')]
    paths += [p for p in OUT.rglob('*') if p.is_file() and p.suffix in ['.json','.parquet','.joblib','.csv']]
    previous.base.dump(REPORT/'source_manifest.json',{str(p):previous.base.digest(p) for p in paths})
    columns = ['case','annualized_excess_sharpe','entries','trims','exits','restore_fills',
               'formation_2018_2023_sharpe','recent_2024_partial2025_sharpe']
    (REPORT/'findings.txt').write_text(
        'ONE-YEAR ACTION-OUTPUT FITTED-Q EXPERIMENT\n\n'
        'Same native 25-delta put contracts, all 529 entry proposals, fixed50 entries, daily delta hedging, and original execution/capital limits. Management can hold, trim, close, or restore a trimmed pair to its original allocation. Different leg expiries retain original natural runoff.\n'
        'Change the critic representation first: jointly predict three Q outputs per observed state/inventory rather than put action in a single scalar-Q tree input. Shared trees preserve paired action covariance. Same20 trees, depth5, leaf40, conservative mean-minus-one-ensemble-standard-deviation advantage. Leaf40 now counts inventory-state observations rather than augmented action rows, so effective leaf regularization differs. A tie holds current inventory.\n'
        'Next test64 Bellman fits with weekly gamma .995, versus16 fits and .98. Original16 fits propagate at most roughly80 observed trading days of future rewards;64 covers the typical one-year weekly sequence. Discounted reward remains a surrogate, not exact portfolio Sharpe. A third arm uses the previously declared past-calibrated mean-variance utility.\n'
        'Before observing replay performance, add greedy-versus-conservative inference for all three fitted models. Greedy maximizes mean Q; conservative subtracts one standard deviation of paired tree advantages over holding. Ensemble dispersion is a heuristic, not calibrated statistical uncertainty. The extra arms require no additional fits.\n'
        'Quarterly48-month fits use only completed weekly transitions ending strictly before the fit date. Training commission .65; actual .03; execution0.1 vol points/leg and stock1bp unchanged. No extra leverage or tail-win selection. Local fractional shadow rewards approximate shared portfolio interactions; executable integer ledger is the final evaluation.\n'
        'All history has already been researched. Full-period rankings and block intervals are exploratory and not adjusted for model selection;2024–partial2025 is a diagnostic slice, not an untouched test.\n\n'
        +frame[columns].to_string(index=False)+'\n\n'+pd.DataFrame(comparisons).to_string(index=False)+'\n')
    print(frame[columns].to_string(index=False),flush=True)


def plot_results(frame,returns):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = {'fixed50_hold_control':'Fixed hold',
              'fixed50_fitted_q_conservative_aligned_runoff':'Prior selective RL',
              'enriched_reversible_aligned_warmup':'Prior reversible RL',
              'action_outputs_16':'Explicit action outputs',
              'long_horizon_64':'Full-year learning horizon',
              'long_horizon_risk_64':'Full-year horizon + risk reward',
              'action_outputs_16_greedy':'Action outputs, greedy',
              'long_horizon_64_greedy':'Full-year horizon, greedy',
              'long_horizon_risk_64_greedy':'Full-year + risk, greedy'}
    fig,axes = plt.subplots(1,2,figsize=(14,6))
    cases = list(labels)
    values = frame.set_index('case').loc[cases].annualized_excess_sharpe
    axes[0].barh(np.arange(len(cases)),values,color=['#64748b']*3+['#2563eb','#15803d','#b45309']*2)
    axes[0].set_yticks(np.arange(len(cases)),[labels[c] for c in cases],fontsize=9)
    axes[0].axvline(0,color='black',linewidth=.6)
    axes[0].set_xlabel('Annualized cash-excess Sharpe')
    for i,value in enumerate(values):
        axes[0].text(value+(.01 if value>=0 else -.01),i,f'{value:.3f}',va='center',
                     ha='left' if value>=0 else 'right',fontsize=9)
    for case in ['fixed50_hold_control',*CASES]:
        axes[1].plot(returns[case].index,100*returns[case].cumsum(),label=labels[case],linewidth=1.3)
    axes[1].set_ylabel('Cumulative daily excess returns, arithmetic (%)')
    axes[1].legend(fontsize=8)
    axes[1].axhline(0,color='black',linewidth=.6)
    fig.suptitle('One-year 25-delta puts: action representation and learning horizon')
    fig.text(.5,.01,'2018–August2025; identical entry proposals/costs/limits. Historical exploration; no fresh holdout.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.03,1,.95))
    fig.savefig(REPORT/'comparison.png',dpi=160)
    plt.close(fig)


def audit():
    REPORT.mkdir(parents=True,exist_ok=True)
    packed = joblib.load(previous.OUT/'365D/episodes.joblib')
    records = [r for ep in packed['episodes'] for r in previous.transitions(ep,True)]
    rows = []
    for cutoff in pd.to_datetime(['2020-01-01','2023-01-01','2024-01-01']):
        past = previous.old.causal_transitions(records,cutoff)
        full = [r for r in past if r['fraction']==1. and r['action']==1.]
        take = np.linspace(0,len(full)-1,min(2048,len(full)),dtype=int)
        state = np.stack([full[i]['state'] for i in take])
        fraction = np.ones(len(state))
        model = joblib.load(previous.OUT/f'365D/models/enriched_reversible/{cutoff.date()}.joblib')
        x = previous.inputs(np.repeat(state,3,axis=0),np.repeat(fraction,3),np.tile(previous.ACTIONS,len(state)))
        pred = np.stack([tree.predict(x).reshape(len(state),3) for tree in model.estimators_])
        mean = pred.mean(axis=0)
        rows.append(dict(fit=str(cutoff.date()),sampled_completed_states=len(state),
            action_feature_importance=float(model.feature_importances_[-1]),
            fraction_feature_importance=float(model.feature_importances_[-2]),
            all_action_values_identical_fraction=float(np.mean(np.ptp(mean,axis=1)<1e-12)),
            hold_trim_identical_fraction=float(np.mean(np.abs(mean[:,2]-mean[:,1])<1e-12)),
            per_tree_hold_trim_identical_fraction=float(np.mean(np.abs(pred[:,:,2]-pred[:,:,1])<1e-12)),
            selected_hold_fraction=float(np.mean(previous.choose_actions(model,state,fraction,True)==1.)),
            unpenalized_hold_fraction=float(np.mean(mean.argmax(axis=1)==2)),
            selected_trim_fraction=float(np.mean(previous.choose_actions(model,state,fraction,True)==.5)),
            selected_exit_fraction=float(np.mean(previous.choose_actions(model,state,fraction,True)==0.))))
    pd.DataFrame(rows).to_csv(REPORT/'previous_critic_audit.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    nav = pd.read_parquet('artifacts/contracts/long_tenor_policy_search/P25/365D/replays/fixed_50_hold/daily_nav.parquet')
    trades = pd.read_csv('artifacts/contracts/long_tenor_policy_search/P25/365D/replays/fixed_50_hold/trades.csv',parse_dates=['date'])
    entries = trades.loc[trades.reason.eq('entry')].groupby('cohort').date.min()
    previous.base.dump(REPORT/'coverage.json',dict(native_training_episodes=len(packed['episodes']),
        executed_entry_cohorts=len(entries),entries_by_year=entries.dt.year.value_counts().sort_index().to_dict(),
        last_entry=str(entries.max().date()),last_mark=str(nav.index[-1].date()),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    benchmark_interest = nav.cash_benchmark.diff().fillna(nav.cash_benchmark.iloc[0]-10_000_000)
    attribution = pd.DataFrame(dict(option_before_execution=nav.option_pnl+nav.option_execution_cost,
        hedge_gross=nav.hedge_gross_pnl,dividends=nav.dividend_cash,
        option_execution=-nav.option_execution_cost,stock_execution=-nav.hedge_cost,
        stock_borrow=-nav.stock_borrow_cost,cash_interest_vs_benchmark=nav.interest-benchmark_interest))
    attribution['total_excess_dollars'] = attribution.sum(axis=1)
    expected = (nav.nav-nav.cash_benchmark).diff().fillna(nav.nav.iloc[0]-nav.cash_benchmark.iloc[0])
    np.testing.assert_allclose(attribution.total_excess_dollars,expected,atol=1e-7,rtol=1e-9)
    attribution.groupby(attribution.index.year).sum().to_csv(REPORT/'fixed_hold_annual_attribution.csv',index_label='year')
    labels = []
    for ep in packed['episodes']:
        labels.append(dict(cohort=ep['cohort'],pair=ep['pair'],entry_year=ep['date'].year,
            entry_carry_forecast=float(ep['entry_features'][0]),
            normalized_hold_pnl=float(ep['reward_actual'].sum()/ep['entry_vega']),
            observations=len(ep['dates']),
            accounting='Isolated native-label PnL per entry vega; not shared-portfolio attribution'))
    label_frame = pd.DataFrame(labels)
    label_frame.to_csv(REPORT/'native_label_diagnostics.csv',index=False)
    label_frame.groupby('pair').agg(cohorts=('cohort','size'),
        mean_normalized_hold_pnl=('normalized_hold_pnl','mean'),
        median_normalized_hold_pnl=('normalized_hold_pnl','median'),
        positive_fraction=('normalized_hold_pnl',lambda x:float(x.gt(0).mean())),
        mean_entry_carry=('entry_carry_forecast','mean')).to_csv(REPORT/'native_pair_diagnostics.csv')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['audit','test','fit','replay','summarize','sweep'],required=True)
    parser.add_argument('--variant',choices=VARIANTS)
    args = parser.parse_args()
    if args.phase=='audit':
        audit()
        return
    if args.phase=='test':
        suite = unittest.defaultTestLoader.loadTestsFromNames(['tests.test_year_option_action_critics','tests.test_long_tenor_reversible'])
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
        return
    configure()
    if args.phase in ['fit','sweep']:
        fit(args.variant)
    if args.phase in ['replay','sweep']:
        replay(args.variant)
    if args.phase in ['summarize','sweep'] and args.variant is None:
        summarize()
    configure()


if __name__=='__main__':
    main()
