"""Inventory-conditioned fitted Q learning on causally completed weekly steps.

Entry fixed at the strongest 50% control; management is hold/half/close.
No new data, entry gate, sizing leverage, or alteration to prior replays.
"""
import argparse
import gc
import json
from pathlib import Path
import resource
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from scripts import run_long_tenor_full_universe as full

base = full.base
OUT = Path('artifacts/contracts/long_tenor_fitted_q/P25')
REPORT = Path('reports/long_tenor_fitted_q')
FEATURES = ['carry_forecast','carry252','carry252_available',
            'gap_convergence','age_fraction','delta_gap']
ACTIONS = np.array([0., .5, 1.])
DISCOUNT = .98
ITERATIONS = 16
WINDOW = 48
PENALTIES = [0., .5]


def configure():
    full.setup('P25')
    base.OUT, base.REPORT = OUT, REPORT
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    base.protected()
    baseline_file = OUT/'baseline_sources.json'
    if not baseline_file.exists():
        sources = {}
        for tenor in base.TENORS:
            control = Path('artifacts/contracts/long_tenor_policy_search/P25')/f'{tenor}D/replays/fixed_50_hold'
            sources.update({str(p):base.digest(p) for p in control.iterdir() if p.is_file()})
        base.dump(baseline_file,sources)
    assert all(base.digest(p)==want for p,want in json.loads(baseline_file.read_text()).items())


def block_reward(ep, start, end, current, requested):
    """Decision at start; fill at start+1, then hold through end close."""
    assert 0 <= requested <= current
    fill = start+1
    available = bool(ep['can_exit'][fill])
    if requested == .5 and ep['split_seen'][start]:
        available = False
    retained = requested if available else current
    # full-universe labels doubled elective exit costs; remove that extra
    # research penalty here. Keep the existing .65 training commission.
    rewards = np.zeros(end-start)
    for j in range(fill, end+1):
        previous = current if j == fill else retained
        change = retained*ep['hedge'][j]-previous*ep['hprev'][j]
        hedge_cost = (np.abs(change)*ep['spot'][j]).sum()*.0001
        closing = (previous-retained)*ep['exit_cost_training'][j]/2
        rewards[j-fill] = previous*ep['gross_training'][j]-hedge_cost-closing
    return float(rewards.sum()/ep['entry_vega']), retained


def episode_transitions(ep):
    n = len(ep['dates'])
    decisions = np.flatnonzero(ep['state_known'] & (np.arange(n)>=5) & (np.arange(n)%5==0) & (np.arange(n)<n-1))
    records = []
    for k, start in enumerate(decisions):
        next_step = int(decisions[k+1]) if k+1 < len(decisions) else n-1
        for current in [1., .5]:
            for action in ACTIONS[ACTIONS<=current]:
                reward, retained = block_reward(ep, int(start), next_step, current, float(action))
                terminal = retained == 0 or k+1 == len(decisions)
                records.append(dict(cohort=ep['cohort'], start=ep['dates'][start],
                    end=ep['dates'][next_step], state=ep['states'][start], fraction=current,
                    action=float(action), reward=reward, next_state=ep['states'][next_step],
                    next_fraction=retained, terminal=terminal,
                    discount=DISCOUNT**((next_step-start)/5)))
    return records


def causal_transitions(records, cutoff, months=WINDOW):
    start = cutoff-pd.DateOffset(months=months)
    result = [r for r in records if r['start']>=start and r['end']<cutoff]
    assert all(r['start']<r['end']<cutoff for r in result)
    return result


def inputs(states, fractions, actions):
    return np.column_stack([states, fractions, actions])


def action_values(model, states, fractions):
    count = len(states)
    actions = np.tile(ACTIONS, count)
    x = inputs(np.repeat(states, 3, axis=0),np.repeat(fractions, 3),actions)
    values = model.predict(x).reshape(count,3)
    values[ACTIONS[None,:]>fractions[:,None]+1e-12] = -np.inf
    return values


def choose_actions(model, states, fractions, conservative=False):
    values = action_values(model, states, fractions)
    if conservative:
        # Penalize unstable *advantages* over holding, preserving covariance
        # between predictions from the same tree. Dispersion is a model
        # disagreement measure, not a calibrated confidence interval.
        x = inputs(np.repeat(states,3,axis=0),np.repeat(fractions,3),np.tile(ACTIONS,len(states)))
        hold_index = np.where(fractions==1.,2,1)
        predictions = np.stack([tree.predict(x).reshape(len(states),3) for tree in model.estimators_])
        advantage = predictions-predictions[:,np.arange(len(states)),hold_index][:,:,None]
        values = advantage.mean(axis=0)-advantage.std(axis=0,ddof=1)
        values[ACTIONS[None,:]>fractions[:,None]+1e-12] = -np.inf
    # Ties preserve the larger inventory; no accidental exit from equal Q.
    choices = 2-np.argmax(values[:,::-1],axis=1)
    return ACTIONS[choices]


def fitted_q(records, penalty):
    states = np.stack([r['state'] for r in records])
    fractions = np.array([r['fraction'] for r in records])
    actions = np.array([r['action'] for r in records])
    reward = np.array([r['reward'] for r in records])
    hold = actions == fractions
    scale = max(float(reward[hold].std()),1e-6)
    # A declared per-step mean/variance utility, not a claim of directly
    # differentiating the final shared-account Sharpe.
    utility = reward-penalty*reward**2/(2*scale)
    next_states = np.stack([r['next_state'] for r in records])
    next_fractions = np.array([r['next_fraction'] for r in records])
    terminal = np.array([r['terminal'] for r in records])
    discount = np.array([r['discount'] for r in records])
    x = inputs(states,fractions,actions)
    cohort = pd.Series([r['cohort'] for r in records])
    weights = (1/cohort.map(cohort.value_counts())).to_numpy()
    weights /= weights.mean()
    y = utility.copy()
    model = None
    for iteration in range(ITERATIONS):
        model = ExtraTreesRegressor(n_estimators=20,max_depth=5,
            min_samples_leaf=40,n_jobs=1,random_state=2026)
        model.fit(x,y,sample_weight=weights)
        continuation = action_values(model,next_states,next_fractions).max(axis=1)
        continuation[terminal] = 0
        y = utility+discount*continuation
        assert np.isfinite(y).all()
    return dict(model=model, penalty=penalty, reward_scale=scale,
                iterations=ITERATIONS, discount=DISCOUNT), dict(
        transitions=len(records), cohorts=int(cohort.nunique()),
        last_training_step=str(max(r['end'] for r in records).date()),
        reward_scale=scale, training_utility_mean=float(utility.mean()))


def fit(tenor):
    packed = joblib.load(full.OUT/'P25'/f'{tenor}D/episodes.joblib')
    cal = packed['calendar']
    records = [r for ep in packed['episodes'] for r in episode_transitions(ep)]
    for cutoff in pd.date_range('2018-01-01',cal[-1],freq='QS-JAN'):
        for penalty in PENALTIES:
            folder = OUT/f'{tenor}D/models/risk{penalty:g}'
            path = folder/f'{cutoff.date()}.joblib'
            if path.exists():
                continue
            use = causal_transitions(records,cutoff)
            cohorts = len({r['cohort'] for r in use})
            model = None
            details = dict(transitions=len(use),cohorts=cohorts,last_training_step=None)
            if cohorts>=16 and len(use)>=640:
                model,details = fitted_q(use,penalty)
            folder.mkdir(parents=True,exist_ok=True)
            joblib.dump(model,path)
            base.dump(path.with_suffix('.json'),dict(fit=str(cutoff.date()),
                tenor=tenor,penalty=penalty,window_months=WINDOW,
                fitted=model is not None,**details))
            print(f'FQI {tenor}D risk{penalty:g} {cutoff.date()}: {cohorts} past cohorts, {len(use)} action transitions',flush=True)
        gc.collect()
    return cal


def callback(schedule, rv, cal, conservative=False):
    positions = {d:i for i,d in enumerate(cal)}
    def policy(date,held):
        if len(held)!=2:
            # Shadow labels and fixed control let each selected leg run off
            # at its own expiry. A missing counterpart is not a learned exit.
            return 1.
        fitted = schedule[max(k for k in schedule if k<=date)]
        if fitted is None:
            return 1.
        age = positions[date]-positions[held[0]['entry_date']]
        if age<5 or age%5:
            return 1.
        values = full.slow_features(date,held,rv)
        if values is None:
            return 1.
        current = min(p['management_fraction'] for p in held)
        state = np.array([[values[field] for field in FEATURES]])
        target = float(choose_actions(fitted['model'],state,np.array([current]),conservative)[0])
        if target==.5 and any(abs(p['quantity']) != abs(p['initial_quantity'])*p['management_fraction'] for p in held):
            return 1.
        return target
    return policy


def replay(tenor):
    rows = pd.read_parquet(full.OUT/'P25'/f'{tenor}D/signal_rows.parquet').assign(learned_level=2)
    market = base.market_for(tenor,'fqi_calendar')
    cal = market.rates.index[market.rates.index>='2018-01-01']
    rv = base.load_rv()
    for penalty in PENALTIES:
        schedule = {pd.Timestamp(p.stem):joblib.load(p) for p in sorted((OUT/f'{tenor}D/models/risk{penalty:g}').glob('*.joblib'))}
        base.save_replay(tenor,f'fixed50_fitted_q_risk{penalty:g}_aligned_runoff',rows,callback(schedule,rv,cal))
    schedule = {pd.Timestamp(p.stem):joblib.load(p) for p in sorted((OUT/f'{tenor}D/models/risk0').glob('*.joblib'))}
    base.save_replay(tenor,'fixed50_fitted_q_conservative_aligned_runoff',rows,callback(schedule,rv,cal,conservative=True))


def summarize():
    rows, years, hashes = [], [], {}
    for tenor in base.TENORS:
        control = Path('artifacts/contracts/long_tenor_policy_search/P25')/f'{tenor}D/replays/fixed_50_hold'
        folders = [control]+sorted((OUT/f'{tenor}D/replays').glob('*_aligned_runoff'))
        for folder in folders:
            status = json.loads((folder/'status.json').read_text())
            nav = pd.read_parquet(folder/'daily_nav.parquet')
            r = nav.excess_return.iloc[1:]
            status['formation_2018_2023_sharpe'] = base.LONG.base.sharpe(r.loc[:'2023-12-31'])
            status['recent_2024_partial2025_sharpe'] = base.LONG.base.sharpe(r.loc['2024-01-01':])
            assert np.isclose(base.LONG.base.sharpe(r),status['annualized_excess_sharpe'])
            status['source_folder'] = str(folder)
            rows.append(status)
            years.append(pd.read_csv(folder/'annual.csv').assign(tenor=tenor,case=folder.name))
            for p in folder.iterdir():
                if p.is_file():
                    hashes[str(p)] = base.digest(p)
    frame = pd.DataFrame(rows)
    frame.to_csv(REPORT/'comparison.csv',index=False)
    pd.concat(years,ignore_index=True).to_csv(REPORT/'annual.csv',index=False)
    fits = [json.loads(p.read_text()) for p in OUT.glob('*D/models/*/*.json')]
    assert all(r['last_training_step'] is None or pd.Timestamp(r['last_training_step'])<pd.Timestamp(r['fit']) for r in fits)
    source_paths = [Path(__file__),Path('tests/test_long_tenor_fitted_q.py'),
                   Path('scripts/run_signal_discovery_guarded.py'),
                   Path('scripts/run_long_tenor_full_universe.py'),
                   Path('scripts/run_long_tenor_policy_search.py'),
                   Path('scripts/report_long_tenor_search.py'),OUT/'baseline_sources.json']
    source_paths += list(OUT.glob('*D/models/*/*'))
    source_paths += [full.OUT/'P25'/f'{tenor}D/episodes.joblib' for tenor in base.TENORS]
    source_paths += [full.OUT/'P25'/f'{tenor}D/signal_rows.parquet' for tenor in base.TENORS]
    hashes.update({str(p):base.digest(p) for p in source_paths})
    base.dump(REPORT/'source_manifest.json',hashes)
    baseline_file = OUT/'baseline_sources.json'
    assert baseline_file.exists()
    assert all(base.digest(p)==want for p,want in json.loads(baseline_file.read_text()).items())
    exit_diagnostics()
    comparison_diagnostics(frame)
    base.dump(REPORT/'audit.json',dict(causal_completed_transition_purge=True,
        original_checkpoints_unchanged=True,entry_sizes_fixed_50=True,
        original_costs_and_risk_limits=True, one_worker=True,cpu_core_fraction=.5,
        each_leg_natural_runoff_matches_training_and_fixed_control=True,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        model='ExtraTrees fitted Q iteration, 16 Bellman updates, gamma .98 per five sessions',
        states=FEATURES+['retained_fraction','requested_action'],
        training_commission=.65,actual_commission=.03,
        training_additional_exit_cost_multiplier=1, full_universe_native_labels=True))
    (REPORT/'findings.txt').write_text(
        'LONG-TENOR FITTED Q MANAGEMENT\n\n'
        'Same 529 original pair/date/direction proposals. Fixed 50% entry on 180D/365D P25 contracts. The shared integer cash ledger, fees, daily hedges and all entry risk limits are unchanged.\n'
        'Quarterly 48-month fits use past completed five-session transitions, including past portions of still-open contracts. Every transition end precedes its fit. No future expiry payoff is a training target. Cold start holds until 16 distinct cohorts and 640 action transitions.\n'
        'State: current slow/recent expected carry, slow-RV availability, IV gap convergence, age fraction, relative delta and retained inventory. Actions: irreversible hold/half/close, checked weekly and filled next session only with executable actual quotes.\n'
        'Natural runoff follows the fixed control and shadow labels: if one leg expires first, the remaining leg continues to its own expiry unless a prior learned paired exit has closed it. Entry vega neutrality is not continuously rebalanced. Earlier unaligned replay folders are retained for diagnosis and excluded from the final comparison.\n'
        'Fitted Q iteration uses 20 shallow ExtraTrees, 16 Bellman updates and discount .98 per five sessions. Two utilities: expected step reward and a .5 standardized quadratic risk penalty. An adaptive follow-up after the initial 180D replay penalizes risk0 action advantage by one ensemble standard deviation; this discourages unstable exits and is not a calibrated confidence interval. Training option commissions remain .65 and actual .03; removes the former extra 2x elective-exit training penalty. Weekly shadow rewards use isolated fractional hedge/borrow costs; real evaluation is the reconciled shared-account integer replay.\n'
        'This isolates sequential management from entry sizing. Additional opportunity selection, re-expansion, rolls, dynamic gamma/vega rebalancing and portfolio-level capital state are not part of this test. All history has been explored; no fresh holdout or multiple-comparison-adjusted significance claim.\n\n'
        +frame[['tenor','case','annualized_excess_sharpe','entries','trims','exits',
                'formation_2018_2023_sharpe','recent_2024_partial2025_sharpe']].to_string(index=False)+'\n\n'
        +'Diagnostics: paired_comparison.csv, exit_continuation_summary.csv, management_comparison.png and portfolio_followup.json. The latter is an exploratory optimized saved-return blend, without joint portfolio replay.\n')
    print(frame[['tenor','case','annualized_excess_sharpe','entries','trims','exits',
                'recent_2024_partial2025_sharpe']].to_string(index=False),flush=True)


def exit_diagnostics():
    records = []
    for tenor in base.TENORS:
        episodes = {ep['cohort']:ep for ep in joblib.load(full.OUT/'P25'/f'{tenor}D/episodes.joblib')['episodes']}
        for folder in sorted((OUT/f'{tenor}D/replays').glob('*_aligned_runoff')):
            trades = pd.read_csv(folder/'trades.csv',parse_dates=['date'])
            exits = trades.loc[trades.reason.eq('learned_policy_exit')].drop_duplicates('cohort')
            for trade in exits.itertuples():
                ep = episodes.get(trade.cohort)
                record = dict(tenor=tenor,case=folder.name,cohort=trade.cohort,exit_date=trade.date,
                              native_episode_available=ep is not None)
                if ep is not None:
                    future = ep['dates']>trade.date
                    # Diagnostic only: continuation *after* the actual fill,
                    # excluding the fill-day benefit/cost and capital reuse.
                    reward = ep['reward_actual'][future]/ep['entry_vega']
                    record.update(remaining_hold_pnl_per_entry_vega=float(reward.sum()),
                                  next20_hold_pnl_per_entry_vega=float(reward[:20].sum()),
                                  avoided_subsequent_loss=bool(reward.sum()<0))
                records.append(record)
    result = pd.DataFrame(records)
    result.to_csv(REPORT/'exit_continuation_diagnostics.csv',index=False)
    if not result.empty:
        known = result.loc[result.native_episode_available]
        grouped = known.groupby(['tenor','case']).agg(
            diagnosed_exit_cohorts=('cohort','size'),
            fraction_followed_by_hold_loss=('avoided_subsequent_loss','mean'),
            mean_remaining_hold_pnl_per_entry_vega=('remaining_hold_pnl_per_entry_vega','mean'))
        grouped.to_csv(REPORT/'exit_continuation_summary.csv')


def comparison_diagnostics(frame):
    from scripts.report_long_tenor_search import paired_bootstrap,optimal_nonnegative_weights,sharpe
    comparisons = []
    panel = pd.DataFrame({'30D':pd.read_parquet('artifacts/contracts/p25_long_management/replays/30D_manage_36M/daily_nav.parquet').excess_return.iloc[1:]})
    components = {'30D':'previous managed 30D checkpoint'}
    for tenor in base.TENORS:
        rows = frame.loc[frame.tenor.eq(tenor)]
        fixed = rows.loc[rows.case.eq('fixed_50_hold')].iloc[0]
        b = pd.read_parquet(Path(fixed.source_folder)/'daily_nav.parquet').excess_return.iloc[1:]
        for row in rows.loc[~rows.case.eq('fixed_50_hold')].itertuples():
            a = pd.read_parquet(Path(row.source_folder)/'daily_nav.parquet').excess_return.iloc[1:]
            pd.testing.assert_index_equal(a.index,b.index)
            lo,hi = paired_bootstrap(a,b)
            comparisons.append(dict(tenor=tenor,case=row.case,sharpe=sharpe(a),
                fixed_sharpe=sharpe(b),sharpe_gain=sharpe(a)-sharpe(b),
                unadjusted_gain_95_low=lo,unadjusted_gain_95_high=hi))
        chosen = rows.loc[rows.annualized_excess_sharpe.idxmax()]
        panel[f'{tenor}D'] = pd.read_parquet(Path(chosen.source_folder)/'daily_nav.parquet').excess_return.iloc[1:]
        components[f'{tenor}D'] = chosen.case
    pd.DataFrame(comparisons).to_csv(REPORT/'paired_comparison.csv',index=False)
    weights = optimal_nonnegative_weights(panel)
    r = panel@weights
    base.dump(REPORT/'portfolio_followup.json',dict(annualized_excess_sharpe=sharpe(r),
        weights=dict(zip(panel.columns,weights)),components=components,
        previous_optimized_blend_sharpe=1.0949854373230519,
        full_history_optimized=True,shared_portfolio_replayed=False,no_fresh_holdout=True))
    r.to_csv(REPORT/'portfolio_followup_daily.csv',header=['excess_return'])
    pd.DataFrame([dict(year=int(year),annualized_excess_sharpe=sharpe(v)) for year,v in r.groupby(r.index.year)]).to_csv(REPORT/'portfolio_followup_annual.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(11,5),sharey=True)
    order = ['fixed_50_hold','fixed50_fitted_q_risk0_aligned_runoff',
             'fixed50_fitted_q_risk0.5_aligned_runoff','fixed50_fitted_q_conservative_aligned_runoff']
    for ax,tenor in zip(axes,base.TENORS):
        selected = frame.loc[frame.tenor.eq(tenor)].set_index('case').loc[order]
        ax.bar(np.arange(4),selected.annualized_excess_sharpe,color=['#94a3b8','#2563eb','#93c5fd','#166534'])
        ax.set_xticks(np.arange(4),['Fixed\nhold','Fitted Q\nmean','Fitted Q\nrisk penalty','Fitted Q\ndisagreement'],fontsize=9)
        ax.set_title(f'{tenor}-day 25-delta puts')
        ax.axhline(0,color='black',linewidth=.6)
        for i,v in enumerate(selected.annualized_excess_sharpe):
            ax.text(i,v+.015,f'{v:.3f}',ha='center',fontsize=10)
    axes[0].set_ylabel('Annualized daily excess Sharpe')
    fig.suptitle('Sequential management: same half-size entries, costs and risk limits')
    fig.text(.5,.015,'2018–partial 2025; all history previously examined. Disagreement arm is an adaptive follow-up.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.95))
    fig.savefig(REPORT/'management_comparison.png',dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase',choices=['test','fit','replay','summarize','sweep'],required=True)
    p.add_argument('--tenor',type=int,choices=base.TENORS)
    args = p.parse_args()
    if args.phase=='test':
        suite = unittest.defaultTestLoader.loadTestsFromNames(['tests.test_long_tenor_fitted_q','tests.test_p25_management'])
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
        return
    configure()
    for tenor in ([args.tenor] if args.tenor else base.TENORS):
        if args.phase in ['fit','sweep']:
            fit(tenor)
        if args.phase in ['replay','sweep']:
            replay(tenor)
    if args.phase in ['summarize','sweep']:
        summarize()
    base.protected()


if __name__=='__main__':
    main()
