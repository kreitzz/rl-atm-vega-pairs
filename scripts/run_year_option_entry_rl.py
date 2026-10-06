"""Native one-year entry Q, bootstrapped from the weekly management critic."""
import argparse
from functools import partial
import json
from pathlib import Path
import resource
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from scripts import run_year_option_action_critics as critic

previous,base = critic.previous,critic.previous.base
OUT = Path('artifacts/contracts/year_option_entry_rl/P25')
REPORT = Path('reports/year_option_entry_rl')
FEATURES = ['carry_forecast','carry252','carry252_available','term_gap',
            'iv_gap_native','iv_mean_native','correlation','vix']
MODES = {'native_mean_entry':'long_horizon_64','native_risk_entry':'long_horizon_risk_64'}


def configure():
    critic.configure()
    base.OUT,base.REPORT = OUT,REPORT
    OUT.mkdir(parents=True,exist_ok=True)
    REPORT.mkdir(parents=True,exist_ok=True)
    base.dump(OUT/'design.json',dict(modes=MODES,features=FEATURES,
        entry_actions='no entry, or original50 allocation; no extra leverage',
        opening_value='First completed5-session block including opening fees, plus discounted optimal management Q',
        entry_tree=dict(trees=20,depth=3,min_leaf=6,seed=2026),
        refits='Quarterly48-month first-week-completed entry labels; parent Q uses past-only weekly transitions',
        acting='Greedy entry and greedy management; no additional confidence boundary',
        candidates='Same529 frozen pair/date/direction proposals; decide eligibility for native365D contracts',
        selection='Adaptive follow-up after management-only failure. No fresh holdout; all variants retained.'))


def first_decision(ep):
    steps = np.flatnonzero(ep['state_known'] & (np.arange(len(ep['dates']))>=5)
                           & (np.arange(len(ep['dates']))%5==0)
                           & (np.arange(len(ep['dates']))<len(ep['dates'])-1))
    return int(steps[0]) if len(steps) else None


def completed_entries(episodes,cutoff):
    start = cutoff-pd.DateOffset(months=48)
    result = []
    for ep in episodes:
        j = first_decision(ep)
        if j is not None and start<=ep['date']<cutoff and ep['dates'][j]<cutoff:
            result.append((ep,j))
    return result


def opening_target(reward,continuation,coefficient,discount):
    return reward-.5*coefficient*reward**2+discount*continuation


def fit():
    episodes = joblib.load(previous.OUT/'365D/episodes.joblib')['episodes']
    for mode,parent in MODES.items():
        for model_path in sorted((critic.OUT/'365D/models'/parent).glob('*.joblib')):
            path = OUT/'models'/mode/model_path.name
            if path.exists():
                continue
            cutoff = pd.Timestamp(model_path.stem)
            management = joblib.load(model_path)
            parent_note = json.loads(model_path.with_suffix('.json').read_text())
            available = completed_entries(episodes,cutoff)
            note = dict(fit=str(cutoff.date()),mode=mode,parent=parent,entry_cohorts=len(available),
                last_entry_label=str(max(ep['dates'][j] for ep,j in available).date()) if available else None,
                risk_coefficient=parent_note['risk_coefficient'])
            opener = None
            if management is not None and len(available)>=16:
                state = np.stack([ep['states'][j] for ep,j in available])
                continuation = management.predict(critic.features(state,np.ones(len(state)))).max(axis=1)
                reward = np.array([ep['reward_training'][:j+1].sum()/ep['entry_vega'] for ep,j in available])
                discount = np.array([critic.VARIANTS[parent]['discount']**(j/5) for _,j in available])
                target = opening_target(reward,continuation,parent_note['risk_coefficient'],discount)
                x = np.stack([ep['entry_features'] for ep,_ in available])
                assert x.shape[1]==len(FEATURES) and np.isfinite(target).all()
                opener = ExtraTreesRegressor(n_estimators=20,max_depth=3,min_samples_leaf=6,n_jobs=1,random_state=2026)
                opener.fit(x,target)
                note.update(mean_training_target=float(target.mean()),positive_target_fraction=float(np.mean(target>0)))
            note['fitted'] = opener is not None
            path.parent.mkdir(parents=True,exist_ok=True)
            joblib.dump(opener,path)
            base.dump(path.with_suffix('.json'),note)
            print(f'{mode} {cutoff.date()}: {len(available)} first-week-completed entry labels',flush=True)


def entry_level(model,state):
    # No-position action has exactly zero excess P&L and is absorbing for this
    # candidate. Entry Q already includes opening costs; never subtract twice.
    return 2 if model is None or float(model.predict(state[None,:])[0])>0 else 0


def scored_rows(mode):
    rows = pd.read_parquet(previous.full.OUT/'P25/365D/signal_rows.parquet').assign(learned_level=2)
    schedule = {pd.Timestamp(p.stem):joblib.load(p) for p in sorted((OUT/'models'/mode).glob('*.joblib'))}
    scores = []
    for index,row in rows.iterrows():
        fit_date = max(d for d in schedule if d<=row.date)
        model = schedule[fit_date]
        state = row[FEATURES].to_numpy(float)
        available = bool(row.native_features_available) and np.isfinite(state).all()
        if available:
            rows.loc[index,'learned_level'] = entry_level(model,state)
        scores.append(dict(signal_date=str(row.date.date()),pair=row.pair,wing=row.wing,
                           fit=str(fit_date.date()),native_state_available=available,
                           opening_q=float(model.predict(state[None,:])[0]) if available and model is not None else None,
                           level=int(rows.loc[index,'learned_level'])))
    pd.DataFrame(scores).to_csv(REPORT/f'{mode}_entry_scores.csv',index=False)
    return rows


def replay():
    market = base.market_for(365,'native_entry_calendar')
    cal = market.rates.index[market.rates.index>='2018-01-01']
    rv,cfg = base.load_rv(),base.source.configuration(365)
    old_engine,old_simulate = base.LONG.base.engine,previous.engine.simulate
    try:
        base.LONG.base.engine = previous.engine
        previous.engine.simulate = partial(old_simulate,reversible_management=True)
        for mode,parent in MODES.items():
            rows = scored_rows(mode)
            schedule = {pd.Timestamp(p.stem):joblib.load(p) for p in sorted((critic.OUT/'365D/models'/parent).glob('*.joblib'))}
            for holding in [False,True]:
                name = mode+'_hold' if holding else mode
                policy = (lambda date,held:min(p['management_fraction'] for p in held)) if holding else critic.callback(schedule,rv,cal,cfg,penalty=0.)
                base.save_replay(365,name,rows,policy)
                folder = OUT/'365D/replays'/name
                trades = pd.read_csv(folder/'trades.csv',parse_dates=['date'])
                restored = trades.loc[trades.reason.eq('learned_policy_restore')]
                if len(restored):
                    np.testing.assert_allclose(restored.commission,restored.quantity.abs()*cfg.option_commission)
                    assert (pd.to_datetime(restored.exit_signal_date)<restored.date).all()
                    assert restored.groupby(['cohort','date']).secid.nunique().eq(2).all()
                status = json.loads((folder/'status.json').read_text())
                status.update(restore_fills=len(restored),declined_proposals=int(rows.learned_level.eq(0).sum()),
                              fixed_hold_same_entry_decisions=holding)
                base.dump(folder/'status.json',status)
    finally:
        previous.engine.simulate = old_simulate
        base.LONG.base.engine = old_engine


def summarize():
    from scripts.report_long_tenor_search import sharpe,paired_bootstrap
    records,annual,returns = [],[],{}
    folders = [critic.OUT/'365D/replays/fixed50_hold_control']
    folders += [critic.OUT/'365D/replays'/(name+'_greedy') for name in MODES.values()]
    folders += [OUT/'365D/replays'/name for mode in MODES for name in [mode,mode+'_hold']]
    for folder in folders:
        status = json.loads((folder/'status.json').read_text())
        r = pd.read_parquet(folder/'daily_nav.parquet').excess_return.iloc[1:]
        assert np.isclose(sharpe(r),status['annualized_excess_sharpe'])
        returns[status['case']] = r
        status.update(source_folder=str(folder),recent_2024_partial2025_sharpe=sharpe(r.loc['2024-01-01':]))
        status.setdefault('restore_fills',0)
        records.append(status)
        annual.append(pd.read_csv(folder/'annual.csv').assign(case=status['case']))
    frame = pd.DataFrame(records)
    frame.to_csv(REPORT/'comparison.csv',index=False)
    pd.concat(annual).to_csv(REPORT/'annual.csv',index=False)
    comparisons = []
    for mode in MODES:
        for control in ['fixed50_hold_control',mode+'_hold']:
            lo,hi = paired_bootstrap(returns[mode],returns[control])
            comparisons.append(dict(case=mode,control=control,sharpe_gain=sharpe(returns[mode])-sharpe(returns[control]),
                                    unadjusted_gain_95_low=lo,unadjusted_gain_95_high=hi))
    pd.DataFrame(comparisons).to_csv(REPORT/'paired_comparison.csv',index=False)
    notes = [json.loads(p.read_text()) for p in (OUT/'models').glob('*/*.json')]
    assert all(n['last_entry_label'] is None or pd.Timestamp(n['last_entry_label'])<pd.Timestamp(n['fit']) for n in notes)
    base.dump(REPORT/'audit.json',dict(past_only_entry_labels=True,all529_proposals_explicit=True,
        costs_and_capital_limits_unchanged=True,one_worker=True,cpu_core_fraction=.5,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,no_fresh_holdout=True,
        training_shadow='Same fractional local cohort rewards; actual evaluation shared-cash integer book',
        cold_start='Keep original fixed50 entry until16 completed first-week labels and fitted parent Q are available'))
    paths = [Path(__file__),Path('tests/test_year_option_entry_rl.py'),Path(critic.__file__),Path(previous.__file__),
             Path(previous.engine.__file__),Path('scripts/run_signal_discovery_guarded.py'),previous.OUT/'365D/episodes.joblib']
    paths += [p for p in OUT.rglob('*') if p.is_file() and p.suffix in ['.parquet','.csv','.json','.joblib']]
    paths += [p for parent in MODES.values() for p in (critic.OUT/'365D/models'/parent).glob('*') if p.is_file()]
    base.dump(REPORT/'source_manifest.json',{str(p):base.digest(p) for p in paths})
    columns = ['case','annualized_excess_sharpe','entries','trims','exits','restore_fills','recent_2024_partial2025_sharpe']
    (REPORT/'findings.txt').write_text(
        'NATIVE ONE-YEAR BOOTSTRAPPED ENTRY Q\n\n'
        'After management-only variants failed to improve the fixed hold baseline, allow RL to choose no entry or the original50 allocation using native long-tenor inputs. Candidate pairs/dates/directions remain the529 short-policy proposals; this is not a complete new one-year opportunity universe.\n'
        'Entry Q target is observed first5-session block utility including .65 training opening commissions and unchanged spreads, plus discounted optimal weekly-management Q. Parent full-year64-fit critic is trained only on completed past weekly transitions. Entry label needs only its first weekly decision completed strictly before the refit, not future maturity payoff. Quarterly48-month entry fits;20 trees, depth3, leaf6. Fewer than16 completed entry labels or no parent critic keeps the original fixed50 entry.\n'
        'Entry state is signal-close long-tenor carry forecast, slow carry/availability, long-versus-short term gap, native signed IV gap/level, return correlation and VIX. Greedy entry compares learned opening utility to the zero-reward no-position action. Greedy management uses the matching parent critic. No extra confidence rule, leverage, or changes to real .03 commission,0.1-vol-point option penalties and1bp stock costs.\n'
        'Replay learned entries both with weekly management and fixed holding to separate entry selection from management. Entry proposal scores are identical within each pair of replays; realized fills can differ through shared risk limits and account equity. Local fractional shadow training approximates integer global book interactions.\n'
        'This follow-up was chosen after earlier failures. All history is explored; confidence intervals are unadjusted for search, and2024–partial2025 is not an untouched holdout.\n\n'
        +frame[columns].to_string(index=False)+'\n\n'+pd.DataFrame(comparisons).to_string(index=False)+'\n')
    print(frame[columns].to_string(index=False),flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase',choices=['test','fit','replay','summarize','sweep'],required=True)
    args = p.parse_args()
    if args.phase=='test':
        suite = unittest.defaultTestLoader.loadTestsFromNames(['tests.test_year_option_entry_rl'])
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
        return
    configure()
    if args.phase in ['fit','sweep']:
        fit()
    if args.phase in ['replay','sweep']:
        replay()
    if args.phase in ['summarize','sweep']:
        summarize()
    configure()


if __name__=='__main__':
    main()
