"""Past-prefix episodic Sharpe policy search for native one-year sizing."""
import argparse
import json
from pathlib import Path
import resource
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from scripts import run_year_option_entry_rl as entry

base,previous = entry.base,entry.previous
OUT = Path('artifacts/contracts/year_option_sharpe_policy/P25')
REPORT = Path('reports/year_option_sharpe_policy')
VARIANTS = {'native_2_48M':(2,48,False),'native_3_48M':(3,48,False),
            'native_3_24M':(3,24,False),'pair_native_3_48M':(3,48,True)}
LEVELS = [2,3,4]


def configure():
    entry.configure()
    base.OUT,base.REPORT = OUT,REPORT
    OUT.mkdir(parents=True,exist_ok=True)
    REPORT.mkdir(parents=True,exist_ok=True)
    base.dump(OUT/'design.json',dict(variants=VARIANTS,levels=LEVELS,
        algorithm='Model-based episodic contextual policy search; not sequential fitted-Q or a neural RL algorithm',
        objective='Daily portfolio Sharpe of causally observed past cohort-prefix returns',
        tree='Native entry-state partition; minimum12 cohorts/leaf; maximum2 or3 states',
        warmup='At least16 cohorts with20 observed holding sessions, and252 historical daily sessions',
        management='Hold to original leg expiries; no elective exits or trims',
        entry_rules='Same529 pair/date/direction/contract proposals, sizes50/75/100%; no new signal boundary',
        costs_and_risk_limits='Original execution and global/pair/wing limits unchanged',
        selection='Adaptive follow-up after fitted-Q failures; no fresh holdout'))


def state(ep,pairs,pair_features):
    value = ep['entry_features']
    if pair_features:
        value = np.r_[value,[float(ep['pair']==p) for p in pairs]]
    return np.asarray(value,float)


def observed_prefixes(episodes,cutoff,months):
    start = cutoff-pd.DateOffset(months=months)
    result = []
    for ep in episodes:
        if not start<=ep['date']<cutoff:
            continue
        take = np.flatnonzero(ep['dates']<cutoff)
        if len(take)>=21:
            result.append((ep,take))
    return result


def select_policy(book,grid):
    scores = base.daily_sharpes(book@grid.T)
    # Uniform leverage has the same shadow Sharpe. Break numerical ties in
    # favor of the smallest allocation rather than invent a leverage benefit.
    tied = np.flatnonzero(scores>=np.max(scores)-1e-10)
    selected = int(tied[np.argmin(grid[tied].sum(axis=1))])
    return grid[selected].astype(int),float(scores[selected])


def fit():
    packed = joblib.load(previous.OUT/'365D/episodes.joblib')
    episodes,cal = packed['episodes'],packed['calendar']
    rows = pd.read_parquet(previous.full.OUT/'P25/365D/signal_rows.parquet')
    pairs = sorted(rows.pair.unique())
    for name,(leaf_count,months,pair_features) in VARIANTS.items():
        for cutoff in pd.date_range('2018-01-01',cal[-1],freq='QS-JAN'):
            path = OUT/'models'/name/f'{cutoff.date()}.joblib'
            if path.exists():
                continue
            prefix = observed_prefixes(episodes,cutoff,months)
            history = cal[(cal>=cutoff-pd.DateOffset(months=months))&(cal<cutoff)]
            note = dict(fit=str(cutoff.date()),variant=name,prefix_cohorts=len(prefix),
                        history_sessions=len(history),last_observed_reward=str(max(ep['dates'][take[-1]] for ep,take in prefix).date()) if prefix else None)
            policy = None
            if len(prefix)>=16 and len(history)>=252:
                x = np.stack([state(ep,pairs,pair_features) for ep,_ in prefix])
                # Cohort targets partition states; allocation is learned from
                # daily portfolio Sharpe, not from ranking this label alone.
                y = np.array([ep['reward_training'][take].sum()/ep['entry_vega']*252/(len(take)-1) for ep,take in prefix])
                low,high = np.quantile(y,[.01,.99])
                tree = DecisionTreeRegressor(max_leaf_nodes=leaf_count,min_samples_leaf=12,random_state=2026)
                tree.fit(x,np.clip(y,low,high))
                leaves = np.unique(tree.apply(x))
                states = np.searchsorted(leaves,tree.apply(x))
                book = np.zeros((len(history),len(leaves)))
                for (ep,take),column in zip(prefix,states):
                    dates = ep['dates'][take]
                    idx = history.get_indexer(dates)
                    assert (idx>=0).all()
                    np.add.at(book[:,column],idx,ep['reward_training'][take]/ep['original_level'])
                grid = base.action_grid(leaves,LEVELS)
                choices,score = select_policy(book,grid)
                fixed_score = float(base.daily_sharpes((book.sum(axis=1)*2)[:,None])[0])
                assert score>=fixed_score-1e-9
                policy = dict(tree=tree,leaves=leaves,choices=choices,pairs=pairs,pair_features=pair_features)
                note.update(shadow_training_sharpe=score,shadow_fixed50_sharpe=fixed_score,choices=choices.tolist(),states=len(leaves))
            note['fitted'] = policy is not None
            path.parent.mkdir(parents=True,exist_ok=True)
            joblib.dump(policy,path)
            base.dump(path.with_suffix('.json'),note)
            print(f'{name} {cutoff.date()}: {len(prefix)} observed prefixes, {note.get("choices",[2])}',flush=True)


def score_row(row,policy):
    if policy is None or not row.native_features_available:
        return 2
    ep = dict(entry_features=row[entry.FEATURES].to_numpy(float),pair=row.pair)
    x = state(ep,policy['pairs'],policy['pair_features'])
    leaf = policy['tree'].apply(x[None,:])[0]
    return int(policy['choices'][np.searchsorted(policy['leaves'],leaf)])


def replay():
    rows = pd.read_parquet(previous.full.OUT/'P25/365D/signal_rows.parquet')
    # Reuse immutable50/100 controls;75 has not previously been executed.
    base.save_replay(365,'fixed75_hold',rows.assign(learned_level=3))
    for name in VARIANTS:
        schedule = {pd.Timestamp(p.stem):joblib.load(p) for p in sorted((OUT/'models'/name).glob('*.joblib'))}
        scored = rows.copy().assign(learned_level=2)
        for index,row in rows.iterrows():
            policy = schedule[max(d for d in schedule if d<=row.date)]
            scored.loc[index,'learned_level'] = score_row(row,policy)
        assert scored.learned_level.isin(LEVELS).all()
        base.save_replay(365,name,scored)


def summarize():
    from scripts.report_long_tenor_search import sharpe,paired_bootstrap
    folders = [Path('artifacts/contracts/long_tenor_policy_search/P25/365D/replays')/name
               for name in ['fixed_50_hold','fixed_100_hold']]
    folders += [OUT/'365D/replays'/name for name in ['fixed75_hold',*VARIANTS]]
    records,annual,returns = [],[],{}
    for folder in folders:
        status = json.loads((folder/'status.json').read_text())
        r = pd.read_parquet(folder/'daily_nav.parquet').excess_return.iloc[1:]
        assert np.isclose(sharpe(r),status['annualized_excess_sharpe'])
        returns[status['case']] = r
        status.update(source_folder=str(folder),recent_2024_partial2025_sharpe=sharpe(r.loc['2024-01-01':]))
        records.append(status)
        annual.append(pd.read_csv(folder/'annual.csv').assign(case=status['case']))
    frame = pd.DataFrame(records)
    frame.to_csv(REPORT/'comparison.csv',index=False)
    pd.concat(annual).to_csv(REPORT/'annual.csv',index=False)
    comparisons = []
    best_fixed = frame.loc[frame.case.isin(['fixed_50_hold','fixed75_hold','fixed_100_hold'])].sort_values('annualized_excess_sharpe').iloc[-1]['case']
    for name in VARIANTS:
        lo,hi = paired_bootstrap(returns[name],returns[best_fixed])
        comparisons.append(dict(case=name,control=best_fixed,sharpe_gain=sharpe(returns[name])-sharpe(returns[best_fixed]),
                                unadjusted_gain_95_low=lo,unadjusted_gain_95_high=hi))
    pd.DataFrame(comparisons).to_csv(REPORT/'paired_comparison.csv',index=False)
    notes = [json.loads(p.read_text()) for p in (OUT/'models').glob('*/*.json')]
    assert all(n['last_observed_reward'] is None or pd.Timestamp(n['last_observed_reward'])<pd.Timestamp(n['fit']) for n in notes)
    base.dump(REPORT/'audit.json',dict(past_prefix_rewards_only=True,all529_proposals_explicit=True,
        original_controls_unchanged=True,costs_and_capital_limits_unchanged=True,
        one_worker=True,cpu_core_fraction=.5,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        no_fresh_holdout=True,algorithm='Episodic contextual allocation policy optimization, not sequential Q learning',
        shadow='Fractional independent cohort sizes; actual evaluation includes integer matching and shared cash/risk'))
    paths = [Path(__file__),Path('tests/test_year_option_sharpe_policy.py'),Path('scripts/run_signal_discovery_guarded.py'),
             previous.OUT/'365D/episodes.joblib',previous.full.OUT/'P25/365D/signal_rows.parquet']
    paths += [p for p in OUT.rglob('*') if p.is_file() and p.suffix in ['.parquet','.csv','.json','.joblib']]
    base.dump(REPORT/'source_manifest.json',{str(p):base.digest(p) for p in paths})
    columns = ['case','annualized_excess_sharpe','entries','net_excess_pnl','recent_2024_partial2025_sharpe']
    (REPORT/'findings.txt').write_text(
        'PAST-PREFIX ONE-YEAR SHARPE ALLOCATION POLICY\n\n'
        'Adaptively tested after management-Q and native-entry-Q failures. Reuse all230 native label episodes without waiting for maturity: only daily rewards strictly before each quarterly refit are visible. Entry cohorts need20 observed holding sessions; at least16 cohorts and252 training-calendar sessions.\n'
        'An entry-state tree partitions native long-tenor signals into2/3 states; a discrete episodic policy search assigns50/75/100% allocations to maximize daily portfolio Sharpe on the observed training prefixes. Pair-aware arm adds one-hot pair identities from the known candidate universe. This is contextual episodic allocation policy optimization; not a claim of sequential Q learning or neural RL.\n'
        'Preserve all529 pair/date/direction proposals and hold each leg to its original expiry. No new confidence/entry gate, no elective exits. Quarterly24/48-month windows. Same real .03 option commissions,0.1-vol-point penalties,1bp stock hedges, floating cash and stock borrow, and global/pair/wing risk limits. Training .65 commissions unchanged. Compare against fixed50/75/100; higher sizing can affect integer eligibility and shared risk admission.\n'
        'Shadow portfolio sums fractional independent cohort histories. Final evaluation replays integer actual contracts and shared cash, financing, costs and constraints. All tested histories were previously explored; all variants reported and intervals are unadjusted for adaptive selection.\n\n'
        +frame[columns].to_string(index=False)+'\n\n'+pd.DataFrame(comparisons).to_string(index=False)+'\n')
    print(frame[columns].to_string(index=False),flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase',choices=['test','fit','replay','summarize','sweep'],required=True)
    args = p.parse_args()
    if args.phase=='test':
        suite = unittest.defaultTestLoader.loadTestsFromNames(['tests.test_year_option_sharpe_policy'])
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
