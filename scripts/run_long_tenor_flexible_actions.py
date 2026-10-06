"""Full-universe learned abstention/sizing; explicit identical-skip controls."""
import argparse
import gc
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from scripts import run_long_tenor_full_universe as full

base = full.base
OUT = Path('artifacts/contracts/long_tenor_flexible_actions')
REPORT = Path('reports/long_tenor_flexible_actions')


def partition(x, y, min_leaf, weights=None):
    low, high = np.quantile(y, [.01, .99])
    return DecisionTreeRegressor(max_leaf_nodes=4, min_samples_leaf=min_leaf,
         random_state=2026).fit(x, np.clip(y, low, high), sample_weight=weights)


def flexible_sizes(episodes, positions, history_len):
    x = np.stack([e['entry_features'] for e in episodes])
    y = np.array([e['reward_training'].sum()/e['entry_vega'] for e in episodes])
    tree = partition(x, y, max(8, len(episodes)//4))
    leaves = np.unique(tree.apply(x))
    states = np.searchsorted(leaves, tree.apply(x))
    grid = base.action_grid(leaves, [0, 1, 2, 3, 4])
    # Aggregate before enumerating policies: tiny state matrix instead of
    # expanding every cohort/day across hundreds of action combinations.
    book = np.zeros((history_len, len(leaves)))
    original = np.zeros(history_len)
    for ep, state in zip(episodes, states):
        idx = positions[ep['positions']]
        np.add.at(book[:, state], idx, ep['reward_training']/ep['original_level'])
        np.add.at(original, idx, ep['reward_training'])
    returns = book @ grid.T
    scores = base.daily_sharpes(returns)
    # Zero exposure is a neutral training utility, not a nonzero-return claim.
    scores[(grid == 0).all(axis=1)] = 0.
    best = int(np.argmax(scores))
    return dict(tree=tree, leaves=leaves, choices=grid[best].astype(int)), dict(
        shadow_training_sharpe=float(scores[best]),
        shadow_original_sharpe=float(base.daily_sharpes(original[:, None])[0]),
        states=len(leaves), choices=grid[best].astype(int).tolist(),
        action_menu=[0, 1, 2, 3, 4], exact_policies_evaluated=len(grid))


def allocations(rows, schedule):
    result = rows.copy()
    for i, row in result.iterrows():
        model = base.model_at(schedule, row.date)['sizing']
        if model is None or not row.native_features_available:
            continue
        leaf = model['tree'].apply(row[base.ENTRY_FEATURES].to_numpy(float)[None, :])[0]
        result.loc[i, 'learned_level'] = model['choices'][np.searchsorted(model['leaves'], leaf)]
    assert result.learned_level.between(0, 4).all()
    assert len(result) == len(rows)
    return result


def setup(instrument):
    full.setup(instrument)
    base.OUT, base.REPORT = OUT/instrument, REPORT/instrument
    base.OUT.mkdir(parents=True, exist_ok=True)
    base.REPORT.mkdir(parents=True, exist_ok=True)
    base.protected()
    base.tree_for = partition
    base.optimize_sizes = flexible_sizes
    base.sized_rows = allocations


def inputs(tenor):
    origin = full.OUT/base.INSTRUMENT/f'{tenor}D'
    target = base.OUT/f'{tenor}D'
    target.mkdir(parents=True, exist_ok=True)
    for name in ['signal_rows.parquet','episodes.joblib','prepared.json']:
        if not (target/name).exists():
            os.link(origin/name, target/name)
    for case in ['fixed_50_hold','fixed_100_hold'] + (['transferred_hold'] if base.INSTRUMENT == 'C75' else []):
        folder = target/'replays'/case
        folder.mkdir(parents=True, exist_ok=True)
        for path in (origin/'replays'/case).iterdir():
            if not (folder/path.name).exists():
                os.link(path, folder/path.name)


def replay(tenor):
    schedule = base.load_schedule(tenor, 48)
    rows = pd.read_parquet(base.OUT/f'{tenor}D/signal_rows.parquet')
    sized = allocations(rows, schedule)
    rv = base.load_rv()
    cal = pd.read_parquet(base.source.OUT/f'P/{tenor}D/market/rates.parquet').index
    base.save_replay(tenor, 'flexible_size_48M', sized)
    base.save_replay(tenor, 'flexible_joint_48M', sized,
        base.management_callback(schedule, rv, cal, joint=True))
    for level in [2, 4]:
        control = sized.assign(learned_level=np.where(sized.learned_level.eq(0), 0, level))
        base.save_replay(tenor, f'same_learned_skip_fixed_{level*25}', control)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['sweep','fit','replay','summarize'], default='sweep')
    p.add_argument('--instrument', choices=['P25','C75'])
    p.add_argument('--tenor', type=int, choices=base.TENORS)
    args = p.parse_args()
    for instrument in ([args.instrument] if args.instrument else ['P25','C75']):
        setup(instrument)
        for tenor in ([args.tenor] if args.tenor else base.TENORS):
            inputs(tenor)
            if args.phase in ['sweep','fit']:
                base.fit(tenor, 48)
            if args.phase in ['sweep','replay']:
                replay(tenor)
        if args.phase in ['sweep','summarize']:
            base.summarize()
            base.dump(base.REPORT/'extension_method.json', dict(
                entry_actions=[0,.25,.5,.75,1.], management_actions=[0,.5,1.],
                maximum_leaves=4, learned_zero_is_explicit=True, external_entry_gates_added=False,
                all_original_proposals=529, identical_learned_skip_fixed_controls=[.5,1.],
                full_universe_labels_from=str(full.OUT/instrument),
                final_costs_and_risk_limits='Unchanged', source_sha256=base.digest(__file__)))
        base.protected()
        gc.collect()


if __name__ == '__main__':
    main()
