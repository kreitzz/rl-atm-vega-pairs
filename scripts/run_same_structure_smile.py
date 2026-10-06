"""Train and backtest one common model structure for ATM/P25/P75."""
from dataclasses import replace
import argparse
import gc
import json
from pathlib import Path
import resource
import time
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from scripts import same_structure_smile as policy
from scripts import uniform_smile_data as cached
from scripts import uniform_smile_engine as engine
from scripts.uniform_smile_risk import book_risk
from scripts.run_uniform_smile import UNIFORM_CORPORATE_EVENTS, aggregate, verify
from scripts.joint_smile_data import registry
from scripts.tail_put25_pooled import completed_mask
from scripts.curve_trading.config import config, RUNOFF
from scripts.select_atm_contracts import load_calendar
from scripts.run_broad_contract_replay import ROOT
from scripts.run_atm65_ablation import write_json
from scripts.run_contract_walkforward import digest
from scripts.run_aligned_contract_learning import mark_book
from scripts.run_selected_pair_backtest import scores

OUT = Path('artifacts/contracts/same_structure_smile')
REPORT = Path('reports/same_structure_smile')
WINGS = cached.WINGS
VARIANTS = ['fixed_50', 'fixed_75', 'fixed_100', 'floor', 'flex']
PROTOCOL = (
    'All345 roster pairs; independent wing policies using one identical P25-based '
    'architecture for ATM calls,25-delta puts,75-delta puts at approximately30D. '
    'Same 252-session IV-spread rank quintile entry/direction rule,27 oriented '
    'features and availability flags, beta0.85-1.15/correlation>=0.5 screen, '
    'original common pair formation/context calendar, positive91/365-day '
    'completed directional-profit gates with minimum21/63 observations. '
    'Same monthly pooled five-year completed-outcome training, max8-leaf tree, '
    'adaptive minleaf max(8,min(63,n//8)), direct-Sharpe CEM50x192 seed2026, '
    'full-size anchor, and global full-size fallback below16 pooled training '
    'rows. No per-pair validation, employee-only input or activation rule. '
    'Two declared common ladders: floor50/75/100% and flex0/25/50/75/100%. '
    'Each ladder applies unchanged to all three wings. Fixed50/75/100 controls '
    'receive identical qualified spread proposals. Flex can additionally skip '
    'on its allocation output; a same-skip full-size diagnostic isolates sizing. '
    'Training commission0.65 and actual0.03 percontract on everywing; mid plus '
    '0.1 vol-point perleg, stock hedge1bp, financing/borrow/dividends unchanged. '
    'Same $1m/20k per pair-wing and $3m/60k pair budgets; shared cash and net '
    'stock hedge, matched initial integer vega within10%, daily deltahedge, '
    'hold own expiry with previously audited causal corporate exits. '
    'Per-wing-only replays use identical scaled funding/risk settings. '
    'Reuse frozen standardized cached features and labels without any '
    'future-label filter on evaluation candidates. Only finite completed '
    'outcomes enter training. All proposals/decisions retained. '
    'Previously inspected history and retrospective roster/calendar conditioning '
    'are preserved; no new holdout. Not sequential inventory-trained RL. '
    'One worker,50% of one core,one BLAS/Arrow thread,3GiB RSS stop.'
)


class Market(cached.ReplayMarket):
    def rate(self, date):
        # Preserve the frozen study's files; its inherited progress callback
        # would otherwise write to the original uniform_smile directory.
        if self.reported_year != date.year:
            self.reported_year = date.year
            print(f'{self.variant}: {date.date()} ({time.monotonic() - self.started:.1f}s)', flush=True)
        return float(self.rates.loc[date])


def progress(stage, **details):
    record = dict(stage=stage, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    record.update(details)
    write_json(OUT / 'progress.json', record)


def prepare(sources):
    coverage = []
    for wing in WINGS:
        path = sources.track(cached.OUT / f'{wing}_rows.parquet')
        raw = pd.read_parquet(path)
        rows = policy.oriented_states(raw)
        rows.to_parquet(OUT / f'{wing}_rows.parquet', index=False)
        for pair in rows.pair.unique():
            sources.track(cached.OUT / 'pairs' / pair / f'{wing}_labels.npy')
        evaluated = rows.date.ge('2018-01-01') & rows.date.le('2025-06-30')
        coverage.append(dict(wing=wing, candidate_rows=len(rows),
            gated_training_opportunities=int(rows.signal_eligible.sum()),
            evaluation_observed_eligible_context_rows=int(evaluated.sum()),
            evaluation_rule_proposals=int((evaluated & rows.direction.ne(0)).sum()),
            evaluation_qualified_proposals=int((evaluated & rows.signal_eligible).sum())))
        print(f'{wing}: {len(rows):,} candidate rows, {coverage[-1]["evaluation_qualified_proposals"]:,} qualified test proposals', flush=True)
        del rows, raw
        gc.collect()
    pd.DataFrame(coverage).to_csv(REPORT / 'coverage.csv', index=False)
    # Quantify data differences from the strongest old P25 model rather than
    # silently treating the standardized cached dataset as an exact rerun.
    rows = pd.read_parquet(OUT / 'P25_rows.parquet')
    old = []
    for path in sorted(Path('artifacts/contracts/tail_put25_pooled').glob('*/training_rows.parquet')):
        old.append(pd.read_parquet(sources.track(path)))
    old = pd.concat(old, ignore_index=True)
    new = rows.loc[rows.direction.ne(0)].set_index(['date', 'pair'])
    old = old.set_index(['date', 'pair'])
    common = new.index.intersection(old.index)
    differences = []
    for col in policy.FEATURES:
        a, b = new.loc[common, col].to_numpy(float), old.loc[common, col].to_numpy(float)
        different = ~np.isclose(a, b, atol=1e-9, rtol=1e-9)
        differences.append(dict(feature=col, matched_rows=len(common),
            differing_rows=int(different.sum()), max_abs_difference=float(np.max(np.abs(a-b)))))
    pd.DataFrame(differences).to_csv(REPORT / 'p25_input_comparison.csv', index=False)


def train(cal):
    logs = []
    for wing in WINGS:
        meta = pd.read_parquet(OUT / f'{wing}_rows.parquet')
        for month in pd.period_range('2018-01-01', cal[-1] - pd.Timedelta(days=RUNOFF), freq='M'):
            cutoff, name = month.start_time, str(month.start_time.date())
            rows = meta.loc[meta.signal_eligible & meta.finite_label &
                completed_mask(meta.date, meta.end, cutoff)].reset_index(drop=True)
            tree, leaves, book = None, np.array([0]), None
            if len(rows) >= 16:
                tree = DecisionTreeRegressor(max_leaf_nodes=8,
                    min_samples_leaf=max(8, min(63, len(rows)//8)), random_state=2026).fit(rows[policy.FEATURES], rows.target)
                leaves = np.unique(tree.apply(rows[policy.FEATURES]))
                states = np.searchsorted(leaves, tree.apply(rows[policy.FEATURES]))
                first = int(rows.position.min()+1)
                last = int(cal.get_indexer(pd.DatetimeIndex(rows.end)).max())
                book = np.zeros((last-first+1, len(leaves)*5))
                for pair, idx in rows.groupby('pair', sort=False).groups.items():
                    idx = np.asarray(idx)
                    disk = np.load(cached.OUT / 'pairs' / pair / f'{wing}_labels.npy', mmap_mode='r')
                    raw = disk[rows.loc[idx, 'label_row'].to_numpy(int)]
                    labels = policy.directional_paths(raw, rows.loc[idx, 'direction'].to_numpy(int))
                    aggregate(book, states[idx], labels, rows.loc[idx, 'position'].to_numpy(int), first)
                    del disk, raw, labels
            for variant in policy.MENUS:
                folder = OUT / 'models' / wing / variant
                folder.mkdir(parents=True, exist_ok=True)
                stamp = folder / f'{name}.json'
                if stamp.exists():
                    logs.append(json.loads(stamp.read_text()))
                    continue
                choices, details = (np.array([4]), {}) if tree is None else policy.fit_sizes(book, cal[first:last+1], tree, leaves, variant)
                assert np.isin(choices, policy.MENUS[variant]).all()
                record = dict(wing=wing, variant=variant, fit=name, training_rows=len(rows),
                    training_pairs=int(rows.pair.nunique()), leaves=len(leaves), choices=choices.tolist(),
                    last_training_outcome=str(pd.Timestamp(rows.end.max()).date()) if len(rows) else None,
                    status='enabled' if tree is not None else 'common_global_fullsize_fallback', **details)
                assert not len(rows) or pd.Timestamp(rows.end.max()) < cutoff
                joblib.dump(dict(tree=tree, leaves=leaves, choices=choices, fit=name), folder / f'{name}.joblib')
                write_json(stamp, record)
                logs.append(record)
            if month.month in [1, 7]:
                print(f'Common fits {wing} {name}: {len(rows):,} completed gated rows', flush=True)
            progress('train', wing=wing, fit=name, models_complete=len(logs))
            del rows, tree, book
            gc.collect()
        del meta
        gc.collect()
    pd.DataFrame(logs).to_csv(REPORT / 'fits.csv', index=False)


def allocations(variant, cal):
    pieces = []
    for wing in WINGS:
        rows = pd.read_parquet(OUT / f'{wing}_rows.parquet')
        rows = rows.loc[rows.signal_eligible & rows.date.ge('2018-01-01') &
            rows.date.le(cal[-1] - pd.Timedelta(days=RUNOFF))].reset_index(drop=True)
        if variant.startswith('fixed_'):
            rows['base_level'] = {'fixed_50': 2, 'fixed_75': 3, 'fixed_100': 4}[variant]
        else:
            rows['base_level'] = 4
            for month, idx in rows.groupby(rows.date.dt.to_period('M')).groups.items():
                model = joblib.load(OUT / 'models' / wing / variant / f'{month.start_time.date()}.joblib')
                assert pd.Timestamp(model['fit']) <= rows.loc[idx, 'date'].min()
                rows.loc[idx, 'base_level'] = policy.predict(model, rows.loc[idx])
        pieces.append(rows)
    return pd.concat(pieces, ignore_index=True).sort_values(['date', 'pair', 'wing'], kind='stable').reset_index(drop=True)


def replay(cal):
    market = Market()
    evaluated = cal[cal >= '2018-01-01']
    arms = [(name, None) for name in VARIANTS] + [('flex_selected_full', None)]
    arms += [(variant, wing) for variant in policy.MENUS for wing in WINGS]
    for variant, wing in arms:
        name = variant if wing is None else f'{variant}_{wing}'
        folder = OUT / 'replays' / name
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / 'status.json').exists():
            continue
        rows = allocations('flex' if variant == 'flex_selected_full' else variant, cal)
        if variant == 'flex_selected_full':
            rows['base_level'] = np.where(rows.base_level.gt(0), 4, 0)
        if wing is not None:
            rows = rows.loc[rows.wing.eq(wing)].copy()
        rows.to_parquet(folder / 'scored_allocations.parquet', index=False)
        signals = rows.loc[rows.base_level.gt(0)].copy()
        multiplier = 3 if wing is None else 1
        cfg = replace(config(), option_commission=.03, capital=1_000_000*multiplier*345,
                      max_gross_vega=20_000*multiplier*345)
        market.variant, market.started, market.reported_year = name, time.monotonic(), None
        print(f'Replay {name}: {len(signals):,} proposals', flush=True)
        saved = engine.book_risk
        try:
            engine.book_risk = book_risk
            nav, trades, decisions, _ = engine.simulate(market, signals, evaluated, cfg,
                1_000_000*multiplier, wing_capital=1_000_000, corporate_events=UNIFORM_CORPORATE_EVENTS)
        finally:
            engine.book_risk = saved
        mark_book(nav, market.rates, cfg.capital)
        audit = verify(nav, trades, decisions, signals, cfg)
        nav.to_parquet(folder / 'daily_nav.parquet')
        trades.to_csv(folder / 'trades.csv', index=False)
        decisions.to_csv(folder / 'decisions.csv', index=False)
        write_json(folder / 'status.json', dict(status='complete', variant=variant, wing=wing,
            proposals=len(signals), qualified_candidates=len(rows), learned_skips=int(rows.base_level.eq(0).sum()),
            net_excess_pnl=float(nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1]),
            elapsed_seconds=time.monotonic()-market.started, **audit))
        progress('replay', arm=name, **audit)
        print(f'{name}: Sharpe {audit["annualized_excess_sharpe"]:.6f}, {audit["entries"]} entries', flush=True)


def summarize(sources):
    comparison, annual, counts, books = [], [], [], {}
    identity = None
    for folder in sorted((OUT / 'replays').iterdir()):
        record = json.loads((folder / 'status.json').read_text())
        assert record['status'] == 'complete'
        comparison.append(dict(model=folder.name, **record))
        nav = pd.read_parquet(folder / 'daily_nav.parquet')
        rows = pd.read_parquet(folder / 'scored_allocations.parquet')
        if record['wing'] is None:
            books[folder.name] = nav
            columns = ['date', 'pair', 'wing', 'direction', 'optionid_a', 'optionid_b']
            if identity is None:
                identity = rows[columns].reset_index(drop=True)
            else:
                pd.testing.assert_frame_equal(rows[columns].reset_index(drop=True), identity)
        for year, g in nav.groupby(nav.index.year):
            value = g.excess_return.iloc[1:] if year == 2018 else g.excess_return
            annual.append(dict(model=folder.name, year=int(year),
                annualized_excess_sharpe=float(np.sqrt(252)*value.mean()/value.std()) if value.std()>1e-12 else None))
        decisions = pd.read_csv(folder / 'decisions.csv')
        for wing, d in decisions.groupby('wing'):
            e = d.loc[d.status.eq('entered')]
            counts.append(dict(model=folder.name, wing=wing, entries=len(e),
                active_pairs=int(e.pair.nunique()), mean_allocation=float(e.level.mean()/4)))
    a = pd.read_parquet(OUT / 'replays/flex/scored_allocations.parquet')
    b = pd.read_parquet(OUT / 'replays/flex_selected_full/scored_allocations.parquet')
    pd.testing.assert_frame_equal(a.drop(columns='base_level'), b.drop(columns='base_level'))
    assert b.base_level.equals(pd.Series(np.where(a.base_level.gt(0), 4, 0), index=a.index, name='base_level'))
    result = pd.DataFrame(comparison)
    result.to_csv(REPORT / 'comparison.csv', index=False)
    pd.DataFrame(annual).to_csv(REPORT / 'annual.csv', index=False)
    pd.DataFrame(annual).pivot(index='year', columns='model', values='annualized_excess_sharpe').to_csv(REPORT / 'annual_comparison.csv')
    pd.DataFrame(counts).to_csv(REPORT / 'wing_entries.csv', index=False)
    pairs = [('floor', control) for control in ['fixed_50', 'fixed_75', 'fixed_100']]
    pairs += [('flex', 'fixed_100'), ('flex', 'flex_selected_full')]
    uplift = []
    for candidate, control in pairs:
        a, b = books[candidate].excess_return.iloc[1:].to_numpy(), books[control].excess_return.iloc[1:].to_numpy()
        rng = np.random.default_rng(2026)
        n, block, draws = len(a), 60, []
        def ratio(x):
            return float(np.sqrt(252)*x.mean()/x.std(ddof=1)) if x.std(ddof=1)>1e-12 else 0.
        for _ in range(1000):
            starts = rng.integers(0, n-block+1, size=int(np.ceil(n/block)))
            idx = (starts[:, None]+np.arange(block)).ravel()[:n]
            draws.append(ratio(a[idx])-ratio(b[idx]))
        uplift.append(dict(candidate=candidate, control=control, sharpe_difference=ratio(a)-ratio(b),
            ci_lower=float(np.quantile(draws, .025)), ci_upper=float(np.quantile(draws, .975))))
    pd.DataFrame(uplift).to_csv(REPORT / 'sizing_uplift.csv', index=False)
    fits = pd.read_csv(REPORT / 'fits.csv')
    assert len(fits) == 540
    known = fits.last_training_outcome.notna()
    assert (pd.to_datetime(fits.loc[known, 'last_training_outcome']) < pd.to_datetime(fits.loc[known, 'fit'])).all()
    assert fits.groupby(['wing', 'variant']).size().eq(90).all()
    write_json(REPORT / 'structure_audit.json', dict(
        identical_entry_rules=True, identical_27_feature_definitions=True,
        identical_training_method=True, identical_monthly_schedule=True,
        identical_gates=True, identical_costs=True, identical_risk_limits=True,
        identical_action_ladders_within_each_variant=True, wing_specific_knobs=['contract_type_and_delta_bucket'],
        model_fits=len(fits), latest_training_labels_strictly_before_each_fit=True,
        same_signal_sizing_controls_verified=True, flex_same_skip_control_verified=True))
    print(result[['model', 'annualized_excess_sharpe', 'entries']].to_string(index=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['test', 'prepare', 'train', 'replay', 'summarize', 'all'], default='all')
    args = parser.parse_args()
    if args.phase == 'test':
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests', pattern='test_same_structure_smile.py'))
        if not result.wasSuccessful():
            raise SystemExit(1)
        return
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    sources = registry()
    sources.expected.update(json.loads(Path('reports/uniform_smile/output_hashes.json').read_text()))
    if (OUT / 'sources.json').exists():
        for path, value in json.loads((OUT / 'sources.json').read_text()).items():
            assert digest(path) == value, f'Frozen source changed: {path}'
            sources.track(path)
    code = [Path('scripts/same_structure_smile.py'), Path('scripts/run_same_structure_smile.py'),
            Path('tests/test_same_structure_smile.py'), Path('scripts/run_tail_put25_residual_rl.py'),
            Path('scripts/tail_put25_pooled.py'), Path('scripts/uniform_smile_data.py'),
            Path('scripts/uniform_smile_engine.py'), Path('scripts/uniform_smile_risk.py'),
            Path('scripts/run_uniform_smile.py')]
    for path in code:
        sources.track(path)
    for path in (cached.OUT / 'market').glob('*.parquet'):
        sources.track(path)
    cal = load_calendar(ROOT)
    cal = cal[(cal >= '2010-01-01') & (cal <= '2025-08-29')]
    write_json(OUT / 'inputs.json', dict(protocol=PROTOCOL, wings=WINGS, features=policy.FEATURES,
        training_commission=.65, actual_commission=.03, variants=VARIANTS, same_structure_across_wings=True))
    if args.phase in ['prepare', 'all']:
        prepare(sources)
        write_json(OUT / 'sources.json', sources)
    if args.phase in ['train', 'all']:
        train(cal)
    if args.phase in ['replay', 'all']:
        replay(cal)
    history_path = OUT / 'phase_history.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history.append(dict(phase=args.phase, elapsed_seconds=time.monotonic()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    write_json(history_path, history)
    if args.phase in ['replay', 'summarize', 'all']:
        result = summarize(sources)
        for path, value in sources.items():
            assert digest(path) == value, f'Frozen source changed: {path}'
        audit = dict(status='complete', common_structure_across_all_wings=True, pairs=345,
            monthly_models=540, variants=5, same_skip_diagnostic=1, separate_wing_replays=6,
            frozen_source_files=len(sources), elapsed_seconds=sum(x['elapsed_seconds'] for x in history),
            peak_rss_bytes=max(x['peak_rss_bytes'] for x in history), no_new_holdout=True)
        write_json(REPORT / 'audit.json', audit)
        (REPORT / 'findings.txt').write_text(result.to_string(index=False)+'\n\n'+PROTOCOL+'\n\n'+json.dumps(audit, indent=2)+'\n')
        progress('complete', **audit)
        paths = [*code, *[p for root in [OUT, REPORT] for p in root.rglob('*') if p.is_file() and p.name!='output_hashes.json']]
        write_json(REPORT / 'output_hashes.json', {str(p): digest(p) for p in paths})
    else:
        progress(args.phase+'_complete', elapsed_seconds=time.monotonic()-started)


if __name__ == '__main__':
    main()
