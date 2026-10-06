"""Original pair-date coverage applied equally to every standardized wing.

This is a retrospective coverage-control diagnostic on already inspected data,
not a new pair selector or an untouched test. No wing-specific sizing protocol.
"""
from dataclasses import replace
import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd

from scripts import run_same_structure_smile as common
from scripts import uniform_smile_engine as engine
from scripts.uniform_smile_risk import book_risk
from scripts.run_uniform_smile import verify, UNIFORM_CORPORATE_EVENTS
from scripts.run_atm65_ablation import read_actions, write_json
from scripts.run_contract_walkforward import digest
from scripts.curve_trading.config import config
from scripts.run_aligned_contract_learning import mark_book
from scripts.select_atm_contracts import load_calendar
from scripts.run_broad_contract_replay import ROOT

OUT = Path('artifacts/contracts/same_structure_calendar')
REPORT = Path('reports/same_structure_calendar')


def formation_dates():
    defs_path = Path('artifacts/contracts/tail_put25_pooled/inputs.json')
    sources = {str(defs_path): digest(defs_path)}
    defs = json.loads(defs_path.read_text())['definitions']
    dates = {}
    for d in defs:
        path = Path('artifacts/contracts/tail_put25_strategy') / d['pair'] / 'rule/actions.csv'
        if path.exists():
            dates[d['pair']] = read_actions(path).index
            sources[str(path)] = digest(path)
        else:
            dates[d['pair']] = pd.DatetimeIndex([])
    return dates, sources


def filter_calendar(rows, calendars):
    permitted = np.zeros(len(rows), bool)
    for pair, idx in rows.groupby('pair', sort=False).groups.items():
        idx = np.asarray(idx)
        permitted[idx] = pd.DatetimeIndex(rows.loc[idx, 'date']).isin(calendars[pair])
    return permitted


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['audit', 'replay', 'all'], default='all')
    args = p.parse_args()
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    cal = load_calendar(ROOT)
    cal = cal[(cal >= '2010-01-01') & (cal <= '2025-08-29')]
    calendars, sources = formation_dates()
    for path in [Path(__file__), Path('scripts/run_same_structure_smile.py'), Path('scripts/same_structure_smile.py')]:
        sources[str(path)] = digest(path)
    for path in (common.OUT / 'models').rglob('*.joblib'):
        sources[str(path)] = digest(path)
    for wing in common.WINGS:
        path = common.OUT / f'{wing}_rows.parquet'
        sources[str(path)] = digest(path)
    protocol = (
        'One identical IV-spread/oriented27-feature/monthly pooled sizing model '
        'structure across ATM/P25/P75, preserving all settings and all540 models '
        'from same_structure_smile. Apply one identical calendar predicate to '
        'all three wings: date belongs to the original rule-action DATE INDEX '
        'for that pair. Ignore original action values, allocations and realized '
        'profitability; this retains original formation/observable-pair coverage '
        'without copying the original wing-specific signals. All345 roster '
        'pairs remain; missing original calendar means no admitted date in this '
        'coverage-control arm and is explicitly counted. No retraining. '
        'Report broad and original-calendar results separately; do not mix the '
        'best sizing method or calendar independently by wing. Actual0.03 '
        'commissions and all common execution, shared-book accounting, risk '
        'budgets and corporate exits unchanged. Fixed100% control receives '
        'the same qualified proposals. Old formation/calendar membership is '
        'retrospective dataset conditioning, not a newly verified causal pair '
        'selection rule. This is an audit-driven diagnostic after implementation '
        'of the common model, on already inspected 2018-Aug2025 history. '
        'One worker,50% of one core,one BLAS/Arrow thread,3GiB RSS stop.'
    )
    write_json(OUT / 'inputs.json', dict(protocol=protocol, same_structure_across_wings=True,
        no_retraining=True, source_models=str(common.OUT / 'models'), pairs=345,
        calendar_predicate='signal date in original rule action-date index for pair; original code values ignored'))
    base = common.allocations('fixed_100', cal)
    permitted = filter_calendar(base, calendars)
    coverage = []
    for wing, g in base.groupby('wing'):
        okay = permitted[g.index]
        coverage.append(dict(wing=wing, broad_proposals=len(g), original_calendar_proposals=int(okay.sum()),
            extra_dates_removed=int((~okay).sum()), original_calendar_pairs=int(g.loc[okay, 'pair'].nunique())))
    pd.DataFrame(coverage).to_csv(REPORT / 'calendar_coverage.csv', index=False)
    frozen_path = Path('artifacts/contracts/joint_smile/training_rows.parquet')
    sources[str(frozen_path)] = digest(frozen_path)
    old = pd.read_parquet(frozen_path)
    keys = ['date', 'pair', 'wing', 'direction']
    baseline = old.loc[old.wing.eq('P25'), keys]
    match = base.loc[base.wing.eq('P25') & permitted, keys]
    merged = match.merge(baseline, on=keys, how='outer', indicator=True)
    merged.to_csv(REPORT / 'p25_proposal_identity.csv', index=False)
    write_json(REPORT / 'calendar_audit.json', dict(
        pairs=345, pairs_with_original_calendar=sum(bool(len(d)) for d in calendars.values()),
        pairs_without_original_calendar=sum(not len(d) for d in calendars.values()),
        old_p25_proposals=len(baseline), common_p25_original_calendar_proposals=len(match),
        matching_p25_proposals=int(merged._merge.eq('both').sum()),
        added_p25_proposals=int(merged._merge.eq('left_only').sum()),
        omitted_p25_proposals=int(merged._merge.eq('right_only').sum()),
        original_actions_values_not_used=True, same_calendar_function_for_every_wing=True))
    print(pd.DataFrame(coverage).to_string(index=False), flush=True)
    if args.phase == 'audit':
        write_json(OUT / 'sources.json', sources)
        return
    market = common.Market()
    evaluated = cal[cal >= '2018-01-01']
    arms = [(variant, None) for variant in ['fixed_100', 'floor', 'flex']]
    arms += [(variant, wing) for variant in ['fixed_100', 'floor', 'flex'] for wing in common.WINGS]
    results, annual = [], []
    identity = None
    for variant, wing in arms:
        name = variant if wing is None else f'{variant}_{wing}'
        folder = OUT / 'replays' / name
        folder.mkdir(parents=True, exist_ok=True)
        rows = common.allocations(variant, cal)
        keep = filter_calendar(rows, calendars)
        rows = rows.loc[keep].copy()
        if wing is not None:
            rows = rows.loc[rows.wing.eq(wing)].copy()
        rows = rows.reset_index(drop=True)
        if wing is None:
            if identity is None:
                identity = rows[keys + ['optionid_a', 'optionid_b']]
            else:
                pd.testing.assert_frame_equal(identity, rows[keys + ['optionid_a', 'optionid_b']])
        rows.to_parquet(folder / 'scored_allocations.parquet', index=False)
        signals = rows.loc[rows.base_level.gt(0)].copy()
        multiplier = 3 if wing is None else 1
        cfg = replace(config(), option_commission=.03, capital=1_000_000*multiplier*345,
                      max_gross_vega=20_000*multiplier*345)
        market.variant, market.started, market.reported_year = 'original_calendar_' + name, time.monotonic(), None
        print(f'Original calendar {name}: {len(signals):,} proposals', flush=True)
        cached_result = (folder / 'status.json').exists()
        if cached_result:
            nav = pd.read_parquet(folder / 'daily_nav.parquet')
            trades = pd.read_csv(folder / 'trades.csv', parse_dates=['date'])
            decisions = pd.read_csv(folder / 'decisions.csv', parse_dates=['signal_date', 'entry_date'])
        else:
            saved = engine.book_risk
            try:
                engine.book_risk = book_risk
                nav, trades, decisions, _ = engine.simulate(market, signals, evaluated, cfg,
                    1_000_000*multiplier, wing_capital=1_000_000, corporate_events=UNIFORM_CORPORATE_EVENTS)
            finally:
                engine.book_risk = saved
            mark_book(nav, market.rates, cfg.capital)
        audit = verify(nav, trades, decisions, signals, cfg)
        if not cached_result:
            nav.to_parquet(folder / 'daily_nav.parquet')
            trades.to_csv(folder / 'trades.csv', index=False)
            decisions.to_csv(folder / 'decisions.csv', index=False)
        result = dict(model=name, status='complete', proposals=len(signals),
            net_excess_pnl=float(nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1]), **audit)
        write_json(folder / 'status.json', result)
        results.append(result)
        pd.DataFrame(results).to_csv(REPORT / 'comparison.csv', index=False)
        for year, g in nav.groupby(nav.index.year):
            returns = g.excess_return.iloc[1:] if year == 2018 else g.excess_return
            annual.append(dict(model=name, year=int(year), annualized_excess_sharpe=
                float(np.sqrt(252)*returns.mean()/returns.std()) if returns.std()>1e-12 else None))
        print(f'{name}: Sharpe {audit["annualized_excess_sharpe"]:.6f}, {audit["entries"]} entries', flush=True)
    pd.DataFrame(annual).to_csv(REPORT / 'annual.csv', index=False)
    pd.DataFrame(annual).pivot(index='year', columns='model', values='annualized_excess_sharpe').to_csv(REPORT / 'annual_comparison.csv')
    uplift = []
    for suffix in ['', '_ATM', '_P25', '_P75']:
        baseline = pd.read_parquet(OUT / 'replays' / ('fixed_100' + suffix) / 'daily_nav.parquet').excess_return.iloc[1:].to_numpy()
        for variant in ['floor', 'flex']:
            candidate = pd.read_parquet(OUT / 'replays' / (variant + suffix) / 'daily_nav.parquet').excess_return.iloc[1:].to_numpy()
            def ratio(x):
                return float(np.sqrt(252)*x.mean()/x.std(ddof=1)) if x.std(ddof=1)>1e-12 else 0.
            rng = np.random.default_rng(2026)
            n, block, draws = len(candidate), 60, []
            for _ in range(1000):
                starts = rng.integers(0, n-block+1, size=int(np.ceil(n/block)))
                idx = (starts[:, None]+np.arange(block)).ravel()[:n]
                draws.append(ratio(candidate[idx])-ratio(baseline[idx]))
            uplift.append(dict(candidate=variant+suffix, control='fixed_100'+suffix,
                sharpe_difference=ratio(candidate)-ratio(baseline),
                ci_lower=float(np.quantile(draws, .025)), ci_upper=float(np.quantile(draws, .975))))
    pd.DataFrame(uplift).to_csv(REPORT / 'sizing_uplift.csv', index=False)
    for path, value in sources.items():
        assert digest(path) == value, f'Frozen model or input changed: {path}'
    write_json(OUT / 'sources.json', sources)
    audit = dict(status='complete', identical_structure_across_all_wings=True,
        no_retraining=True, fixed_common_model_and_gates=True, original_date_index_applied_identically=True,
        original_action_values_not_used=True, pairs=345, retrospective_coverage_diagnostic=True,
        source_files_verified=len(sources), elapsed_seconds=time.monotonic()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_json(REPORT / 'audit.json', audit)
    (REPORT / 'findings.txt').write_text(pd.DataFrame(results).to_string(index=False)+'\n\n'+protocol+'\n\n'+json.dumps(audit, indent=2)+'\n')
    paths = [Path(__file__), *[p for root in [OUT, REPORT] for p in root.rglob('*') if p.is_file() and p.name!='output_hashes.json']]
    write_json(REPORT / 'output_hashes.json', {str(p): digest(p) for p in paths})


if __name__ == '__main__':
    main()
