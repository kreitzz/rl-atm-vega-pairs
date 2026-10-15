"""Small, regularized contextual entry policies for selling ATM straddles.

Reuse exact integer-contract 50/75/100% counterfactual reward paths. Monthly
fits only see trades completed before the cutoff. Sparse shadow validation
screens procedures; actual funded validation chooses the retained procedure.
Every live proposal remains present, including those without training labels.
Run numerical phases through run_signal_discovery_guarded, one phase at a time.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import atm_rv_straddle_bandit as source
from scripts.run_capped_straddle_book import native_ladder
from paper_code.metrics import sharpe, period_metrics, paired_interval

CONFIG = Path('config/straddle_entry_refinement.json')
CFG = json.loads(CONFIG.read_text())
OUT, REPORT = Path(CFG['output_root']), Path(CFG['report_root'])
DATA = Path(CFG['source_root'])
ACTIONS = np.array(CFG['actions'], int)
FEATURES = {'core': ['residual_memory', 'iv', 'vix'],
            'economic': ['memory_yield', 'iv_rv_gap', 'iv', 'vix']}
PERIODS = [('validation', '2018-01-01', CFG['validation_end']),
           ('evaluation', '2022-01-01', CFG['evaluation_end']),
           ('full', '2018-01-01', CFG['evaluation_end'])]


def dump(path, value):
    source.base.dump(path, value)


def check_sources():
    paths = [Path(__file__), CONFIG, DATA/'training.parquet', DATA/'proposals.parquet',
             Path(source.__file__), Path(source.bandit.__file__), Path(source.base.__file__),
             Path(source.engine.__file__), Path('scripts/run_capped_straddle_book.py'),
             Path('paper_code/execution.json')]
    paths += sorted((DATA/'labels').glob('*.npz'))
    record = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    marker = OUT/'sources.json'
    if marker.exists():
        assert json.loads(marker.read_text()) == record, 'Changed sources: use a new experiment root'
    else:
        dump(marker, record)


def add_features(rows):
    rows = rows.copy()
    rows['memory_yield'] = -rows.residual_memory / rows.iv.clip(lower=.01)
    rows['iv_rv_gap'] = (rows.iv - rows.rv21) / rows.iv.clip(lower=.01)
    return rows


def prepare():
    rows = add_features(pd.read_parquet(DATA/'proposals.parquet', use_threads=False))
    meta = pd.read_parquet(DATA/'training.parquet', use_threads=False)
    features = rows[['date', 'secid', 'memory_yield', 'iv_rv_gap']]
    assert not features.duplicated(['date', 'secid']).any()
    meta = meta.merge(features, on=['date', 'secid'], how='left', validate='many_to_one')
    assert len(meta) == 1080 and int(meta.finite_label.sum()) == 518
    assert np.isfinite(meta.loc[meta.finite_label, FEATURES['economic']]).all().all()
    rows.to_parquet(OUT/'proposals.parquet', index=False)
    meta.to_parquet(OUT/'training.parquet', index=False)
    dump(REPORT/'protocol.json', dict(configuration=CFG, proposals=len(rows),
         training_samples=len(meta), completed_usable_labels=int(meta.finite_label.sum()),
         features=FEATURES, target='Completed 100% daily excess-dollar profit divided by its path L2 norm; clipped to [-3,3].',
         sizing_objective='Exhaustive leaf-action search on combined daily completed training profit paths; Sharpe objective.',
         warmup='50% until 64 completed examples; same for every new procedure.',
         state_regularization='Minimum 32 completed examples per leaf; at most 2-4 states or ridge-score terciles.',
         unchanged_costs_and_risk_limits=True))
    print(json.dumps(dict(phase='prepare', columns=list(rows), valid=int(meta.finite_label.sum()))), flush=True)


def load_labels():
    meta = pd.read_parquet(OUT/'training.parquet', use_threads=False)
    paths = np.zeros((len(meta), 3, source.bandit.PATH_LENGTH))
    for sid, idx in meta.groupby('secid').groups.items():
        with np.load(DATA/'labels'/f'{sid}.npz') as z:
            paths[idx] = z['paths'][meta.loc[idx, 'label_row'].to_numpy(int)]
    return meta, paths


def risk_adjusted_target(paths):
    full = paths[:, 2]
    scale = np.sqrt((full*full).sum(axis=1))
    return np.clip(np.divide(full.sum(axis=1), scale, out=np.zeros(len(full)), where=scale>1e-9), -3, 3)


def map_actions(book, occupancy):
    """Exhaust <=81 maps; numerical ties favor lower exposure, not spurious precision."""
    states = len(occupancy)
    maps = np.array(list(itertools.product(range(3), repeat=states)), int)
    columns = maps + 3*np.arange(states)[None, :]
    cov = np.atleast_2d(np.cov(book, rowvar=False, ddof=1))
    means = book.mean(axis=0)[columns].sum(axis=1)
    variance = cov[columns[:, :, None], columns[:, None, :]].sum(axis=(1, 2))
    scores = np.sqrt(252)*np.divide(means, np.sqrt(np.maximum(variance, 0)),
                                  out=np.zeros(len(maps)), where=variance>1e-15)
    best = np.flatnonzero(scores >= scores.max()-1e-8)
    winner = min(best, key=lambda i: (float((ACTIONS[maps[i]]*occupancy).sum()), int(i)))
    return ACTIONS[maps[winner]], float(scores[winner])


def fit_model(meta, paths, cutoff, spec, cal):
    cutoff = pd.Timestamp(cutoff)
    eligible = (meta.finite_label & meta.date.ge(cutoff-pd.DateOffset(months=spec['months']))
                & meta.date.lt(cutoff) & meta.end.lt(cutoff))
    idx = np.flatnonzero(eligible.to_numpy())
    rows = meta.iloc[idx]
    names = FEATURES[spec['features']]
    model = dict(cutoff=str(cutoff.date()), specification=spec, features=names,
                 training_samples=len(rows), choices=[CFG['warmup_action']],
                 last_completed=str(rows.end.max().date()) if len(rows) else None)
    if len(rows) < CFG['minimum_training_samples']:
        return model
    values = rows[names].to_numpy(float)
    target = risk_adjusted_target(paths[idx])
    if spec['kind'] == 'tree':
        leaves = source.bandit.reward_partition(values, target,
                     CFG['minimum_leaf_samples'], max_leaves=spec['leaves'])
        states = source.bandit.tree_contexts(values, leaves)
        model['tree'] = leaves
    else:
        center, scale = values.mean(axis=0), values.std(axis=0)
        scale = np.maximum(scale, .001)
        design = np.column_stack([np.ones(len(values)), (values-center)/scale])
        penalty = np.eye(design.shape[1])*spec['ridge']; penalty[0, 0] = 0
        beta = np.linalg.solve(design.T@design+penalty, design.T@target)
        score = design@beta
        cuts = np.quantile(score, [1/3, 2/3])
        states = np.searchsorted(cuts, score, side='left')
        model.update(center=center.tolist(), scale=scale.tolist(), beta=beta.tolist(), cuts=cuts.tolist())
    occupancy = np.bincount(states)
    first = cal.searchsorted(cutoff-pd.DateOffset(months=spec['months']))
    stop = cal.searchsorted(cutoff)
    book = np.zeros((stop-first, 3*len(occupancy)))
    days = rows.entry_position.to_numpy(int)[:, None] + np.arange(paths.shape[2])[None]-first
    valid = (days>=0) & (days<len(book))
    for state in range(len(occupancy)):
        selected = np.flatnonzero(states==state)
        for action in range(3):
            np.add.at(book[:, 3*state+action], days[selected][valid[selected]],
                      paths[idx[selected], action][valid[selected]])
    choices, score = map_actions(book, occupancy)
    model.update(choices=choices.tolist(), occupancy=occupancy.tolist(), training_shadow_sharpe=score)
    assert pd.Timestamp(model['last_completed']) < cutoff
    return model


def predict(model, rows):
    if len(model['choices']) == 1:
        return np.full(len(rows), model['choices'][0], int)
    values = rows[model['features']].to_numpy(float)
    finite = np.isfinite(values).all(axis=1)
    safe = np.where(np.isfinite(values), values, 0.)
    if 'tree' in model:
        states = source.bandit.tree_contexts(safe, model['tree'])
    else:
        design = np.column_stack([np.ones(len(rows)), (safe-model['center'])/model['scale']])
        states = np.searchsorted(model['cuts'], design@model['beta'], side='left')
    levels = np.asarray(model['choices'])[states]
    levels[~finite] = CFG['warmup_action']
    return levels


def allocations(name, end):
    spec = next(s for s in CFG['specifications'] if s['name']==name)
    rows = pd.read_parquet(OUT/'proposals.parquet', use_threads=False)
    full = rows.copy()
    rows = rows.loc[rows.date.le(end)].copy()
    meta, paths = load_labels(); cal = source.base.calendar(); parts = []
    for month, group in rows.groupby(rows.date.dt.to_period('M'), sort=True):
        marker = OUT/'models'/name/f'{month}.json'
        if marker.exists(): model = json.loads(marker.read_text())
        else:
            model = fit_model(meta, paths, month.start_time, spec, cal); dump(marker, model)
        group = group.copy(); group['learned_level'] = predict(model, group); parts.append(group)
    rows = pd.concat(parts).sort_index()
    full['learned_level'] = CFG['warmup_action']
    full.loc[rows.index, 'learned_level'] = rows.learned_level
    folder = OUT/'cases'/name/'allocations'; folder.mkdir(parents=True, exist_ok=True)
    full.to_parquet(folder/'tree18_test.parquet', index=False)
    return rows


def shadow(name):
    marker = REPORT/'shadow'/f'{name}.json'
    if marker.exists(): return
    rows = allocations(name, CFG['validation_end'])
    meta, paths = load_labels(); cal = source.base.calendar()
    lookup = rows.set_index(['date', 'secid']).learned_level
    levels = lookup.reindex(pd.MultiIndex.from_frame(meta[['date', 'secid']])).fillna(2).to_numpy(int)
    pnl = np.zeros(len(cal)); controls = {a: np.zeros(len(cal)) for a in ACTIONS}
    for i in np.flatnonzero((meta.finite_label & meta.date.le(CFG['validation_end'])).to_numpy()):
        days = int(meta.entry_position.iloc[i])+np.arange(paths.shape[2])
        valid = days<len(cal)
        pnl[days[valid]] += paths[i, np.flatnonzero(ACTIONS==levels[i])[0], valid]
        for a in ACTIONS: controls[a][days[valid]] += paths[i, np.flatnonzero(ACTIONS==a)[0], valid]
    dates = (cal>=pd.Timestamp('2018-01-01')) & (cal<=pd.Timestamp(CFG['validation_end']))
    row = dict(name=name, shadow_validation_sharpe=sharpe(pnl[dates]),
               fixed_shadow_sharpes={str(a): sharpe(v[dates]) for a, v in controls.items()},
               size_counts=rows.learned_level.value_counts().to_dict(),
               sparse_unfunded_shadow=True)
    dump(marker, row); print(json.dumps(row), flush=True)


def shortlist():
    rows = [json.loads((REPORT/'shadow'/f'{s["name"]}.json').read_text()) for s in CFG['specifications']]
    ranked = sorted(rows, key=lambda r: (-r['shadow_validation_sharpe'], r['name']))
    record = dict(candidates=ranked, finalists=[r['name'] for r in ranked[:CFG['shortlist_size']]],
                  selection_data='2018-2021 sparse shadow only; actual funded comparison follows')
    dump(REPORT/'shortlist.json', record); pd.DataFrame(rows).to_csv(REPORT/'shadow_candidates.csv', index=False)
    print(json.dumps(record), flush=True)


def book(name, months, later=False):
    # Use the proven native 25%-increment execution API; .75 is unsupported by
    # the old scaling hook. Never change existing experiments' output files.
    if later:
        assert json.loads((REPORT/'selection.json').read_text())['selected'] == name
        allocations(name, CFG['evaluation_end'])
    source.OUT = OUT/'cases'/name
    if not later:
        done = len(list((source.OUT/'replays/tree18/months').glob('????-??/checkpoint.joblib')))
        months = min(months, 48-done)
        if months <= 0:
            print(json.dumps(dict(phase='validation_complete', case=name)), flush=True)
            return
    original = source.engine.simulate
    def simulate(*args, **kwargs): return native_ladder(original, *args, **kwargs)
    source.engine.simulate = simulate
    try: source.replay(months)
    finally: source.engine.simulate = original


def funded_nav(name, end):
    from scripts.cheap_vega_runtime import mark_book
    folder = OUT/'cases'/name/'replays/tree18'
    parts = sorted(p.parent for p in (folder/'months').glob('????-??/checkpoint.joblib')
                   if p.parent.name <= end[:7])
    nav = pd.concat([pd.read_parquet(p/'daily_nav.parquet', use_threads=False) for p in parts])
    assert nav.index.max() >= source.base.calendar()[source.base.calendar()<=pd.Timestamp(end)][-1]
    rates = pd.read_parquet(source.base.MARKET_SOURCE/'rates.parquet', use_threads=False).rate
    mark_book(nav, rates, 10_000_000)
    d = pd.concat([pd.read_pickle(p/'decisions.pkl') for p in parts], ignore_index=True)
    return nav, d


def select():
    names = json.loads((REPORT/'shortlist.json').read_text())['finalists']
    rows = []
    for name in names:
        nav, d = funded_nav(name, CFG['validation_end'])
        rows.append(dict(name=name, **period_metrics(nav, d, '2018-01-01', CFG['validation_end'])))
    ranked = sorted(rows, key=lambda r: (-r['annualized_excess_sharpe'], r['name']))
    record = dict(selected=ranked[0]['name'], candidates=ranked, selection_data='Funded 2018-2021 only',
                  evaluation_status=CFG['evaluation_status'])
    path = REPORT/'selection.json'
    if path.exists(): assert json.loads(path.read_text()) == record
    else: dump(path, record)
    print(json.dumps(record), flush=True)


def report():
    selected = json.loads((REPORT/'selection.json').read_text())['selected']
    roots = {'fixed50': Path('artifacts/contracts/atm_rv_straddle_fixed50/replays/tree18'),
             'fixed75': Path('artifacts/contracts/straddle_capped_books/fixed75/replays/tree18'),
             'fixed100': Path('artifacts/contracts/atm_rv_straddle_full/replays/straddle100'),
             'previous_bandit': DATA/'replays/tree18',
             selected: OUT/'cases'/selected/'replays/tree18'}
    results, annual, intervals, navs, audit = [], [], [], {}, {}
    for name, folder in roots.items():
        nav = pd.read_parquet(folder/'daily_nav.parquet', use_threads=False)
        d = pd.read_csv(folder/'decisions.csv', parse_dates=['entry_date', 'signal_date'])
        navs[name] = nav
        reconciled = nav.option_pnl+nav.hedge_gross_pnl+nav.dividend_cash+nav.interest-nav.hedge_cost-nav.stock_borrow_cost
        error = float((reconciled-nav.pnl).abs().max()); assert error < 1e-5
        audit[name] = dict(independent_daily_ledger_error=error, decisions=len(d))
        for period, start, stop in PERIODS:
            results.append(dict(case=name, period=period, **period_metrics(nav, d, start, stop)))
        for y in range(2018, 2026):
            annual.append(dict(case=name, year=y, **period_metrics(nav, d, f'{y}-01-01', min(f'{y}-12-31', CFG['evaluation_end']))))
    for control in ['fixed50', 'fixed75', 'fixed100', 'previous_bandit']:
        intervals.append(dict(control=control, period='evaluation', **paired_interval(
            navs[selected].excess_return.loc['2022':], navs[control].excess_return.loc['2022':], block=63, draws=1000)))
    models = [json.loads(p.read_text()) for p in sorted((OUT/'models'/selected).glob('*.json'))]
    assert all(m['last_completed'] is None or pd.Timestamp(m['last_completed']) < pd.Timestamp(m['cutoff']) for m in models)
    allocations = pd.read_parquet(OUT/'cases'/selected/'allocations/tree18_test.parquet', use_threads=False)
    source_rows = pd.read_parquet(DATA/'proposals.parquet', use_threads=False)
    pd.testing.assert_frame_equal(allocations[source.bandit.EXECUTION].reset_index(drop=True),
                                  source_rows[source.bandit.EXECUTION].reset_index(drop=True))
    audit['selected_models'] = dict(monthly_fits=len(models), all_outcomes_complete_before_fit=True,
        latest_training_samples=models[-1]['training_samples'], latest_state_occupancy=models[-1].get('occupancy'),
        unchanged_daily_proposals=len(allocations), size_frequencies=allocations.learned_level.value_counts().to_dict())
    pd.DataFrame(results).to_csv(REPORT/'results.csv', index=False)
    pd.DataFrame(annual).to_csv(REPORT/'annual.csv', index=False)
    pd.DataFrame(intervals).to_csv(REPORT/'sharpe_intervals.csv', index=False)
    dump(REPORT/'audit.json', audit)
    table = pd.DataFrame(results)[['case', 'period', 'annualized_excess_sharpe', 'annualized_growth_relative_to_cash_pct', 'entries']]
    findings = ('SHORT-STRADDLE ENTRY BANDIT REFINEMENT\n\n'+table.to_string(index=False)+'\n\n'
        +'Selection: eight sparse-shadow validation candidates, two funded validation finalists; selected '+selected+'.\n'
        +'Same daily bottom-decile RV-minus-IV proposals, actual short call+put contracts, daily hedging, costs, and risk limits.\n'
        +'Actions are 50/75/100% total straddle entry vega. No cross-strategy capital Q learner is involved.\n'
        +'Warmup is 50%, rather than the previous learner\'s 100%; constant controls are essential to interpret the change.\n'
        +'These historical dates were already examined. Confidence intervals condition on the selected procedure and do not adjust for search.\n')
    (REPORT/'FINDINGS.txt').write_text(findings); print(findings, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=['test', 'prepare', 'shadow', 'shortlist', 'book', 'select', 'report'])
    parser.add_argument('--case', choices=[s['name'] for s in CFG['specifications']])
    parser.add_argument('--months', type=int, choices=range(1, 9), default=8)
    parser.add_argument('--later', action='store_true')
    args = parser.parse_args()
    if args.phase=='test':
        import unittest
        suite=unittest.defaultTestLoader.loadTestsFromNames(['tests.test_straddle_entry_refinement', 'tests.test_research_memory_guard'])
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful(): raise SystemExit(1)
        return
    OUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True); check_sources()
    if args.phase in ['shadow', 'book']:
        if not args.case: parser.error('--case required')
        if args.phase=='shadow': shadow(args.case)
        else: book(args.case, args.months, args.later)
    else: dict(prepare=prepare, shortlist=shortlist, select=select, report=report)[args.phase]()


if __name__=='__main__': main()
