"""Verify, restore and reproduce the retained submission without a WRDS login.

Verification and packaging stream files using the standard library. Numerical
commands require the existing supervisor; neither downloads nor broad research
searches run implicitly. Run from the repository root.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'data/submission'
MANIFEST = BUNDLE / 'manifest.json'
FEATURE = Path('artifacts/contracts/p25_feature_search')
CAPITAL = Path('artifacts/contracts/p25_capital_sensitivity')
CASES = ['10m_learned', '10m_fixed50', '10m_fixed100']
FINAL_FILES = {'daily_nav.parquet', 'trades.csv', 'decisions.csv', 'annual.csv',
               'status.json', 'config.json', 'scored_allocations.parquet',
               'entries.csv', 'management_decisions.parquet',
               'management_log.json', 'management_log.csv'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def relative_path(value):
    p = Path(value)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError(f'Unsafe bundle path: {value}')
    return p


def package():
    paths = set()
    def add(pattern):
        paths.update(p for p in ROOT.glob(pattern) if p.is_file())
    for pattern in [
        str(FEATURE / 'models/variance_premium/*'),
        str(FEATURE / 'training_rows.parquet'), str(FEATURE / 'signal_rows.parquet'),
        str(FEATURE / 'replays/variance_premium/scored_allocations.parquet'),
        'artifacts/contracts/p25_signal_transfer/market/*',
        'artifacts/contracts/same_structure_smile/P25_rows.parquet',
        'artifacts/contracts/uniform_smile/market/rates.parquet',
        'artifacts/contracts/uniform_smile/pairs/*/P25_labels.npy',
        'artifacts/contracts/year_cheap_vega/models/*',
        'artifacts/contracts/year_cheap_vega/market_cache/*',
        'artifacts/contracts/year_cheap_vega/supplemental_quotes/*.parquet',
        'artifacts/contracts/year_cheap_vega/*.json',
        'artifacts/contracts/year_cheap_vega/*_rows.parquet',
        'artifacts/contracts/year_cheap_vega/name_signals.parquet',
        'artifacts/contracts/year_cheap_vega/transitions.bin',
        'artifacts/contracts/native_year_entry/return_reconciled_v2/market/rates.parquet',
        'artifacts/contracts/native_year_entry/return_reconciled_v2/market/dividends.parquet',
        'artifacts/contracts/native_year_entry/return_reconciled_v2/market/stocks.parquet',
        'artifacts/contracts/native_year_entry/return_reconciled_v2/name_signals.parquet',
        'artifacts/contracts/year_reversion_management/models/*',
        'artifacts/contracts/year_reversion_management/protected_sources.json',
        'artifacts/contracts/year_reversion_management/prepared.json',
    ]:
        add(pattern)
    for folder in [CAPITAL / 'replays',
                   Path('artifacts/contracts/year_reversion_management/replays'),
                   Path('artifacts/contracts/year_cheap_vega/replays')]:
        for p in (ROOT / folder).glob('*/*'):
            if p.is_file() and p.name in FINAL_FILES:
                paths.add(p)
    paper = json.loads((ROOT / 'paper/source_manifest.json').read_text())
    for name in paper['sources']:
        if name.startswith('artifacts/'):
            paths.add(ROOT / name)
    BUNDLE.mkdir(parents=True, exist_ok=True)
    records = []
    for source in sorted(paths):
        original = source.relative_to(ROOT)
        if not source.is_file():
            raise FileNotFoundError(original)
        target = BUNDLE / original
        target.parent.mkdir(parents=True, exist_ok=True)
        checksum = digest(source)
        if not target.exists() or digest(target) != checksum:
            shutil.copy2(source, target)
        records.append(dict(source=str(original), stored=str(target.relative_to(ROOT)),
                            bytes=source.stat().st_size, sha256=checksum))
    # A single SQLite cache is too large for an ordinary Git object. Compress it
    # as a streaming ZIP; extraction is optional and explicitly requested.
    source = ROOT / 'artifacts/contracts/year_cheap_vega/quotes.sqlite'
    target = BUNDLE / 'year_cheap_vega_quotes.zip'
    checksum = digest(source)
    if not target.exists():
        with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            archive.write(source, 'quotes.sqlite')
    else:
        with zipfile.ZipFile(target) as archive, archive.open('quotes.sqlite') as stream:
            h = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
            if h.hexdigest() != checksum:
                raise ValueError('Existing quote archive differs; preserve it before repackaging')
    if target.stat().st_size >= 96 * 1024**2:
        raise ValueError('Quote archive exceeds the submission object budget')
    records.append(dict(source=str(source.relative_to(ROOT)),
                        stored=str(target.relative_to(ROOT)), bytes=source.stat().st_size,
                        sha256=checksum, compression='zip', member='quotes.sqlite',
                        stored_sha256=digest(target), stored_bytes=target.stat().st_size))
    states = ROOT / 'data/p25_variance_premium_inputs.csv'
    checkpoint = ROOT / 'config/p25_submission.json'
    manifest = dict(version=1, created='2026-10-06', files=records,
                    input_csv=str(states.relative_to(ROOT)), input_sha256=digest(states),
                    checkpoint=str(checkpoint.relative_to(ROOT)),
                    checkpoint_sha256=digest(checkpoint),
                    scope='Retained P25 fitting/replay, capital controls, final one-year ledgers and cheap-vega replay cache',
                    excluded='Bulk WRDS raw histories and intermediate search caches; licensed data acquisition requires institutional access')
    MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Packaged {len(records)} cache files; '
          f'{sum(r.get("stored_bytes", r["bytes"]) for r in records) / 1024**2:.1f} MiB', flush=True)


def load_manifest():
    return json.loads(MANIFEST.read_text())


def verify():
    manifest = load_manifest()
    for record in manifest['files']:
        source = relative_path(record['source'])
        if source.parts[0] != 'artifacts':
            raise ValueError('Restored caches must remain under artifacts/')
        stored = ROOT / relative_path(record['stored'])
        if digest(stored) != record.get('stored_sha256', record['sha256']):
            raise ValueError(f'Hash mismatch: {stored.relative_to(ROOT)}')
        if stored.stat().st_size != record.get('stored_bytes', record['bytes']):
            raise ValueError(f'Size mismatch: {stored.relative_to(ROOT)}')
    for path_key, hash_key in [('input_csv', 'input_sha256'), ('checkpoint', 'checkpoint_sha256')]:
        if digest(ROOT / relative_path(manifest[path_key])) != manifest[hash_key]:
            raise ValueError(f'Hash mismatch: {manifest[path_key]}')
    with (ROOT / manifest['input_csv']).open(newline='') as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        if set(columns) & {'target', 'reward', 'finite_label', 'label_row', 'end'}:
            raise ValueError('Forward training labels must not enter the input CSV')
        count = sum(1 for _ in reader)
    if count != 36327:
        raise ValueError(f'Expected 36327 input rows, found {count}')
    checkpoint = json.loads((ROOT / manifest['checkpoint']).read_text())
    if len(checkpoint['features']) != 33 or not set(checkpoint['features']) <= set(columns):
        raise ValueError('Retained 33-state schema differs')
    saved = {r['case']: r for r in csv.DictReader((ROOT / 'reports/p25_capital_sensitivity/comparison.csv').open())}
    for case, expected in checkpoint['results'].items():
        if not math.isclose(float(saved[case]['annualized_excess_sharpe']), expected['sharpe'], abs_tol=1e-12):
            raise ValueError(f'Result changed: {case}')
    for name, wanted in json.loads((ROOT / 'paper/source_manifest.json').read_text())['sources'].items():
        path = BUNDLE / name if name.startswith('artifacts/') else ROOT / name
        if digest(path) != wanted:
            raise ValueError(f'Paper source changed: {name}')
    print(f'Verified {len(manifest["files"])} cache files, {count} input rows, '
          '33 states, retained comparisons and paper sources.', flush=True)


def restore():
    verify()
    count = 0
    # Git checkout resets mtimes. Restore the original stamps only on newly
    # created cache sources so the cheap-vega immutable-cache check remains
    # valid. Their contents were verified by SHA-256 above.
    stamp_path = BUNDLE / 'artifacts/contracts/year_cheap_vega/market_cache/metadata.json'
    stamps = json.loads(stamp_path.read_text())['sources'] if stamp_path.exists() else {}
    for record in load_manifest()['files']:
        target = ROOT / relative_path(record['source'])
        if target.is_symlink() or any(p.is_symlink() for p in target.parents if p != ROOT.parent):
            raise ValueError(f'Refuse symlink cache destination: {target}')
        if target.exists():
            if digest(target) != record['sha256']:
                raise ValueError(f'Existing research cache differs; refusing overwrite: {target.relative_to(ROOT)}')
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.submission-partial')
        try:
            if record.get('compression') == 'zip':
                with zipfile.ZipFile(ROOT / record['stored']) as archive, archive.open(record['member']) as stream, temporary.open('wb') as out:
                    shutil.copyfileobj(stream, out, length=1024 * 1024)
            else:
                shutil.copy2(ROOT / record['stored'], temporary)
            if digest(temporary) != record['sha256']:
                raise ValueError(f'Extracted cache differs: {target}')
            temporary.replace(target)
            stamp = stamps.get(record['source'])
            if stamp is not None:
                if target.stat().st_size != stamp['bytes']:
                    raise ValueError(f'Cache-stamp size differs: {target}')
                os.utime(target, ns=(target.stat().st_atime_ns, stamp['mtime_ns']))
            count += 1
        finally:
            temporary.unlink(missing_ok=True)
    print(f'Restored {count} missing caches; preserved existing research.', flush=True)


def require_supervisor():
    if not os.environ.get('RESEARCH_SUPERVISOR_PID'):
        raise RuntimeError('Run numerical commands through scripts.run_signal_discovery_guarded; see SUBMISSION.md')


def metrics():
    import pandas as pd
    from scripts.cheap_vega_runtime import metrics as calculate, annual_metrics
    saved = pd.read_csv(ROOT / 'reports/p25_capital_sensitivity/comparison.csv').set_index('case')
    annual = pd.read_csv(ROOT / 'reports/p25_capital_sensitivity/annual.csv')
    output = []
    roots = [(case, CAPITAL / 'replays' / case) for case in CASES]
    roots += [(p.name, p) for p in (BUNDLE / 'artifacts/contracts/year_cheap_vega/replays').iterdir() if p.is_dir()]
    for case, folder in roots:
        if not folder.is_absolute():
            folder = BUNDLE / folder
        nav = pd.read_parquet(folder / 'daily_nav.parquet')
        status = json.loads((folder / 'status.json').read_text())
        result = calculate(nav, status['funded_equity'])
        for key, value in result.items():
            wanted = float(status[key])
            if not math.isclose(value, wanted, rel_tol=1e-10, abs_tol=1e-7):
                raise ValueError(f'{case}: recomputed {key} differs: {value} vs {wanted}')
        if case in CASES:
            expected = annual.loc[annual.case.eq(case)].reset_index(drop=True)
            pd.testing.assert_frame_equal(pd.DataFrame(annual_metrics(nav, case, 10_000_000)), expected,
                                          check_dtype=False, rtol=1e-10, atol=1e-7)
            if not math.isclose(result['annualized_excess_sharpe'], saved.loc[case, 'annualized_excess_sharpe'], abs_tol=1e-12):
                raise ValueError(f'{case}: saved comparison differs')
        print(f'{case}: verified Sharpe {result["annualized_excess_sharpe"]:.6f}', flush=True)
        output.append(dict(case=case, **result))
    folder = ROOT / 'artifacts/submission'
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output).to_csv(folder / 'verified_metrics.csv', index=False)


def fit_check():
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.tree import DecisionTreeRegressor
    from scripts import run_p25_action_windows as base
    from scripts.p25_action_windows import fit_menu, score_model
    from scripts.p25_recency import completed_months
    checkpoint = json.loads((ROOT / 'config/p25_submission.json').read_text())
    features = checkpoint['features']
    meta = pd.read_parquet(BUNDLE / FEATURE / 'training_rows.parquet')
    cutoff = pd.Timestamp('2018-01-01')
    rows = meta.loc[meta.signal_eligible & meta.finite_label & completed_months(meta.date, meta.end, cutoff, 18)].reset_index(drop=True)
    tree = DecisionTreeRegressor(max_leaf_nodes=8, min_samples_leaf=max(8, min(63, len(rows) // 8)), random_state=2026).fit(rows[features], rows.target)
    leaves = np.unique(tree.apply(rows[features]))
    states = np.searchsorted(leaves, tree.apply(rows[features]))
    cal = pd.read_parquet(BUNDLE / 'artifacts/contracts/uniform_smile/market/rates.parquet').index
    first = int(rows.position.min() + 1)
    last = int(cal.get_indexer(pd.DatetimeIndex(rows.end)).max())
    book = np.zeros((last - first + 1, len(leaves) * 5))
    for pair, indices in rows.groupby('pair', sort=False).groups.items():
        idx = np.asarray(indices)
        disk = np.load(BUNDLE / 'artifacts/contracts/uniform_smile/pairs' / pair / 'P25_labels.npy', mmap_mode='r')
        labels = base.old.directional_paths(disk[rows.loc[idx, 'label_row'].to_numpy(int)], rows.loc[idx, 'direction'].to_numpy(int))
        base.aggregate(book, states[idx], labels, rows.loc[idx, 'position'].to_numpy(int), first)
        del disk, labels
    choices, details = fit_menu(book, cal[first:last + 1], [2, 3, 4])
    stored = joblib.load(BUNDLE / FEATURE / 'models/variance_premium/2018-01-01.joblib')
    np.testing.assert_array_equal(choices, stored['choices'])
    np.testing.assert_array_equal(leaves, stored['leaves'])
    for field in ['feature', 'threshold', 'children_left', 'children_right', 'value']:
        np.testing.assert_array_equal(getattr(tree.tree_, field), getattr(stored['tree'].tree_, field))
    proposals = pd.read_parquet(BUNDLE / FEATURE / 'replays/variance_premium/scored_allocations.parquet')
    for month, idx in proposals.groupby(proposals.date.dt.to_period('M')).groups.items():
        model = joblib.load(BUNDLE / FEATURE / 'models/variance_premium' / f'{month.start_time.date()}.joblib')
        np.testing.assert_array_equal(score_model(model, proposals.loc[idx], features), proposals.loc[idx, 'learned_level'])
    print(f'Refitted initial fold exactly ({len(rows)} completed examples); verified all 608 frozen allocations. {details}', flush=True)


def replay(policy):
    import gc
    import pandas as pd
    from scripts import run_p25_action_windows as base
    from scripts.option_flex.contract_engine import Settings
    base.cached.OUT = BUNDLE / 'artifacts/contracts/p25_signal_transfer'
    market = base.Market()
    cal = market.rates.index[market.rates.index >= '2018-01-01']
    original = pd.read_parquet(BUNDLE / FEATURE / 'replays/variance_premium/scored_allocations.parquet')
    for case in CASES if policy == 'all' else [f'10m_{policy}']:
        reference = BUNDLE / CAPITAL / 'replays' / case
        rows = original.copy()
        if case.endswith('fixed50'):
            rows['learned_level'] = 2
        elif case.endswith('fixed100'):
            rows['learned_level'] = 4
        cfg = Settings(**json.loads((reference / 'config.json').read_text()))
        saved = base.engine.book_risk
        try:
            base.engine.book_risk = base.book_risk
            nav, trades, decisions, ledger = base.engine.simulate(market, rows, cal, cfg, 1_000_000,
                policy=lambda row, context: int(row['learned_level']),
                corporate_events=base.UNIFORM_CORPORATE_EVENTS, wing_capital=1_000_000)
        finally:
            base.engine.book_risk = saved
        base.mark_book(nav, market.rates, cfg.capital)
        pd.testing.assert_frame_equal(nav, pd.read_parquet(reference / 'daily_nav.parquet'),
                                      check_exact=False, rtol=1e-12, atol=1e-6)
        for frame, name, dates in [(trades, 'trades.csv', ['date']),
                                   (decisions, 'decisions.csv', ['signal_date', 'entry_date'])]:
            pd.testing.assert_frame_equal(frame.reset_index(drop=True), pd.read_csv(reference / name, parse_dates=dates),
                                          check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-6)
        audit = base.verify(nav, trades, decisions, rows, cfg)
        output = ROOT / 'artifacts/submission/replays' / case
        output.mkdir(parents=True, exist_ok=True)
        nav.to_parquet(output / 'daily_nav.parquet')
        (output / 'audit.json').write_text(json.dumps(dict(audit=audit, ledger=ledger), indent=2))
        print(f'{case}: exact funded replay verified', flush=True)
        del nav, trades, decisions, rows
        gc.collect()


def tests(pattern):
    # Separate modules release imports between tests and avoid global fixture
    # leakage. The supervisor still bounds the whole child process tree.
    failures = []
    for path in sorted((ROOT / 'tests').glob(pattern)):
        print(f'Test module: {path.name}', flush=True)
        result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', path.name], cwd=ROOT)
        if result.returncode:
            failures.append(path.name)
    if failures:
        raise RuntimeError('Failed test modules: ' + ', '.join(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_mutually_exclusive_group()
    for name in ['verify', 'package', 'restore-cache', 'metrics', 'fit-check', 'replay', 'tests']:
        commands.add_argument('--' + name, action='store_true')
    parser.add_argument('--policy', choices=['all', 'learned', 'fixed50', 'fixed100'], default='all')
    parser.add_argument('--test-pattern', default='test_*.py')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.package:
        package()
    elif args.restore_cache:
        restore()
    elif args.metrics or args.fit_check or args.replay or args.tests:
        require_supervisor()
        if args.metrics:
            metrics()
        elif args.fit_check:
            fit_check()
        elif args.replay:
            replay(args.policy)
        else:
            tests(args.test_pattern)
    else:
        verify()


if __name__ == '__main__':
    main()
