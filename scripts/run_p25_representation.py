"""Five predeclared P25 representations, causal monthly fits and funded replay.

Windows/POSIX portable. Reads bundled inputs/labels/quotes; preserves all old
checkpoints. One numerical thread, serial worker, 20% CPU duty, 384 MiB RSS stop.
Run with the pinned requirements.txt environment:
python -m scripts.run_p25_representation [--phase train|replay|report|all]
"""
import os
for _key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'ARROW_NUM_THREADS']:
    os.environ[_key] = '1'

import argparse
import gc
import hashlib
import io
import json
from pathlib import Path
import platform
import time
import zipfile

import joblib
import numpy as np
import pandas as pd
import psutil
import sklearn
from sklearn.tree import DecisionTreeRegressor

from scripts.diagnose_p25_pca import CORE, training_window
from scripts.p25_representation import CASES, fit_representation, transform, fit_menu, state_book, score_model
from scripts.submission_market import FrozenMarket, verify_ledger
from scripts.cheap_vega_runtime import mark_book
from scripts.option_flex.contract_engine import Settings
from scripts import uniform_smile_engine as engine
from scripts.uniform_smile_risk import book_risk

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'data/submission'
SOURCE = BUNDLE / 'artifacts/contracts/p25_feature_search/training_rows.parquet'
LABELS = BUNDLE / 'artifacts/contracts/uniform_smile/pairs'
MARKET = BUNDLE / 'artifacts/contracts/p25_signal_transfer/market'
ARCHIVE = ROOT / 'data/p25_validation_split_experiment.zip'
CONFIG = ROOT / 'config/p25_validation_2018_2021.json'
OUT = ROOT / 'artifacts/contracts/p25_representation'
REPORT = ROOT / 'reports/p25_representation'
REFERENCE = 'artifacts/contracts/p25_validation_split/replays/screened_selected/'
MODEL_ROOT = 'artifacts/contracts/p25_feature_search/models/control/'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class Budget:
    def __init__(self, duty=.2, rss_mib=384):
        self.process = psutil.Process()
        self.duty, self.limit, self.peak = duty, rss_mib, 0.
        self.started = time.monotonic()
        cpu = self.process.cpu_times()
        self.cpu_start = cpu.user + cpu.system
        try:
            self.process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == 'nt' else 10)
        except (psutil.Error, AttributeError):
            pass

    def check(self):
        rss = self.process.memory_info().rss / 1024**2
        self.peak = max(self.peak, rss)
        if rss > self.limit:
            raise MemoryError(f'Worker exceeded {self.limit} MiB RSS')
        if (OUT / 'STOP').exists():
            raise InterruptedError('Experiment STOP file exists')

    def rest(self):
        self.check()
        cpu = self.process.cpu_times()
        delay = (cpu.user + cpu.system - self.cpu_start) / self.duty - (time.monotonic() - self.started)
        while delay > 0:
            time.sleep(min(delay, .5))
            self.check()
            delay -= .5


class Sources:
    def __init__(self):
        self.hashes = json.loads((OUT / 'sources.json').read_text()) if (OUT / 'sources.json').exists() else {}
        self.stats = {}

    def track(self, path):
        path = Path(path)
        key = path.relative_to(ROOT).as_posix()
        current = digest(path) if key not in self.stats else self.hashes[key]
        if key in self.hashes and current != self.hashes[key]:
            raise ValueError(f'Source changed: {key}')
        self.hashes[key] = current
        self.stats[key] = (path.stat().st_size, path.stat().st_mtime_ns)
        return path

    def save(self):
        for key, stat in self.stats.items():
            p = ROOT / key
            if (p.stat().st_size, p.stat().st_mtime_ns) != stat:
                raise ValueError(f'Source changed during run: {key}')
        write(OUT / 'sources.json', self.hashes)


def archive_read(archive, member, manifest):
    raw = archive.read(member)
    if hashlib.sha256(raw).hexdigest() != manifest[member]:
        raise ValueError(f'Archive hash mismatch: {member}')
    return raw


def prepare(sources):
    cfg = json.loads(sources.track(CONFIG).read_text())
    sources.track(ARCHIVE)
    manifest = json.loads(sources.track(ROOT / 'reports/p25_validation_split/bundle_manifest.json').read_text())['files']
    inputs = cfg['screened']['inputs']
    columns = inputs + ['date', 'end', 'pair', 'signal_eligible', 'finite_label',
                        'target', 'position', 'label_row', 'direction']
    rows = pd.read_parquet(sources.track(SOURCE), columns=columns)
    cal = pd.read_parquet(sources.track(MARKET / 'rates.parquet')).index
    with zipfile.ZipFile(ARCHIVE) as archive:
        signals = pd.read_parquet(io.BytesIO(archive_read(archive, REFERENCE + 'scored_allocations.parquet', manifest)))
    assert len(signals) == 608 and not signals.duplicated(['date', 'pair']).any()
    features = rows[['date', 'pair'] + inputs]
    missing = [c for c in inputs if c not in signals]
    if missing:
        signals = signals.merge(features[['date', 'pair'] + missing], on=['date', 'pair'], how='left', validate='one_to_one')
    assert np.isfinite(signals[inputs]).all().all()
    assert (cal.get_indexer(rows.date) == rows.position.to_numpy()).all()
    protocol = dict(cases=CASES, core=CORE, inputs=inputs,
        pca='Sample-standardized SVD on 15 economic inputs; flags separate; training-only fit; no whitening.',
        all_pc='All directions of varying inputs retained, including sample-null directions. Constant inputs cannot help training.',
        training='90 monthly 18-month windows Jan2018-Jun2025; signal_eligible & finite_label; signal and end strictly before cutoff.',
        optimizer='CEM50x192 seed2026; full-size anchor; 50/75/100%; <=8 leaves; minleaf=max(8,min(63,n//8)); fallback100 under16 rows.',
        validation='2018-01-02 through2021-12-31; highest funded10m excess Sharpe, lexical case tie break.',
        evaluation='2022-01-03 through2025-08-29; continuing inventory and monthly completed-outcome refits.',
        comparison='Identical608 proposals, contracts, screens, costs,10m capital, risk limits,18month lookback and search budget.',
        bootstrap='Paired circular time blocks21/63/126 sessions,2000 replications per contrast; same dates for both; no search adjustment.',
        scope='Sizing representation only; core-only retains entry filters based on other variables. No pair/universe/signal reselection.',
        untouched_holdout=False, prior_history_inspected=True, automatic_promotion=False,
        runtime=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                     sklearn=sklearn.__version__, joblib=joblib.__version__, threads=1, duty_cycle=.2, rss_limit_mib=384),
        source_sha256=digest(SOURCE), archive_sha256=digest(ARCHIVE),
        scripts={p: digest(ROOT / p) for p in ['scripts/p25_representation.py', 'scripts/run_p25_representation.py']})
    path = REPORT / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Protocol changed; use a new output directory for a new experiment')
    if not path.exists():
        write(path, protocol)
    sources.save()
    return rows, signals, inputs, cal, manifest


def train(meta, inputs, cal, manifest, sources, budget, limit=None):
    months = pd.period_range('2018-01', '2025-06', freq='M')
    if limit:
        months = months[:limit]
    logs = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        for month in months:
            stamp = str(month.start_time.date())
            rows = training_window(meta, month.start_time).reset_index(drop=True)
            assert len(rows) >= 16
            assert rows.end.max() < month.start_time
            reference = joblib.load(io.BytesIO(archive_read(archive, MODEL_ROOT + stamp + '.joblib', manifest)))
            ref_meta = json.loads(archive_read(archive, MODEL_ROOT + stamp + '.json', manifest))
            assert len(rows) == ref_meta['training_rows']
            assert str(rows.end.max().date()) == ref_meta['last_training_outcome']
            for case in CASES:
                folder = OUT / 'models' / case
                path = folder / (stamp + '.json')
                if path.exists():
                    logs.append(json.loads(path.read_text()))
                    continue
                rep = fit_representation(rows, inputs, case)
                x = transform(rows, rep)
                tree = DecisionTreeRegressor(max_leaf_nodes=8, min_samples_leaf=max(8, min(63, len(rows)//8)), random_state=2026).fit(x, rows.target)
                leaf_ids = np.unique(tree.apply(x))
                states = np.searchsorted(leaf_ids, tree.apply(x))
                book = state_book(rows, states, len(leaf_ids), cal, LABELS, sources.track)
                choices, detail = fit_menu(book)
                exact = None
                if case == 'raw_full':
                    np.testing.assert_array_equal(leaf_ids, reference['leaves'])
                    np.testing.assert_array_equal(choices, reference['choices'])
                    for f in ['feature', 'threshold', 'children_left', 'children_right', 'value']:
                        np.testing.assert_array_equal(getattr(tree.tree_, f), getattr(reference['tree'].tree_, f))
                    exact = True
                model = dict(tree=tree, leaves=leaf_ids, choices=choices, fit=stamp,
                             representation=rep, input_features=inputs)
                record = dict(case=case, fit=stamp, training_rows=len(rows),
                    latest_completed_outcome=str(rows.end.max().date()), leaves=len(leaf_ids),
                    economic_dimensions=rep['economic_dimensions'], total_dimensions=x.shape[1],
                    variance_retained=rep.get('variance_retained'), choices=choices.tolist(),
                    varying_flags=int(rows[[f for f in inputs if f.endswith('_available')]].nunique().gt(1).sum()),
                    exact_baseline_fit=exact, **detail)
                folder.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, folder / (stamp + '.joblib'))
                write(path, record)
                logs.append(record)
                del model, tree, book, x
                budget.rest()
            print(f'Fitted {stamp}: {len(rows)} completed outcomes; {len(logs)} models', flush=True)
            write(OUT / 'progress.json', dict(stage='train', fit=stamp, models_complete=len(logs), peak_rss_mib=budget.peak))
            sources.save()
            del rows, reference
            gc.collect()
    pd.DataFrame(logs).to_csv(REPORT / 'fits.csv', index=False)


class ThrottledMarket(FrozenMarket):
    def __init__(self, folder, budget):
        super().__init__(folder)
        self.budget, self.last_check = budget, None

    def rate(self, date):
        if date != self.last_check:
            self.budget.rest()
            self.last_check = date
        return super().rate(date)


def replay(signals, inputs, cal, manifest, sources, budget):
    for p in MARKET.iterdir():
        if p.is_file():
            sources.track(p)
    settings_path = BUNDLE / 'artifacts/contracts/p25_capital_sensitivity/replays/10m_learned/config.json'
    cfg = Settings(**json.loads(sources.track(settings_path).read_text()))
    ck = json.loads(sources.track(ROOT / 'config/p25_submission.json').read_text())
    for case in ['raw_full'] + [c for c in CASES if c != 'raw_full']:
        folder = OUT / 'replays' / case
        if (folder / 'status.json').exists():
            continue
        folder.mkdir(parents=True, exist_ok=True)
        rows = signals.copy()
        for month, indices in rows.groupby(rows.date.dt.to_period('M')).groups.items():
            model = joblib.load(OUT / 'models' / case / f'{month.start_time.date()}.joblib')
            rows.loc[indices, 'learned_level'] = score_model(model, rows.loc[indices])
        assert rows.learned_level.isin([2, 3, 4]).all()
        if case == 'raw_full':
            np.testing.assert_array_equal(rows.learned_level, signals.learned_level)
        rows.to_parquet(folder / 'scored_allocations.parquet', index=False)
        print(f'Funded replay {case}', flush=True)
        market = ThrottledMarket(MARKET, budget)
        original = engine.book_risk
        try:
            engine.book_risk = book_risk
            nav, trades, decisions, ledger = engine.simulate(market, rows, cal[cal >= '2018-01-01'], cfg, 1_000_000,
                policy=lambda row, context: int(row['learned_level']),
                corporate_events=ck['corporate_events'], wing_capital=1_000_000)
        finally:
            engine.book_risk = original
        mark_book(nav, market.rates, cfg.capital)
        audit = verify_ledger(nav, trades, decisions, rows)
        if case == 'raw_full':
            with zipfile.ZipFile(ARCHIVE) as archive:
                old = pd.read_parquet(io.BytesIO(archive_read(archive, REFERENCE + 'daily_nav.parquet', manifest)))
                pd.testing.assert_frame_equal(nav, old, check_exact=False, rtol=1e-12, atol=1e-6)
                for frame, name, dates in [(trades, 'trades.csv', ['date']),
                                          (decisions, 'decisions.csv', ['signal_date', 'entry_date'])]:
                    old = pd.read_csv(io.BytesIO(archive_read(archive, REFERENCE + name, manifest)), parse_dates=dates)
                    pd.testing.assert_frame_equal(frame.reset_index(drop=True), old,
                        check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-6)
        nav.to_parquet(folder / 'daily_nav.parquet')
        trades.to_csv(folder / 'trades.csv', index=False)
        decisions.to_csv(folder / 'decisions.csv', index=False)
        write(folder / 'status.json', dict(status='complete', case=case,
            baseline_reproduced=(case == 'raw_full'), **audit, **ledger))
        print(f'{case}: {audit["entries"]} funded entries; ledger reconciled', flush=True)
        write(OUT / 'progress.json', dict(stage='replay', case=case, peak_rss_mib=budget.peak))
        sources.save()
        del market, rows, nav, trades, decisions
        gc.collect()
        budget.rest()


def sharpe(values):
    r = np.asarray(values, float)
    if not np.isfinite(r).all():
        raise ValueError('Nonfinite returns')
    return float(np.sqrt(252) * r.mean() / r.std(ddof=1)) if r.std(ddof=1) > 1e-12 else 0.


def period_metrics(nav, decisions, start, stop):
    part = nav.loc[start:stop]
    first = nav.index.get_loc(part.index[0])
    before = nav.iloc[max(0, first - 1)]
    returns = part.excess_return.iloc[1:] if first == 0 else part.excess_return
    ratio = pd.concat([pd.Series([before.nav / before.cash_benchmark]),
                       (part.nav / part.cash_benchmark).reset_index(drop=True)], ignore_index=True)
    return dict(start=str(part.index[0].date()), end=str(part.index[-1].date()),
        annualized_excess_sharpe=sharpe(returns),
        profit_above_cash=float((part.nav.iloc[-1] - part.cash_benchmark.iloc[-1]) - (before.nav - before.cash_benchmark)),
        entries=int((decisions.status.eq('entered') & decisions.entry_date.between(part.index[0], part.index[-1])).sum()),
        average_reserve_pct=float(100 * (part.capital_reserve / part.nav).mean()),
        max_cash_relative_drawdown_pct=float(100 * (ratio / ratio.cummax() - 1).min()),
        post_entry_risk_breach_days=int(part.post_trade_risk_breach.sum()))


def report(budget):
    books, decisions = {}, {}
    for case in CASES:
        folder = OUT / 'replays' / case
        if not (folder / 'status.json').exists():
            raise ValueError(f'Incomplete replay: {case}')
        books[case] = pd.read_parquet(folder / 'daily_nav.parquet')
        decisions[case] = pd.read_csv(folder / 'decisions.csv', parse_dates=['entry_date', 'signal_date'])
    # Lock using validation only, before evaluating any later-period metric.
    validation = [dict(case=c, **period_metrics(books[c], decisions[c], '2018-01-01', '2021-12-31')) for c in CASES]
    ranked = sorted(validation, key=lambda r: (-r['annualized_excess_sharpe'], r['case']))
    selection = dict(selected=ranked[0]['case'], ranking=ranked, selected_using='2018-2021 funded excess Sharpe only',
                     untouched_holdout=False, protocol_sha256=digest(REPORT / 'protocol.json'))
    selection_path = REPORT / 'selection.json'
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError('Locked selection changed')
    write(selection_path, selection)
    records, annual = [], []
    for case in CASES:
        for period, start, stop in [('validation', '2018-01-01', '2021-12-31'),
                                  ('evaluation', '2022-01-01', '2025-08-29'),
                                  ('full_history', '2018-01-01', '2025-08-29')]:
            records.append(dict(case=case, period=period, **period_metrics(books[case], decisions[case], start, stop)))
        for year in range(2018, 2026):
            annual.append(dict(case=case, year=year, **period_metrics(books[case], decisions[case], f'{year}-01-01', f'{year}-12-31')))
    frame = pd.DataFrame(records)
    frame.to_csv(REPORT / 'periods.csv', index=False)
    pd.DataFrame(annual).to_csv(REPORT / 'annual.csv', index=False)
    intervals = []
    contrasts = [(c, 'raw_full') for c in CASES if c != 'raw_full'] + [
        ('pca3', 'core3'), ('pca80', 'pca_all'), ('pca_all', 'pca3')]
    for left, right in contrasts:
        a = books[left].loc['2022-01-01':, 'excess_return'].to_numpy()
        b = books[right].loc['2022-01-01':, 'excess_return'].to_numpy()
        assert books[left].index.equals(books[right].index)
        for block in [21, 63, 126]:
            rng = np.random.default_rng(2026)
            changes = []
            for _ in range(2000):
                starts = rng.integers(0, len(a), size=int(np.ceil(len(a)/block)))
                idx = ((starts[:, None] + np.arange(block)[None]) % len(a)).ravel()[:len(a)]
                changes.append(sharpe(a[idx]) - sharpe(b[idx]))
            intervals.append(dict(candidate=left, reference=right, point=sharpe(a)-sharpe(b),
                lower_95=float(np.quantile(changes, .025)), upper_95=float(np.quantile(changes, .975)),
                block_sessions=block, replications=2000, paired=True, circular=True, selection_adjusted=False))
            budget.rest()
    pd.DataFrame(intervals).to_csv(REPORT / 'sharpe_intervals.csv', index=False)
    baseline_rows = pd.read_parquet(OUT / 'replays/raw_full/scored_allocations.parquet')
    allocations = baseline_rows[['date', 'pair']].copy()
    for case in CASES:
        rows = pd.read_parquet(OUT / 'replays' / case / 'scored_allocations.parquet')
        pd.testing.assert_frame_equal(rows[['date', 'pair']], allocations[['date', 'pair']])
        allocations[case] = rows.learned_level.to_numpy() / 4
    allocations.to_csv(REPORT / 'allocation_comparison.csv', index=False)
    evals = frame.loc[frame.period.eq('evaluation')].set_index('case')
    vals = frame.loc[frame.period.eq('validation')].set_index('case')
    lines = ['# P25 sizing: original inputs versus PCA factors', '',
        'Five predeclared representations were refitted across all90 monthly18-month completed-outcome windows. Each reran the same tree and50x192 cross-entropy sizing search. All608 proposals were replayed through the same10m funded ledger. The current full-input baseline reproduces every monthly tree, sizing choice, daily NAV, trade and decision.', '',
        '| Representation |2018–2021 validation Sharpe |2022–Aug2025 evaluation Sharpe | Evaluation profit above cash | Evaluation entries |',
        '|---|---:|---:|---:|---:|']
    for case in CASES:
        r = evals.loc[case]
        lines.append(f'| {case} | {vals.loc[case,"annualized_excess_sharpe"]:.3f} | {r.annualized_excess_sharpe:.3f} | ${r.profit_above_cash:,.0f} | {int(r.entries)} |')
    lines += ['', f'Validation selected **{selection["selected"]}** using only2018–2021 returns. All later results are reported, but later ranking is descriptive; no configuration was selected using later Sharpe.', '',
        '## Paired uncertainty versus the current model', '',
        '| Candidate | Evaluation Sharpe change |95% interval,126-session blocks |', '|---|---:|---:|']
    for r in intervals:
        if r['reference'] == 'raw_full' and r['block_sessions'] == 126:
            lines.append(f'| {r["candidate"]} | {r["point"]:+.3f} | [{r["lower_95"]:+.3f}, {r["upper_95"]:+.3f}] |')
    lines += ['', '## What the comparison tests', '',
        '- core3 versus raw_full tests whether three economic sizing inputs suffice. All availability flags remain; they are constant in every training window.',
        '- pca3 versus core3 tests a compact factor representation against the three economic heuristics.',
        '- pca_all versus raw_full changes splitting coordinates without discarding variation in nonconstant economic inputs.',
        '- pca80 and pca3 versus pca_all test compression within the PCA representation. The80% threshold is fixed in advance; dimension is determined from each training window.',
        '- PCA loadings/means/scales use training inputs only. No target enters the PCA fit. Trees learn the original regression target; CEM optimizes training shadow-book Sharpe; reported performance is independently replayed funded portfolio Sharpe.', '',
        '## Limits', '',
        'The2022–2025 history was already inspected. This is a causal walk-forward comparison on previously researched history, not a fresh untouched holdout. Later completed outcomes may train later monthly fits. Paired bootstrap intervals preserve common dates and serial blocks, but do not adjust for prior strategy search or representation selection. Block-length sensitivity21/63/126 sessions is provided.', '',
        'Entry screens, universe, direction and contracts stay frozen. A three-input sizing model still inherits qualification screens using other economic inputs. This experiment cannot establish that those other variables are irrelevant to trade selection. It also does not test PCA residual signals across stock returns, supervised factors, neural networks, other feature families, or globally optimal trees.', '',
        'The old0.65-per-contract labels, actual0.03 execution commission,0.10 vol-point option penalty,1bp stock hedge costs and all financing/risk conventions are preserved. Continued inventory across the2022 boundary is included.2025 is partial.', '',
        '## Reproduction and files', '',
        '`python -m scripts.run_p25_representation` with the pinned requirements.txt environment plus matplotlib. Single worker, one numerical thread,20% CPU duty,384MiB RSS stop; STOP file under artifacts/contracts/p25_representation interrupts safely. Original models and data are read-only.', '',
        '- protocol.json and selection.json: predeclared cases, date splits, package versions and validation-only selection.',
        '- periods.csv, annual.csv and sharpe_intervals.csv: funded metrics and paired intervals.',
        '- fits.csv:90 monthly fits per representation, dimensions, training-only variance retained and exact baseline checks.',
        '- allocation_comparison.csv: all608 proposed sizing actions.',
        '- comparison.png: validation/evaluation Sharpe and later cash-relative wealth.',
        '- artifacts/contracts/p25_representation: fitted transformations, trees, leaf actions and complete audited funded ledgers.', '']
    (REPORT / 'findings.md').write_text('\n'.join(lines), encoding='utf-8')
    plot(books, vals, evals)
    budget.rest()
    write(REPORT / 'completion.json', dict(status='complete', models=450, funded_replays=5,
        baseline_reproduced=True, peak_rss_mib=budget.peak, selected=selection['selected'],
        source_hashes=json.loads((OUT / 'sources.json').read_text())))
    print(frame.loc[frame.period.isin(['validation','evaluation']),
                    ['case','period','annualized_excess_sharpe','profit_above_cash','entries']].to_string(index=False), flush=True)


def plot(books, vals, evals):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    x = np.arange(len(CASES))
    axes[0].bar(x-.18, [vals.loc[c, 'annualized_excess_sharpe'] for c in CASES], width=.36, label='2018–2021 validation')
    axes[0].bar(x+.18, [evals.loc[c, 'annualized_excess_sharpe'] for c in CASES], width=.36, label='2022–Aug2025 evaluation')
    axes[0].set_xticks(x, CASES, rotation=20)
    axes[0].set_ylabel('Annualized daily cash-excess Sharpe')
    axes[0].legend(fontsize=8)
    axes[0].set_title('Same tree/search capacity; different inputs')
    for case in CASES:
        nav = books[case]
        base = nav.loc[:'2021-12-31'].iloc[-1]
        part = nav.loc['2022-01-01':]
        growth = 100*((part.nav/part.cash_benchmark)/(base.nav/base.cash_benchmark)-1)
        axes[1].plot(part.index, growth, label=case)
    axes[1].set_ylabel('Growth relative to cash (%)')
    axes[1].set_title('Later performance; previously inspected history')
    axes[1].legend(fontsize=8)
    axes[1].tick_params(axis='x', rotation=20)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(REPORT / 'comparison.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['all','train','replay','report'], default='all')
    parser.add_argument('--limit-fits', type=int, help='Training-only preflight; never produces final results')
    args = parser.parse_args()
    if args.limit_fits and args.phase != 'train':
        parser.error('--limit-fits requires --phase train')
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    if sklearn.__version__ != '1.9.0':
        raise RuntimeError('Exact baseline reproduction requires saved sklearn1.9.0')
    budget, sources = Budget(), Sources()
    meta, signals, inputs, cal, manifest = prepare(sources)
    budget.rest()
    if args.phase in ['all','train']:
        train(meta, inputs, cal, manifest, sources, budget, args.limit_fits)
    del meta
    gc.collect()
    if args.phase in ['all','replay']:
        replay(signals, inputs, cal, manifest, sources, budget)
    if args.phase in ['all','report']:
        report(budget)


if __name__ == '__main__':
    main()
