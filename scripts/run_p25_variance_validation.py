"""Seed/objective checks of the selected variance-premium research candidate."""
import argparse
import gc
import json
from pathlib import Path
import resource
import shutil

import joblib
import numpy as np
import pandas as pd

from scripts import run_p25_feature_search as study
from scripts import run_p25_action_windows as base
from scripts.p25_recency import completed_months
from scripts.run_tail_put25_residual_rl import moments, score_choices

OUT = Path('artifacts/contracts/p25_variance_validation')
REPORT = Path('reports/p25_variance_validation')
SOURCE = Path('artifacts/contracts/p25_feature_search')
SIGNALS = study.GROUPS['variance_premium']
SEEDS = [2026, 1, 7, 42, 2027]
CASES = [f'sharpe_seed{s}' for s in SEEDS] + ['mean_pnl', 'local_sharpe', 'diagonal_sharpe']
PROTOCOL = (
    'Diagnostic follow-up after variance-premium feature selection. Same33 '
    'features and original variance-premium trees,18-month completed training '
    'windows and positive50/75/100 actions. Compare CEM seeds2026/1/7/42/2027 '
    '(50x192), profit-only action selection, each leaf standalone Sharpe, and '
    'direct portfolio Sharpe ignoring cross-leaf covariance. Seed2026 remains '
    'the retained candidate regardless of best descriptive seed. These are '
    'contextual sizing objective ablations, not an NN or a new sequential RL '
    'algorithm. Same608 frozen proposals, original costs/risk/cash/hedges; '
    'four fixed controls. Original90 candidate allocations and cash book must '
    'reproduce. Joint circular moving-block bootstrap60/126 sessions,2000 '
    'draws,95% intervals not adjusted for multiple research selection. '
    'Previously explored2018-Aug2025 history and retrospective formation; '
    'no fresh holdout, no universal RL benefit claim. One worker50%onecore,3GiB.'
)


def optimize(stats, ns, seed):
    menu = np.array([2, 3, 4])
    rng = np.random.default_rng(seed)
    probability = np.full((ns, len(menu)), 1 / len(menu))
    best = np.full(ns, 4)
    best_score = float(score_choices(best[None], stats)[0])
    for _ in range(50):
        aliases = (rng.random((192, ns))[:, :, None] > probability.cumsum(axis=1)[None]).sum(axis=2).clip(max=2)
        choices = menu[aliases]
        choices[0] = best
        choices[1:4] = menu[:, None]
        scores = score_choices(choices, stats)
        winner = scores.argmax()
        if scores[winner] > best_score:
            best, best_score = choices[winner].copy(), float(scores[winner])
        elite = choices[np.argsort(scores)[-19:]]
        frequency = np.stack([(elite == level).mean(axis=0) for level in menu], axis=1)
        probability = .97 * (.3 * probability + .7 * frequency) + .03 / len(menu)
    return best, best_score


def train(sources, cal):
    meta = pd.read_parquet(sources.track(SOURCE / 'training_rows.parquet'))
    logs = []
    for month in pd.period_range('2018-01-01', cal[-1] - pd.Timedelta(days=base.RUNOFF), freq='M'):
        cutoff = month.start_time
        stamp = str(cutoff.date())
        original = joblib.load(sources.track(SOURCE / 'models/variance_premium' / f'{stamp}.joblib'))
        rows = meta.loc[meta.signal_eligible & meta.finite_label & completed_months(meta.date, meta.end, cutoff, 18)].reset_index(drop=True)
        stats = None
        if original['tree'] is not None:
            states = np.searchsorted(original['leaves'], original['tree'].apply(rows[original['features']]))
            first = int(rows.position.min() + 1)
            last = int(cal.get_indexer(pd.DatetimeIndex(rows.end)).max())
            book = np.zeros((last - first + 1, len(original['leaves']) * 5))
            for pair, idx in rows.groupby('pair', sort=False).groups.items():
                idx = np.asarray(idx)
                disk = np.load(sources.track(study.NATIVE / 'pairs' / pair / 'P25_labels.npy'), mmap_mode='r')
                labels = base.old.directional_paths(disk[rows.loc[idx, 'label_row'].to_numpy(int)], rows.loc[idx, 'direction'].to_numpy(int))
                base.aggregate(book, states[idx], labels, rows.loc[idx, 'position'].to_numpy(int), first)
                del disk, labels
            stats = moments(book, cal[first:last + 1], False)
            del book
        for case in CASES:
            folder = OUT / 'models' / case
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f'{stamp}.json'
            if path.exists():
                logs.append(json.loads(path.read_text()))
                continue
            choices = np.array([4])
            score = None
            if stats is not None:
                ns = len(original['leaves'])
                mean, cov = stats[0]
                if case.startswith('sharpe_seed'):
                    choices, score = optimize(stats, ns, int(case.removeprefix('sharpe_seed')))
                elif case == 'diagonal_sharpe':
                    choices, score = optimize([(mean, np.diag(np.diag(cov)))], ns, 2026)
                else:
                    menu = np.array([4, 3, 2])
                    cols = np.arange(ns)[:, None] * 5 + menu[None, :]
                    values = mean[cols]
                    if case == 'local_sharpe':
                        sd = np.sqrt(np.maximum(np.diag(cov)[cols], 0))
                        values = np.divide(values, sd, out=np.zeros_like(values), where=sd > 1e-12)
                    choices = menu[values.argmax(axis=1)]
                    score = float(score_choices(choices[None], stats)[0])
            if case == 'sharpe_seed2026':
                np.testing.assert_array_equal(choices, original['choices'])
            model = {**original, 'choices': choices}
            record = dict(model=case, fit=stamp, training_rows=len(rows), last_training_outcome=str(rows.end.max().date()) if len(rows) else None, choices=choices.tolist(), training_objective=score)
            joblib.dump(model, folder / f'{stamp}.joblib')
            base.write_json(path, record)
            logs.append(record)
        if month.month in [1, 4, 7, 10]:
            print(f'Validation fits {stamp}: {len(logs)}', flush=True)
        sources.save()
        base.progress('train', fit=stamp, models_complete=len(logs))
        gc.collect()
    assert len(logs) == 90 * len(CASES)
    pd.DataFrame(logs).to_csv(REPORT / 'fits.csv', index=False)


def diagnostics(sources):
    candidate = pd.read_parquet(OUT / 'replays/sharpe_seed2026/daily_nav.parquet')
    pd.testing.assert_frame_equal(candidate, pd.read_parquet(sources.track(SOURCE / 'replays/variance_premium/daily_nav.parquet')), rtol=1e-12, atol=1e-6)
    for seed in SEEDS:
        pd.testing.assert_frame_equal(candidate, pd.read_parquet(OUT / 'replays' / f'sharpe_seed{seed}' / 'daily_nav.parquet'))
    reference = pd.read_parquet(sources.track(SOURCE / 'replays/control/daily_nav.parquet'))
    fixed = pd.read_parquet(OUT / 'replays/fixed2/daily_nav.parquet')
    fixed100 = pd.read_parquet(OUT / 'replays/fixed4/daily_nav.parquet')
    array = np.column_stack([candidate.excess_return.iloc[1:], reference.excess_return.iloc[1:], fixed.excess_return.iloc[1:], fixed100.excess_return.iloc[1:]])
    observed = np.sqrt(252) * array.mean(axis=0) / array.std(axis=0, ddof=1)
    bootstrap = []
    for block in [60, 126]:
        rng = np.random.default_rng(2026)
        scores = []
        n = len(array)
        for _ in range(2000):
            start = rng.integers(n, size=int(np.ceil(n / block)))
            idx = ((start[:, None] + np.arange(block)) % n).ravel()[:n]
            sample = array[idx]
            scores.append(np.sqrt(252) * sample.mean(axis=0) / sample.std(axis=0, ddof=1))
        scores = np.asarray(scores)
        for label, values, point in [('candidate_sharpe', scores[:, 0], observed[0]), ('vs_previous_rl', scores[:, 0] - scores[:, 1], observed[0] - observed[1]), ('vs_best_fixed', scores[:, 0] - scores[:, 2], observed[0] - observed[2]), ('vs_full_size', scores[:, 0] - scores[:, 3], observed[0] - observed[3])]:
            lo, hi = np.quantile(values, [.025, .975])
            bootstrap.append(dict(comparison=label, block_sessions=block, point=float(point), ci_low=float(lo), ci_high=float(hi), draws=2000, selection_adjusted=False))
    pd.DataFrame(bootstrap).to_csv(REPORT / 'bootstrap.csv', index=False)
    summary = pd.read_csv(REPORT / 'comparison.csv')
    seeds = summary[summary.model.str.startswith('sharpe_seed')]
    base.write_json(REPORT / 'seed_summary.json', dict(retained_seed=2026, no_best_seed_promotion=True, seeds=SEEDS, min_sharpe=float(seeds.sharpe.min()), median_sharpe=float(seeds.sharpe.median()), max_sharpe=float(seeds.sharpe.max())))
    by_year = []
    for year in sorted(set(candidate.index.year)):
        returns = candidate.loc[candidate.index.year != year, 'excess_return'].iloc[1:]
        by_year.append(dict(excluded_year=int(year), descriptive_sharpe=base.sharpe(returns)))
    pd.DataFrame(by_year).to_csv(REPORT / 'leave_year_out.csv', index=False)
    audit = json.loads((REPORT / 'audit.json').read_text())
    audit.pop('original90_models_and_five_cash_books_reproduced', None)
    audit.update(original90_candidate_allocations_and_cash_book_exact=True, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    base.write_json(REPORT / 'audit.json', audit)
    (REPORT / 'interpretation.txt').write_text(PROTOCOL + '\n\n' + summary.to_string(index=False) + '\n\n' + pd.DataFrame(bootstrap).to_string(index=False) + '\n')
    base.progress('complete')
    sources.check()
    sources.save()
    base.write_json(REPORT / 'output_hashes.json', {str(p): base.digest(p) for root in [OUT, REPORT] for p in root.rglob('*') if p.is_file() and p.name != 'output_hashes.json'})
    print(pd.DataFrame(bootstrap).to_string(index=False), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['train', 'replay', 'summarize', 'all'], default='all')
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    base.OUT, base.REPORT = OUT, REPORT
    study.OUT, study.REPORT = OUT, REPORT
    study.GROUPS = {case: SIGNALS for case in CASES}
    sources = base.Sources()
    for path in [Path(__file__), Path('scripts/run_p25_feature_search.py'), Path('scripts/run_tail_put25_residual_rl.py'), Path('scripts/p25_action_windows.py'), Path('scripts/p25_recency.py')]:
        sources.track(path)
    sources.hashes[str(Path(__file__))] = base.digest(Path(__file__))
    cal = pd.read_parquet(sources.track(study.NATIVE / 'market/rates.parquet')).index
    base.write_json(OUT / 'inputs.json', dict(protocol=PROTOCOL, cases=CASES, retained_seed=2026, added_signals=SIGNALS))
    shutil.copyfile(sources.track(SOURCE / 'signal_rows.parquet'), OUT / 'signal_rows.parquet')
    if args.phase in ['train', 'all']:
        train(sources, cal)
    if args.phase in ['replay', 'all']:
        study.replay(sources, cal)
    if args.phase in ['summarize', 'all']:
        study.summarize(sources)
        diagnostics(sources)
    sources.check()
    sources.save()


if __name__ == '__main__':
    main()
