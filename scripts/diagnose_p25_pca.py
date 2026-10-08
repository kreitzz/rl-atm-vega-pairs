"""Inspect selected P25 inputs and frozen monthly trees; never fit a policy.

Run from the repository root: python -m scripts.diagnose_p25_pca
Reads only bundled training columns and hash-verified control models. Outputs
stay in reports/p25_pca; no market restore, model rewrite, or Sharpe search.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import platform
import time
import warnings
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('data/submission/artifacts/contracts/p25_feature_search/training_rows.parquet')
ARCHIVE = Path('data/p25_validation_split_experiment.zip')
CONFIG = Path('config/p25_validation_2018_2021.json')
CORE = ['iv_mean', 'oriented_iv_gap', 'rank_strength']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def training_window(rows, cutoff, months=18):
    """Match the selected policy: signal and outcome strictly precede fit."""
    cutoff = pd.Timestamp(cutoff)
    mask = (rows.signal_eligible & rows.finite_label &
            rows.date.ge(cutoff - pd.DateOffset(months=months)) &
            rows.date.lt(cutoff) & rows.end.lt(cutoff))
    return rows.loc[mask].copy()


def standardized_pca(frame):
    """Correlation PCA, sample-variance scaling; constant fields get zero weight."""
    x = frame.to_numpy(dtype=float)
    if len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('PCA requires at least two finite observations')
    mean = x.mean(axis=0)
    sd = x.std(axis=0, ddof=1)
    active = sd > 1e-12
    if not active.any():
        raise ValueError('No varying inputs')
    z = (x[:, active] - mean[active]) / sd[active]
    _, singular, vectors = np.linalg.svd(z, full_matrices=False)
    # Deterministic signs for display only; signs have no economic meaning.
    for vector in vectors:
        if vector[np.argmax(np.abs(vector))] < 0:
            vector *= -1
    variance = singular ** 2 / (len(x) - 1)
    ratio = variance / variance.sum()
    weights = np.zeros((len(vectors), x.shape[1]))
    weights[:, active] = vectors
    loadings = weights * np.sqrt(variance)[:, None]
    return dict(mean=mean, sd=sd, active=active, weights=weights,
                loadings=loadings, variance=variance, ratio=ratio)


def subspace_overlap(a, b, k=3):
    """Mean squared cosine of principal angles, invariant to sign/PC rotation."""
    k = min(k, len(a), len(b))
    return float(np.linalg.norm(a[:k] @ b[:k].T, 'fro') ** 2 / k)


def tree_splits(model):
    """Count splits, and prune subtrees whose leaves all choose the same size."""
    raw, effective = Counter(), Counter()
    if model['tree'] is None:
        return raw, effective, 1, 0
    tree = model['tree'].tree_
    actions = dict(zip(map(int, model['leaves']), map(int, model['choices'])))

    def visit(node):
        if tree.children_left[node] == -1:
            return {actions[node]}
        feature = model['features'][tree.feature[node]]
        raw[feature] += 1
        sizes = visit(tree.children_left[node]) | visit(tree.children_right[node])
        if len(sizes) > 1:
            effective[feature] += 1
        return sizes

    visit(0)
    return raw, effective, len(actions), sum(raw.values()) - sum(effective.values())


class Budget:
    """Small serial Windows/POSIX diagnostic: one thread, bounded RSS/duty cycle."""
    def __init__(self, duty_cycle=.2, rss_limit_mib=384):
        import psutil
        self.psutil = psutil
        self.process = psutil.Process()
        self.duty = duty_cycle
        self.limit = rss_limit_mib
        self.peak = 0.
        try:
            self.process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if platform.system() == 'Windows' else 10)
        except (psutil.Error, AttributeError):
            pass

    def check(self):
        rss = self.process.memory_info().rss / 1024 ** 2
        self.peak = max(self.peak, rss)
        if rss > self.limit:
            raise MemoryError(f'Diagnostic exceeded {self.limit} MiB RSS')
        memory = self.psutil.virtual_memory()
        # Match run_signal_discovery_guarded: percentage floor is macOS-only.
        if platform.system() == 'Darwin' and memory.available / memory.total < .35:
            raise MemoryError('System available memory below 35%; stopped')

    def rest(self, started):
        self.check()
        time.sleep(max(0., time.monotonic() - started) * (1 / self.duty - 1))


def plot_reports(out, baseline, latest, pearson, spearman, features, summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    for ax, corr, title in zip(axes, [pearson, spearman], ['Pearson', 'Spearman']):
        img = ax.imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r')
        ax.set_xticks(range(len(features)), features, rotation=90)
        ax.set_yticks(range(len(features)), features)
        ax.set_title(title)
    fig.colorbar(img, ax=axes, shrink=.7, label='Correlation')
    fig.suptitle('Selected P25 inputs: January 2022 completed-outcome training window (n=70)')
    fig.savefig(out / 'correlations.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for model, label in [(baseline, 'January 2022'), (latest, 'June 2025')]:
        axes[0].plot(np.arange(len(model['ratio'])) + 1, np.cumsum(model['ratio']), 'o-', label=label)
    axes[0].axhline(.8, color='grey', ls='--', lw=1)
    axes[0].set(xlabel='Number of principal components', ylabel='Cumulative explained variance', ylim=(0, 1.03))
    axes[0].legend()
    dates = pd.to_datetime(summaries['fit'])
    axes[1].plot(dates, summaries['pc3_share'], label='First three PCs')
    axes[1].plot(dates, summaries['adjacent_pc3_subspace_overlap'], label='Adjacent top-3 subspace overlap')
    axes[1].set(ylabel='Fraction / overlap', ylim=(0, 1.03), xlabel='Monthly fit date')
    axes[1].legend()
    fig.suptitle('Variance compression and monthly stability; input diagnostics, not performance')
    fig.savefig(out / 'variance_and_stability.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 7), constrained_layout=True)
    for ax, model, label in zip(axes, [baseline, latest], ['January 2022', 'June 2025']):
        img = ax.imshow(model['loadings'][:3].T, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
        ax.set_xticks(range(3), ['PC1', 'PC2', 'PC3'])
        ax.set_yticks(range(len(features)), features)
        ax.set_title(label)
    fig.colorbar(img, ax=axes, shrink=.7, label='Correlation with component')
    fig.suptitle('Standardized input loadings; component signs are arbitrary')
    fig.savefig(out / 'component_loadings.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('reports/p25_pca'))
    args = parser.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    import joblib
    from sklearn.exceptions import InconsistentVersionWarning
    from threadpoolctl import threadpool_limits
    budget = Budget()
    input_config = json.loads((ROOT / CONFIG).read_text())
    inputs = input_config['screened']['inputs']
    features = [f for f in inputs if not f.endswith('_available')]
    flags = [f for f in inputs if f.endswith('_available')]
    source_sha = digest(ROOT / SOURCE)
    manifest = json.loads((ROOT / 'data/submission/manifest.json').read_text())
    expected = next(f['sha256'] for f in manifest['files'] if f['stored'] == SOURCE.as_posix())
    if source_sha != expected:
        raise ValueError('Training source hash differs from submission manifest')
    rows = pd.read_parquet(ROOT / SOURCE, columns=['date', 'end', 'signal_eligible', 'finite_label'] + inputs,
                           use_threads=False)
    rows['date'] = pd.to_datetime(rows.date)
    rows['end'] = pd.to_datetime(rows.end)
    archive_manifest = json.loads((ROOT / 'reports/p25_validation_split/bundle_manifest.json').read_text())['files']
    records, loadings, explained, missing = [], [], [], []
    splits = {f: Counter() for f in inputs}
    previous = None
    baseline = latest = None
    provenance = {}
    with threadpool_limits(limits=1), zipfile.ZipFile(ROOT / ARCHIVE) as archive:
        names = sorted(n for n in archive.namelist() if '/models/control/' in n and n.endswith('.joblib'))
        for name in names:
            started = time.monotonic()
            raw = archive.read(name)
            meta_name = name[:-7] + '.json'
            meta_raw = archive.read(meta_name)
            for member, content in [(name, raw), (meta_name, meta_raw)]:
                sha = hashlib.sha256(content).hexdigest()
                if sha != archive_manifest[member]:
                    raise ValueError(f'Model hash mismatch: {member}')
                provenance[member] = sha
            meta = json.loads(meta_raw)
            # Only inspect saved arrays; do not invoke model predictions/refitting.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', InconsistentVersionWarning)
                model = joblib.load(io.BytesIO(raw))
            stamp = model['fit']
            if model['features'] != inputs:
                raise ValueError(f'Unexpected feature list: {stamp}')
            train = training_window(rows, stamp, input_config['screened']['lookback_months'])
            if len(train) != meta['training_rows'] or str(train.end.max().date()) != meta['last_training_outcome']:
                raise ValueError(f'Training-window reconstruction failed: {stamp}')
            pca = standardized_pca(train[features])
            core = standardized_pca(train[CORE])
            raw_splits, effective, leaves, redundant = tree_splits(model)
            for f in inputs:
                splits[f].update(raw_splits=raw_splits[f], months_used=int(raw_splits[f] > 0),
                                 action_effective_splits=effective[f], action_effective_months=int(effective[f] > 0))
            records.append(dict(fit=stamp, training_rows=len(train), last_training_outcome=meta['last_training_outcome'],
                                pc1_share=pca['ratio'][0], pc3_share=pca['ratio'][:3].sum(),
                                components_for_80pct=int(np.searchsorted(np.cumsum(pca['ratio']), .8) + 1),
                                core_pc1_share=core['ratio'][0], core_pc2_share=core['ratio'][:2].sum(),
                                adjacent_pc3_subspace_overlap=subspace_overlap(previous['weights'], pca['weights']) if previous is not None else np.nan,
                                tree_leaves=leaves, sizing_redundant_splits=redundant))
            for k, ratio in enumerate(pca['ratio'], 1):
                explained.append(dict(fit=stamp, component=k, explained_variance=pca['variance'][k-1],
                                      variance_share=ratio, cumulative_share=pca['ratio'][:k].sum()))
            for k in range(min(3, len(pca['ratio']))):
                for j, f in enumerate(features):
                    loadings.append(dict(fit=stamp, component=k+1, feature=f, weight=pca['weights'][k, j],
                                         correlation_loading=pca['loadings'][k, j]))
            for f in flags:
                missing.append(dict(fit=stamp, feature=f, available_fraction=float(train[f].mean()),
                                    distinct_values=int(train[f].nunique())))
            if stamp == '2022-01-01':
                baseline = pca
                pearson, spearman = train[features].corr(), train[features].corr(method='spearman')
            latest = pca
            previous = pca
            budget.rest(started)
        if baseline is None:
            raise ValueError('Missing January 2022 baseline fit')
        summaries = pd.DataFrame(records)
        split_table = pd.DataFrame([dict(feature=f, **splits[f]) for f in inputs]).sort_values('months_used', ascending=False)
        summaries.to_csv(out / 'monthly_summary.csv', index=False)
        pd.DataFrame(explained).to_csv(out / 'explained_variance.csv', index=False)
        pd.DataFrame(loadings).to_csv(out / 'component_loadings.csv', index=False)
        missing_table = pd.DataFrame(missing)
        missing_table.to_csv(out / 'availability.csv', index=False)
        split_table.to_csv(out / 'tree_split_usage.csv', index=False)
        pearson.to_csv(out / 'pearson_2022_01.csv')
        spearman.to_csv(out / 'spearman_2022_01.csv')
        started = time.monotonic()
        plot_reports(out, baseline, latest, pearson, spearman, features, summaries)
        budget.rest(started)
    pairs = [(abs(pearson.loc[a, b]), a, b, pearson.loc[a, b], spearman.loc[a, b])
             for i, a in enumerate(features) for b in features[i+1:]]
    strongest = sorted(pairs, reverse=True)[:5]
    baseline_row = summaries.loc[summaries.fit.eq('2022-01-01')].iloc[0]
    latest_row = summaries.iloc[-1]
    lines = ['# PCA diagnostic for the selected P25 sizing policy', '',
             'This inspects the 27-input `features/control` policy associated with 1.263 funded excess Sharpe in 2022–August 2025. No policy was changed or backtest rerun.', '',
             '## Method and scope', '',
             f'- Reconstructed all {len(records)} monthly 18-month training windows exactly; row counts and latest completed outcomes match saved fit metadata.',
             '- Every signal date and outcome end precedes its fit cutoff. The main correlation/loadings snapshot is January 2022; June 2025 is a separate later historical snapshot.',
             '- PCA uses 15 economic inputs, centered and scaled to sample standard deviation within each window. The 12 availability flags are reported separately. Constant economic fields receive zero weight.',
             '- Uses the model\'s existing zero-filled inputs. Observations are pair/date training examples; overlapping trades and repeated pairs are not independent samples.',
             '- Targets are not read as PCA inputs. Only eligibility and completed-outcome metadata reproduce the training population; no profit correlation or significance test is performed.',
             '- PC signs are arbitrary. Adjacent-window stability uses the top-three subspaces, so sign flips or rotations within a subspace do not count as instability.',
             '- Later windows are descriptive. This is not a new holdout, a PCA sizing policy, or evidence of improved Sharpe.', '',
             '## Results', '',
             f'- January 2022: n={baseline_row.training_rows}; first PC explains {baseline_row.pc1_share:.1%}; first three explain {baseline_row.pc3_share:.1%}; {baseline_row.components_for_80pct} PCs are required for 80%.',
             f'- June 2025: n={latest_row.training_rows}; first three explain {latest_row.pc3_share:.1%}; {latest_row.components_for_80pct} PCs are required for 80%.',
             f'- Across monthly windows, first-three variance share ranges from {summaries.pc3_share.min():.1%} to {summaries.pc3_share.max():.1%}; median {summaries.pc3_share.median():.1%}.',
             f'- Adjacent top-three subspace overlap: median {summaries.adjacent_pc3_subspace_overlap.median():.3f}, minimum {summaries.adjacent_pc3_subspace_overlap.min():.3f} (1 means identical subspaces). Overlapping training windows naturally increase apparent stability.',
             f'- Three-core-input check: January 2022 first PC explains {baseline_row.core_pc1_share:.1%}, first two {baseline_row.core_pc2_share:.1%}.',
             f'- Availability flags: {int(missing_table.distinct_values.gt(1).sum())} flag/window combinations vary; minimum available fraction {missing_table.available_fraction.min():.1%}.', '',
             'Strongest January 2022 correlations:', '',
             '| Inputs | Pearson | Spearman |', '|---|---:|---:|']
    lines += [f'| {a} / {b} | {p:.3f} | {s:.3f} |' for _, a, b, p, s in strongest]
    lines += ['', 'Dominant standardized inputs in the January 2022 components:', '']
    for k in range(3):
        indices = np.argsort(np.abs(baseline['loadings'][k]))[-4:][::-1]
        lines.append(f'- PC{k+1}: ' + ', '.join(f"{features[j]} ({baseline['loadings'][k,j]:+.2f})" for j in indices) + '.')
    lines += ['', 'Most frequently used tree split variables:', '',
              '| Variable | Months with a split | Months after pruning constant-size subtrees |', '|---|---:|---:|']
    lines += [f"| {r.feature} | {r.months_used} | {r.action_effective_months} |" for r in split_table.head(8).itertuples()]
    lines += ['', f"There are {int(summaries.sizing_redundant_splits.sum())} splits across the saved models inside subtrees that always choose one size. Those subtrees can be collapsed for understanding the sizing mapping. This does not measure profitability importance, and does not prove every retained split changes allocations on future qualified proposals.", '',
              '## Interpretation', '',
              'There is measurable redundancy: January 2022 IV and RV have correlation 0.897, and two components capture 89.0% of variation within the three core inputs. However, the full economic input set needs six components in January 2022 and seven in June 2025 to reach 80%. These snapshots do not support treating all 15 inputs as just three linear factors.', '',
              'The saved trees draw on profitability and observation history as well as volatility and correlation. Frequent splits and large PCA loadings answer different questions; neither establishes that a feature improves portfolio Sharpe. The 112 constant-size-subtree splits also demonstrate that predicting the regression target and changing the sizing action are distinct.', '',
              'PCA can identify shared variation, but principal components are mixtures rather than feature selections. Use these results to nominate related feature groups for future retrained ablations; no input is dropped on the basis of this diagnostic. Adjacent training windows overlap, and the January 2022 / June 2025 samples have only 70 / 83 observations, so apparent stability is not an independent robustness test.', '',
              '## Files and reproduction', '',
              '`python -m scripts.diagnose_p25_pca` reads bundled data directly. It uses one numerical thread, serial processing, 20% duty cycle and a 384 MiB RSS stop. The 35% system-memory floor applies on macOS, matching the existing supervisor; no cache restoration is required.', '',
              '- `correlations.png`: January 2022 Pearson/Spearman relationships.',
              '- `variance_and_stability.png`: cumulative variance and monthly subspace stability.',
              '- `component_loadings.png`: January 2022 and June 2025 top-three loadings.',
              '- CSVs provide correlations, all variance shares, top-three weights/loadings, monthly summaries, availability, and tree split usage.',
              '- `protocol.json`: source/model hashes, exact procedure, environment and export limitations.', '']
    (out / 'findings.md').write_text('\n'.join(lines), encoding='utf-8')
    import sklearn
    protocol = dict(configuration=str(CONFIG), source=str(SOURCE), source_sha256=source_sha,
                    archive=str(ARCHIVE), archive_sha256=digest(ROOT / ARCHIVE),
                    diagnostic_script_sha256=digest(Path(__file__)), model_hashes=provenance,
                    economic_inputs=features, availability_flags=flags,
                    baseline_fit='2022-01-01', latest_fit=latest_row.fit,
                    procedure='Per-fit signal_eligible & finite_label; signals within past 18 months; signal/end strictly before fit; sample-standardized SVD PCA.',
                    policy_modified=False, backtest_replayed=False,
                    runtime=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
                                 saved_sklearn='1.9.0', threads=1, duty_cycle=.2, peak_rss_mib=budget.peak),
                    model_read_note='Hash-verified saved sklearn arrays inspected only; version warning suppressed. No model fitting/prediction. Use pinned requirements for policy reproduction.')
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')
    print(f'Inspected {len(records)} fits; peak RSS {budget.peak:.1f} MiB; results: {out}', flush=True)


if __name__ == '__main__':
    main()
