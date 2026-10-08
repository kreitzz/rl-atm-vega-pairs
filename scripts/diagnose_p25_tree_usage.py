"""Read-only census of the current policy's90 archived monthly trees.

Records the actual model used by every proposal, feature usage in the whole
tree and along actual decision paths, and constant-size subtree pruning.
No training, sizing search or portfolio replay. Requires saved sklearn1.9.0.
"""
import os
for _key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'ARROW_NUM_THREADS']:
    os.environ[_key] = '1'

import hashlib
import io
import json
from collections import Counter
from pathlib import Path
import time
import zipfile

import joblib
import numpy as np
import pandas as pd
import sklearn

from scripts.diagnose_p25_pca import Budget, tree_splits, digest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/p25_tree_usage'
ARCHIVE = ROOT / 'data/p25_validation_split_experiment.zip'
SOURCE = ROOT / 'data/submission/artifacts/contracts/p25_feature_search/training_rows.parquet'
PREFIX = 'artifacts/contracts/p25_feature_search/models/control/'
REPLAY = 'artifacts/contracts/p25_validation_split/replays/screened_selected/'
LABELS = {
    'profit_365d': 'Past profitability, 365 days',
    'count_365d': 'Observation count, 365 days',
    'correlation': 'Pair correlation',
    'iv_mean': 'Mean implied volatility',
    'profit_91d': 'Past profitability, 91 days',
    'oriented_iv_gap': 'Oriented relative IV gap',
    'oriented_vrp_gap': 'Oriented volatility-premium gap',
    'rank_strength': 'Spread rank strength',
    'beta': 'Pair beta', 'rv_mean': 'Mean realized volatility',
    'commission_vol_points': 'Commission in volatility points',
    'vix': 'VIX', 'count_91d': 'Observation count, 91 days',
    'dte_mean': 'Mean days to expiry', 'oriented_dte_gap': 'Oriented expiry gap',
}


def checked_read(archive, member, manifest):
    raw = archive.read(member)
    sha = hashlib.sha256(raw).hexdigest()
    if sha != manifest[member]:
        raise ValueError(f'Hash mismatch: {member}')
    return raw, sha


def action_sets(model):
    tree = model['tree'].tree_
    leaf_actions = dict(zip(map(int, model['leaves']), map(int, model['choices'])))
    sets = {}

    def visit(node):
        if tree.children_left[node] == -1:
            value = {leaf_actions[node]}
        else:
            value = visit(int(tree.children_left[node])) | visit(int(tree.children_right[node]))
        sets[node] = value
        return value

    visit(0)
    return sets, leaf_actions


def summarize_distribution(values, name, population):
    return [dict(population=population, measure=name, value=k, count=n,
                 percent=100*n/len(values)) for k, n in sorted(Counter(values).items())]


def main():
    if sklearn.__version__ != '1.9.0':
        raise RuntimeError('Use saved sklearn1.9.0 for model application')
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(ROOT / 'artifacts/pca_runtime/matplotlib')
    budget = Budget()
    config = json.loads((ROOT / 'config/p25_validation_2018_2021.json').read_text())
    inputs = config['screened']['inputs']
    manifest = json.loads((ROOT / 'reports/p25_validation_split/bundle_manifest.json').read_text())['files']
    provenance, monthly, path_records = {}, [], []
    raw_counts, effective_counts, roots = Counter(), Counter(), Counter()
    active_counts, active_effective = Counter(), Counter()
    raw_path_counts, effective_path_counts, accepted_effective = Counter(), Counter(), Counter()
    with zipfile.ZipFile(ARCHIVE) as archive:
        raw, _ = checked_read(archive, REPLAY + 'scored_allocations.parquet', manifest)
        signals = pd.read_parquet(io.BytesIO(raw)).reset_index(drop=True)
        raw, _ = checked_read(archive, REPLAY + 'decisions.csv', manifest)
        decisions = pd.read_csv(io.BytesIO(raw), parse_dates=['signal_date', 'entry_date'])
        missing = [f for f in inputs if f not in signals]
        if missing:
            features = pd.read_parquet(SOURCE, columns=['date', 'pair'] + missing)
            signals = signals.merge(features, on=['date', 'pair'], how='left', validate='one_to_one')
            del features
        matched = decisions[['signal_date', 'entry_date', 'pair', 'status']].rename(columns={'signal_date': 'date'})
        signals = signals.merge(matched, on=['date', 'pair'], how='left', validate='one_to_one')
        assert len(signals) == 608 and signals.status.notna().all()
        signals['model_fit'] = signals.date.dt.to_period('M').dt.start_time
        members = sorted(x for x in archive.namelist() if x.startswith(PREFIX) and x.endswith('.joblib'))
        assert len(members) == 90
        for member in members:
            started = time.monotonic()
            raw, sha = checked_read(archive, member, manifest)
            provenance[member] = sha
            model = joblib.load(io.BytesIO(raw))
            raw, _ = checked_read(archive, member.replace('.joblib', '.json'), manifest)
            metadata = json.loads(raw)
            assert model['features'] == inputs
            fit = pd.Timestamp(model['fit'])
            assert pd.Timestamp(metadata['last_training_outcome']) < fit
            proposals = signals.loc[signals.model_fit.eq(fit)]
            tree = model['tree'].tree_
            raw_usage, effective_usage, leaves, redundant = tree_splits(model)
            sets, leaf_actions = action_sets(model)
            root = inputs[int(tree.feature[0])]
            roots[root] += 1
            raw_counts.update(raw_usage.keys())
            effective_counts.update(effective_usage.keys())
            if len(proposals):
                active_counts.update(raw_usage.keys())
                active_effective.update(effective_usage.keys())
                predicted = model['tree'].apply(proposals[inputs])
                np.testing.assert_array_equal(model['choices'][np.searchsorted(model['leaves'], predicted)], proposals.learned_level)
                for position, row in enumerate(proposals.itertuples(index=False)):
                    # sklearn converts prediction features tofloat32 before comparing.
                    x = np.asarray([getattr(row, f) for f in inputs], dtype=np.float32)
                    node, used, retained = 0, set(), set()
                    while tree.children_left[node] != -1:
                        f = inputs[int(tree.feature[node])]
                        used.add(f)
                        if len(sets[node]) > 1:
                            retained.add(f)
                        node = int(tree.children_left[node] if x[tree.feature[node]] <= tree.threshold[node] else tree.children_right[node])
                    assert node == predicted[position]
                    assert leaf_actions[node] == row.learned_level
                    raw_path_counts.update(used)
                    effective_path_counts.update(retained)
                    if row.status == 'entered':
                        accepted_effective.update(retained)
                    path_records.append(dict(signal_date=str(row.date.date()), entry_date=str(row.entry_date.date()),
                        pair=row.pair, model_fit=str(fit.date()), leaf=node, size=row.learned_level/4,
                        status=row.status, raw_path_variables=len(used), retained_path_variables=len(retained),
                        raw_features=';'.join(sorted(used)), retained_features=';'.join(sorted(retained))))
            monthly.append(dict(fit=str(fit.date()), period='validation' if fit.year <= 2021 else 'evaluation',
                training_rows=metadata['training_rows'], latest_training_outcome=metadata['last_training_outcome'],
                leaves=leaves, split_nodes=sum(raw_usage.values()), root_feature=root,
                unique_split_variables=len(raw_usage), retained_variables=len(effective_usage), redundant_split_nodes=redundant,
                proposals=len(proposals), entered=int(proposals.status.eq('entered').sum()),
                first_signal=str(proposals.date.min().date()) if len(proposals) else None,
                last_signal=str(proposals.date.max().date()) if len(proposals) else None,
                first_execution=str(proposals.entry_date.min().date()) if len(proposals) else None,
                last_execution=str(proposals.entry_date.max().date()) if len(proposals) else None,
                split_variables=';'.join(sorted(raw_usage)), retained_split_variables=';'.join(sorted(effective_usage)),
                leaf_sizes=';'.join(f'{node}:{action/4:g}' for node, action in sorted(leaf_actions.items()))))
            budget.rest(started)
    monthly = pd.DataFrame(monthly)
    paths = pd.DataFrame(path_records).sort_values(['signal_date', 'pair']).reset_index(drop=True)
    assert len(paths) == 608 and monthly.proposals.sum() == 608 and monthly.entered.sum() == 207
    n_active = int(monthly.proposals.gt(0).sum())
    usage = pd.DataFrame([dict(feature=f, label=LABELS.get(f, f), trees=raw_counts[f],
        percent_trees=100*raw_counts[f]/90, retained_trees=effective_counts[f],
        active_trees=active_counts[f], percent_active_trees=100*active_counts[f]/n_active,
        active_retained_trees=active_effective[f], first_split_trees=roots[f],
        proposal_paths=raw_path_counts[f], retained_proposal_paths=effective_path_counts[f],
        retained_entered_paths=accepted_effective[f]) for f in inputs]).sort_values(['trees', 'feature'], ascending=[False, True])
    # Independently reproduce the earlier tree census, including pruning.
    old = pd.read_csv(ROOT / 'reports/p25_pca/tree_split_usage.csv').set_index('feature')
    for r in usage.itertuples():
        assert r.trees == old.loc[r.feature, 'months_used']
        assert r.retained_trees == old.loc[r.feature, 'action_effective_months']
    distribution = []
    for frame, population in [(monthly, 'all90fits'), (monthly.loc[monthly.proposals.gt(0)], 'fits_with_proposals')]:
        for col in ['unique_split_variables', 'retained_variables', 'leaves']:
            distribution.extend(summarize_distribution(frame[col].tolist(), col, population))
    for col in ['raw_path_variables', 'retained_path_variables']:
        distribution.extend(summarize_distribution(paths[col].tolist(), col, 'all608proposals'))
    monthly.to_csv(OUT / 'monthly_trees.csv', index=False)
    usage.to_csv(OUT / 'feature_usage.csv', index=False)
    paths.to_csv(OUT / 'proposal_tree_mapping.csv', index=False)
    pd.DataFrame(distribution).to_csv(OUT / 'distributions.csv', index=False)
    summary = dict(monthly_fits=90, fits_with_proposals=n_active, fits_with_entries=int(monthly.entered.gt(0).sum()),
        proposals=608, entries=207, raw_variables_min=int(monthly.unique_split_variables.min()),
        raw_variables_max=int(monthly.unique_split_variables.max()), raw_variables_median=float(monthly.unique_split_variables.median()),
        raw_variables_mean=float(monthly.unique_split_variables.mean()),
        raw_trees_exactly_three=int(monthly.unique_split_variables.eq(3).sum()),
        raw_trees_more_than_three=int(monthly.unique_split_variables.gt(3).sum()),
        retained_variables_min=int(monthly.retained_variables.min()), retained_variables_max=int(monthly.retained_variables.max()),
        retained_variables_median=float(monthly.retained_variables.median()),
        split_nodes=int(monthly.split_nodes.sum()), redundant_split_nodes=int(monthly.redundant_split_nodes.sum()),
        leaves_distribution=dict(sorted(Counter(map(int, monthly.leaves)).items())),
        cross_month_execution_proposals=int((signals.date.dt.to_period('M') != signals.entry_date.dt.to_period('M')).sum()))
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    started = time.monotonic()
    plot(usage, monthly, distribution)
    budget.rest(started)
    write_findings(summary, usage, monthly, distribution)
    (OUT / 'protocol.json').write_text(json.dumps(dict(
        scope='Current selected27-input policy only; original archived90 models, not five experiment arms.',
        model_hashes=provenance, archive_sha256=digest(ARCHIVE), feature_source_sha256=digest(SOURCE),
        script_sha256=digest(Path(__file__)),
        procedure='Original model assigned by signal month; verify all608 saved sizing actions and manual decision-path leaves; count tree-wide and actual-path feature usage.',
        pruning='Collapse any subtree whose leaves all choose the same size; retained counts are structural usage, not causal PnL importance.',
        attribution='Entry occurs next session, so month-end signals can execute using the previous month policy. Accepted inventory keeps original entry sizing.',
        unchanged='No fitting, CEM, portfolio replay, data mutation or policy changes.',
        runtime=dict(sklearn=sklearn.__version__, threads=1, duty_cycle=.2, peak_rss_mib=budget.peak)), indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    print(usage.loc[usage.trees.gt(0), ['feature','trees','retained_trees','first_split_trees','retained_proposal_paths']].to_string(index=False), flush=True)


def plot(usage, monthly, distribution):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    table = usage.loc[usage.trees.gt(0)].iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), constrained_layout=True)
    y = np.arange(len(table))
    axes[0].barh(y+.18, table.trees, height=.36, label='Any split in tree')
    axes[0].barh(y-.18, table.retained_trees, height=.36, label='After collapsing constant-size branches')
    axes[0].set_yticks(y, table.label)
    axes[0].set_xlim(0, 52)
    axes[0].set_xlabel('Number of monthly trees, out of 90')
    axes[0].set_title('Which variables appear in the trees?')
    axes[0].legend(loc='lower right', fontsize=8)
    for pos, value in enumerate(table.trees):
        axes[0].text(value+.4, pos+.18, str(value), va='center', fontsize=8)
    axes[1].barh(y, table.first_split_trees, color='#517c9e')
    axes[1].set_yticks(y, table.label)
    axes[1].set_xlabel('Number of monthly trees, out of 90')
    axes[1].set_title('Which variable is the first split?')
    for pos, value in enumerate(table.first_split_trees):
        if value:
            axes[1].text(value+.2, pos, str(value), va='center', fontsize=8)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(OUT / 'feature_usage.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for col, offset, label in [('unique_split_variables', -.18, 'Whole fitted tree'),
                               ('retained_variables', .18, 'After constant-size pruning')]:
        counts = Counter(monthly[col])
        x = np.arange(0, 8)
        axes[0].bar(x+offset, [counts[i] for i in x], width=.36, label=label)
    axes[0].set_xticks(np.arange(0, 8))
    axes[0].set_xlabel('Number of distinct variables per tree')
    axes[0].set_ylabel('Number of monthly trees')
    axes[0].legend(fontsize=8)
    axes[0].set_title('The trees are not limited to three variables')
    counts = Counter(monthly.leaves)
    axes[1].bar(sorted(counts), [counts[i] for i in sorted(counts)], color='#517c9e')
    axes[1].set_xticks(range(4, 9))
    axes[1].set_xlabel('Number of leaves (states)')
    axes[1].set_ylabel('Number of monthly trees')
    axes[1].set_title('Eight is the maximum state count')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(OUT / 'tree_structure.png', dpi=160)
    plt.close(fig)
    economic = usage.loc[~usage.feature.str.endswith('_available')]
    heat = np.array([[f in set(row.split_variables.split(';')) for row in monthly.itertuples()] for f in economic.feature])
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.8), constrained_layout=True, gridspec_kw={'height_ratios':[3,1]})
    axes[0].imshow(heat, aspect='auto', interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    axes[0].set_yticks(range(len(economic)), economic.label)
    ticks = np.arange(0, len(monthly), 6)
    for ax in axes:
        ax.set_xticks(ticks, monthly.fit.iloc[ticks].str.slice(0,7), rotation=35)
    axes[0].set_title('Variables used by each monthly tree (blue = at least one split)')
    axes[1].bar(np.arange(90), monthly.proposals, label='Qualified proposals', color='#b3cde3')
    axes[1].bar(np.arange(90), monthly.entered, label='Funded entries', color='#225588')
    axes[1].set_xlim(-.5,89.5)
    axes[1].set_ylabel('Proposals / entries')
    axes[1].legend(fontsize=8)
    fig.savefig(OUT / 'monthly_usage.png', dpi=160)
    plt.close(fig)


def write_findings(summary, usage, monthly, distribution):
    lines = ['# Which monthly P25 trees and variables were used?', '',
        f'The current strategy has 90 saved monthly fits, January 2018–June 2025. **{summary["fits_with_proposals"]} trees scored at least one qualified proposal**, and **{summary["fits_with_entries"]} trees produced at least one funded entry**. All 608 proposals and 207 entries were mapped to their original policy. This is a read-only census of the current strategy, not the five representation-experiment variants.', '',
        '## Variables per tree', '',
        f'The full fitted trees use **{summary["raw_variables_min"]}–{summary["raw_variables_max"]} distinct variables**, with median **{summary["raw_variables_median"]:g}** and mean **{summary["raw_variables_mean"]:.2f}**. Only {summary["raw_trees_exactly_three"]} of 90 trees use exactly three variables; {summary["raw_trees_more_than_three"]} use more than three.', '',
        '| Distinct variables in whole fitted tree | Trees | Share of 90 |', '|---|---:|---:|']
    for r in distribution:
        if r['population'] == 'all90fits' and r['measure'] == 'unique_split_variables':
            lines.append(f'| {r["value"]} | {r["count"]} | {r["percent"]:.1f}% |')
    lines += ['', '## Most frequently used variables', '',
        'The first count includes any split in the whole tree. The second collapses branches whose leaves all choose the same size. Counts are structural usage, not measures of incremental Sharpe or causal feature importance.', '',
        '| Variable | Trees with any split | Share of 90 | Trees after constant-size pruning | First split |', '|---|---:|---:|---:|---:|']
    for r in usage.loc[usage.trees.gt(0)].itertuples():
        lines.append(f'| {r.label} (`{r.feature}`) | {r.trees} | {r.percent_trees:.1f}% | {r.retained_trees} | {r.first_split_trees} |')
    lines += ['', f'There are {summary["split_nodes"]} raw split nodes across all 90 trees; {summary["redundant_split_nodes"]} belong to subtrees that always choose the same size. Pruning yields {summary["retained_variables_min"]}–{summary["retained_variables_max"]} distinct variables per sizing tree, median {summary["retained_variables_median"]:g}. A value of zero means that monthly policy chooses one size regardless of its tree partition.', '',
        '## Which tree was used for a trade?', '',
        'The model is selected by the **signal month**. The signal is evaluated at its close, and execution is on the next trading session. A month-end signal can therefore execute in the next month using the earlier month policy. The model is never chosen by later performance. Each accepted trade retains its original entry sizing while the book is marked and hedged.', '',
        f'{summary["cross_month_execution_proposals"]} proposals cross a calendar-month boundary between signal and execution. Monthly dates with zero proposals represent fitted models that did not score a qualified proposal. The daily book continues during months without new entries and through the August 2025 runoff.', '',
        'The full 90-row **monthly_trees.csv** lists fit date, proposal and funded-entry counts, first/last signal and execution dates, training cutoff, states, first split, every split variable and every leaf sizing action. **proposal_tree_mapping.csv** lists all 608 proposals with their exact model, leaf, size, admission result and variables visited along their decision path.', '',
        '## Whole tree versus one decision path', '',
        'Different branches can use different variables. A trade visits one root-to-leaf path, so the number of variables it encounters can be smaller than the number in the whole tree. The following counts track actual proposal paths rather than all possible branches.', '',
        '| Distinct variables on a proposal path | Proposals | Share of 608 |', '|---|---:|---:|']
    for r in distribution:
        if r['population'] == 'all608proposals' and r['measure'] == 'raw_path_variables':
            lines.append(f'| {r["value"]} | {r["count"]} | {r["percent"]:.1f}% |')
    lines += ['', '## Checks and reproduction', '',
        'All 90 original models were read from the hash-verified archive with the saved sklearn version. The earlier feature-frequency census was reproduced exactly. Tree application and independent path traversal reproduce all 608 saved leaf assignments and sizing actions. No model was fitted, no CEM search was run, and no portfolio was replayed.', '',
        '`python -m scripts.diagnose_p25_tree_usage` in the pinned environment. One numerical thread, 20% duty cycle, 384 MiB RSS stop. Charts: feature_usage.png, tree_structure.png and monthly_usage.png. Raw statistics: feature_usage.csv, monthly_trees.csv, distributions.csv and proposal_tree_mapping.csv. Provenance: protocol.json and summary.json.', '']
    (OUT / 'findings.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
