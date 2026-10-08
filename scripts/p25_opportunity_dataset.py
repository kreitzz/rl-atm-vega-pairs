"""Counterfactual full-size P25 opportunity data, without rank/memory entry gates.

Archived labels are cash-relative standalone dollars divided by the reference
$450-per-vol-point vega target, with $0.65 option commission. They are NOT
funded portfolio returns or returns divided by actual entry gross vega. The
compact archive lacks the per-contract entry quantities needed to recost all
labels exactly to $0.03. Keep that distinction explicit in every experiment.
"""
from pathlib import Path
import hashlib
import json
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/submission/artifacts/contracts/same_structure_smile/P25_rows.parquet'
LABELS = ROOT / 'data/submission/artifacts/contracts/uniform_smile/pairs'
CALENDAR = ROOT / 'reports/p25_memory_ablation/formation_calendar.csv'
FEATURES = [
    'iv_mean', 'oriented_iv_gap', 'oriented_rank', 'oriented_vrp_gap',
    'rv_mean', 'beta', 'correlation', 'dte_mean', 'oriented_dte_gap',
    'commission_vol_points', 'vix', 'profit_91d', 'count_91d',
    'profit_365d', 'count_365d',
]
LABEL_CONVENTION = {
    'target': 'Lifetime standalone cash-relative net P&L / reference entry vega 450',
    'option_commission': 0.65,
    'option_penalty_vol_points': 0.1,
    'stock_hedge_bps': 1,
    'actual_entry_vega_available': False,
    'recosted_to_live_commission': False,
    'known_no_entry_is_not_a_training_outcome': True,
    'missing_or_unresolved_outcome_is_not_zero': True,
    'features': 'Signal-close state; memory contains strictly prior completed outcomes',
    'universe_limitation': 'Inherited retrospective roster and formation/context calendar',
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def orient_candidates(meta):
    """Two choices per signal, with every directional feature recomputed.

    Middle-rank rows had legacy direction zero. Reusing their old oriented
    fields would erase the signal and accidentally select the short memory.
    Future label completeness is retained as training metadata only.
    """
    meta = meta.reset_index(drop=True).copy()
    if meta.duplicated(['date', 'pair']).any():
        raise ValueError('Duplicate source signal identity')
    meta['candidate_id'] = np.arange(len(meta), dtype=np.int64)
    parts = []
    for direction, side, label_action in [(1, 'long', 8), (-1, 'short', 1)]:
        x = meta.copy()
        x['direction'] = direction
        x['label_action'] = label_action
        x['oriented_iv_gap'] = direction * x.iv_gap_relative
        x['oriented_rank'] = direction * (x.iv_rank - 0.5)
        x['rank_strength'] = abs(x.iv_rank - 0.5) * 2
        x['oriented_vrp_gap'] = direction * x.vrp_gap_relative
        x['oriented_dte_gap'] = direction * x.dte_gap
        for days in [91, 365]:
            for prefix in ['profit', 'count']:
                x[f'{prefix}_{days}d'] = x[f'{prefix}_{side}_{days}d']
                name = f'{prefix}_{side}_{days}d_available'
                if name in x:
                    x[f'{prefix}_{days}d_available'] = x[name]
        x['target_net_per_reference_vega'] = x[f'target_{side}'].where(x.finite_label)
        x['target'] = x.target_net_per_reference_vega
        x['outcome_end'] = x.end
        x['train_eligible'] = x.finite_label & np.isfinite(x.target_net_per_reference_vega)
        x['legacy_selected'] = x.signal_eligible & x.rule_direction.eq(direction)
        x['legacy_rank_selected'] = x.rule_direction.eq(direction)
        x['base_level'] = 4
        if not np.isfinite(x[FEATURES].to_numpy(dtype=float)).all():
            raise ValueError('Nonfinite model input')
        parts.append(x)
    result = pd.concat(parts, ignore_index=True)
    return result.sort_values(['date', 'pair', 'direction']).reset_index(drop=True)


def evaluation_mask(meta, calendar, start='2018-01-02', end='2025-06-30'):
    """Signal eligibility never uses future completion or next-session entry."""
    keys = pd.MultiIndex.from_frame(calendar[['pair', 'date']]).unique()
    identity = pd.MultiIndex.from_frame(meta[['pair', 'date']])
    return (meta.observed & meta.eligible & meta.context &
            meta.date.between(start, end) & identity.isin(keys))


def load_paths(rows, label_root=LABELS, check=None):
    """Return archived daily excess-P&L/reference-vega paths in input row order.

    NaN unresolved paths remain NaN. Callers must not silently sum these into
    apparent zero-return observations, or omit them from prediction eligibility.
    """
    rows = rows.reset_index(drop=True)
    output = None
    for pair, indexes in rows.groupby('pair', sort=False).groups.items():
        labels = np.load(Path(label_root) / pair / 'P25_labels.npy', mmap_mode='r')
        if output is None:
            output = np.empty((len(rows), labels.shape[-1]), dtype=np.float32)
        g = rows.loc[indexes]
        output[indexes] = labels[g.label_row.to_numpy(int), g.label_action.to_numpy(int)]
        del labels
        if check:
            check()
    return output if output is not None else np.empty((0, 57), dtype=np.float32)


def build_dataset(output_dir, check=None):
    """Write auditable candidates and coverage; no fitting or policy selection."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(SOURCE)
    calendar = pd.read_csv(CALENDAR, parse_dates=['date'])
    meta['evaluation_eligible'] = evaluation_mask(meta, calendar)
    meta['archived_path_status'] = ''
    label_hashes = {}
    max_error = 0.0
    for pair, g in meta.groupby('pair', sort=False):
        path = LABELS / pair / 'P25_labels.npy'
        labels = np.load(path, mmap_mode='r')
        native = labels[g.label_row.to_numpy(int)]
        all_finite = np.isfinite(native).all(axis=(1, 2))
        # A rejected size can leave ~1e-12 cash-minus-benchmark roundoff when
        # another ladder size entered. Classification alone uses tolerance;
        # stored paths are never silently changed.
        all_zero = all_finite & (abs(native) <= 1e-8).all(axis=(1, 2))
        full_zero = all_finite & (abs(native[:, [1, 8]]) <= 1e-8).all(axis=(1, 2))
        status = np.where(g.finite_label.to_numpy(bool), 'complete_full_size',
                 np.where(~all_finite, 'unresolved_nonfinite',
                 np.where(all_zero, 'all_actions_zero_no_entry',
                 np.where(full_zero, 'full_size_zero_other_actions_nonzero',
                          'finite_nonzero_but_invalid_full_size'))))
        meta.loc[g.index, 'archived_path_status'] = status
        for action, side in [(8, 'long'), (1, 'short')]:
            actual = labels[g.label_row.to_numpy(int), action].sum(axis=1)
            expected = g[f'target_{side}'].to_numpy()
            np.testing.assert_allclose(actual, expected, rtol=0, atol=0, equal_nan=True)
            valid = g.finite_label.to_numpy(bool)
            if not np.isfinite(actual[valid]).all():
                raise ValueError('Finite-label flag includes unavailable outcome')
            if valid.any():
                max_error = max(max_error, float(np.max(abs(actual[valid] - expected[valid]))))
        label_hashes[str(path.relative_to(ROOT))] = digest(path)
        del labels, native
        if check:
            check()
    rows = orient_candidates(meta)
    rows.to_parquet(out / 'candidates.parquet', index=False)
    evaluation = meta.evaluation_eligible
    archive = ROOT / 'data/p25_memory_ablation_experiment.zip'
    with zipfile.ZipFile(archive) as z:
        preparation = json.loads(z.read('artifacts/contracts/p25_memory_ablation/prepared.json'))
    cached_ids = set(preparation['quote_ids'])
    contract_coverage = meta.optionid_a.isin(cached_ids) & meta.optionid_b.isin(cached_ids)
    # Coverage is an audit output only. Never let retrospective quote coverage
    # change the candidate set, the model inputs, or the signal eligibility.
    summary = dict(
        source_pair_dates=len(meta), source_pairs=int(meta.pair.nunique()),
        complete_pair_dates=int(meta.finite_label.sum()),
        candidate_direction_rows=len(rows),
        evaluation_pair_dates=int(evaluation.sum()),
        evaluation_pairs=int(meta.loc[evaluation, 'pair'].nunique()),
        evaluation_complete_pair_dates=int((evaluation & meta.finite_label).sum()),
        evaluation_middle_rank_pair_dates=int((evaluation & meta.rule_direction.eq(0)).sum()),
        evaluation_middle_rank_complete_pair_dates=int((evaluation & meta.rule_direction.eq(0) & meta.finite_label).sum()),
        evaluation_legacy_selected_pair_dates=int((evaluation & meta.signal_eligible).sum()),
        funded_quote_contract_coverage=dict(
            both_contract_ids=int((evaluation & contract_coverage).sum()),
            middle_rank_both_contract_ids=int((evaluation & contract_coverage & meta.rule_direction.eq(0)).sum()),
            missing_at_least_one_contract=int((evaluation & ~contract_coverage).sum()),
            note='Contract ID presence only; not a day-by-day completeness or entry-validity test. Not used to select candidates.'),
        label_target_max_absolute_error=max_error,
        path_zero_classification_tolerance=1e-8,
        archived_path_status_all=meta.archived_path_status.value_counts().to_dict(),
        archived_path_status_evaluation=meta.loc[evaluation, 'archived_path_status'].value_counts().to_dict(),
        features=FEATURES, label_convention=LABEL_CONVENTION,
        sources={str(p.relative_to(ROOT)): digest(p) for p in [SOURCE, CALENDAR, archive, Path(__file__)]},
        label_sources=label_hashes,
    )
    (out / 'dataset_audit.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return rows, summary
