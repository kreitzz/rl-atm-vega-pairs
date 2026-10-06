"""Read-only causal-state, common-input and realized-entry coverage checks."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_contract_walkforward import digest
from scripts.run_tail_put25_atm_transfer import OUT, REPORT
from scripts.run_atm65_volume_oi import EMPLOYEE
from scripts.tail_put25_atm_features import columns

DEST = Path('reports/tail_put25_atm_transfer_diagnostics')


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    hashes = json.loads((REPORT / 'output_hashes.json').read_text())
    for path, sha in hashes.items():
        assert digest(path) == sha, f'Finished output changed: {path}'
    cov = pd.read_csv(REPORT / 'coverage.csv')
    old_cov = pd.read_csv('reports/tail_put25_strategy/coverage.csv')
    fits = pd.read_csv(REPORT / 'fits.csv')
    states = {}; checks = []; models = ['rl_atm_state', 'rl_atm_state_rank']
    for path in OUT.glob('*/feature_columns.json'):
        folder = path.parent; state = pd.read_parquet(folder / 'states.parquet')
        original = pd.read_parquet(EMPLOYEE / folder.name / 'states.parquet')
        state_columns = json.loads(path.read_text()); states[folder.name] = state
        changed = set(['iv_amd', 'iv_nvda', 'spread', 'rv_adjusted_value', 'spread_reversion',
                       'commission_vol_entry', 'commission_a', 'commission_b',
                       'dte_a', 'dte_b', 'moneyness_a', 'moneyness_b'])
        changed.update(c for c in original if c.startswith(('profit_', 'count_')))
        context = [c for c in columns(original, employee=len(state_columns[models[0]]) == 49) if c not in changed]
        pd.testing.assert_frame_equal(state[context], original[context])
        # Future-mark finiteness does not determine deployment observation.
        common = np.isfinite(state[state_columns[models[0]]]).all(axis=1)
        ranked = np.isfinite(state[state_columns[models[1]]]).all(axis=1)
        np.testing.assert_array_equal(common & ranked, state.feature_available)
        actions = [pd.read_csv(folder / m / 'actions.csv', index_col=0, parse_dates=[0]) for m in models]
        identical = actions[0].equals(actions[1])
        checks.append(dict(pair=folder.name, inherited_context_columns=len(context),
                           feature_rows=int(state.feature_available.sum()),
                           extra_rank_warmup_rows=int((common & ~ranked).sum()), actions_identical=identical))
    # Diagnose the fixed rule's actual entries, without selecting or refitting
    # a policy using those outcomes.
    entries = pd.read_csv('reports/tail_put25_strategy/entered_cohorts.csv', parse_dates=['signal_date'])
    entries = entries[(entries.model == 'rule') & (entries.period == 'full')].copy()
    observed = []
    for row in entries.itertuples():
        state = states.get(row.pair)
        observed.append(bool(state is not None and row.signal_date in state.index
                             and state.loc[row.signal_date, 'feature_available']))
    entries['full_state_observed'] = observed
    entries.to_csv(DEST / 'rule_entry_state_coverage.csv', index=False)
    pd.DataFrame(checks).to_csv(DEST / 'state_checks.csv', index=False)
    pairs = pd.read_csv(REPORT / 'pairs.csv')
    pairs[(pairs.model == models[0]) & (pairs.period == 'full') & pairs.entries.gt(0)].to_csv(DEST / 'active_pairs.csv', index=False)
    summary = dict(status='complete', finished_output_files_verified=len(hashes),
        inherited_context_panels_verified=len(checks),
        identical_actions_all_full_state_pairs=bool(all(r['actions_identical'] for r in checks)),
        previous_spread_state_rows=int(old_cov.observed_rows.fillna(0).sum()),
        full_state_rows=int(cov.observed_rows.fillna(0).sum()),
        previous_evaluation_memory_permitted_rows=int(old_cov.evaluated_memory_permitted_rows.fillna(0).sum()),
        full_state_evaluation_memory_permitted_rows=int(cov.evaluated_memory_permitted_rows.fillna(0).sum()),
        fixed_rule_entries=len(entries), fixed_rule_entries_with_full_state=int(sum(observed)),
        full_state_fits={m: fits[fits.model.eq(m)].status.value_counts().to_dict() for m in models},
        outcome_reconciliation_cohorts=int(cov.recovered_scan_cohorts.fillna(0).sum()),
        maximum_old_outcome_error=float(cov.max_label_error.fillna(0).max()))
    (DEST / 'audit.json').write_text(json.dumps(summary, indent=2) + '\n')
    (DEST / 'source_hashes.json').write_text(json.dumps({str(p): digest(p) for p in [
        Path(__file__), REPORT / 'output_hashes.json', REPORT / 'coverage.csv',
        Path('reports/tail_put25_strategy/coverage.csv'), Path('reports/tail_put25_strategy/entered_cohorts.csv')
    ]}, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
