"""Compare fee-sensitive permissions, fits and realized pair results."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.retrain_atm_commission import OUTPUT, REPORT, SOURCE, FROZEN_REPORT
from scripts.curve_trading.policy import ATM
from scripts.run_contract_walkforward import digest


def main():
    assert json.loads((OUTPUT / 'task_status.json').read_text())['status'] == 'complete'
    inputs = json.loads((OUTPUT / 'inputs.json').read_text())
    for path, expected in inputs['code_and_sources'].items():
        assert digest(path) == expected, f'Changed source: {path}'
    defs = inputs['definitions']; rows = []; old_fits = []; ledger_error = memory_error = 0.
    for d in defs:
        folder = OUTPUT / d['pair']; old_folder = SOURCE / d['pair']
        for period in ['full', 'recent']:
            status = json.loads((folder / period / 'status.json').read_text())
            assert status['status'] == 'complete'
            ledger_error = max(ledger_error, status.get('ledger_error', 0.))
        if not (folder / 'gates.npz').exists(): continue
        check = json.loads((folder / 'rebuild_checks.json').read_text())
        memory_error = max(memory_error, check['original_memory_max_error'])
        with np.load(folder / 'gates.npz') as new, np.load(old_folder / 'gates_control.npz') as old:
            np.testing.assert_array_equal(new['dates'], old['dates'])
            np.testing.assert_array_equal(new['available'], old['available'][ATM])
            np.testing.assert_array_equal(new['ends'], old['ends'][ATM])
            before, after = old['gates'][ATM], new['gates']
            assert not (before & ~after).any(), 'Cheaper fees removed a memory permission'
            panel = pd.read_parquet(folder / 'states.parquet', columns=['eligible'])
            valid = panel.eligible.to_numpy(bool)[:, None] & new['available'][:, None]
            row = dict(pair=d['pair'], label=d['label'],
                       permitted_direction_days_065=int((before & valid).sum()),
                       permitted_direction_days_003=int((after & valid).sum()),
                       newly_permitted_direction_days=int((~before & after & valid).sum()))
            rows.append(row)
        try:
            fits = pd.read_csv(old_folder / 'atm_control/fits.csv')
        except pd.errors.EmptyDataError: continue
        if len(fits): old_fits.append(fits)
    pd.DataFrame(rows).to_csv(REPORT / 'memory_gate_changes.csv', index=False)
    previous = pd.concat(old_fits, ignore_index=True)
    current = pd.read_csv(REPORT / 'fits.csv')
    counts = pd.concat([previous.status.value_counts().rename('trained_065'),
                        current.status.value_counts().rename('trained_003')], axis=1).fillna(0).astype(int)
    counts.to_csv(REPORT / 'gate_comparison.csv')
    new_pairs = pd.read_csv(REPORT / 'pairs.csv')
    old_pairs = pd.read_csv(FROZEN_REPORT / 'pairs.csv')
    changes = []
    for period in ['full', 'recent']:
        before = old_pairs[old_pairs.variant.eq(f'{period}_003')].drop(columns='variant')
        after = new_pairs[new_pairs.period.eq(period)].drop(columns='period')
        together = before.merge(after, on=['pair', 'label'], suffixes=('_frozen', '_retrained'), validate='one_to_one')
        assert len(together) == len(defs)
        together['period'] = period
        together['entry_change'] = together.entries_retrained - together.entries_frozen
        together['pnl_change'] = together.net_excess_pnl_retrained - together.net_excess_pnl_frozen
        together['newly_active_pair'] = together.entries_frozen.eq(0) & together.entries_retrained.gt(0)
        together['newly_inactive_pair'] = together.entries_frozen.gt(0) & together.entries_retrained.eq(0)
        changes.append(together)
    changes = pd.concat(changes, ignore_index=True)
    changes.to_csv(REPORT / 'pair_changes.csv', index=False)
    audit = dict(status='complete', pairs=len(defs), memory_pairs_checked=len(rows),
                 largest_original_memory_error=memory_error, largest_ledger_error=ledger_error,
                 old_permitted_direction_days=sum(r['permitted_direction_days_065'] for r in rows),
                 new_permitted_direction_days=sum(r['permitted_direction_days_003'] for r in rows),
                 newly_permitted_direction_days=sum(r['newly_permitted_direction_days'] for r in rows),
                 lost_memory_permissions=0,
                 new_active_full=int(changes.loc[changes.period.eq('full'), 'newly_active_pair'].sum()),
                 lost_active_full=int(changes.loc[changes.period.eq('full'), 'newly_inactive_pair'].sum()))
    (REPORT / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2), flush=True)
    print(counts.to_string(), flush=True)


if __name__ == '__main__': main()
