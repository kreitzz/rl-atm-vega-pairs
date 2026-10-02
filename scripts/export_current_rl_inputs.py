"""Stream the frozen best's decision inputs into a shareable CSV.

No future rewards, option outcome labels or new fits. One pair in memory.
"""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_atm65_ablation import NN, features
from scripts.run_atm65_employee_sentiment import OUT, REPORT, ADDED
from scripts.run_curve_trading import screen_dates


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def decision_inputs(panel, definition, gates, employee_enabled):
    columns = features(panel) + ADDED
    assert len(columns) == 49 and len(set(columns)) == 49
    # Entry IVs and timing are useful context, although not direct model states.
    extra = ['iv_amd', 'iv_nvda', 'commission_a', 'commission_b', 'vix_date', 'eligible']
    result = panel[columns + extra].copy()
    result.insert(0, 'date', panel.index)
    result.insert(1, 'pair', definition['pair'])
    result.insert(2, 'label', definition['label'])
    result.insert(3, 'secid_a', definition['secids'][0])
    result.insert(4, 'secid_b', definition['secids'][1])
    result['employee_features_enabled'] = employee_enabled
    result['screen_allowed'] = screen_dates(definition, panel.index)
    result['contract_available'] = gates['available']
    result['short_memory_gate'] = gates['gates'][:, 0]
    result['long_memory_gate'] = gates['gates'][:, 1]
    assert not any(c.startswith(('reward', 'forward_', 'target_')) for c in result)
    return result.reset_index(drop=True)


def main():
    inputs = json.loads((OUT / 'inputs.json').read_text())
    coverage = pd.read_csv(REPORT / 'coverage.csv').set_index('pair')
    dest = Path('data/current_rl_inputs.csv.gz')
    temporary = dest.with_suffix('.tmp')
    sources = {}; roster = []; total = 0; columns = None
    with temporary.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, compresslevel=1, mtime=0) as compressed:
            for definition in inputs['definitions']:
                pair = definition['pair']; path = OUT / pair / 'states.parquet'
                employee = bool(coverage.loc[pair, 'retrain'])
                row = dict(pair=pair, label=definition['label'], secid_a=definition['secids'][0],
                           secid_b=definition['secids'][1], employee_features_enabled=employee,
                           state_rows=0, first_date=None, last_date=None,
                           screen_years=json.dumps(definition.get('screen_years')))
                if path.exists():
                    panel = pd.read_parquet(path)
                    gate_path = NN / pair / 'fee_065/gates.npz'
                    with np.load(gate_path) as gates:
                        result = decision_inputs(panel, definition, gates, employee)
                    if columns is None: columns = list(result)
                    assert list(result) == columns
                    result.to_csv(compressed, index=False, header=total == 0, date_format='%Y-%m-%d')
                    row.update(state_rows=len(result), first_date=str(panel.index.min().date()),
                               last_date=str(panel.index.max().date()))
                    total += len(result)
                    sources[str(path)] = digest(path); sources[str(gate_path)] = digest(gate_path)
                roster.append(row)
    temporary.replace(dest)
    pd.DataFrame(roster).to_csv('data/current_rl_pairs.csv', index=False)
    manifest = dict(checkpoint='config/atm30_fee065_employee_2026-10-01_checkpoint.json',
                    rows=total, roster_pairs=len(roster), pairs_with_states=sum(r['state_rows'] > 0 for r in roster),
                    csv=str(dest), bytes=dest.stat().st_size, sha256=digest(dest), columns=columns,
                    model_features=features(panel), optional_employee_features=ADDED,
                    employee_features_rule='Only sleeves flagged employee_features_enabled use the two review states.',
                    naming='Legacy amd/nvda column suffixes denote first/second legs, not ticker identities.',
                    labels='No forward rewards or outcome targets. profit_/count_ columns are past profitability memory used as state.',
                    availability='Backward lags enforced; archived historical vendor vintages remain unverified.',
                    replay='Decision inputs only. Rebuilding rewards and the option replay also requires licensed historical contract, stock, dividend and rate data.',
                    structural_cash='Roster pairs without state rows are structural cash sleeves, not missing training records.',
                    license='Academic use only; vendor and institutional terms apply. No additional redistribution rights granted.',
                    source_sha256=sources)
    Path('data/current_rl_data_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: manifest[k] for k in ['rows','roster_pairs','pairs_with_states','bytes','sha256']}), flush=True)


if __name__ == '__main__':
    main()
