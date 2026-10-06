"""Compare new spread directions at the actual frozen rule's entry dates."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scripts.run_atm65_ablation import read_actions, write_json
from scripts.run_contract_walkforward import digest


def main():
    report = Path('reports/tail_signal_identity'); report.mkdir(parents=True, exist_ok=True)
    source = Path('reports/tail_put25_strategy/entered_cohorts.csv')
    entries = pd.read_csv(source, parse_dates=['signal_date'])
    entries = entries[entries.model.eq('rule') & entries.period.eq('full')]
    rows = []; sources = {str(source): digest(source)}
    for pair, group in entries.groupby('pair'):
        old = Path('artifacts/contracts/tail_put25_strategy') / pair
        new = Path('artifacts/contracts/tail_put25_pooled') / pair
        actions = Path('artifacts/contracts/tail_put25_pooled_complete') / pair / 'ungated_spread/actions.csv'
        for p in [old / 'rule/actions.csv', old / 'states.parquet', new / 'states.parquet', actions]:
            sources[str(p)] = digest(p)
        legacy = read_actions(old / 'rule/actions.csv')
        broad = read_actions(actions)
        before = pd.read_parquet(old / 'states.parquet')
        after = pd.read_parquet(new / 'states.parquet')
        for date in group.signal_date:
            a, b = int(legacy.loc[date]), int(broad.get(date, 0))
            status = 'same' if a == b else ('flat' if b == 0 else 'reversed')
            if after.loc[date, 'direction'] != 0:
                np.testing.assert_allclose(before.loc[date, 'wing_iv_gap'],
                    after.loc[date, 'iv_mean'] * after.loc[date, 'oriented_iv_gap'] * after.loc[date, 'direction'], atol=1e-12)
            rows.append(dict(pair=pair, signal_date=str(date.date()), status=status,
                old_code=a, broad_code=b, old_rank=float(before.loc[date, 'wing_iv_gap_pct']),
                broad_rank=float(after.loc[date, 'rank']), broad_beta_eligible=bool(after.loc[date, 'eligible'])))
    pd.DataFrame(rows).to_csv(report / 'frozen_entry_signal_overlap.csv', index=False)
    counts = pd.Series([r['status'] for r in rows]).value_counts().to_dict()
    assert sum(counts.values()) == len(entries) == 129
    for p, sha in sources.items(): assert digest(p) == sha
    write_json(report / 'audit.json', dict(status='complete', old_actual_entries=len(rows),
        broad_signal_status=counts, sources_verified=len(sources)))
    write_json(report / 'source_hashes.json', sources)
    print(json.dumps(counts))


if __name__ == '__main__': main()
