"""Trade counts and inventory rejection diagnostics for the completed silo."""
from pathlib import Path
import hashlib
import json

import pandas as pd


def main():
    root = Path('artifacts/contracts/tail_put10_strategy')
    report = Path('reports/tail_put10_strategy')
    audit = json.loads((report / 'audit.json').read_text())
    assert audit['status'] == 'complete' and not audit['unresolved']
    yearly, decisions, entries = [], [], []
    for model in ['rule', 'rl_sized']:
        for period in ['full', 'recent']:
            frames = []
            for path in root.glob(f'*/{model}/{period}/decisions.csv'):
                frame = pd.read_csv(path, parse_dates=['signal_date', 'entry_date'])
                frame['pair'] = path.parents[2].name
                frames.append(frame)
            frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            if frame.empty:
                continue
            proposed = frame[frame.action.ne(0)]
            counts = proposed.status.value_counts().rename_axis('status').reset_index(name='count')
            decisions.append(counts.assign(model=model, period=period))
            entered = proposed[proposed.status.eq('entered')].copy()
            entered['year'] = entered.entry_date.dt.year
            entries.append(entered.assign(model=model, period=period))
            counts = entered.groupby('year').size().reindex(range(2018 if period == 'full' else 2022, 2026), fill_value=0)
            yearly.append(counts.rename_axis('year').reset_index(name='entries').assign(model=model, period=period))
    pd.concat(yearly, ignore_index=True).to_csv(report / 'entries_by_year.csv', index=False)
    pd.concat(decisions, ignore_index=True).to_csv(report / 'decisions.csv', index=False)
    pd.concat(entries, ignore_index=True).to_csv(report / 'entered_cohorts.csv', index=False)
    # Hash outputs so later revisions cannot silently replace this first result.
    paths = [Path(__file__), Path('tests/test_tail_put_policy.py')]
    paths += list(root.rglob('*.json')) + list(root.rglob('*.csv')) + list(root.rglob('*.parquet'))
    paths += list(root.rglob('*.npz')) + list(root.rglob('*.joblib')) + list(report.glob('*.csv'))
    paths += [report / 'audit.json', report / 'findings.txt']
    hashes = {}
    for path in sorted(set(paths)):
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(chunk)
        hashes[str(path)] = h.hexdigest()
    (report / 'output_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')


if __name__ == '__main__':
    main()
