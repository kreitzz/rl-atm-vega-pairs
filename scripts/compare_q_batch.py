"""Re-run existing Q+P&L and CEM references and paired block-bootstrap differences.

Run after run_q_batch. Reference implementations are intentionally unchanged.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.q_panel import build_pair
from scripts.run_q_batch import respect_gaps, slices
from scripts.run_q_learning import (Market, cohort_pnl, daily_sharpe, folds,
                                    run_current, run_hybrid, simulate)


def paired_interval(a, b, seed=2026, block=63, draws=2000):
    """Same resampled calendar blocks for both policies, including flat days."""
    rng = np.random.default_rng(seed)
    values = []
    n = len(a)
    for _ in range(draws):
        start = rng.integers(n, size=n//block+1)
        ix = ((start[:, None]+np.arange(block)) % n).ravel()[:n]
        values.append(daily_sharpe(a[ix])-daily_sharpe(b[ix]))
    return list(np.quantile(values, [.025, .975]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pair', nargs=2, default=['AMD', 'NVDA'])
    ap.add_argument('--out', type=Path, default=Path('artifacts/q_batch'))
    ap.add_argument('--cached', action='store_true', help='Reuse reference daily P&L already in output directory')
    ap.add_argument('--references', nargs='+', choices=['legacy_q_pnl', 'current_cem'], default=['legacy_q_pnl', 'current_cem'])
    ap.add_argument('--respect-gaps', action='store_true', help='Gap sensitivity for CEM; requires --references current_cem')
    args = ap.parse_args()
    out = args.out/'_'.join(args.pair)
    if args.respect_gaps and 'legacy_q_pnl' in args.references:
        ap.error('--respect-gaps currently supports only the CEM reference')
    if not args.cached:
        parts = {name: [] for name in args.references}
        panel = build_pair(*args.pair)
        if args.respect_gaps:
            panel = respect_gaps(panel, *args.pair)
        for year, tr, va, te in folds(panel):
            if 'legacy_q_pnl' in parts:
                dec, _, _ = run_hybrid(tr, va, te, use_pnl=True)
                qp, _ = simulate(Market(te), dec)
                parts['legacy_q_pnl'].append(pd.Series(qp, index=te.index, name='pnl'))
            if 'current_cem' in parts:
                sizes, _ = run_current(tr, va, te)
                cp = np.concatenate([cohort_pnl(Market(te.iloc[s]), sizes[s]) for s in slices(te)])
                parts['current_cem'].append(pd.Series(cp, index=te.index, name='pnl'))
            print(year, {name: daily_sharpe(chunks[-1]) for name, chunks in parts.items()}, flush=True)
        for name, chunks in parts.items():
            pd.concat(chunks).to_csv(out/f'daily_{name}.csv', index_label='date')
    result = []
    selected = pd.read_csv(out/'daily_validation_selected.csv', index_col=0).pnl
    for name in args.references:
        reference = pd.read_csv(out/f'daily_{name}.csv', index_col=0).pnl
        if not selected.index.equals(reference.index):
            raise ValueError('Reference and new policy dates differ')
        result.append({'reference': name, 'reference_sharpe': daily_sharpe(reference),
                       'selected_sharpe': daily_sharpe(selected),
                       'difference': daily_sharpe(selected)-daily_sharpe(reference),
                       'difference_ci': paired_interval(selected.to_numpy(), reference.to_numpy())})
    (out/'comparison.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
