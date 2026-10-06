"""Combine frozen ATM and put-tail rule books; no fitting or weight search."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from scripts.run_contract_walkforward import digest
from scripts.run_selected_pair_backtest import scores


ATM = Path('reports/atm65_employee_sentiment')
TAIL = Path('reports/tail_put10_strategy')
OUT = Path('reports/atm_tail_stack')
PERIODS = ['full', 'recent']


def add_returns(book):
    result = book[['nav', 'cash_benchmark']].copy()
    result['return_'] = result.nav.pct_change().fillna(0)
    result['excess_return'] = result.return_ - result.cash_benchmark.pct_change().fillna(0)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path('config/atm30_fee065_employee_2026-10-01_checkpoint.json')
    checkpoint = json.loads(checkpoint_path.read_text())
    tail_hashes_path = TAIL / 'output_hashes.json'
    tail_hashes = json.loads(tail_hashes_path.read_text())
    main_sources = [Path(__file__), checkpoint_path, tail_hashes_path,
                    ATM / 'comparison.csv', TAIL / 'comparison.csv',
                    ATM / 'pairs.csv', TAIL / 'pairs.csv', TAIL / 'audit.json']
    for period in PERIODS:
        main_sources += [ATM / f'all_rl_sentiment_{period}_portfolio.csv', TAIL / f'rule_{period}_portfolio.csv']
    signature = {str(p): digest(p) for p in main_sources}
    for p in main_sources:
        expected = checkpoint['files'].get(str(p)) or tail_hashes.get(str(p))
        if expected:
            assert signature[str(p)] == expected, f'Saved input changed: {p}'
    assert json.loads((TAIL / 'audit.json').read_text())['status'] == 'complete'
    atm_summary = pd.read_csv(ATM / 'comparison.csv').query("universe == 'all' and model == 'rl_sentiment'")
    tail_summary = pd.read_csv(TAIL / 'comparison.csv').query("model == 'rule'")
    atm_pairs = pd.read_csv(ATM / 'pairs.csv').query("universe == 'all' and model == 'rl_sentiment'")
    tail_pairs = pd.read_csv(TAIL / 'pairs.csv').query("model == 'rule'")
    summaries, annual, dependence, concentration, pair_tables = [], [], [], [], []
    for period in PERIODS:
        a = pd.read_csv(ATM / f'all_rl_sentiment_{period}_portfolio.csv', index_col=0, parse_dates=[0])
        t = pd.read_csv(TAIL / f'rule_{period}_portfolio.csv', index_col=0, parse_dates=[0])
        assert a.index.equals(t.index) and a.index.is_unique and a.index.is_monotonic_increasing
        np.testing.assert_allclose(a.cash_benchmark, t.cash_benchmark, atol=1e-6, rtol=0)
        assert np.isfinite(a[['nav', 'cash_benchmark']]).all().all()
        assert np.isfinite(t[['nav', 'cash_benchmark']]).all().all()
        capital = float(a.nav.iloc[0]); assert abs(capital - 345_000_000) < 1e-6
        books = {'atm': add_returns(a), 'tail_rule': add_returns(t),
                 'combined': add_returns(a[['nav', 'cash_benchmark']] + t[['nav', 'cash_benchmark']])}
        # Independent capital buffers preserve the recorded sizing, costs, funding,
        # and fills. Combined capital is the sum of both original book capitals.
        ap = atm_pairs[atm_pairs.period.eq(period)].copy()
        tp = tail_pairs[tail_pairs.period.eq(period)].copy()
        assert set(ap.pair) == set(tp.pair) and len(ap) == len(tp) == 345
        pc = ap[['pair', 'label', 'entries', 'net_excess_pnl']].merge(
            tp[['pair', 'entries', 'net_excess_pnl']], on='pair', suffixes=('_atm', '_tail'), validate='one_to_one')
        pc['entries_combined'] = pc.entries_atm + pc.entries_tail
        pc['net_excess_pnl_combined'] = pc.net_excess_pnl_atm + pc.net_excess_pnl_tail
        pair_tables.append(pc.assign(period=period))
        ar = atm_summary[atm_summary.period.eq(period)].iloc[0]
        tr = tail_summary[tail_summary.period.eq(period)].iloc[0]
        for model, book in books.items():
            base = capital * (2 if model == 'combined' else 1)
            statistics = scores(book, base)
            if model != 'combined':
                saved = ar if model == 'atm' else tr
                np.testing.assert_allclose(statistics['annualized_excess_sharpe'], saved.annualized_excess_sharpe, atol=1e-10, rtol=0)
                np.testing.assert_allclose(book.nav.iloc[-1] - book.cash_benchmark.iloc[-1], saved.net_excess_pnl, atol=1e-6, rtol=0)
            if model == 'combined':
                count = int(ar.entries + tr.entries)
                active = int(pc.entries_combined.gt(0).sum())
            else:
                row = ar if model == 'atm' else tr
                count, active = int(row.entries), int(row.active_pairs)
            pnl = book.nav - book.cash_benchmark
            summaries.append(dict(model=model, period=period, initial_capital=base,
                entries=count, active_pairs=active, net_excess_pnl=float(pnl.iloc[-1]),
                excess_return_percent=statistics['excess_return'] * 100,
                max_excess_drawdown_dollars=float((pnl - pnl.cummax().clip(lower=0)).min()), **statistics))
            book.to_csv(OUT / f'{model}_{period}_portfolio.csv')
            benchmark_returns = book.cash_benchmark.pct_change().fillna(0)
            for year, g in book.groupby(book.index.year):
                sd = g.excess_return.std()
                annual.append(dict(model=model, period=period, year=int(year),
                    annualized_excess_sharpe=float(np.sqrt(252) * g.excess_return.mean() / sd) if sd > 1e-12 else None,
                    return_percent=100 * ((1 + g.return_).prod() - 1),
                    excess_return_percent=100 * ((1 + g.return_).prod() - (1 + benchmark_returns.loc[g.index]).prod()),
                    net_excess_pnl_change=float(pnl.diff().fillna(pnl.iloc[0]).loc[g.index].sum())))
        for label, rows in [('whole_period', a.index[1:]), ('either_stream_active',
            a.index[1:][(a.excess_return.iloc[1:].abs() > 1e-12) | (t.excess_return.iloc[1:].abs() > 1e-12)])]:
            dependence.append(dict(period=period, sample=label, observations=len(rows),
                daily_excess_return_correlation=a.loc[rows, 'excess_return'].corr(t.loc[rows, 'excess_return'])))
        combined_pnl = books['combined'].nav - books['combined'].cash_benchmark
        np.testing.assert_allclose(combined_pnl, (a.nav - a.cash_benchmark) + (t.nav - t.cash_benchmark), atol=1e-6, rtol=0)
        np.testing.assert_allclose(combined_pnl.iloc[-1], pc.net_excess_pnl_combined.sum(), atol=1e-6, rtol=0)
        # Ex-post concentration checks keep both original capital buffers invested.
        for component, pairs in [('tail_only', tp), ('combined_pair', pc)]:
            field = 'net_excess_pnl' if component == 'tail_only' else 'net_excess_pnl_combined'
            winners = pairs[pairs[field] > 0].sort_values(field, ascending=False)
            for number in [1, 3]:
                top = winners.head(number); removed = pd.Series(0., index=a.index)
                for row in top.itertuples():
                    path = Path('artifacts/contracts/tail_put10_strategy') / row.pair / 'rule' / period / 'daily_nav.parquet'
                    # An inactive tail sleeve contributes zero excess P&L.
                    status_path = path.parent / 'status.json'
                    status = json.loads(status_path.read_text())
                    assert digest(status_path) == tail_hashes[str(status_path)]
                    signature[str(status_path)] = digest(status_path)
                    if not status.get('cash_only'):
                        assert digest(path) == tail_hashes[str(path)]
                        signature[str(path)] = digest(path)
                        nav = pd.read_parquet(path, columns=['nav', 'cash_benchmark'])
                        assert nav.index.equals(a.index)
                        removed += nav.nav - nav.cash_benchmark
                    if component == 'combined_pair':
                        # Archived ATM daily ledgers may be stored in either the
                        # sentiment sleeve or the frozen predecessor replay.
                        employee = Path('artifacts/contracts/atm65_employee_sentiment') / row.pair / 'rl_sentiment' / period
                        frozen = Path('artifacts/contracts/frozen_atm_commission_003') / row.pair / f'{period}_003'
                        folder = employee if (employee / 'status.json').exists() else frozen
                        status_path = folder / 'status.json'; saved = json.loads(status_path.read_text())
                        assert digest(status_path) == checkpoint['files'][str(status_path)]
                        signature[str(status_path)] = digest(status_path)
                        if not saved.get('cash_only'):
                            path = Path(saved.get('nav_source', folder / 'daily_nav.parquet'))
                            assert digest(path) == checkpoint['files'][str(path)]
                            signature[str(path)] = digest(path)
                            nav = pd.read_parquet(path, columns=['nav', 'cash_benchmark'])
                            assert nav.index.equals(a.index)
                            removed += nav.nav - nav.cash_benchmark
                replacement = books['combined'][['nav', 'cash_benchmark']].copy()
                replacement['nav'] -= removed
                replacement = add_returns(replacement)
                concentration.append(dict(period=period, removal=component, count=len(top),
                    pairs='; '.join(top.label), net_excess_pnl=float(replacement.nav.iloc[-1] - replacement.cash_benchmark.iloc[-1]),
                    annualized_excess_sharpe=scores(replacement, 2 * capital)['annualized_excess_sharpe']))
    for name, rows in [('comparison', summaries), ('annual', annual), ('dependence', dependence), ('concentration', concentration)]:
        pd.DataFrame(rows).to_csv(OUT / f'{name}.csv', index=False)
    pd.concat(pair_tables, ignore_index=True).to_csv(OUT / 'pairs.csv', index=False)
    for path, sha in signature.items():
        assert digest(path) == sha, f'Input changed while combining: {path}'
    manifest = dict(status='complete', code_and_sources=signature, refits=0, weight_search=False,
        method='Sum both saved NAV and cash-benchmark books. Equal original capital allocation: $345m ATM + $345m tail = $690m. Both component positions and costs retained.',
        limitations='Frozen separate-sleeve arithmetic combination, not a shared-capital inventory replay. No cross-strategy hedge netting, risk-limit reassessment, or collateral reuse. Previously inspected history; tail signal chosen using it. Concentration replacements are ex post.',
        output_sha256={str(p): digest(p) for p in OUT.glob('*.csv')})
    (OUT / 'audit.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(pd.DataFrame(summaries)[['model', 'period', 'annualized_excess_sharpe', 'entries', 'net_excess_pnl']].to_string(index=False))
    print(pd.DataFrame(concentration).to_string(index=False))


if __name__ == '__main__':
    main()
