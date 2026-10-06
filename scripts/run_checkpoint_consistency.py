"""Sizing-only controls on the frozen 0.778 shared-book checkpoint.

Wing-specific signal models are preserved. All sizing arms receive exactly the
same proposals and use the original costs, calendar, ledger and risk budgets.
"""
from dataclasses import replace
import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd

from scripts.joint_smile_data import SharedMarket, registry
from scripts.joint_smile_engine import simulate
from scripts.curve_trading.config import config
from scripts.run_broad_contract_replay import ROOT
from scripts.select_atm_contracts import load_calendar
from scripts.run_aligned_contract_learning import mark_book
from scripts.run_selected_pair_backtest import scores
from scripts.run_atm65_ablation import write_json
from scripts.run_contract_walkforward import digest
from scripts.research_opposite_wing_signals import CORPORATE_EVENTS

FROZEN = Path('artifacts/contracts/joint_smile')
OUT = Path('artifacts/contracts/checkpoint_consistency')
REPORT = Path('reports/checkpoint_consistency')
PROTOCOL = (
    'Sizing-only comparison on the exact 2,133 nonzero frozen proposals from the '
    '0.7782445992546072 shared-book checkpoint. Preserve proposal dates, selected '
    'contracts, directions, wing-specific causal entry models and original cash '
    'decisions. Fixed allocations 50%, 75%, 100%, original individual sizing and '
    'original joint sizing all use the same original shared cash account, net '
    'stock hedges, per-pair/per-wing $1m capital and 20k gross-vega reservation, '
    '$3m/60k pair budget and summed global limits. No retraining or new signals. '
    'Actual option commission $0.03 per contract, original mid plus 0.1 vol-point '
    'per leg, daily delta hedge, integer vega matching within 10%, original '
    'financing, stock borrow, dividends and corporate exits. Hold to own expiry. '
    'Different allocations can change risk admissions and accepted trade counts. '
    'Joint sizing may choose additional cash; this is explicitly an allocation '
    'decision on the same proposal universe. Training methods and feature sets '
    'remain wing-specific: this does not claim identical entry models across '
    'ATM, P25 and P75. Same already inspected 2018-August 2025 history; exploratory '
    'comparisons, no untouched holdout and no supervised-NN comparison. '
    'One worker, 50% of one core, one BLAS/Arrow thread, 3 GiB RSS stop.'
)


def audit_book(nav, trades, decisions, signals, cal, cfg):
    assert nav.index.equals(cal) and len(decisions) == len(signals)
    assert decisions.signal_date.tolist() == sorted(decisions.signal_date.tolist())
    wanted = signals.sort_values(['date', 'pair', 'wing'], kind='stable')
    wanted = wanted.rename(columns={'date': 'signal_date'})
    columns = ['signal_date', 'pair', 'wing', 'base_level']
    pd.testing.assert_frame_equal(decisions[columns].reset_index(drop=True),
                                  wanted[columns].reset_index(drop=True),
                                  check_dtype=False)
    cashflow = (-trades.quantity * 100 * trades.price - trades.commission -
                trades.execution_penalty).groupby(trades.date).sum().reindex(cal, fill_value=0)
    expected = (cashflow + nav.option_value.diff().fillna(nav.option_value) +
                nav.hedge_gross_pnl - nav.hedge_cost + nav.interest -
                nav.stock_borrow_cost + nav.dividend_cash)
    error = float((expected - nav.pnl).abs().max())
    assert error < 1e-5
    assert trades.groupby(['cohort', 'optionid']).quantity.sum().eq(0).all()
    assert abs(nav.option_value.iloc[-1]) + abs(nav.stock_value.iloc[-1]) < 1e-7
    paid = trades.reason.isin(['entry', 'announced_corporate_action_exit'])
    np.testing.assert_allclose(trades.loc[paid, 'commission'],
                               trades.loc[paid, 'quantity'].abs() * .03,
                               rtol=0, atol=1e-10)
    entered = decisions.loc[decisions.status.eq('entered')]
    assert ((entered.vega_a - entered.vega_b).abs() /
            entered[['vega_a', 'vega_b']].max(axis=1)).le(.10 + 1e-12).all()
    wanted['cohort'] = (wanted.signal_date.dt.strftime('%Y-%m-%d') + '|' +
                        wanted.pair + '|' + wanted.wing)
    proposal = wanted.set_index('cohort')
    for trade in trades.loc[trades.reason.eq('entry')].itertuples():
        row = proposal.loc[trade.cohort]
        leg = 'a' if trade.secid == row.secid_a else 'b'
        assert trade.optionid == row['optionid_' + leg]
        assert np.sign(trade.quantity) == row.direction * (1 if leg == 'a' else -1)
        assert trade.date == cal[cal.get_loc(row.signal_date) + 1]
    return dict(independent_ledger_error=error, inventory_closed=True,
                identical_proposals=True, contracts_and_directions_preserved=True,
                entry_vega_matching_verified=True)


def sharpe(x):
    return np.sqrt(252) * np.mean(x) / np.std(x, ddof=1)


def bootstrap(books):
    """Paired daily-return moving blocks; report every predeclared size control."""
    names = list(books)
    # Match scores(): the initial valuation session has no return observation.
    returns = np.stack([books[name].excess_return.iloc[1:].to_numpy() for name in names])
    draws = np.empty((1000, len(names)))
    rng = np.random.default_rng(2026)
    block = 60
    n = returns.shape[1]
    for i in range(len(draws)):
        starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
        indices = (starts[:, None] + np.arange(block)).ravel()[:n]
        sample = returns[:, indices]
        draws[i] = np.sqrt(252) * sample.mean(axis=1) / sample.std(axis=1, ddof=1)
    rows = []
    for candidate in ['individual_sizing', 'joint_sizing']:
        for control in ['fixed_50', 'fixed_75', 'fixed_100']:
            a, b = names.index(candidate), names.index(control)
            delta = draws[:, a] - draws[:, b]
            rows.append(dict(candidate=candidate, control=control,
                             sharpe_difference=sharpe(returns[a]) - sharpe(returns[b]),
                             ci_lower=float(np.quantile(delta, .025)),
                             ci_upper=float(np.quantile(delta, .975)),
                             bootstrap_probability_positive=float(np.mean(delta > 0)),
                             block_sessions=block, draws=len(draws)))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    sources = registry()
    sources.expected.update(json.loads(Path('reports/joint_smile/output_hashes.json').read_text()))
    sources.expected.update(json.loads(Path('reports/smile_flat_size/output_hashes.json').read_text()))
    code = [Path(__file__).relative_to(Path.cwd()), Path('config/checkpoint_consistency.json'),
            Path('scripts/joint_smile_engine.py'),
            Path('scripts/joint_smile_data.py'), Path('scripts/option_flex/contract_engine.py'),
            Path('scripts/option_flex/contract_risk.py')]
    for path in code:
        sources.track(path)
    signals = pd.read_parquet(sources.track(FROZEN / 'training_rows.parquet'))
    cal = load_calendar(ROOT)
    cal = cal[(cal >= '2018-01-01') & (cal <= '2025-08-29')]
    cfg = replace(config(), option_commission=.03, capital=3_000_000 * 345,
                  max_gross_vega=60_000 * 345)
    for path in (FROZEN / 'market').glob('*.parquet'):
        sources.track(path)
    market = SharedMarket()
    write_json(OUT / 'inputs.json', dict(protocol=PROTOCOL, pairs=345,
                                        proposals=len(signals), actual_commission=.03))
    rows, annual, wings, checks, counts, books = [], [], [], [], [], {}
    for name, level, existing in [
        ('individual_sizing', None, None),
        ('fixed_50', 2, None),
        ('fixed_75', 3, None),
        ('fixed_100', 4, Path('artifacts/contracts/smile_flat_size')),
        ('joint_sizing', None, FROZEN / 'shared_joint_native_limits'),
    ]:
        arm_start = time.monotonic()
        folder = OUT / name
        folder.mkdir(exist_ok=True)
        print(f'{name}: identical-proposal ledger replay/control', flush=True)
        if existing is None:
            policy = None if level is None else lambda row, context, size=level: size
            nav, trades, decisions, _ = simulate(market, signals, cal, cfg, 3_000_000,
                                                policy=policy, corporate_events=CORPORATE_EVENTS,
                                                wing_capital=1_000_000)
            mark_book(nav, market.rates, cfg.capital)
        else:
            nav = pd.read_parquet(sources.track(existing / 'daily_nav.parquet'))
            trades = pd.read_csv(sources.track(existing / 'trades.csv'), parse_dates=['date'])
            decisions = pd.read_csv(sources.track(existing / 'decisions.csv'),
                                    parse_dates=['signal_date', 'entry_date'])
        if level is not None:
            assert decisions.level.eq(level).all()
        checks.append(dict(model=name, **audit_book(nav, trades, decisions, signals, cal, cfg)))
        if name == 'individual_sizing':
            original = pd.read_parquet(sources.track(FROZEN / 'shared_frozen_native_limits/daily_nav.parquet'))
            np.testing.assert_allclose(nav.nav, original.nav, rtol=0, atol=1e-5)
            np.testing.assert_allclose(scores(nav, cfg.capital)['annualized_excess_sharpe'],
                                       .7782445992546072, rtol=0, atol=1e-8)
        nav.to_parquet(folder / 'daily_nav.parquet')
        trades.to_csv(folder / 'trades.csv', index=False)
        decisions.to_csv(folder / 'decisions.csv', index=False)
        entered = decisions.loc[decisions.status.eq('entered')]
        rows.append(dict(model=name, entries=len(entered), active_pairs=int(entered.pair.nunique()),
                         net_excess_pnl=float(nav.nav.iloc[-1] - nav.cash_benchmark.iloc[-1]),
                         commission=float(trades.commission.sum()),
                         option_cost=float(nav.option_execution_cost.sum()),
                         hedge_cost=float(nav.hedge_cost.sum()), **scores(nav, cfg.capital)))
        books[name] = nav
        excess_pnl = nav.pnl - nav.cash_benchmark.diff().fillna(nav.cash_benchmark.iloc[0] - cfg.capital)
        for year, g in nav.groupby(nav.index.year):
            annual.append(dict(model=name, year=int(year), annualized_excess_sharpe=sharpe(g.excess_return),
                               net_excess_pnl=float(excess_pnl.loc[g.index].sum())))
        for wing, g in decisions.groupby('wing'):
            e = g.loc[g.status.eq('entered')]
            wings.append(dict(model=name, wing=wing, proposals=len(g), entries=len(e),
                              mean_entry_allocation=float(e.level.mean() / 4)))
        for status, count in decisions.status.value_counts().items():
            counts.append(dict(model=name, status=status, proposals=int(count)))
        pd.DataFrame(rows).to_csv(REPORT / 'comparison.csv', index=False)
        write_json(OUT / 'progress.json', dict(completed=[r['model'] for r in rows],
                                              elapsed_seconds=time.monotonic() - started))
        print(f'{name}: Sharpe {rows[-1]["annualized_excess_sharpe"]:.6f}, '
              f'{len(entered)} entries, {time.monotonic() - arm_start:.1f}s', flush=True)
    pd.DataFrame(annual).to_csv(REPORT / 'annual.csv', index=False)
    pd.DataFrame(annual).pivot(index='year', columns='model', values='annualized_excess_sharpe').to_csv(REPORT / 'annual_comparison.csv')
    pd.DataFrame(wings).to_csv(REPORT / 'wings.csv', index=False)
    pd.DataFrame(checks).to_csv(REPORT / 'ledger_audit.csv', index=False)
    pd.DataFrame(counts).to_csv(REPORT / 'decisions.csv', index=False)
    bootstrap(books).to_csv(REPORT / 'matched_sizing_uplift.csv', index=False)
    print('Verifying frozen source hashes', flush=True)
    for path, value in sources.items():
        assert digest(path) == value, f'Frozen source changed: {path}'
    write_json(OUT / 'sources.json', sources)
    record = dict(status='complete', pairs=345, proposals=len(signals), no_retraining=True,
                  signal_dates_and_directions_preserved=True, original_cash_preserved=True,
                  original_checkpoint_reproduced=True, scope='same rules across sizing benchmarks',
                  wing_specific_entry_models_preserved=True, actual_commission=.03,
                  frozen_sources_verified=len(sources.verified), elapsed_seconds=time.monotonic() - started,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_json(REPORT / 'audit.json', record)
    (REPORT / 'findings.txt').write_text(pd.DataFrame(rows).to_string(index=False) + '\n\n' +
                                         PROTOCOL + '\n\n' + json.dumps(record, indent=2) + '\n')
    write_json(REPORT / 'output_hashes.json', {str(p): digest(p) for p in [*code, *[
        p for root in [OUT, REPORT] for p in root.rglob('*')
        if p.is_file() and p.name != 'output_hashes.json']]})
    print(pd.DataFrame(rows)[['model', 'annualized_excess_sharpe', 'entries']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
