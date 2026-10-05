"""Rebuild ATM30 rewards, native memory and monthly policies at $0.03.

Retains the entire original roster and the original optimizer and gates. Only
commission changes. Execute through run_signal_discovery_guarded, one worker.
"""
import argparse
import copy
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd

from scripts.curve_trading.config import config, COMBOS
from scripts.curve_trading.policy import ATM, Replay, memory_gates, gate
from scripts.curve_trading.unit_paths import UnitPaths, pair_labels
from scripts.train_expiry_contracts import settings
from scripts.run_curve_trading import train, BASE, NEW_FEATURES, EXTRA, FEATURES
from scripts.replay_frozen_atm_commission import SOURCE, load_pair, cash_book, save_result
from scripts.run_broad_contract_replay import Market, ROOT
from scripts.run_aligned_contract_learning import mark_book
from scripts.run_selected_pair_backtest import scores
from scripts.run_contract_walkforward import digest
from scripts.select_atm_contracts import load_calendar

OUTPUT = Path('artifacts/contracts/atm_retrained_commission_003')
REPORT = Path('reports/atm_retrained_commission')
FROZEN_REPORT = Path('reports/frozen_atm_commission')
FEE = .03


def fast_memory(dates, ends, rewards, entered):
    """Same completed-cohort means and counts as profitability_memory."""
    out = pd.DataFrame(index=dates)
    times = pd.DatetimeIndex(dates).asi8
    finish = pd.DatetimeIndex(ends).asi8
    for side, name in enumerate(['short', 'long']):
        good = entered[:, side] & np.isfinite(rewards[:, side])
        order = np.argsort(finish[good], kind='stable')
        e, r = finish[good][order], rewards[good, side][order]
        cumulative = np.r_[0., np.cumsum(r)]
        right = np.searchsorted(e, times, side='left')
        for days in [91, 365]:
            left = np.searchsorted(e, times - pd.Timedelta(days=days).value, side='left')
            count = right - left
            total = cumulative[right] - cumulative[left]
            out[f'profit_{name}_{days}d'] = np.divide(total, count, out=np.zeros(len(times)), where=count > 0)
            out[f'count_{name}_{days}d'] = count
    return out[[f'{prefix}_{name}_{days}d' for days in [91, 365]
                for name in ['short', 'long'] for prefix in ['profit', 'count']]]


def cheaper_units(unit, cfg):
    """Reprice unit entry flows; all holding paths and hedges stay identical."""
    result = copy.copy(unit)
    result.flow_long = unit.flow_long.copy()
    result.flow_short = unit.flow_short.copy()
    saving = (unit.cfg.option_commission - cfg.option_commission) * unit.entered
    result.flow_long[:, 0] += saving
    result.flow_short[:, 0] += saving
    result.cfg = cfg
    return result


class ATMLabels:
    """Original train() indexing without allocating unused smile actions."""
    def __init__(self, daily):
        self.daily = daily.astype(np.float32)
        self.shape = (len(COMBOS), *self.daily.shape)

    def __getitem__(self, combo):
        assert combo == ATM
        return self.daily


def rebuild(panel, q, s, selection, divs, cal, cfg):
    secids = list(s.secid.unique())
    native_cfg = settings()
    native = [UnitPaths(q[q.secid.eq(sid)], s[s.secid.eq(sid)], selection[selection.secid.eq(sid)],
                        cal, panel.index, divs[divs.secid.eq(sid)], native_cfg) for sid in secids]
    old_daily, old_ends, old_vegas, _ = pair_labels(*native, (30, 'C', 'atm', 'atm'), native_cfg)
    old_reward = np.divide(old_daily[:, [0, -1]].sum(axis=2) * cfg.entry_vega_per_leg,
                           old_vegas[:, [0, -1]], out=np.full((len(panel), 2), np.nan),
                           where=old_vegas[:, [0, -1]] > 0)
    old_memory = fast_memory(panel.index, cal[old_ends], old_reward, old_vegas[:, [0, -1]] > 0)
    # Rebuilding at the original fee must recover the inherited state definition.
    for col in old_memory:
        np.testing.assert_allclose(old_memory[col], panel[col], atol=1e-6, rtol=0, err_msg=col)
    memory_error = float(abs(old_memory - panel[old_memory.columns]).to_numpy().max(initial=0.))
    new_native_cfg = replace(native_cfg, option_commission=FEE)
    new_native = [cheaper_units(u, new_native_cfg) for u in native]
    memory_daily, ends, vegas, _ = pair_labels(*new_native, (30, 'C', 'atm', 'atm'), new_native_cfg)
    np.testing.assert_array_equal(ends, old_ends)
    reward = np.divide(memory_daily[:, [0, -1]].sum(axis=2) * cfg.entry_vega_per_leg,
                       vegas[:, [0, -1]], out=np.full((len(panel), 2), np.nan),
                       where=vegas[:, [0, -1]] > 0)
    memory = fast_memory(panel.index, cal[ends], reward, vegas[:, [0, -1]] > 0)
    changed = panel.copy()
    changed[memory.columns] = memory
    for col in ['commission_a', 'commission_b', 'commission_vol_entry']:
        changed[col] *= FEE / native_cfg.option_commission
    other = [c for c in panel if c not in memory and not c.startswith('commission_')]
    pd.testing.assert_frame_equal(changed[other], panel[other])
    del native, new_native, memory_daily, old_daily
    units = [UnitPaths(q[q.secid.eq(sid)], s[s.secid.eq(sid)], selection[selection.secid.eq(sid)],
                       cal, panel.index, divs[divs.secid.eq(sid)], cfg) for sid in secids]
    daily, finish, entry_vegas, statuses = pair_labels(*units, (30, 'C', 'atm', 'atm'), cfg)
    rewards = np.divide(daily[:, [0, -1]].sum(axis=2) * cfg.entry_vega_per_leg,
                        entry_vegas[:, [0, -1]], out=np.full((len(panel), 2), np.nan),
                        where=entry_vegas[:, [0, -1]] > 0)
    own_gates = memory_gates(panel.index, cal[finish], rewards, entry_vegas[:, [0, -1]] > 0)
    available = (units[0].ids(30, 'C', 'atm') >= 0) & (units[1].ids(30, 'C', 'atm') >= 0)
    gates = np.zeros((len(COMBOS), len(panel), 2), bool); gates[ATM] = own_gates
    avail = np.zeros((len(COMBOS), len(panel)), bool); avail[ATM] = available
    all_ends = np.broadcast_to(finish, (len(COMBOS), len(panel)))
    return changed, ATMLabels(daily), all_ends, gates, avail, dict(
        original_memory_max_error=memory_error,
        label_status={str(k): int(v) for k, v in pd.Series(statuses).value_counts().items()},
        old_memory_permitted_rows=int(((old_memory.profit_short_91d > 0) & (old_memory.profit_short_365d > 0)).sum()),
        changed_memory_rows=int((abs(memory - old_memory).max(axis=1) > 1e-8).sum()))


def check_fits(fits):
    for outcome, cutoff in [('last_training_outcome', 'validation_start'), ('last_recent_outcome', 'fit')]:
        if outcome in fits:
            valid = fits[outcome].notna()
            assert (pd.to_datetime(fits.loc[valid, outcome]) < pd.to_datetime(fits.loc[valid, cutoff])).all()


def report(defs, market, cfg):
    results, annual, pairs, fit_tables = [], [], [], []
    for period, start in [('full', '2018-01-01'), ('recent', '2022-01-01')]:
        cal = market.calendar[market.calendar >= start]
        cash = cash_book(cal, market.rates, cfg.capital)
        book = cash[['nav', 'cash_benchmark']] * len(defs)
        entries = active = 0; commission = option_cost = hedge_cost = 0.
        pnl_paths = {}
        for d in defs:
            folder = OUTPUT / d['pair']
            row = json.loads((folder / period / 'status.json').read_text())
            assert row['status'] == 'complete'
            nav = cash if row.get('cash_only') else pd.read_parquet(folder / period / 'daily_nav.parquet')
            assert nav.index.equals(cal)
            book += nav[['nav', 'cash_benchmark']] - cash[['nav', 'cash_benchmark']]
            pnl = float(nav.nav.iloc[-1] - nav.cash_benchmark.iloc[-1])
            pairs.append(dict(period=period, pair=d['pair'], label=d['label'], entries=row['entries'], net_excess_pnl=pnl))
            if row['entries']: pnl_paths[d['pair']] = nav.nav - nav.cash_benchmark
            entries += row['entries']; active += int(row['entries'] > 0)
            commission += row.get('commission', 0.); option_cost += row.get('option_cost', 0.)
            hedge_cost += row.get('hedge_cost', 0.)
            if period == 'full' and (folder / 'fits.csv').exists():
                try: fits = pd.read_csv(folder / 'fits.csv')
                except pd.errors.EmptyDataError: fits = pd.DataFrame()
                if len(fits):
                    check_fits(fits)
                    fits['pair'] = d['pair']; fits['label'] = d['label']; fit_tables.append(fits)
        book['return_'] = book.nav.pct_change().fillna(0)
        book['excess_return'] = book.return_ - book.cash_benchmark.pct_change().fillna(0)
        book.to_csv(REPORT / f'{period}_portfolio.csv')
        row = dict(variant=f'{period}_retrained_003', period=period, start=str(cal[0].date()), end=str(cal[-1].date()),
                   pairs=len(defs), entries=entries, active_pairs=active, option_commission=commission,
                   option_cost=option_cost, hedge_cost=hedge_cost,
                   net_excess_pnl=float(book.nav.iloc[-1] - book.cash_benchmark.iloc[-1]),
                   **scores(book, cfg.capital * len(defs)))
        winners = sorted([r for r in pairs if r['period'] == period and r['net_excess_pnl'] > 1e-7], key=lambda r: -r['net_excess_pnl'])
        if winners:
            top = winners[0]; without = book.nav - pnl_paths[top['pair']]
            excess = without.pct_change().fillna(0) - book.cash_benchmark.pct_change().fillna(0)
            row.update(largest_winner=top['label'], largest_winner_pnl=top['net_excess_pnl'],
                       top_three_share_positive_pnl=sum(r['net_excess_pnl'] for r in winners[:3]) / sum(r['net_excess_pnl'] for r in winners),
                       sharpe_largest_winner_replaced_by_cash=float(np.sqrt(252) * excess.iloc[1:].mean() / excess.iloc[1:].std()))
        results.append(row)
        for year, g in book.groupby(book.index.year):
            annual.append(dict(variant=row['variant'], year=year, total_return_percent=100 * ((1 + g.return_).prod() - 1),
                               cash_return_percent=100 * ((1 + book.cash_benchmark.pct_change().fillna(0).loc[g.index]).prod() - 1),
                               annualized_excess_sharpe=float(np.sqrt(252) * g.excess_return.mean() / g.excess_return.std()) if g.excess_return.std() > 1e-12 else None))
    frozen = pd.read_csv(FROZEN_REPORT / 'comparison.csv')
    frozen = frozen[frozen.variant.isin(['full_003', 'recent_003'])].copy()
    comparison = pd.concat([frozen, pd.DataFrame(results)], ignore_index=True)
    comparison.to_csv(REPORT / 'comparison.csv', index=False)
    pd.DataFrame(annual).to_csv(REPORT / 'annual.csv', index=False)
    pd.DataFrame(pairs).to_csv(REPORT / 'pairs.csv', index=False)
    fits = pd.concat(fit_tables, ignore_index=True)
    fits.to_csv(REPORT / 'fits.csv', index=False)
    fits.groupby('status').size().rename('months').to_csv(REPORT / 'fit_summary.csv')
    lines = ['# ATM strategy retrained at $0.03 commission', '',
             'All 346 original pairs retained: 147 passed the structural screen and were processed for retraining; 199 remain cash under the unchanged monthly beta 0.85–1.15 and correlation ≥0.5 screen. Pairs without usable states or enough training remain cash. Rebuilds net actual-contract training paths, commission state inputs, native profitability-memory features, directional memory gates and prior-year validation at $0.03 per contract. Same eight-leaf direct Sharpe policy search, monthly five-year completed-outcome fitting, signed sizing ladder and risk limits.', '',
             '| Policy | Period | Annualized excess Sharpe | Entries | Active pairs | P&L above cash |',
             '|---|---|---:|---:|---:|---:|']
    for r in comparison.itertuples():
        lines.append(f'| {r.variant} | {r.start} to {r.end} | {r.annualized_excess_sharpe:.3f} | {r.entries} | {r.active_pairs} | ${r.net_excess_pnl:,.0f} |')
    lines += ['', 'Both policies use the same $0.03 execution assumption. Full history starts flat in 2018; the separate recent test starts flat in 2022 using only actions generated by historical monthly fits. The 2022 replay does not carry positions or portfolio NAV from earlier years. One $1m sleeve per pair, including cash sleeves. Total portfolio returns therefore include substantial cash interest; Sharpe uses daily excess returns versus that cash benchmark, sample standard deviation and √252.', '',
              'Midpoint ±0.10 vol point per option leg at entry, 1 bp stock hedging, daily delta hedges, initial integer vega mismatch ≤10%, fixed option quantities until each leg expires (apart from splits). No daily option-vega resizing. Original gates remain: positive completed-cohort memory over 91/365 days with at least 21/63 shadow observations and positive prior-year validation Sharpe. Validation still evaluates the proposal before the recent-five-year refit, matching the original algorithm.', '',
              'The old $0.65 native memory is reconstructed and checked against saved input states before changing fees. Label and state dates are preserved; unresolved past labels are excluded from fitting, never used to filter deployment. Ledger, expiry, initial-vega and completed-outcome timing checks run for every simulated pair. This uses previously examined research history, not a new holdout. Largest-winner removal is an ex-post diagnostic.', '',
              '[Annual results](atm_retrained_commission/annual.csv) · [Pair results](atm_retrained_commission/pairs.csv) · [Training gates](atm_retrained_commission/fit_summary.csv) · [Comparison](atm_retrained_commission/comparison.csv)', '']
    REPORT.with_suffix('.md').write_text('\n'.join(lines))
    print(comparison[['variant', 'annualized_excess_sharpe', 'entries', 'active_pairs', 'net_excess_pnl']].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--check', action='store_true')
    parser.add_argument('--pair'); parser.add_argument('--report-only', action='store_true'); args = parser.parse_args()
    if args.check:
        import unittest
        suite = unittest.defaultTestLoader.loadTestsFromNames(['tests.test_atm_commission_training', 'tests.test_curve_trading'])
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful(): raise SystemExit(1)
        return
    OUTPUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    defs = json.loads((SOURCE / 'control_inputs.json').read_text())['definitions']
    cfg = replace(config(), option_commission=FEE)
    paths = [Path(__file__), Path('scripts/run_curve_trading.py'), Path('scripts/curve_trading/unit_paths.py'),
             Path('scripts/curve_trading/policy.py'), Path('scripts/option_flex/contract_engine.py'),
             Path('scripts/curve_trading/config.py'), Path('scripts/replay_frozen_atm_commission.py'),
             Path('reports/aligned_contract_learning/manifest.json'), SOURCE / 'control_inputs.json',
             Path('data/curve_trading/C_history_manifest.json'), Path('data/curve_trading/C_select_manifest.json')]
    paths += [BASE / d['pair'] / '30d/states.parquet' for d in defs if (BASE / d['pair'] / '30d/states.parquet').exists()]
    signature = dict(settings=asdict(cfg), definitions=defs, code_and_sources={str(p): digest(p) for p in paths})
    manifest = OUTPUT / 'inputs.json'
    if manifest.exists(): assert json.loads(manifest.read_text()) == json.loads(json.dumps(signature)), 'Inputs changed'
    else: manifest.write_text(json.dumps(signature, indent=2) + '\n')
    cal = load_calendar(ROOT); cal = cal[(cal >= '2010-01-01') & (cal <= '2025-08-29')]
    market = Market(cal); started = time.monotonic()
    if not args.report_only:
        for n, d in enumerate(defs, 1):
            if args.pair and d['pair'] != args.pair: continue
            folder = OUTPUT / d['pair']; folder.mkdir(parents=True, exist_ok=True)
            if all((folder / p / 'status.json').exists() for p in ['full', 'recent']): continue
            prior = json.loads((SOURCE / d['pair'] / 'atm_control/status.json').read_text())
            assert prior['status'] == 'complete'
            if prior.get('structural_cash'):
                for period in ['full', 'recent']:
                    dest = folder / period; dest.mkdir(exist_ok=True)
                    (dest / 'status.json').write_text(json.dumps(dict(status='complete', pair=d['pair'], label=d['label'], entries=0, cash_only=True, structural_cash=True)) + '\n')
                continue
            panel = pd.read_parquet(BASE / d['pair'] / '30d/states.parquet')
            if panel.empty:
                for period in ['full', 'recent']:
                    dest = folder / period; dest.mkdir(exist_ok=True)
                    (dest / 'status.json').write_text(json.dumps(dict(status='complete', pair=d['pair'], label=d['label'], entries=0, cash_only=True, no_state_rows=True)) + '\n')
                continue
            q, s, selection, divs = load_pair(market, d['secids'])
            panel, labels, ends, gates, available, checks = rebuild(panel, q, s, selection, divs, cal, cfg)
            panel.to_parquet(folder / 'states.parquet')
            np.savez_compressed(folder / 'gates.npz', dates=panel.index.values, gates=gates[ATM], available=available[ATM], ends=ends[ATM])
            (folder / 'rebuild_checks.json').write_text(json.dumps(checks, indent=2) + '\n')
            replay = Replay(q, s, selection, d['secids'], divs, cfg)
            features = NEW_FEATURES + [c if c != 'commission_vol_roundtrip' else 'commission_vol_entry' for c in EXTRA]
            features += [c for c in panel if c.startswith(('profit_', 'count_'))] + FEATURES + ['vix_previous_close']
            print(f'{n}/{len(defs)} {d["label"]}: rebuilt {len(panel)} states; training', flush=True)
            if (folder / 'actions.csv').exists():
                actions = pd.read_csv(folder / 'actions.csv', index_col=0, parse_dates=[0]).code.astype(int)
                fits = pd.read_csv(folder / 'fits.csv')
            else:
                actions, fits = train(d, panel, labels, ends, gates, available, cal, replay, market.rates, folder, features, 'atm_control')
                actions.to_csv(folder / 'actions.csv'); fits.to_csv(folder / 'fits.csv', index=False)
            check_fits(fits)
            pd.testing.assert_series_equal(actions, gate(actions, actions.index, panel, gates, available), check_names=False)
            for period, start in [('full', '2018-01-01'), ('recent', '2022-01-01')]:
                dest = folder / period; dest.mkdir(exist_ok=True)
                if (dest / 'status.json').exists(): continue
                own = actions[actions.index >= start]
                if not own.ne(0).any():
                    (dest / 'status.json').write_text(json.dumps(dict(status='complete', pair=d['pair'], label=d['label'], entries=0, cash_only=True)) + '\n')
                    continue
                nav, t, dec = replay.run(own, cal[cal >= start]); mark_book(nav, market.rates, cfg.capital)
                row = save_result(dest, nav, t, dec, cfg, s, d)
                print(f'  {d["label"]} {period}: {row["entries"]} entries', flush=True)
                del nav, t, dec
            del panel, labels, ends, gates, available, replay, q, s, selection, divs, actions, fits
            gc.collect()
            (OUTPUT / 'progress.json').write_text(json.dumps(dict(completed_roster_position=n, pairs=len(defs), last_pair=d['label'], elapsed_seconds=time.monotonic()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)) + '\n')
            print(f'{n}/{len(defs)} completed {d["label"]}, {time.monotonic()-started:.0f}s', flush=True)
    if args.pair: return
    report(defs, market, cfg)
    progress = json.loads((OUTPUT / 'progress.json').read_text())
    elapsed = time.monotonic()-started
    (OUTPUT / 'task_status.json').write_text(json.dumps(dict(status='complete', pairs=len(defs),
        peak_rss_bytes=max(progress['peak_rss_bytes'], resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        elapsed_seconds=progress['elapsed_seconds']+elapsed if args.report_only else elapsed)) + '\n')


if __name__ == '__main__': main()
