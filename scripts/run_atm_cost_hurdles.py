"""Training-cost sensitivity with actual execution fixed at $0.03.

Two predeclared intermediate hurdles; same original monthly optimizer, roster
and risk rules. One worker, checkpointed by pair/hurdle, no parameter search
over evaluation outcomes. Run through run_signal_discovery_guarded.
"""
import argparse
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
from scripts.retrain_atm_commission import fast_memory, cheaper_units, ATMLabels, check_fits
from scripts.run_curve_trading import train, BASE, NEW_FEATURES, EXTRA, FEATURES
from scripts.replay_frozen_atm_commission import SOURCE, load_pair, cash_book, save_result
from scripts.run_broad_contract_replay import Market, ROOT
from scripts.run_aligned_contract_learning import mark_book
from scripts.run_selected_pair_backtest import scores
from scripts.run_contract_walkforward import digest
from scripts.select_atm_contracts import load_calendar

OUTPUT = Path('artifacts/contracts/atm_cost_hurdles')
REPORT = Path('reports/atm_cost_hurdles')
HURDLES = [.15, .30]
EXECUTION_FEE = .03
PERIODS = [('full', '2018-01-01'), ('recent', '2022-01-01')]


def fee_name(fee):
    return f'fee_{round(100 * fee):03d}'


def read_fits(path):
    try: return pd.read_csv(path)
    except pd.errors.EmptyDataError: return pd.DataFrame()


def make_units(panel, q, s, selection, divs, cal, cfg):
    return [UnitPaths(q[q.secid.eq(sid)], s[s.secid.eq(sid)], selection[selection.secid.eq(sid)],
                      cal, panel.index, divs[divs.secid.eq(sid)], cfg) for sid in s.secid.unique()]


def reward_and_memory(units, cfg):
    daily, ends, vegas, statuses = pair_labels(*units, (30, 'C', 'atm', 'atm'), cfg)
    reward = np.divide(daily[:, [0, -1]].sum(axis=2) * cfg.entry_vega_per_leg,
                       vegas[:, [0, -1]], out=np.full((len(ends), 2), np.nan),
                       where=vegas[:, [0, -1]] > 0)
    memory = fast_memory(units[0].dates, units[0].calendar[ends], reward, vegas[:, [0, -1]] > 0)
    return daily, ends, vegas, statuses, reward, memory


def prepare_hurdle(panel, native, curve, fee):
    native_cfg = replace(native[0].cfg, option_commission=fee)
    own_native = [cheaper_units(u, native_cfg) for u in native]
    _, _, _, _, _, memory = reward_and_memory(own_native, native_cfg)
    changed = panel.copy(); changed[memory.columns] = memory
    for col in ['commission_a', 'commission_b', 'commission_vol_entry']:
        changed[col] *= fee / native[0].cfg.option_commission
    unchanged = [c for c in panel if c not in memory and not c.startswith('commission_')]
    pd.testing.assert_frame_equal(changed[unchanged], panel[unchanged])
    cfg = replace(curve[0].cfg, option_commission=fee)
    own_curve = [cheaper_units(u, cfg) for u in curve]
    daily, ends, vegas, statuses, reward, _ = reward_and_memory(own_curve, cfg)
    gates = np.zeros((len(COMBOS), len(panel), 2), bool)
    gates[ATM] = memory_gates(panel.index, curve[0].calendar[ends], reward, vegas[:, [0, -1]] > 0)
    avail = np.zeros((len(COMBOS), len(panel)), bool)
    avail[ATM] = (curve[0].ids(30, 'C', 'atm') >= 0) & (curve[1].ids(30, 'C', 'atm') >= 0)
    return changed, ATMLabels(daily), np.broadcast_to(ends, (len(COMBOS), len(panel))), gates, avail


def verify_reference(d, panel, native, curve):
    _, _, _, _, _, memory = reward_and_memory(native, native[0].cfg)
    np.testing.assert_allclose(memory, panel[memory.columns], atol=1e-6, rtol=0)
    error = float(abs(memory - panel[memory.columns]).to_numpy().max(initial=0.))
    _, ends, vegas, _, reward, _ = reward_and_memory(curve, curve[0].cfg)
    gate_reference = memory_gates(panel.index, curve[0].calendar[ends], reward, vegas[:, [0, -1]] > 0)
    with np.load(SOURCE / d['pair'] / 'gates_control.npz') as z:
        np.testing.assert_array_equal(z['dates'], panel.index.values)
        np.testing.assert_array_equal(z['ends'][ATM], ends)
        np.testing.assert_array_equal(z['gates'][ATM], gate_reference)
    return error


def summary(defs, market, cfg):
    rows, annual, pairs, fits_all = [], [], [], []
    # Endpoints are completed historical monthly policies evaluated at $0.03.
    baseline = [('reports/atm_retrained_commission', 'full_retrained_003', 'recent_retrained_003', .03),
                ('reports/frozen_atm_commission', 'full_003', 'recent_003', .65)]
    for root, full_name, recent_name, fee in baseline:
        table = pd.read_csv(Path(root) / 'comparison.csv')
        yearly = pd.read_csv(Path(root) / 'annual.csv')
        for period, name in [('full', full_name), ('recent', recent_name)]:
            record = table[table.variant.eq(name)].iloc[0].to_dict()
            record.update(training_hurdle=fee, execution_commission=EXECUTION_FEE, period=period, source='saved_policy')
            rows.append(record)
            own = yearly[yearly.variant.eq(name)].copy()
            own['training_hurdle'] = fee; own['period'] = period
            annual.extend(own.to_dict('records'))
    for fee in HURDLES:
        name = fee_name(fee)
        for period, start in PERIODS:
            cal = market.calendar[market.calendar >= start]
            cash = cash_book(cal, market.rates, cfg.capital)
            book = cash[['nav', 'cash_benchmark']] * len(defs)
            entries = active = 0; commission = option_cost = hedge_cost = 0.
            pnl_paths = {}
            for d in defs:
                folder = OUTPUT / d['pair'] / name
                status = json.loads((folder / period / 'status.json').read_text())
                assert status['status'] == 'complete'
                assert status['execution_commission'] == EXECUTION_FEE and status['training_hurdle'] == fee
                nav = cash if status.get('cash_only') else pd.read_parquet(folder / period / 'daily_nav.parquet')
                assert nav.index.equals(cal)
                book += nav[['nav', 'cash_benchmark']] - cash[['nav', 'cash_benchmark']]
                pnl = float(nav.nav.iloc[-1] - nav.cash_benchmark.iloc[-1])
                pairs.append(dict(training_hurdle=fee, period=period, pair=d['pair'], label=d['label'], entries=status['entries'], net_excess_pnl=pnl))
                if status['entries']: pnl_paths[d['pair']] = nav.nav - nav.cash_benchmark
                entries += status['entries']; active += int(status['entries'] > 0)
                commission += status.get('commission', 0.); option_cost += status.get('option_cost', 0.)
                hedge_cost += status.get('hedge_cost', 0.)
                if period == 'full' and (folder / 'fits.csv').exists():
                    fits = read_fits(folder / 'fits.csv'); check_fits(fits)
                    if len(fits):
                        fits['pair'] = d['pair']; fits['training_hurdle'] = fee; fits_all.append(fits)
            book['return_'] = book.nav.pct_change().fillna(0)
            book['excess_return'] = book.return_ - book.cash_benchmark.pct_change().fillna(0)
            book.to_csv(REPORT / f'{name}_{period}_portfolio.csv')
            row = dict(training_hurdle=fee, execution_commission=EXECUTION_FEE, period=period, source='new_fit',
                       start=str(cal[0].date()), end=str(cal[-1].date()), pairs=len(defs), entries=entries,
                       active_pairs=active, option_commission=commission, option_cost=option_cost, hedge_cost=hedge_cost,
                       net_excess_pnl=float(book.nav.iloc[-1] - book.cash_benchmark.iloc[-1]),
                       cash_return=float(book.cash_benchmark.iloc[-1] / (cfg.capital * len(defs)) - 1),
                       **scores(book, cfg.capital * len(defs)))
            winners = sorted([r for r in pairs if r['training_hurdle'] == fee and r['period'] == period and r['net_excess_pnl'] > 1e-7], key=lambda r: -r['net_excess_pnl'])
            if winners:
                top = winners[0]; without = book.nav - pnl_paths[top['pair']]
                excess = without.pct_change().fillna(0) - book.cash_benchmark.pct_change().fillna(0)
                row.update(largest_winner=top['label'], largest_winner_pnl=top['net_excess_pnl'],
                           top_three_share_positive_pnl=sum(r['net_excess_pnl'] for r in winners[:3]) / sum(r['net_excess_pnl'] for r in winners),
                           sharpe_largest_winner_replaced_by_cash=float(np.sqrt(252) * excess.iloc[1:].mean() / excess.iloc[1:].std()))
            rows.append(row)
            for year, g in book.groupby(book.index.year):
                annual.append(dict(training_hurdle=fee, period=period, year=year,
                                   total_return_percent=100 * ((1 + g.return_).prod() - 1),
                                   cash_return_percent=100 * ((1 + book.cash_benchmark.pct_change().fillna(0).loc[g.index]).prod() - 1),
                                   annualized_excess_sharpe=float(np.sqrt(252) * g.excess_return.mean() / g.excess_return.std()) if g.excess_return.std() > 1e-12 else None))
    comparison = pd.DataFrame(rows).sort_values(['period', 'training_hurdle'])
    comparison.to_csv(REPORT / 'comparison.csv', index=False)
    pd.DataFrame(annual).sort_values(['period', 'training_hurdle', 'year']).to_csv(REPORT / 'annual.csv', index=False)
    pd.DataFrame(pairs).to_csv(REPORT / 'pairs.csv', index=False)
    pd.concat(fits_all, ignore_index=True).to_csv(REPORT / 'fits.csv', index=False)
    lines = ['# Training cost as a profitability hurdle', '',
             'Predeclared grid: $0.03, $0.15, $0.30, $0.65 per option contract. All evaluation books pay $0.03. The two intermediate policies rebuild rewards, commission features, completed-outcome profitability memory, directional gates and prior-year validation using the higher training cost. The extra cost is a conservative profitability hurdle, not a calibrated probability of profit.', '',
             '| Training hurdle | Full-history Sharpe | Recent Sharpe | Full entries | Active pairs | Full P&L above cash |',
             '|---|---:|---:|---:|---:|---:|']
    for fee in [.03, .15, .30, .65]:
        full = comparison[comparison.training_hurdle.eq(fee) & comparison.period.eq('full')].iloc[0]
        recent = comparison[comparison.training_hurdle.eq(fee) & comparison.period.eq('recent')].iloc[0]
        lines.append(f'| ${fee:.2f} | {full.annualized_excess_sharpe:.3f} | {recent.annualized_excess_sharpe:.3f} | {int(full.entries)} | {int(full.active_pairs)} | ${full.net_excess_pnl:,.0f} |')
    lines += ['', 'Full history: January 2018–August 2025. Recent test: January 2022–August 2025, starting flat with the same $1m per pair. All 346 original pairs remain, including cash sleeves. Same ATM30 call/call contracts, signed 25/50/75/100% ladder, monthly beta 0.85–1.15 and correlation ≥0.5 screen, actual-contract rewards, monthly eight-leaf direct Sharpe search, completed-outcome purging, prior-year validation and recent-five-year refit.', '',
              'Training validation pays the selected hurdle; evaluation resets commission to $0.03. The 0.10-vol-point option entry penalty, 1 bp daily stock hedge execution, financing, borrowing and risk limits are identical everywhere. Option quantities are fixed after initially matching vega within 10%, apart from supported splits, and each leg settles at its own expiry. No daily option-vega rebalancing. Training-cost changes can alter directions, dates and sizes, so this is more than filtering a frozen prediction by a confidence threshold.', '',
              'The $0.03 and $0.65 endpoints reuse their saved monthly policies; $0.15 and $0.30 are fresh fits of the same algorithm. Original $0.65 memory and curve gates are checked before fitting. Accounting, expiry and commission checks run on every simulated book. This is exploratory sensitivity on previously examined history; selecting the highest reported Sharpe would not constitute untouched validation. Largest-winner replacement is an ex-post concentration diagnostic.', '',
              '[Annual results](atm_cost_hurdles/annual.csv) · [Comparison](atm_cost_hurdles/comparison.csv) · [New pair results](atm_cost_hurdles/pairs.csv)', '']
    REPORT.with_suffix('.md').write_text('\n'.join(lines))
    print(comparison[['training_hurdle', 'period', 'annualized_excess_sharpe', 'entries', 'active_pairs', 'net_excess_pnl']].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--check', action='store_true')
    parser.add_argument('--report-only', action='store_true'); args = parser.parse_args()
    if args.check:
        import unittest
        suite = unittest.defaultTestLoader.loadTestsFromNames(['tests.test_atm_cost_hurdles', 'tests.test_atm_commission_training'])
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful(): raise SystemExit(1)
        return
    OUTPUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    defs = json.loads((SOURCE / 'control_inputs.json').read_text())['definitions']
    base_cfg = config(); actual_cfg = replace(base_cfg, option_commission=EXECUTION_FEE)
    paths = [Path(__file__), Path('scripts/retrain_atm_commission.py'), Path('scripts/run_curve_trading.py'),
             Path('scripts/curve_trading/unit_paths.py'), Path('scripts/curve_trading/policy.py'),
             Path('scripts/option_flex/contract_engine.py'), Path('scripts/curve_trading/config.py'),
             Path('scripts/replay_frozen_atm_commission.py'), SOURCE / 'control_inputs.json',
             Path('reports/aligned_contract_learning/manifest.json'),
             Path('data/curve_trading/C_history_manifest.json'), Path('data/curve_trading/C_select_manifest.json'),
             Path('reports/frozen_atm_commission/comparison.csv'), Path('reports/atm_retrained_commission/comparison.csv')]
    paths += [BASE / d['pair'] / '30d/states.parquet' for d in defs if (BASE / d['pair'] / '30d/states.parquet').exists()]
    signature = dict(grid=[.03, *HURDLES, .65], new_fits=HURDLES, execution_settings=asdict(actual_cfg),
                     definitions=defs, code_and_sources={str(p): digest(p) for p in paths})
    manifest = OUTPUT / 'inputs.json'
    if manifest.exists(): assert json.loads(manifest.read_text()) == json.loads(json.dumps(signature)), 'Inputs changed'
    else: manifest.write_text(json.dumps(signature, indent=2) + '\n')
    cal = load_calendar(ROOT); cal = cal[(cal >= '2010-01-01') & (cal <= '2025-08-29')]
    market = Market(cal); started = time.monotonic()
    if not args.report_only:
        for n, d in enumerate(defs, 1):
            pending = [fee for fee in HURDLES if not all((OUTPUT / d['pair'] / fee_name(fee) / p / 'status.json').exists() for p, _ in PERIODS)]
            if not pending: continue
            old = json.loads((SOURCE / d['pair'] / 'atm_control/status.json').read_text())
            assert old['status'] == 'complete'
            panel = None if old.get('structural_cash') else pd.read_parquet(BASE / d['pair'] / '30d/states.parquet')
            if panel is None or panel.empty:
                for fee in pending:
                    for period, _ in PERIODS:
                        dest = OUTPUT / d['pair'] / fee_name(fee) / period; dest.mkdir(parents=True, exist_ok=True)
                        (dest / 'status.json').write_text(json.dumps(dict(status='complete', pair=d['pair'], label=d['label'], entries=0, cash_only=True, training_hurdle=fee, execution_commission=EXECUTION_FEE)) + '\n')
                continue
            q, s, selection, divs = load_pair(market, d['secids'])
            native = make_units(panel, q, s, selection, divs, cal, settings())
            curve = make_units(panel, q, s, selection, divs, cal, base_cfg)
            memory_error = verify_reference(d, panel, native, curve)
            replay = Replay(q, s, selection, d['secids'], divs, actual_cfg)
            for fee in pending:
                folder = OUTPUT / d['pair'] / fee_name(fee); folder.mkdir(parents=True, exist_ok=True)
                state, labels, ends, gates, available = prepare_hurdle(panel, native, curve, fee)
                state.to_parquet(folder / 'states.parquet')
                np.savez_compressed(folder / 'gates.npz', dates=state.index.values, gates=gates[ATM], available=available[ATM], ends=ends[ATM])
                (folder / 'reference_check.json').write_text(json.dumps(dict(original_memory_max_error=memory_error, original_curve_gates_match=True)) + '\n')
                replay.cfg = replace(base_cfg, option_commission=fee)
                print(f'{n}/{len(defs)} {d["label"]} hurdle ${fee:.2f}: monthly training', flush=True)
                if (folder / 'actions.csv').exists():
                    actions = pd.read_csv(folder / 'actions.csv', index_col=0, parse_dates=[0]).code.astype(int)
                    fits = read_fits(folder / 'fits.csv')
                else:
                    features = NEW_FEATURES + [c if c != 'commission_vol_roundtrip' else 'commission_vol_entry' for c in EXTRA]
                    features += [c for c in state if c.startswith(('profit_', 'count_'))] + FEATURES + ['vix_previous_close']
                    actions, fits = train(d, state, labels, ends, gates, available, cal, replay, market.rates, folder, features, 'atm_control')
                    actions.to_csv(folder / 'actions.csv'); fits.to_csv(folder / 'fits.csv', index=False)
                check_fits(fits)
                pd.testing.assert_series_equal(actions, gate(actions, actions.index, state, gates, available), check_names=False)
                replay.cfg = actual_cfg
                for period, start in PERIODS:
                    dest = folder / period; dest.mkdir(exist_ok=True)
                    if (dest / 'status.json').exists(): continue
                    own = actions[actions.index >= start]
                    if not own.ne(0).any():
                        row = dict(status='complete', pair=d['pair'], label=d['label'], entries=0, cash_only=True)
                    else:
                        nav, t, dec = replay.run(own, cal[cal >= start]); mark_book(nav, market.rates, actual_cfg.capital)
                        row = save_result(dest, nav, t, dec, actual_cfg, s, d)
                        del nav, t, dec
                    row.update(training_hurdle=fee, execution_commission=EXECUTION_FEE)
                    (dest / 'status.json').write_text(json.dumps(row, indent=2) + '\n')
                    print(f'  {d["label"]} ${fee:.2f} {period}: {row["entries"]} entries at $0.03 execution', flush=True)
                del state, labels, ends, gates, available, actions, fits
                gc.collect()
            del panel, q, s, selection, divs, native, curve, replay
            gc.collect()
            (OUTPUT / 'progress.json').write_text(json.dumps(dict(roster_position=n, pairs=len(defs), last_pair=d['label'], elapsed_seconds=time.monotonic()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)) + '\n')
            print(f'{n}/{len(defs)} {d["label"]} both hurdles complete, {time.monotonic()-started:.0f}s', flush=True)
    summary(defs, market, actual_cfg)
    (OUTPUT / 'task_status.json').write_text(json.dumps(dict(status='complete', pairs=len(defs), new_hurdles=HURDLES, execution_commission=EXECUTION_FEE, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, elapsed_seconds=time.monotonic()-started)) + '\n')


if __name__ == '__main__': main()
