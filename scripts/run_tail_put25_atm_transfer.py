"""Frozen ATM-style full-state RL transferred to 30D 25-delta puts."""
import argparse
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import resource
import time

import joblib
import numpy as np
import pandas as pd

from scripts.tail_put25_policy import COMBO, CODES, FEATURES, states as spread_states, fit_tail, permitted
from scripts.tail_put25_atm_features import states as full_states, columns, actions
from scripts.run_atm65_volume_oi import employee_roster
from scripts.retrain_atm_commission import fast_memory
from scripts.train_expiry_contracts import settings
from scripts.curve_trading.config import config, DATA, RUNOFF
from scripts.curve_trading.policy import Replay, memory_gates
from scripts.curve_trading.unit_paths import UnitPaths, pair_labels
from scripts.retrain_atm_commission import cheaper_units, check_fits
from scripts.run_atm65_volume_oi import configure, EMPLOYEE, CHECKPOINT, WORKFLOW
from scripts.run_atm65_ablation import write_json, PERIODS
from scripts.replay_frozen_atm_commission import cash_book, save_result
from scripts.run_broad_contract_replay import Market, ROOT
from scripts.run_curve_trading import screen_dates
from scripts.run_aligned_contract_learning import mark_book
from scripts.run_selected_pair_backtest import scores, audit
from scripts.run_contract_walkforward import digest
from scripts.select_atm_contracts import load_calendar
from scripts.option_flex.contract_engine import UnresolvedPosition

OUT = Path('artifacts/contracts/tail_put25_atm_transfer')
REPORT = Path('reports/tail_put25_atm_transfer')
EXPLORATION = Path('artifacts/contracts/tail_feature_exploration')
MODELS = ['rl_atm_state', 'rl_atm_state_rank']


class PutMarket:
    def __init__(self, market, cfg, sources):
        self.market, self.cfg, self.sources, self.cache = market, cfg, sources, {}

    def retain(self, secids):
        for sid in list(self.cache):
            if sid not in secids:
                del self.cache[sid]
        gc.collect()

    def asset(self, sid):
        if sid in self.cache:
            return self.cache[sid]
        cal = self.market.calendar; parts = []
        for year in cal.year.unique():
            path = DATA / 'selection/P' / str(year) / f'{sid}.parquet'
            if path.exists():
                if str(path) not in self.sources:
                    self.sources[str(path)] = digest(path)
                s = pd.read_parquet(path)
                parts.append(s[s.tenor.eq(30) & s.bucket.eq('25') & s.cp_flag.eq('P')])
        sel = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=['date', 'bs_iv'])
        sel['date'] = pd.to_datetime(sel.date)
        assert not sel.date.duplicated().any()
        ids = sel.optionid.unique().tolist() if len(sel) else []
        quote_parts = []
        if ids:
            for year in cal.year.unique():
                path = DATA / 'history/P' / str(year) / f'{sid}.parquet'
                if path.exists():
                    if str(path) not in self.sources:
                        self.sources[str(path)] = digest(path)
                    quote_parts.append(pd.read_parquet(path, filters=[('optionid', 'in', ids)]))
        q = pd.concat(quote_parts, ignore_index=True) if quote_parts else pd.DataFrame()
        stock = self.market.stocks[self.market.stocks.secid.eq(sid)].set_index('date').reindex(cal)
        stock['secid'] = sid; stock['rate'] = self.market.rates; stock.index.name = 'date'
        divs = self.market.divs[self.market.divs.secid.eq(sid)]
        self.cache[sid] = dict(q=q, s=stock.reset_index(), sel=sel, divs=divs)
        return self.cache[sid]


def train_pair(d, state, daily, ends, gates, available, cal, replay, rates, folder, feature_columns):
    stop = cal[-1] - pd.Timedelta(days=RUNOFF)
    wanted = (state.index >= '2018-01-01') & (state.index <= stop) & screen_dates(d, state.index)
    idx = state.index[wanted]
    output = pd.Series(0, index=idx, name='code', dtype=int); logs = []
    checkpoint = folder / 'month_checkpoint.json'
    if checkpoint.exists():
        old = json.loads(checkpoint.read_text())
        assert old['dates'] == [str(t) for t in idx]
        output.iloc[:] = old['actions']; logs = old['fits']
    completed = {r['fit'] for r in logs}
    positions = cal.get_indexer(state.index); completion = cal[ends]
    finite = np.isfinite(daily).all(axis=(1, 2)) & state.feature_available.to_numpy(bool)
    for month in pd.period_range('2018-01-01', stop, freq='M'):
        cutoff = month.start_time; fitdate = str(cutoff.date())
        test = (idx >= cutoff) & (idx < (month + 1).start_time)
        if not test.any() or fitdate in completed:
            continue
        vstart = cutoff - pd.DateOffset(years=1)
        use = (state.index >= vstart - pd.DateOffset(years=5)) & (completion < vstart) & finite
        recent = (state.index >= cutoff - pd.DateOffset(years=5)) & (completion < cutoff) & finite
        vcal = cal[(cal >= vstart) & (cal < cutoff)]
        val = (state.index >= vcal[0]) & (state.index <= vcal[-1] - pd.Timedelta(days=RUNOFF))
        opportunity = (state.eligible.to_numpy(bool) & state.feature_available.to_numpy(bool) & available & gates.any(axis=1) & val).any()
        row = dict(fit=fitdate, training_rows=int(use.sum()), recent_rows=int(recent.sum()),
                   validation_start=str(vcal[0].date()), validation_end=str(vcal[-1].date()))
        row['status'] = 'insufficient_training' if use.sum() < 504 or len(vcal) < 200 else 'flat_no_validation_opportunities'
        if use.sum() >= 504 and len(vcal) >= 200 and opportunity:
            labels = daily[use].copy(); labels[~state.loc[use, 'eligible'].to_numpy(bool)] = 0
            model, details = fit_tail(state.loc[use, feature_columns], labels, positions[use], ends[use])
            row['last_training_outcome'] = str(completion[use].max().date())
            row.update({'proposal_' + k: v for k, v in details.items()})
            proposal = actions(model, state.index[val], state, gates, available, feature_columns)
            score = None; error = None
            try:
                nav, trades, _ = replay.run(proposal, vcal)
                mark_book(nav, rates, replay.cfg.capital)
                audit(nav, trades, replay.s, replay.cfg, d['secids'])
                score = scores(nav, replay.cfg.capital)['annualized_excess_sharpe']
            except (UnresolvedPosition, ValueError) as exc:
                error = str(exc)
            enabled = score is not None and score > 0 and recent.sum() >= 504
            row.update(status='enabled' if enabled else 'flat_validation_gate',
                       validation_sharpe=score, validation_error=error)
            if enabled:
                own = daily[recent].copy(); own[~state.loc[recent, 'eligible'].to_numpy(bool)] = 0
                model, details = fit_tail(state.loc[recent, feature_columns], own, positions[recent], ends[recent])
                output.loc[idx[test]] = actions(model, idx[test], state, gates, available, feature_columns)
                row['last_recent_outcome'] = str(completion[recent].max().date())
                row.update({'refit_' + k: v for k, v in details.items()})
                row['split_features'] = [feature_columns[int(f)] for f in model[0].tree_.feature if f >= 0]
                row['leaf_action_codes'] = model[2].tolist()
                dest = folder / 'models'; dest.mkdir(exist_ok=True)
                joblib.dump(model, dest / f'{fitdate}.joblib')
        logs.append(row)
        write_json(checkpoint, dict(dates=[str(t) for t in idx], actions=output.to_list(), fits=logs))
        if month.month in [1, 7] and opportunity:
            print(f'  {d["label"]} {month}: {row["status"]}', flush=True)
    fits = pd.DataFrame(logs) if logs else pd.DataFrame(columns=['fit', 'status', 'training_rows'])
    check_fits(fits); fits.to_csv(folder / 'fits.csv', index=False)
    pd.testing.assert_series_equal(output, permitted(output, idx, state, gates, available))
    return output


def cash_status(d, model, period, reason):
    dest = OUT / d['pair'] / model / period; dest.mkdir(parents=True, exist_ok=True)
    write_json(dest / 'status.json', dict(status='complete', pair=d['pair'], label=d['label'],
               entries=0, cash_only=True, reason=reason, training_hurdle=.65, execution_commission=.03))


def process(d, market, loader, actual, training):
    folder = OUT / d['pair']; folder.mkdir(parents=True, exist_ok=True)
    source = EMPLOYEE / d['pair'] / 'states.parquet'
    if not source.exists():
        for model in MODELS:
            for period, _ in PERIODS:
                cash_status(d, model, period, 'no_state_panel')
        write_json(folder / 'coverage.json', dict(pair=d['pair'], label=d['label'], state_rows=0))
        return
    panel = pd.read_parquet(source); cal = market.calendar
    loader.retain(d['secids']); a, b = [loader.asset(sid) for sid in d['secids']]
    state = spread_states(a['sel'], b['sel'], panel, cal)
    state.to_parquet(folder / 'states.parquet')
    if not state.feature_available.any() or a['q'].empty or b['q'].empty:
        for model in MODELS:
            for period, _ in PERIODS:
                cash_status(d, model, period, 'no_observed_put25_state')
        write_json(folder / 'coverage.json', dict(pair=d['pair'], label=d['label'], state_rows=len(state), observed_rows=0))
        return
    unit = [UnitPaths(x['q'], x['s'], x['sel'], cal, state.index, x['divs'], actual) for x in [a, b]]
    actual_daily, ends, vegas, statuses = pair_labels(*unit, COMBO, actual)
    # Reconcile against the prior independent feature scan before fitting.
    old = pd.read_parquet(EXPLORATION / 'pairs' / d['pair'] / 'P25.parquet').set_index('date')
    common = state.index.intersection(old.index)
    np.testing.assert_allclose(state.loc[common, FEATURES], old.loc[common, FEATURES], atol=1e-12, rtol=0, equal_nan=True)
    complete = old.index[old.resolved]
    pos = state.index.get_indexer(complete)
    rewards = np.divide(actual_daily[:, [0, -1]].sum(axis=2) * actual.entry_vega_per_leg,
                        vegas[:, [0, -1]], out=np.full((len(state), 2), np.nan), where=vegas[:, [0, -1]] > 0)
    np.testing.assert_allclose(rewards[pos], old.loc[complete, ['short_net', 'long_net']], atol=1e-6, rtol=0)
    label_error = float(np.nanmax(abs(rewards[pos] - old.loc[complete, ['short_net', 'long_net']].to_numpy()))) if len(pos) else 0.
    del actual_daily
    training_units = [cheaper_units(u, training) for u in unit]
    labels, finish, training_vegas, training_status = pair_labels(*training_units, COMBO, training)
    np.testing.assert_array_equal(ends, finish); np.testing.assert_array_equal(vegas, training_vegas)
    reward = np.divide(labels[:, [0, -1]].sum(axis=2) * training.entry_vega_per_leg,
                       vegas[:, [0, -1]], out=np.full((len(state), 2), np.nan), where=vegas[:, [0, -1]] > 0)
    gates = memory_gates(state.index, cal[finish], reward, vegas[:, [0, -1]] > 0)
    # ATM's original state memory used its native 21-45D contract setup,
    # whereas its separate action gates used curve execution eligibility.
    # Preserve that distinction, allowing the required OTM put strikes.
    native_cfg = replace(settings(), option_commission=.65, max_moneyness_distance=1_000_000.)
    native_units = [UnitPaths(x['q'], x['s'], x['sel'], cal, state.index, x['divs'], native_cfg) for x in [a, b]]
    native_daily, native_finish, native_vegas, _ = pair_labels(*native_units, COMBO, native_cfg)
    native_reward = np.divide(native_daily[:, [0, -1]].sum(axis=2) * native_cfg.entry_vega_per_leg,
        native_vegas[:, [0, -1]], out=np.full((len(state), 2), np.nan), where=native_vegas[:, [0, -1]] > 0)
    memory = fast_memory(state.index, cal[native_finish], native_reward, native_vegas[:, [0, -1]] > 0)
    state = full_states(a['sel'], b['sel'], panel, cal, memory, .65)
    employee = d['pair'] in employee_roster()
    feature_sets = {m: columns(state, employee=employee, rank=m.endswith('_rank')) for m in MODELS}
    state['feature_available'] &= np.isfinite(state[feature_sets[MODELS[-1]]]).all(axis=1)
    state.to_parquet(folder / 'states.parquet')
    write_json(folder / 'feature_columns.json', feature_sets)
    # Frozen two-feature experiment is a labels/gates control, never a source
    # of put-state profitability inputs or newly learned actions.
    baseline = Path('artifacts/contracts/tail_put25_strategy') / d['pair']
    with np.load(baseline / 'gates.npz') as z:
        np.testing.assert_array_equal(z['dates'], state.index.values)
        np.testing.assert_array_equal(z['gates'], gates)
        np.testing.assert_array_equal(z['ends'], finish)
        np.testing.assert_allclose(z['reward'], reward, atol=1e-9, rtol=0, equal_nan=True)
    del native_daily, native_units, native_reward
    gc.collect()
    available = (unit[0].ids(*COMBO[:3]) >= 0) & (unit[1].ids(*COMBO[:3]) >= 0)
    np.savez_compressed(folder / 'gates.npz', dates=state.index.values, gates=gates, available=available,
                        ends=finish, reward=reward, training_status=training_status)
    daily = np.zeros((len(state), 9, labels.shape[-1]), dtype=np.float32); daily[:, 1:] = labels
    q = pd.concat([a['q'], b['q']], ignore_index=True)
    s = pd.concat([a['s'], b['s']], ignore_index=True)
    sel = pd.concat([a['sel'], b['sel']], ignore_index=True)
    divs = pd.concat([a['divs'], b['divs']], ignore_index=True)
    replay = Replay(q, s, sel, d['secids'], divs, training)
    del labels, unit, training_units; gc.collect()
    idx = state.index[(state.index >= '2018-01-01') & (state.index <= cal[-1] - pd.Timedelta(days=RUNOFF)) & screen_dates(d, state.index)]
    for model in MODELS:
        dest = folder / model; dest.mkdir(exist_ok=True)
        replay.cfg = training
        rlpath = dest / 'actions.csv'
        codes = pd.read_csv(rlpath, index_col=0, parse_dates=[0]).code.astype(int) if rlpath.exists() else train_pair(
            d, state, daily, finish, gates, available, cal, replay, market.rates, dest, feature_sets[model])
        codes.index = pd.DatetimeIndex(codes.index)
        assert codes.index.equals(idx)
        pd.testing.assert_series_equal(codes, permitted(codes, idx, state, gates, available))
        codes.to_csv(rlpath)
        replay.cfg = actual
        for period, start in PERIODS:
            output = dest / period; output.mkdir(exist_ok=True)
            if (output / 'status.json').exists():
                continue
            own = codes[codes.index >= start]
            if not own.ne(0).any():
                cash_status(d, model, period, 'policy_flat')
                continue
            try:
                nav, trades, decisions = replay.run(own, cal[cal >= start]); mark_book(nav, market.rates, actual.capital)
                result = save_result(output, nav, trades, decisions, actual, s, d)
                assert trades.empty or trades.optionid.isin(sel.optionid).all()
                result['decision_counts'] = decisions.status.value_counts().to_dict()
                result['maximum_gross_vega'] = float(nav.gross_vega.max())
                result['risk_breach_sessions'] = int(nav.post_trade_risk_breach.sum())
                result.update(training_hurdle=.65, execution_commission=.03)
                write_json(output / 'status.json', result)
            except UnresolvedPosition as exc:
                write_json(output / 'status.json', dict(status='incomplete', pair=d['pair'], label=d['label'], error=str(exc)))
                if hasattr(exc, 'partial_nav'):
                    exc.partial_nav.to_parquet(output / 'partial_nav.parquet')
    tables = [pd.read_csv(folder / m / 'fits.csv').assign(model=m) for m in MODELS]
    pd.concat(tables, ignore_index=True).to_csv(folder / 'fits.csv', index=False)
    write_json(folder / 'coverage.json', dict(pair=d['pair'], label=d['label'], state_rows=len(state),
        employee_features=employee, base_feature_count=len(feature_sets[MODELS[0]]), rank_feature_count=len(feature_sets[MODELS[1]]),
        original_tail_observed_rows=int(spread_states(a['sel'], b['sel'], panel, cal).feature_available.sum()),
        observed_rows=int(state.feature_available.sum()), screened_rows=int((state.feature_available & state.eligible).sum()),
        signal_known_rows=int(available.sum()), memory_permitted_rows=int((available & gates.any(axis=1)).sum()),
        resolved_labels=int((statuses == 'complete').sum()), unresolved_labels=int((statuses == 'unresolved').sum()),
        recovered_scan_cohorts=len(pos), max_label_error=label_error,
        evaluated_memory_permitted_rows=int((state.loc[idx, 'feature_available'].to_numpy(bool) & state.loc[idx, 'eligible'].to_numpy(bool) & available[state.index.get_indexer(idx)] & gates[state.index.get_indexer(idx)].any(axis=1)).sum())))


def report(defs, market, cfg):
    comparisons, annual, pairs, coverage, fits, concentration = [], [], [], [], [], []
    unresolved = []
    for d in defs:
        folder = OUT / d['pair']; coverage.append(json.loads((folder / 'coverage.json').read_text()))
        if (folder / 'fits.csv').exists():
            fit = pd.read_csv(folder / 'fits.csv'); check_fits(fit); fits.append(fit.assign(pair=d['pair']))
    for model in MODELS:
        for period, start in PERIODS:
            cal = market.calendar[market.calendar >= start]
            cash = cash_book(cal, market.rates, cfg.capital)
            book = cash[['nav', 'cash_benchmark']] * len(defs); entries = active = 0; paths = {}
            costs = dict(commission=0., option_cost=0., hedge_cost=0.)
            for d in defs:
                folder = OUT / d['pair'] / model / period
                status = json.loads((folder / 'status.json').read_text())
                if status['status'] != 'complete':
                    unresolved.append(dict(model=model, period=period, pair=d['pair'], label=d['label'], error=status.get('error')))
                    continue
                nav = cash if status.get('cash_only') else pd.read_parquet(folder / 'daily_nav.parquet')
                assert nav.index.equals(cal)
                book += nav[['nav', 'cash_benchmark']] - cash[['nav', 'cash_benchmark']]
                pnl = nav.nav - nav.cash_benchmark
                if status['entries']:
                    paths[d['pair']] = pnl
                pairs.append(dict(model=model, period=period, pair=d['pair'], label=d['label'], entries=status['entries'], net_excess_pnl=float(pnl.iloc[-1])))
                entries += status['entries']; active += int(status['entries'] > 0)
                for k in costs:
                    costs[k] += status.get(k, 0.)
            if any(r['model'] == model and r['period'] == period for r in unresolved):
                comparisons.append(dict(model=model, period=period, status='incomplete', pairs=len(defs)))
                continue
            book['return_'] = book.nav.pct_change().fillna(0)
            cash_returns = book.cash_benchmark.pct_change().fillna(0)
            book['excess_return'] = book.return_ - cash_returns
            book.to_csv(REPORT / f'{model}_{period}_portfolio.csv')
            excess_pnl = book.nav - book.cash_benchmark
            comparisons.append(dict(model=model, period=period, status='complete', pairs=len(defs), entries=entries,
                active_pairs=active, net_excess_pnl=float(excess_pnl.iloc[-1]),
                max_excess_drawdown_dollars=float((excess_pnl - excess_pnl.cummax().clip(lower=0)).min()),
                **costs, **scores(book, cfg.capital * len(defs))))
            for year, g in book.groupby(book.index.year):
                sd = g.excess_return.std()
                annual.append(dict(model=model, period=period, year=year,
                    return_percent=100 * ((1 + g.return_).prod() - 1),
                    excess_return_percent=100 * ((1 + g.return_).prod() - (1 + cash_returns.loc[g.index]).prod()),
                    annualized_excess_sharpe=float(np.sqrt(252) * g.excess_return.mean() / sd) if sd > 1e-12 else None))
            winners = sorted([r for r in pairs if r['model'] == model and r['period'] == period and r['net_excess_pnl'] > 0], key=lambda r: -r['net_excess_pnl'])
            for count in [1, 3]:
                top = winners[:count]; without = book.nav.copy()
                for r in top:
                    without -= paths[r['pair']]
                excess = without.pct_change().fillna(0) - cash_returns
                concentration.append(dict(model=model, period=period, winners_replaced_with_cash=len(top),
                    winner_names='; '.join(r['label'] for r in top),
                    removed_net_excess_pnl=sum(r['net_excess_pnl'] for r in top),
                    net_excess_pnl=float(without.iloc[-1] - book.cash_benchmark.iloc[-1]),
                    annualized_excess_sharpe=float(np.sqrt(252) * excess.iloc[1:].mean() / excess.iloc[1:].std()) if excess.iloc[1:].std() > 1e-12 else None))
    pd.DataFrame(comparisons).to_csv(REPORT / 'comparison.csv', index=False)
    pd.DataFrame(annual).to_csv(REPORT / 'annual.csv', index=False)
    pd.DataFrame(pairs).to_csv(REPORT / 'pairs.csv', index=False)
    pd.DataFrame(coverage).to_csv(REPORT / 'coverage.csv', index=False)
    pd.DataFrame(concentration).to_csv(REPORT / 'concentration.csv', index=False)
    if fits:
        pd.concat(fits, ignore_index=True).to_csv(REPORT / 'fits.csv', index=False)
    write_json(REPORT / 'accounting_coverage.json', dict(requested_pairs=len(defs), unresolved=unresolved))
    pd.read_csv('reports/atm65_employee_sentiment/comparison.csv').query("universe == 'all' and model == 'rl_sentiment'").to_csv(REPORT / 'frozen_atm_reference.csv', index=False)
    print(pd.DataFrame(comparisons).to_string(index=False), flush=True)
    return unresolved


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--limit', type=int); parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args(); configure()
    OUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic(); actual = replace(config(), option_commission=.03); training = replace(actual, option_commission=.65)
    defs = json.loads((EMPLOYEE / 'inputs.json').read_text())['definitions']
    cal = load_calendar(ROOT); cal = cal[(cal >= '2010-01-01') & (cal <= '2025-08-29')]
    files = [Path(__file__), Path('scripts/tail_put25_atm_features.py'), Path('tests/test_tail_put25_atm_features.py'), Path('scripts/tail_put25_policy.py'), Path('scripts/run_tail_put25_strategy.py'),
        Path('scripts/run_atm65_employee_sentiment.py'), Path('scripts/run_atm65_ablation.py'), Path('scripts/run_atm65_volume_oi.py'),
        Path('reports/atm65_employee_sentiment/coverage.csv'), Path('reports/aligned_contract_learning/manifest.json'),
        Path('reports/tail_put25_strategy/output_hashes.json'), Path('reports/tail_put25_strategy/comparison.csv'), Path('scripts/run_tail_put_strategy.py'),
        Path('scripts/tail_put_policy.py'), Path('scripts/curve_trading/policy.py'),
        Path('scripts/curve_trading/unit_paths.py'), Path('scripts/curve_trading/config.py'), Path('scripts/retrain_atm_commission.py'),
        Path('scripts/option_flex/contract_engine.py'), Path('scripts/option_flex/contract_risk.py'),
        Path('scripts/run_aligned_contract_learning.py'), Path('scripts/replay_frozen_atm_commission.py'),
        CHECKPOINT, WORKFLOW, EMPLOYEE / 'inputs.json', EXPLORATION / 'inputs.json', EXPLORATION / 'status.json',
        ROOT / 'zero_curve.parquet', ROOT / 'distributions.parquet',
        *[ROOT / 'underlying' / f'year={year}.parquet' for year in cal.year.unique()]]
    for d in defs:
        files.extend(p for p in [EMPLOYEE / d['pair'] / 'states.parquet', EXPLORATION / 'pairs' / d['pair'] / 'P25.parquet'] if p.exists())
    signature = dict(actual=asdict(actual), training=asdict(training), combo=list(COMBO), codes=CODES, models=MODELS,
        features='47 frozen ATM definitions, plus original employee features on original 28 sleeves; second variant adds causal put-spread historical rank',
        definitions=defs, sources={str(p): digest(p) for p in files}, seed=2026,
        protocol='Two predeclared RL-only variants: exact frozen ATM state definitions adapted to put25 contracts, and the same state plus put-spread historical rank. Same 8-leaf Sharpe kernel, 25/50/75/100% ladder, monthly completed-outcome 5Y refits, positive prior-year actual inventory validation, directional profitability gates, $0.65 training hurdle and $0.03 actual execution/risk settings. Both can choose direction/entry/size as original ATM. Native state memory remains 21-45D with OTM moneyness allowed; action gates retain curve 7-55D settings. Stock/earnings/VIX/surface/employee context and candidate calendar are frozen. No NN, no parameter grid or new inventory-aware algorithm. All 345 original sleeves retained; inspected history, not untouched holdout.')
    manifest = OUT / 'inputs.json'
    if manifest.exists():
        assert json.loads(manifest.read_text()) == signature, 'Inputs changed'
    else:
        write_json(manifest, signature)
    source_path = OUT / 'market_sources.json'
    sources = json.loads(source_path.read_text()) if source_path.exists() else {}
    market = Market(cal); loader = PutMarket(market, actual, sources); newly = 0
    if not args.report_only:
        for n, d in enumerate(defs, 1):
            if all((OUT / d['pair'] / m / p / 'status.json').exists() for m in MODELS for p, _ in PERIODS):
                continue
            pair_started = time.monotonic()
            if (EMPLOYEE / d['pair'] / 'states.parquet').exists():
                print(f'{n}/{len(defs)} {d["label"]}: ATM-state put25 transfer', flush=True)
            process(d, market, loader, actual, training)
            write_json(source_path, sources); newly += 1
            write_json(OUT / 'progress.json', dict(roster_position=n, pairs=len(defs), last_pair=d['label'],
                last_pair_seconds=time.monotonic() - pair_started, elapsed_seconds=time.monotonic() - started,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            print(f'{n}/{len(defs)} {d["label"]}: complete {time.monotonic() - pair_started:.1f}s, peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.0f} MiB', flush=True)
            gc.collect()
            if args.limit and newly >= args.limit:
                return
    loader.cache.clear(); gc.collect()
    unresolved = report(defs, market, actual)
    for path, sha in signature['sources'].items():
        assert digest(path) == sha, f'Source changed: {path}'
    for path, sha in sources.items():
        assert digest(path) == sha, f'Market input changed: {path}'
    previous = json.loads(Path('reports/tail_put25_strategy/output_hashes.json').read_text())
    for path, sha in previous.items():
        assert digest(path) == sha, f'Prior put25 output changed: {path}'
    reference = json.loads(CHECKPOINT.read_text())
    assert digest(reference['archive']) == reference['archive_sha256']
    for path, sha in reference['files'].items():
        assert digest(path) == sha, f'Frozen reference changed: {path}'
    fitpath = REPORT / 'fits.csv'
    fits = pd.read_csv(fitpath) if fitpath.exists() else pd.DataFrame()
    audit_record = dict(status='complete' if not unresolved else 'accounting_incomplete', roster=len(defs),
        unresolved=unresolved, fitting_features=signature['features'], nn_refits=0,
        monthly_fits_logged=len(fits), enabled_monthly_refits=int(fits.status.eq('enabled').sum()) if len(fits) else 0,
        main_sources_verified=len(signature['sources']), market_sources_verified=len(sources),
        frozen_files_verified=len(reference['files']), prior_tail_files_verified=len(previous), reference_archive_verified=True,
        elapsed_seconds=time.monotonic() - started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        allocation='Separate $1m sleeves for each original pair, matching ATM reference. Risk limits applied per sleeve, not cross-sleeve netting. No strategy stacking.',
        limitations='Inspected history; sample correlations were not portfolio Sharpe. RL optimizer uses additive standalone shadow labels, validated and evaluated with actual capped inventory. No new holdout. Puts use existing European BS marks; early-exercise/assignment is not modeled. Initial vega neutrality, daily delta hedging, own-expiry hold; no ongoing option-vega rebalance or forced risk exit.')
    write_json(REPORT / 'audit.json', audit_record); write_json(OUT / 'task_status.json', audit_record)
    print(json.dumps(audit_record, indent=2), flush=True)
    finish_report()


def finish_report():
    current = pd.read_csv(REPORT / 'comparison.csv')
    old = pd.read_csv('reports/tail_put25_strategy/comparison.csv')
    combined = pd.concat([old.assign(source='frozen_two_feature'), current.assign(source='atm_state_transfer')], ignore_index=True)
    combined.to_csv(REPORT / 'reference_comparison.csv', index=False)
    annual = pd.concat([pd.read_csv('reports/tail_put25_strategy/annual.csv').assign(source='frozen_two_feature'),
                        pd.read_csv(REPORT / 'annual.csv').assign(source='atm_state_transfer')], ignore_index=True)
    annual.to_csv(REPORT / 'reference_annual.csv', index=False)
    fits = pd.read_csv(REPORT / 'fits.csv')
    usage = []
    for path in OUT.glob('*/rl_atm_state*/models/*.joblib'):
        model = joblib.load(path)
        feature_set = json.loads((path.parents[2] / 'feature_columns.json').read_text())[path.parents[1].name]
        for f in model[0].tree_.feature:
            if f >= 0:
                usage.append(dict(pair=path.parents[2].name, model=path.parents[1].name, feature=feature_set[int(f)]))
    if usage:
        pd.DataFrame(usage).groupby(['model', 'feature']).size().rename('splits').reset_index().to_csv(REPORT / 'feature_usage.csv', index=False)
    text = ['ATM-state RL transfer to 30D 25-delta puts', '',
        'Two predeclared variants; no NN retraining. Actual costs, risk limits, optimizer, monthly validation, quarter sizing ladder, and full 345-sleeve roster match the frozen ATM checkpoint.',
        'Contract IV, IV/RV value, shifted 252-session reversion, DTE, moneyness, commission and completed-outcome native profitability memory are recomputed for puts. Other context is frozen. Employee features remain active only on the original 28 review-covered sleeves.',
        'Native state memory uses the original 21-45D bounds with OTM strikes allowed; independent action gates use the original curve 7-55D execution bounds. Both histories exclude outcomes completing on or after the signal date.',
        'The second variant adds only the known put-spread percentile. Both variants use identical finite input masks. Original ATM momentum is not present in this checkpoint; no new momentum feature was introduced.',
        'Policies may choose direction, entry and size, matching ATM; this is not a fixed-rule sizing overlay or an inventory-aware sequential agent.', '',
        combined[['source', 'model', 'period', 'entries', 'active_pairs', 'net_excess_pnl', 'annualized_excess_sharpe']].to_string(index=False), '',
        'All Sharpe ratios use daily cash-excess returns and sqrt(252). 2025 ends August 29. Full 2018-2025 and recent 2022-2025 books use separate identical cash starts.',
        'Previously inspected history; neither signal selection nor this comparison is an untouched holdout. Baseline rule has no learned validation gate, so comparison with rule is not a pure sizing ablation.',
        'Actual quote-close selection executes next session, initial integer vega neutrality, daily delta hedging, hold each option to its expiry. Dividend cash on the hedge, financing and borrow remain included. European put approximation; American assignment not modeled.',
        'Risk caps apply within independent $1m pair sleeves; portfolio is their sum with cash-only sleeves retained. No shared-name netting or cross-sleeve inventory optimization.', '',
        fits.groupby(['model', 'status']).size().to_string(), '']
    (REPORT / 'findings.txt').write_text('\n'.join(text))
    paths = [Path(__file__), Path('scripts/tail_put25_atm_features.py'), Path('tests/test_tail_put25_atm_features.py')]
    paths += [p for p in OUT.rglob('*') if p.is_file()]
    paths += [p for p in REPORT.glob('*') if p.is_file() and p.name != 'output_hashes.json']
    write_json(REPORT / 'output_hashes.json', {str(p): digest(p) for p in sorted(set(paths))})


if __name__ == '__main__':
    main()
