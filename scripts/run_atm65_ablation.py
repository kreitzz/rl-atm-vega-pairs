"""Single-hurdle attribution, bounded worker and atomic monthly checkpoints.

Reuse fixed-NN validation decisions; recover only enabled refits for confidence
sizing. Retrain fixed-action Sharpe and supervised tree policies on identical
tree partitions. Also replay frozen RL directions at full size.
"""
import argparse
import gc
import json
from pathlib import Path
import resource
import time
from dataclasses import asdict, replace
import numpy as np
import pandas as pd
from scripts.curve_trading.config import config, COMBOS
from scripts.curve_trading.policy import ATM, Replay, fit as fit_rl, predict, gate
from scripts.run_atm_cost_hurdles import make_units, reward_and_memory
from scripts.replay_frozen_atm_commission import load_pair, cash_book, save_result
from scripts.run_broad_contract_replay import Market, ROOT
from scripts.run_curve_trading import RUNOFF, screen_dates, NEW_FEATURES, EXTRA, FEATURES
from scripts.run_contract_walkforward import fit_nn, digest
from scripts.run_selected_pair_backtest import scores
from scripts.run_aligned_contract_learning import mark_book
from scripts.retrain_atm_commission import check_fits
from scripts.select_atm_contracts import load_calendar
from scripts.option_flex.contract_engine import UnresolvedPosition

OUT = Path('artifacts/contracts/atm65_ablation')
REPORT = Path('reports/atm65_ablation')
NN = Path('artifacts/contracts/atm_nn_cost_hurdles')
CURVE = Path('artifacts/contracts/curve_trading')
FROZEN = Path('artifacts/contracts/frozen_atm_commission_003')
MODELS = ['rl_fixed', 'tree_fixed', 'nn_sized', 'rl_fixed_frozen']
PERIODS = [('full', '2018-01-01'), ('recent', '2022-01-01')]
SHORT = 1 + ATM * 8
LONG = SHORT + 7
FIXED = [0, SHORT, LONG]

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n'); temp.replace(path)

def full_size(codes):
    c = np.asarray(codes, dtype=int)
    return np.where(c == 0, 0, np.where((c - 1 - ATM * 8) < 4, SHORT, LONG))

def read_actions(path):
    result = pd.read_csv(path, index_col=0, parse_dates=[0]).code.astype(int)
    # pandas cannot infer datetime dtype when an action CSV has no rows.
    result.index = pd.DatetimeIndex(result.index)
    return result

def tree_baseline(model, x, labels):
    tree, leaves, _ = model
    assignment = tree.apply(x)
    target = labels[:, [1, -1]].sum(axis=2)
    means = np.array([target[assignment == leaf].mean(axis=0) for leaf in leaves])
    choice = means.argmax(axis=1)
    policy = np.where(means.max(axis=1) > 0, np.array([SHORT, LONG])[choice], 0)
    return tree, leaves, policy

def nn_values(model, x):
    scaler, network, center, scale = model
    return network.predict(scaler.transform(x)) * scale + center

def confidence_thresholds(predictions, allowed):
    choice = predictions.argmax(axis=1)
    best = predictions[np.arange(len(predictions)), choice]
    good = (best > 0) & allowed[np.arange(len(predictions)), choice] & np.isfinite(best)
    return np.quantile(best[good], [.25, .5, .75]) if good.any() else np.array([np.inf] * 3)

def confidence_codes(predictions, thresholds):
    side = predictions.argmax(axis=1)
    best = predictions[np.arange(len(predictions)), side]
    rank = np.searchsorted(thresholds, best, side='right')
    # short slots -1,-.75,-.5,-.25; long slots .25,.5,.75,1.
    codes = SHORT + np.where(side == 0, 3 - rank, 4 + rank)
    return np.where(best > 0, codes, 0).astype(int)

def features(panel):
    columns = NEW_FEATURES + [c if c != 'commission_vol_roundtrip' else 'commission_vol_entry' for c in EXTRA]
    return columns + [c for c in panel if c.startswith(('profit_', 'count_'))] + FEATURES + ['vix_previous_close']

def train_pair(d, panel, daily, finish, gates, available, cal, replay, rates, folder):
    stop = cal[-1] - pd.Timedelta(days=RUNOFF)
    wanted = (panel.index >= '2018-01-01') & (panel.index <= stop) & screen_dates(d, panel.index)
    idx = panel.index[wanted]; columns = features(panel)
    actions = pd.DataFrame(0, index=idx, columns=MODELS, dtype=int)
    logs = []; checkpoint = folder / 'month_checkpoint.json'
    if checkpoint.exists():
        old = json.loads(checkpoint.read_text())
        assert old['dates'] == [str(t) for t in idx]
        actions.loc[:, MODELS] = np.asarray(old['actions'], dtype=int); logs = old['fits']
    completed = {r['fit'] for r in logs}
    fixed_nn = read_actions(NN / d['pair'] / 'fee_065/actions.csv')
    try:
        nn_fits = pd.read_csv(NN / d['pair'] / 'fee_065/fits.csv').set_index('fit')
    except pd.errors.EmptyDataError:
        assert len(idx) == 0, 'Nonempty test period lacks NN fit records'
        nn_fits = pd.DataFrame(columns=['fit', 'status']).set_index('fit')
    saved_rl = read_actions(CURVE / d['pair'] / 'atm_control/actions.csv')
    assert fixed_nn.index.equals(idx) and saved_rl.index.equals(idx)
    actions['rl_fixed_frozen'] = full_size(saved_rl)
    positions = cal.get_indexer(panel.index)
    completion = cal[finish]
    finite = np.isfinite(daily).all(axis=(1, 2))
    x = panel[columns]
    for month in pd.period_range('2018-01-01', stop, freq='M'):
        cutoff = month.start_time; fitdate = str(cutoff.date())
        test = (idx >= cutoff) & (idx < (month + 1).start_time)
        if not test.any() or fitdate in completed: continue
        vstart = cutoff - pd.DateOffset(years=1)
        use = (panel.index >= vstart - pd.DateOffset(years=5)) & (completion < vstart) & finite
        vcal = cal[(cal >= vstart) & (cal < cutoff)]
        val = (panel.index >= vcal[0]) & (panel.index <= vcal[-1] - pd.Timedelta(days=RUNOFF))
        recent = (panel.index >= cutoff - pd.DateOffset(years=5)) & (completion < cutoff) & finite
        dates = idx[test]
        info = dict(fit=fitdate, training_rows=int(use.sum()), validation_start=str(vcal[0].date()),
                    validation_end=str(vcal[-1].date()))
        row = dict(info); permitted = {}
        opportunity = (panel.eligible.to_numpy(bool) & available[ATM] & gates[ATM].any(axis=1) & val).any()
        if use.sum() >= 504 and len(vcal) >= 200 and opportunity:
            labels = daily[use].copy(); labels[~panel.loc[use, 'eligible'].to_numpy(bool)] = 0
            model, _ = fit_rl(x.loc[use], labels[:, [0, 1, -1]], positions[use], finish[use], FIXED)
            tree = tree_baseline(model, x.loc[use], labels)
            row['last_training_outcome'] = str(completion[use].max().date())
            for name, proposal_model in [('rl_fixed', model), ('tree_fixed', tree)]:
                proposal = gate(predict(proposal_model, x.loc[val]), panel.index[val], panel, gates, available)
                score = None; error = None
                if proposal.ne(0).any():
                    try:
                        nav, _, _ = replay.run(proposal, vcal); mark_book(nav, rates, replay.cfg.capital)
                        score = scores(nav, replay.cfg.capital)['annualized_excess_sharpe']
                    except UnresolvedPosition as exc: error = str(exc)
                permitted[name] = score is not None and score > 0 and recent.sum() >= 504
                row[name + '_status'] = 'enabled' if permitted[name] else 'flat_validation_gate'
                row[name + '_validation_sharpe'] = score; row[name + '_validation_error'] = error
            if any(permitted.values()):
                own = daily[recent].copy(); own[~panel.loc[recent, 'eligible'].to_numpy(bool)] = 0
                model, _ = fit_rl(x.loc[recent], own[:, [0, 1, -1]], positions[recent], finish[recent], FIXED)
                tree = tree_baseline(model, x.loc[recent], own)
                row['last_recent_outcome'] = str(completion[recent].max().date())
                for name, m in [('rl_fixed', model), ('tree_fixed', tree)]:
                    if permitted[name]: actions.loc[dates, name] = gate(predict(m, x.loc[dates]), dates, panel, gates, available)
        else:
            reason = 'insufficient_training' if use.sum() < 504 or len(vcal) < 200 else 'flat_no_validation_opportunities'
            row.update(rl_fixed_status=reason, tree_fixed_status=reason)
        # Recover only the NN refits that were actually enabled in the saved run.
        original = nn_fits.loc[fitdate]
        row['nn_sized_status'] = original['status']
        if original['status'] == 'enabled':
            assert recent.sum() >= 504
            own = daily[recent].copy(); own[~panel.loc[recent, 'eligible'].to_numpy(bool)] = 0
            model, detail = fit_nn(x.loc[recent], own[:, 1:])
            train_prediction = nn_values(model, x.loc[recent]); values = nn_values(model, x.loc[dates])
            allowed = gates[ATM, recent] & panel.loc[recent, 'eligible'].to_numpy(bool)[:, None] & available[ATM, recent, None]
            thresholds = confidence_thresholds(train_prediction, allowed)
            sized = gate(confidence_codes(values, thresholds), dates, panel, gates, available)
            # This must reconstruct the original NN's directions, not silently change the model.
            recovered = gate(full_size(confidence_codes(values, np.array([-np.inf] * 3))), dates, panel, gates, available)
            np.testing.assert_array_equal(recovered.to_numpy(), fixed_nn.loc[dates].to_numpy())
            np.testing.assert_array_equal(full_size(sized), fixed_nn.loc[dates].to_numpy())
            actions.loc[dates, 'nn_sized'] = sized
            row.update(nn_converged=detail['converged'], nn_iterations=int(detail['iterations']),
                       nn_last_recent_outcome=str(completion[recent].max().date()),
                       nn_confidence_thresholds=thresholds.tolist(), nn_fixed_directions_recovered=True)
        logs.append(row)
        write_json(checkpoint, dict(dates=[str(t) for t in idx], actions=actions.to_numpy().tolist(), fits=logs))
        if month.month in [1, 7]:
            print(f'  {d["label"]} {month}: RL {row["rl_fixed_status"]}, tree {row["tree_fixed_status"]}, NN {row["nn_sized_status"]}', flush=True)
    for name in MODELS:
        dest = folder / name; dest.mkdir(exist_ok=True)
        a = actions[name].rename('code')
        pd.testing.assert_series_equal(a, gate(a, a.index, panel, gates, available), check_names=False)
        a.to_csv(dest / 'actions.csv')
    fit_table = pd.DataFrame(logs) if logs else pd.DataFrame(columns=['fit', 'nn_sized_status', 'rl_fixed_status', 'tree_fixed_status'])
    fit_table.to_csv(folder / 'fits.csv', index=False)
    return actions

def source_folder(pair, model, period):
    if model == 'nn_fixed': return NN / pair / 'fee_065' / period
    if model == 'rl_sized': return FROZEN / pair / (period + '_003')
    return OUT / pair / model / period

def summary(defs, market, cfg):
    all_models = ['nn_fixed', 'nn_sized', 'tree_fixed', 'rl_fixed_frozen', 'rl_fixed', 'rl_sized']
    statuses = {(d['pair'], m, p): json.loads((source_folder(d['pair'], m, p) / 'status.json').read_text())
                for d in defs for m in all_models for p, _ in PERIODS}
    complete = [d for d in defs if all(statuses[d['pair'], m, p]['status'] == 'complete' for m in all_models for p, _ in PERIODS)]
    unresolved = [dict(pair=d['pair'], label=d['label'], model=m, period=p, error=statuses[d['pair'], m, p].get('error'))
                  for d in defs for m in all_models for p, _ in PERIODS if statuses[d['pair'], m, p]['status'] != 'complete']
    write_json(REPORT / 'coverage.json', dict(requested_pairs=len(defs), common_complete_pairs=len(complete), unresolved=unresolved))
    rows = []; annual = []; pair_rows = []
    for m in all_models:
        for p, start in PERIODS:
            cal = market.calendar[market.calendar >= start]; cash = cash_book(cal, market.rates, cfg.capital)
            book = cash[['nav', 'cash_benchmark']] * len(complete); entries = active = 0
            for d in complete:
                status = statuses[d['pair'], m, p]; folder = source_folder(d['pair'], m, p)
                nav = cash if status.get('cash_only') else pd.read_parquet(status.get('nav_source', folder / 'daily_nav.parquet'), columns=['nav', 'cash_benchmark'])
                assert nav.index.equals(cal)
                book += nav[['nav', 'cash_benchmark']] - cash[['nav', 'cash_benchmark']]
                entries += status['entries']; active += int(status['entries'] > 0)
                pair_rows.append(dict(model=m, period=p, pair=d['pair'], label=d['label'], entries=status['entries'],
                                      net_excess_pnl=float(nav.nav.iloc[-1] - nav.cash_benchmark.iloc[-1])))
            book['return_'] = book.nav.pct_change().fillna(0)
            book['excess_return'] = book.return_ - book.cash_benchmark.pct_change().fillna(0)
            book.to_csv(REPORT / f'{m}_{p}_portfolio.csv')
            rows.append(dict(model=m, period=p, pairs=len(complete), entries=entries, active_pairs=active,
                             net_excess_pnl=float(book.nav.iloc[-1]-book.cash_benchmark.iloc[-1]), **scores(book, cfg.capital * len(complete))))
            for year, g in book.groupby(book.index.year):
                annual.append(dict(model=m, period=p, year=year, return_percent=100*((1+g.return_).prod()-1),
                    annualized_excess_sharpe=float(np.sqrt(252)*g.excess_return.mean()/g.excess_return.std()) if g.excess_return.std()>1e-12 else None))
    comparison = pd.DataFrame(rows); comparison.to_csv(REPORT / 'comparison.csv', index=False)
    pd.DataFrame(annual).to_csv(REPORT / 'annual.csv', index=False); pd.DataFrame(pair_rows).to_csv(REPORT / 'pairs.csv', index=False)
    fits = [pd.read_csv(OUT / d['pair'] / 'fits.csv').assign(pair=d['pair']) for d in defs if (OUT / d['pair'] / 'fits.csv').exists()]
    pd.concat(fits, ignore_index=True).to_csv(REPORT / 'fits.csv', index=False)
    lines = ['# $0.65 policy ablation', '',
        'Training hurdle $0.65; actual commission $0.03. Same checkpoint features, costs, eligibility, risk limits and monthly timing. The starting roster is the previously matched 345 pairs, including cash sleeves. Further unresolved paths are explicitly reported and excluded from every model, not assigned zero P&L.', '',
        '| Model | Full Sharpe | Recent Sharpe | Full entries |', '|---|---:|---:|---:|']
    for m in all_models:
        a = comparison[comparison.model.eq(m)&comparison.period.eq('full')].iloc[0]
        b = comparison[comparison.model.eq(m)&comparison.period.eq('recent')].iloc[0]
        lines.append(f'| {m} | {a.annualized_excess_sharpe:.3f} | {b.annualized_excess_sharpe:.3f} | {int(a.entries)} |')
    lines += ['', 'Full: 2018–August 2025. Recent: 2022–August 2025 starting flat. Cash-excess daily Sharpes annualized with √252.', '',
        'NN-sized recovers the original enabled monthly refits, verifies identical fixed-size directions, and maps positive predicted net P&L into 25/50/75/100% size using training-only confidence quartiles among eligible, memory-permitted rows. Original NN validation gates are frozen to isolate sizing; this is not a new validation test of the sized policy. No NN fitting is repeated for disabled months. Architecture, optimizer and full-size payoff targets are unchanged.', '',
        'RL-fixed-frozen maps the saved RL signals to 100% size without changing signal direction or dates. The simulator reapplies risk limits, so realized entries can change. RL-fixed is separately retrained with only short full size, long full size or cash, using the original Sharpe search. Tree-fixed uses exactly the same training tree partitions, choosing the leaf direction with highest positive mean net payoff rather than direct Sharpe search. The tree and RL-fixed variants use their own prior-year validation and recent-five-year refits.', '',
        'This remains exploratory attribution on inspected history, with the inherited validation-then-refit procedure. NN convergence warnings are retained. Raw data, the $0.65 checkpoint and saved baseline results are unchanged.', '',
        '[Comparison](atm65_ablation/comparison.csv) · [Annual](atm65_ablation/annual.csv) · [Pairs](atm65_ablation/pairs.csv) · [Fits](atm65_ablation/fits.csv) · [Coverage](atm65_ablation/coverage.json)', '']
    REPORT.with_suffix('.md').write_text('\n'.join(lines))
    print(comparison[['model', 'period', 'pairs', 'annualized_excess_sharpe', 'entries']].to_string(index=False), flush=True)

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--limit', type=int); parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args(); OUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    inputs = json.loads((NN / 'inputs.json').read_text())
    roster = pd.read_csv('reports/atm_nn_cost_hurdles/matched_roster.csv')
    included = set(roster.loc[roster.included, 'pair'])
    defs = [d for d in inputs['definitions'] if d['pair'] in included]
    actual = replace(config(), option_commission=.03); training = replace(actual, option_commission=.65)
    assert asdict(actual) == inputs['execution_settings'], 'Execution settings differ from checkpoint'
    paths = [Path(__file__), Path('scripts/run_atm65_ablation_audit.py'), NN/'inputs.json',
             Path('reports/atm_nn_cost_hurdles/matched_roster.csv'), Path('config/atm30_fee065_checkpoint.json')]
    paths += list(Path('scripts/curve_trading').glob('*.py'))
    paths += [Path('scripts/run_contract_walkforward.py'), Path('scripts/option_flex/contract_engine.py'),
              Path('scripts/run_atm_cost_hurdles.py'), Path('scripts/replay_frozen_atm_commission.py')]
    for d in defs:
        for p in [NN/d['pair']/'fee_065/states.parquet', NN/d['pair']/'fee_065/gates.npz',
                  NN/d['pair']/'fee_065/actions.csv', NN/d['pair']/'fee_065/fits.csv',
                  CURVE/d['pair']/'atm_control/actions.csv']:
            if p.exists(): paths.append(p)
    signature = dict(definitions=defs, actual=asdict(actual), training=asdict(training), models=MODELS,
                     confidence_rule='training-only positive best net payoff quartiles, frozen NN monthly gates',
                     sources={str(p): digest(p) for p in paths})
    if (OUT/'inputs.json').exists(): assert json.loads((OUT/'inputs.json').read_text()) == json.loads(json.dumps(signature)), 'Inputs changed'
    else: write_json(OUT/'inputs.json', signature)
    cal = load_calendar(ROOT); cal = cal[(cal >= '2010-01-01') & (cal <= '2025-08-29')]
    market = Market(cal); started = time.monotonic(); newly = 0
    if not args.report_only:
        for n, d in enumerate(defs, 1):
            folder = OUT / d['pair']; folder.mkdir(exist_ok=True)
            if all((folder/m/p/'status.json').exists() for m in MODELS for p, _ in PERIODS): continue
            original = json.loads((CURVE/d['pair']/'atm_control/status.json').read_text())
            source = NN / d['pair'] / 'fee_065'
            panel = pd.read_parquet(source/'states.parquet') if (source/'states.parquet').exists() else None
            if original.get('structural_cash') or panel is None or panel.empty:
                for m in MODELS:
                    for p, _ in PERIODS:
                        write_json(folder/m/p/'status.json', dict(status='complete', pair=d['pair'], label=d['label'], entries=0,
                                   cash_only=True, training_hurdle=.65, execution_commission=.03))
                continue
            pair_started = time.monotonic(); print(f'{n}/{len(defs)} {d["label"]}: single-hurdle ablations', flush=True)
            q, s, selections, divs = load_pair(market, d['secids'])
            with np.load(source/'gates.npz') as z:
                gates = np.zeros((len(COMBOS), len(panel), 2), bool); gates[ATM] = z['gates']
                available = np.zeros((len(COMBOS), len(panel)), bool); available[ATM] = z['available']
                saved_ends = z['ends'].copy()
            units = make_units(panel, q, s, selections, divs, cal, training)
            labels, finish, vegas, statuses, rewards, _ = reward_and_memory(units, training)
            np.testing.assert_array_equal(finish, saved_ends)
            from scripts.curve_trading.policy import memory_gates
            np.testing.assert_array_equal(gates[ATM], memory_gates(panel.index, cal[finish], rewards, vegas[:, [0, -1]]>0))
            daily = np.zeros((len(panel), 9, labels.shape[-1]), dtype=np.float32); daily[:, 1:] = labels
            del units, labels; gc.collect()
            replay = Replay(q, s, selections, d['secids'], divs, training)
            actions = train_pair(d, panel, daily, finish, gates, available, cal, replay, market.rates, folder)
            replay.cfg = actual
            for m in MODELS:
                for p, start in PERIODS:
                    dest = folder/m/p; dest.mkdir(parents=True, exist_ok=True)
                    if (dest/'status.json').exists(): continue
                    a = actions.loc[actions.index >= start, m]
                    if not a.ne(0).any(): result = dict(status='complete', pair=d['pair'], label=d['label'], entries=0, cash_only=True)
                    else:
                        try:
                            nav, trades, decisions = replay.run(a, cal[cal >= start]); mark_book(nav, market.rates, actual.capital)
                            result = save_result(dest, nav, trades, decisions, actual, s, d)
                        except UnresolvedPosition as exc:
                            result = dict(status='incomplete', pair=d['pair'], label=d['label'], error=str(exc))
                    result.update(training_hurdle=.65, execution_commission=.03); write_json(dest/'status.json', result)
            newly += 1
            elapsed = time.monotonic() - pair_started
            write_json(OUT/'progress.json', dict(roster_position=n, pairs=len(defs), last_pair=d['label'],
                last_pair_seconds=elapsed, elapsed_seconds=time.monotonic()-started,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            print(f'{d["label"]}: completed in {elapsed:.0f}s; peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30:.2f} GiB', flush=True)
            del panel, q, s, selections, divs, replay, daily, actions; gc.collect()
            if args.limit and newly >= args.limit: return
    summary(defs, market, actual)
    write_json(OUT/'task_status.json', dict(status='complete', pairs=len(defs), training_hurdle=.65, execution_commission=.03))

if __name__ == '__main__': main()
