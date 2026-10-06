"""Three trade-quality RL state ablations on the frozen 0.368 reference."""
import argparse
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import resource
import time
import numpy as np
import pandas as pd
from scripts.run_atm65_ablation import features, read_actions, write_json, NN, CURVE, PERIODS
from scripts.run_atm65_volume_oi import configure, employee_roster, baseline_actions, baseline_results, EMPLOYEE, EMPLOYEE_REPORT, EMPLOYEE_ADDED, CODES, CHECKPOINT, WORKFLOW
from scripts.atm65_trade_quality_features import ADDED, GROUPS, collect, add_signals, LEG_SOURCE
from scripts.curve_trading.config import config, COMBOS
from scripts.curve_trading.policy import ATM, Replay, fit, predict, gate, memory_gates
from scripts.run_atm_cost_hurdles import make_units, reward_and_memory
from scripts.replay_frozen_atm_commission import load_pair, cash_book, save_result
from scripts.run_broad_contract_replay import Market, ROOT
from scripts.run_curve_trading import RUNOFF, screen_dates
from scripts.run_contract_walkforward import digest
from scripts.run_selected_pair_backtest import scores, audit
from scripts.run_aligned_contract_learning import mark_book
from scripts.select_atm_contracts import load_calendar
from scripts.option_flex.contract_engine import UnresolvedPosition
from scripts.retrain_atm_commission import check_fits

OUT = Path('artifacts/contracts/atm65_rl_trade_quality')
REPORT = Path('reports/atm65_rl_trade_quality')
MODELS = list(GROUPS)


def prepare(defs, market, employees):
    cal = market.calendar
    legs = collect(defs, EMPLOYEE, market.stocks, cal)
    rows = []
    for d in defs:
        source = EMPLOYEE / d['pair'] / 'states.parquet'
        folder = OUT / d['pair']; folder.mkdir(exist_ok=True)
        row = dict(pair=d['pair'], label=d['label'], employee_features=d['pair'] in employees, state_rows=0, retrain=False)
        row.update({m+'_retrain': False for m in MODELS})
        if source.exists():
            panel = pd.read_parquet(source)
            a, b = d['secids']; state = add_signals(panel, legs[a], legs[b])
            state.to_parquet(folder / 'states.parquet')
            test = (panel.index >= '2018-01-01') & (panel.index <= cal[-1]-pd.Timedelta(days=RUNOFF)) & screen_dates(d, panel.index)
            with np.load(NN / d['pair'] / 'fee_065/gates.npz') as z:
                permitted = panel.eligible.to_numpy(bool) & z['available'] & z['gates'].any(axis=1)
            row['state_rows'] = len(state)
            for m, columns in GROUPS.items():
                observed = state[[c for c in columns if c.endswith('_available')]].gt(0).any(axis=1).to_numpy()
                count = int((observed & test & permitted).sum())
                row[m+'_observed_permitted_rows'] = count
                row[m+'_retrain'] = count > 0
            row['retrain'] = any(row[m+'_retrain'] for m in MODELS)
        rows.append(row)
    coverage = pd.DataFrame(rows)
    coverage.to_csv(REPORT / 'coverage.csv', index=False)
    write_json(OUT / 'prepared.json', dict(pairs=len(defs), retrained_pairs=int(coverage.retrain.sum()),
        model_features=GROUPS, timing='Signal-session close; next-session execution. No future labels or outcome-filtered membership.',
        name_memory='Prior completed 365-calendar-day standalone ATM30 outcomes at actual $0.03 costs; per-name nonoverlapping intervals selected before reading labels; minimum four resolved blocks; separate signed means and dispersion/sqrt(count), plus count/availability.',
        iv_correlation='126-session IV-change correlation relative to its prior 252-session distribution, minimum 63/126 observations.',
        gamma='Log ratio of observed signal-contract BS dollar gamma per quoted vega; same dividend adjustment, no new fit.',
        baseline='Frozen employee-sentiment reference; employee inputs active on original 28 sleeves only.'))
    print(coverage.groupby('retrain').size().to_string(), flush=True)
    print({m: int(coverage[m+'_retrain'].sum()) for m in MODELS}, flush=True)


def split_counts(model, columns, row, prefix):
    chosen = [columns[int(f)] for f in model[0].tree_.feature if f >= 0]
    added = [c for c in columns if c in ADDED]
    row[prefix+'_new_value_splits'] = sum(chosen.count(c) for c in added if not c.endswith('_available'))
    row[prefix+'_new_missing_splits'] = sum(chosen.count(c) for c in added if c.endswith('_available'))
    for c in added: row[prefix+'_splits_'+c] = chosen.count(c)


def train_pair(d, panel, daily, finish, gates, available, cal, replay, rates, folder, employee, coverage):
    stop = cal[-1]-pd.Timedelta(days=RUNOFF)
    idx = panel.index[(panel.index >= '2018-01-01') & (panel.index <= stop) & screen_dates(d, panel.index)]
    original = features(panel)+(EMPLOYEE_ADDED if employee else [])
    assert len(original) == (49 if employee else 47)
    saved = read_actions(baseline_actions(d['pair'], 'rl', employee)); assert saved.index.equals(idx)
    path = EMPLOYEE/d['pair']/'fits.csv' if employee else CURVE/d['pair']/'atm_control/fits.csv'
    table = pd.read_csv(path).set_index('fit')
    baseline_fits = table['rl_sentiment_status' if employee else 'status']
    actions = pd.DataFrame(0, index=idx, columns=MODELS, dtype=int); logs = []
    checkpoint = folder/'month_checkpoint.json'
    if checkpoint.exists():
        old = json.loads(checkpoint.read_text()); assert old['dates'] == [str(t) for t in idx]
        actions.iloc[:, :] = np.asarray(old['actions'], dtype=int); logs = old['fits']
    completed = {r['fit'] for r in logs}; positions = cal.get_indexer(panel.index); completion = cal[finish]
    finite = np.isfinite(daily).all(axis=(1, 2))
    for month in pd.period_range('2018-01-01', stop, freq='M'):
        cutoff = month.start_time; fitdate = str(cutoff.date())
        test = (idx >= cutoff) & (idx < (month+1).start_time)
        if not test.any() or fitdate in completed: continue
        vstart = cutoff-pd.DateOffset(years=1)
        use = (panel.index >= vstart-pd.DateOffset(years=5)) & (completion < vstart) & finite
        recent = (panel.index >= cutoff-pd.DateOffset(years=5)) & (completion < cutoff) & finite
        vcal = cal[(cal >= vstart) & (cal < cutoff)]
        val = (panel.index >= vcal[0]) & (panel.index <= vcal[-1]-pd.Timedelta(days=RUNOFF))
        opportunity = (panel.eligible.to_numpy(bool) & available[ATM] & gates[ATM].any(axis=1) & val).any()
        dates = idx[test]
        row = dict(fit=fitdate, training_rows=int(use.sum()), recent_rows=int(recent.sum()),
            validation_start=str(vcal[0].date()), validation_end=str(vcal[-1].date()))
        # Recover the frozen refit once per month; all candidates share labels and seeds.
        if baseline_fits.loc[fitdate] == 'enabled':
            assert recent.sum() >= 504
            own = daily[recent].copy(); own[~panel.loc[recent, 'eligible'].to_numpy(bool)] = 0
            model, _ = fit(panel.loc[recent, original], own, positions[recent], finish[recent], CODES)
            recovered = gate(predict(model, panel.loc[dates, original]), dates, panel, gates, available)
        else: recovered = pd.Series(0, index=dates, dtype=int)
        np.testing.assert_array_equal(recovered.to_numpy(), saved.loc[dates].to_numpy(),
            err_msg=f'{d["label"]} {fitdate}: frozen baseline was not reproduced')
        row['baseline_reproduced'] = True
        labels = daily[use].copy(); labels[~panel.loc[use, 'eligible'].to_numpy(bool)] = 0
        for m in MODELS:
            if not coverage[m+'_retrain']:
                actions.loc[dates, m] = saved.loc[dates]
                row[m+'_status'] = 'baseline_passthrough'; continue
            columns = original+GROUPS[m]
            row[m+'_status'] = 'insufficient_training' if use.sum() < 504 or len(vcal) < 200 else 'flat_no_validation_opportunities'
            if use.sum() < 504 or len(vcal) < 200 or not opportunity: continue
            model, details = fit(panel.loc[use, columns], labels, positions[use], finish[use], CODES)
            row['last_training_outcome'] = str(completion[use].max().date())
            row.update({m+'_proposal_'+k: v for k, v in details.items()})
            split_counts(model, columns, row, m+'_proposal')
            proposal = gate(predict(model, panel.loc[val, columns]), panel.index[val], panel, gates, available)
            score = None; error = None
            try:
                nav, trades, _ = replay.run(proposal, vcal); mark_book(nav, rates, replay.cfg.capital)
                audit(nav, trades, replay.s, replay.cfg, d['secids'])
                score = scores(nav, replay.cfg.capital)['annualized_excess_sharpe']
            except (UnresolvedPosition, ValueError) as exc: error = str(exc)
            enabled = score is not None and score > 0 and recent.sum() >= 504
            row[m+'_status'] = 'enabled' if enabled else 'flat_validation_gate'
            row[m+'_validation_sharpe'] = score; row[m+'_validation_error'] = error
            if enabled:
                own = daily[recent].copy(); own[~panel.loc[recent, 'eligible'].to_numpy(bool)] = 0
                model, details = fit(panel.loc[recent, columns], own, positions[recent], finish[recent], CODES)
                actions.loc[dates, m] = gate(predict(model, panel.loc[dates, columns]), dates, panel, gates, available)
                row['last_recent_outcome'] = str(completion[recent].max().date())
                row.update({m+'_refit_'+k: v for k, v in details.items()})
                split_counts(model, columns, row, m+'_refit')
        logs.append(row)
        write_json(checkpoint, dict(dates=[str(t) for t in idx], actions=actions.to_numpy().tolist(), fits=logs))
        if month.month in [1, 7]: print(f'  {d["label"]} {month}: '+', '.join(m+' '+row[m+'_status'] for m in MODELS), flush=True)
    table = pd.DataFrame(logs); check_fits(table); table.to_csv(folder/'fits.csv', index=False)
    for m in MODELS:
        dest = folder/m; dest.mkdir(exist_ok=True); a = actions[m].rename('code')
        pd.testing.assert_series_equal(a, gate(a, a.index, panel, gates, available), check_names=False)
        a.to_csv(dest/'actions.csv')
    return actions


def summary(defs, coverage, market, cfg, employees):
    selected = set(coverage.loc[coverage.retrain, 'pair']); models = ['rl_baseline']+MODELS
    def source(d, m, p):
        return OUT/d['pair']/m/p if m in MODELS and d['pair'] in selected else baseline_results(d['pair'], 'rl', p, d['pair'] in employees)
    statuses = {(d['pair'], m, p): json.loads((source(d, m, p)/'status.json').read_text()) for d in defs for m in models for p, _ in PERIODS}
    unresolved = [dict(pair=d['pair'], label=d['label'], model=m, period=p, error=statuses[d['pair'], m, p].get('error'))
        for d in defs for m in models for p, _ in PERIODS if statuses[d['pair'], m, p]['status'] != 'complete']
    write_json(REPORT/'accounting_coverage.json', dict(requested_pairs=len(defs), unresolved=unresolved,
        complete_by_model_period={m: {p: sum(statuses[d['pair'], m, p]['status']=='complete' for d in defs) for p, _ in PERIODS} for m in models}))
    rows = []; annual = []; pairs = []
    for universe in ['all', 'common_complete']:
        for p, start in PERIODS:
            roster = defs if universe == 'all' else [d for d in defs if all(statuses[d['pair'], m, p]['status']=='complete' for m in models)]
            for m in models:
                if any(statuses[d['pair'], m, p]['status'] != 'complete' for d in roster):
                    rows.append(dict(universe=universe, model=m, period=p, pairs=len(roster), status='incomplete')); continue
                cal = market.calendar[market.calendar >= start]; cash = cash_book(cal, market.rates, cfg.capital)
                book = cash[['nav', 'cash_benchmark']]*len(roster); entries = active = 0
                for d in roster:
                    status = statuses[d['pair'], m, p]; folder = source(d, m, p)
                    nav = cash if status.get('cash_only') else pd.read_parquet(status.get('nav_source', folder/'daily_nav.parquet'), columns=['nav', 'cash_benchmark'])
                    assert nav.index.equals(cal); book += nav[['nav', 'cash_benchmark']]-cash[['nav', 'cash_benchmark']]
                    entries += status['entries']; active += int(status['entries'] > 0)
                    pairs.append(dict(universe=universe, model=m, period=p, pair=d['pair'], label=d['label'], entries=status['entries'],
                        net_excess_pnl=float(nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1])))
                book['return_'] = book.nav.pct_change().fillna(0)
                book['excess_return'] = book.return_-book.cash_benchmark.pct_change().fillna(0)
                book.to_csv(REPORT/f'{universe}_{m}_{p}_portfolio.csv')
                rows.append(dict(universe=universe, model=m, period=p, pairs=len(roster), status='complete', entries=entries, active_pairs=active,
                    net_excess_pnl=float(book.nav.iloc[-1]-book.cash_benchmark.iloc[-1]), **scores(book, cfg.capital*len(roster))))
                for year, g in book.groupby(book.index.year):
                    sd = g.excess_return.std()
                    annual.append(dict(universe=universe, model=m, period=p, year=year, return_percent=100*((1+g.return_).prod()-1),
                        annualized_excess_sharpe=float(np.sqrt(252)*g.excess_return.mean()/sd) if sd > 1e-12 else None))
    comparison = pd.DataFrame(rows); comparison.to_csv(REPORT/'comparison.csv', index=False)
    pd.DataFrame(annual).to_csv(REPORT/'annual.csv', index=False); pd.DataFrame(pairs).to_csv(REPORT/'pairs.csv', index=False)
    saved = pd.read_csv(EMPLOYEE_REPORT/'comparison.csv')
    for p, _ in PERIODS:
        got = comparison[(comparison.universe=='all') & (comparison.model=='rl_baseline') & (comparison.period==p)].iloc[0]
        expected = saved[(saved.universe=='all') & (saved.model=='rl_sentiment') & (saved.period==p)].iloc[0]
        np.testing.assert_allclose(got.annualized_excess_sharpe, expected.annualized_excess_sharpe, atol=1e-12)
        assert got.entries == expected.entries
    tables = [pd.read_csv(OUT/d['pair']/'fits.csv').assign(pair=d['pair']) for d in defs if d['pair'] in selected]
    if tables: pd.concat(tables, ignore_index=True).to_csv(REPORT/'fits.csv', index=False)
    notes = ['Three fixed trade-quality RL state ablations against the frozen 0.368 reference',
        'Name-level role-specific blocked profitability memory, IV-change-correlation history regime, and dollar-gamma/vega geometry, tested separately.',
        'Name memory uses hypothetical standalone outcomes at actual $0.03 costs and includes standalone financing; this is a predictor, not attribution of paired trades.',
        'Memory is sampled using known scheduled intervals, including missing-outcome intervals; outcomes enter only strictly after completion. No cross-pair duplicate weighting.',
        'At least four resolved nonoverlapping blocks within 365 days for means/uncertainty. Dispersion/sqrt(count) is an uncertainty proxy, not a formal independent-observation standard error.',
        'Signal-close inputs; next-session execution. Features recomputed from all observable original states and stock bars, independent of future payoff resolution.',
        'All original states unchanged. Employee leadership states remain on original 28 sleeves.',
        'Same monthly prior-year validation, five-year completed-outcome refits, 8-leaf partition, fixed seed 2026, Sharpe policy optimizer and 25/50/75/100% ladder.',
        'Same beta/correlation, profitability and inventory risk gates. ATM30 calls held to expiry with daily delta hedging.',
        'Training commission $0.65, actual commission $0.03. Mid + 0.1-vol-point entry penalty; stock costs, funding and risk unchanged.',
        'All 345 sleeves retained. Unavailable/unfitted sleeves retain original actions/cash; any unresolved accounting is explicit, never silently dropped.',
        'Full history starts flat in 2018; recent independently starts flat in 2022; both end August 2025.',
        'Sharpe is annualized daily excess-return Sharpe, sqrt(252), above the cash benchmark.',
        'RL only; no NN refits. Enabled baseline fits recovered and compared with saved historical actions.',
        'Signals were chosen after examining this history; this comparison is exploratory, not untouched confirmation. No joint or threshold search.',
        'Reference checkpoint preserved, no automatic promotion.', '', comparison.to_string(index=False)]
    (REPORT/'summary.txt').write_text('\n'.join(notes)+'\n')
    print(comparison[['universe', 'model', 'period', 'pairs', 'status', 'entries', 'annualized_excess_sharpe']].to_string(index=False), flush=True)
    return unresolved


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--limit', type=int); parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args(); configure()
    OUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    inputs = json.loads((EMPLOYEE/'inputs.json').read_text()); defs = inputs['definitions']; employees = employee_roster()
    actual = replace(config(), option_commission=.03); training = replace(actual, option_commission=.65)
    assert asdict(actual) == inputs['actual']
    mined = json.loads(Path('artifacts/contracts/atm_variance_mining/inputs.json').read_text())
    mined_audit = json.loads(Path('reports/atm_variance_mining/audit.json').read_text())
    assert mined['settings'] == asdict(actual)
    assert digest(LEG_SOURCE) == mined_audit['source_and_output_sha256'][str(LEG_SOURCE)]
    cal = load_calendar(ROOT); cal = cal[(cal >= '2010-01-01') & (cal <= '2025-08-29')]
    files = [Path(__file__), WORKFLOW, CHECKPOINT, Path('config/current_contract_reference.json'), Path('scripts/atm65_trade_quality_features.py'), Path('scripts/run_atm65_volume_oi.py'),
        LEG_SOURCE, Path('artifacts/contracts/atm_variance_mining/inputs.json'), Path('reports/atm_variance_mining/audit.json'),
        Path('data/curve_trading/C_history_manifest.json'), Path('data/curve_trading/C_select_manifest.json'),
        Path('scripts/single_leg_policy.py'), Path('scripts/mine_atm_variance_signals.py'), Path('scripts/option_flex/black_scholes.py'),
        EMPLOYEE/'inputs.json', EMPLOYEE_REPORT/'coverage.csv', EMPLOYEE_REPORT/'comparison.csv',
        Path('scripts/curve_trading/policy.py'), Path('scripts/curve_trading/unit_paths.py'), Path('scripts/run_atm_cost_hurdles.py'),
        Path('scripts/retrain_atm_commission.py'), Path('scripts/replay_frozen_atm_commission.py'), Path('scripts/option_flex/contract_engine.py'),
        ROOT/'zero_curve.parquet', ROOT/'distributions.parquet',
        *[ROOT/'underlying'/f'year={year}.parquet' for year in cal.year.unique()]]
    for d in defs:
        files.extend(p for p in [EMPLOYEE/d['pair']/'states.parquet', NN/d['pair']/'fee_065/gates.npz',
            baseline_actions(d['pair'], 'rl', d['pair'] in employees)] if p.exists())
    signature = dict(models=MODELS, workflow=json.loads(WORKFLOW.read_text()), definitions=defs, baseline_checkpoint=str(CHECKPOINT),
        actual=asdict(actual), training=asdict(training), model_features=GROUPS, code_and_sources={str(p): digest(p) for p in files})
    if (OUT/'inputs.json').exists(): assert json.loads((OUT/'inputs.json').read_text()) == signature, 'Inputs changed'
    else: write_json(OUT/'inputs.json', signature)
    market = Market(cal)
    if not (OUT/'prepared.json').exists(): prepare(defs, market, employees)
    if args.prepare: return
    coverage = pd.read_csv(REPORT/'coverage.csv'); selected = set(coverage.loc[coverage.retrain, 'pair'])
    started = time.monotonic(); newly = 0
    if not args.report_only:
        for n, d in enumerate([x for x in defs if x['pair'] in selected], 1):
            folder = OUT/d['pair']
            if all((folder/m/p/'status.json').exists() for m in MODELS for p, _ in PERIODS): continue
            pair_started = time.monotonic(); print(f'{n}/{len(selected)} {d["label"]}: three RL state ablations', flush=True)
            panel = pd.read_parquet(folder/'states.parquet'); q, s, selections, divs = load_pair(market, d['secids'])
            with np.load(NN/d['pair']/'fee_065/gates.npz') as z:
                gates = np.zeros((len(COMBOS), len(panel), 2), bool); gates[ATM] = z['gates']
                available = np.zeros((len(COMBOS), len(panel)), bool); available[ATM] = z['available']; saved_ends = z['ends'].copy()
            units = make_units(panel, q, s, selections, divs, cal, training)
            labels, finish, vegas, statuses, rewards, _ = reward_and_memory(units, training)
            np.testing.assert_array_equal(finish, saved_ends)
            np.testing.assert_array_equal(gates[ATM], memory_gates(panel.index, cal[finish], rewards, vegas[:, [0, -1]] > 0))
            daily = np.zeros((len(panel), 9, labels.shape[-1]), dtype=np.float32); daily[:, 1:] = labels
            del units, labels; gc.collect(); replay = Replay(q, s, selections, d['secids'], divs, training)
            c = coverage.set_index('pair').loc[d['pair']]
            actions = train_pair(d, panel, daily, finish, gates, available, cal, replay, market.rates, folder, d['pair'] in employees, c)
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
                        except UnresolvedPosition as exc: result = dict(status='incomplete', pair=d['pair'], label=d['label'], error=str(exc))
                    result.update(training_hurdle=.65, execution_commission=.03); write_json(dest/'status.json', result)
            newly += 1
            write_json(OUT/'progress.json', dict(completed_retrained_pairs=n, retrained_pairs=len(selected), last_pair=d['label'],
                last_pair_seconds=time.monotonic()-pair_started, elapsed_seconds=time.monotonic()-started,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            print(f'{d["label"]}: complete in {time.monotonic()-pair_started:.1f}s; peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20:.0f} MiB', flush=True)
            del panel, q, s, selections, divs, replay, daily, actions; gc.collect()
            if args.limit and newly >= args.limit: return
    unresolved = summary(defs, coverage, market, actual, employees)
    write_json(OUT/'task_status.json', dict(status='complete' if not unresolved else 'complete_with_explicit_data_gap', models=MODELS,
        pairs=len(defs), retrained_pairs=len(selected), unresolved=unresolved, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__ == '__main__': main()
