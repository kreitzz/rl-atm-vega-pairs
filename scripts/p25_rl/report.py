"""Analyze frozen inventory-Q ledgers after the serial experiment finishes.

Run ``python -m scripts.p25_rl.report``. This module never fits a model,
chooses a policy using later returns, or splices portfolio return histories.
"""
import os
for _key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'ARROW_NUM_THREADS']:
    os.environ[_key] = '1'

import gc
import json
from pathlib import Path
import sqlite3
import time

import joblib
import numpy as np
import pandas as pd

from scripts.p25_rl import run
from scripts.p25_rl.policy import ACTION_NAMES, FEATURES, SPECS
from scripts.analyze_p25_opportunity_filter import (
    paired_sharpe_interval, pair_diagnostics, period_metrics, sharpe,
)
from scripts.attribute_p25_opportunity_filter import QUOTE_FIELDS, memory_estimate
from scripts import shared_trade_ledger as attribution


NAMES = {
    'fixed100': 'Original threshold, full size',
    'q_capacity': 'Q: capital capacity',
    'q_risk': 'Q: risk utility',
    'q_conservative': 'Q: conservative',
    'q_portfolio': 'Q: portfolio diversification',
    'myopic_risk': 'Myopic risk utility',
    'q_no_inventory': 'Q: no inventory inputs',
    'fixed50': 'Fixed half size',
    'diversified': 'Unheld-pair rule',
    'congestion': 'Congestion rule',
    'tree_cem': 'Archived tree + CEM',
    'learned5': 'Archived learned 5% threshold',
}
NEW_CASES = list(SPECS) + ['fixed50', 'diversified', 'congestion']
CANDIDATES = ['q_capacity', 'q_risk', 'q_conservative', 'q_portfolio']
ORDER = ['fixed100'] + NEW_CASES + ['tree_cem', 'learned5']
PERIODS = [('earlier', '2018-01-01', '2021-12-31'),
           ('evaluation', '2022-01-01', '2025-08-29'),
           ('full', '2018-01-01', '2025-08-29')]
THRESHOLD = run.ROOT / 'artifacts/contracts/p25_threshold_search'


def replay_paths():
    paths = {case: run.OUT / 'replays' / case for case in NEW_CASES}
    paths['fixed100'] = run.OUT / 'replays/baseline_verification'
    paths['tree_cem'] = run.prior.OUT / 'replays/tree_cem'
    paths['learned5'] = THRESHOLD / 'replays/selected_fixed'
    return {case: paths[case] for case in ORDER}


def absolute_sharpe_interval(values, block, replications=2000, seed=2026,
                             rest=lambda: None):
    """Conditional circular-block percentile interval for one saved strategy."""
    if not values.index.is_unique or not values.index.is_monotonic_increasing:
        raise ValueError('Sharpe interval needs unique, ordered calendar dates')
    a = values.to_numpy(float)
    point = sharpe(a)
    if not 1 <= block <= len(a) or replications < 2:
        raise ValueError('Invalid block length or replication count')
    rng = np.random.default_rng(seed)
    estimates = np.empty(replications)
    for j in range(replications):
        starts = rng.integers(0, len(a), size=int(np.ceil(len(a) / block)))
        ix = ((starts[:, None] + np.arange(block)) % len(a)).ravel()[:len(a)]
        estimates[j] = sharpe(a[ix])
        if j % 100 == 99:
            rest()
    return dict(point=point, lower95=float(np.quantile(estimates, .025)),
                upper95=float(np.quantile(estimates, .975)),
                block_sessions=block, replications=replications, seed=seed)


def verify_selection(validation, selected):
    """Independently reproduce the frozen earlier-period selection rule."""
    reference = float(validation.loc[validation.case.eq('fixed100'),
                                     'annualized_excess_sharpe'].iloc[0])
    eligible = validation.loc[
        validation.case.isin(CANDIDATES)
        & validation.entries.ge(20) & validation.profit_2020.ge(0)
        & validation.profit_2021.ge(0)
        & validation.annualized_excess_sharpe.gt(reference + .01)]
    expected = ('fixed100' if eligible.empty else
                eligible.sort_values(['annualized_excess_sharpe', 'case'],
                                     ascending=[False, True]).case.iloc[0])
    if selected['selected'] != expected or selected['validation_only'] is not True:
        raise ValueError('Saved selection differs from the earlier-period rule')
    best_q = validation.loc[validation.case.isin(CANDIDATES)].sort_values(
        'annualized_excess_sharpe', ascending=False).case.iloc[0]
    if selected['best_q_by_validation'] != best_q:
        raise ValueError('Saved best Q differs from earlier validation ranking')
    return expected


def period_returns(nav, start, stop):
    values = nav.loc[start:stop, 'excess_return']
    if values.empty:
        raise ValueError('No returns in the requested period')
    return values.iloc[1:] if values.index[0] == nav.index[0] else values


def cost_metrics(nav, trades, start, stop):
    own = nav.loc[start:stop]
    fills = trades.loc[trades.date.between(start, stop)]
    return dict(option_commission=float(fills.commission.sum()),
                option_execution_penalty=float(fills.execution_penalty.sum()),
                stock_execution_cost=float(own.hedge_cost.sum()),
                stock_borrow_cost=float(own.stock_borrow_cost.sum()),
                interest=float(own.interest.sum()),
                peak_stress_pct=float(100 * (own.stress_loss / own.nav).max()))


def tracked_hash(path, hashes, budget):
    hashes[str(Path(path).relative_to(run.ROOT))] = run.base.digest(path)
    budget.rest()


def verify_decision_links(signals, decisions, actions=None):
    """Connect unique proposal identities, stored requests and policy actions."""
    scored = signals[['date', 'pair', 'learned_level', 'direction']].copy()
    scored['date'] = pd.to_datetime(scored.date)
    scored = scored.rename(columns={'date': 'signal_date'})
    decided = decisions.copy()
    decided['signal_date'] = pd.to_datetime(decided.signal_date)
    keys = ['signal_date', 'pair']
    if scored.duplicated(keys).any() or decided.duplicated(keys).any():
        raise ValueError('Proposal/decision identities must be unique')
    linked = decided.merge(scored, on=keys, how='outer', indicator=True,
                            validate='one_to_one')
    if not linked['_merge'].eq('both').all():
        raise ValueError('Decision identities differ from scored proposals')
    np.testing.assert_array_equal(linked.learned_level.to_numpy(), linked.level.to_numpy())
    np.testing.assert_allclose(linked.action, linked.direction * linked.level / 4,
                               atol=1e-12, rtol=0)
    if actions is not None:
        policy = actions[['date', 'action']].copy()
        policy['date'] = pd.to_datetime(policy.date)
        if policy.date.duplicated().any() or set(policy.date) != set(decided.signal_date):
            raise ValueError('Policy action dates differ from proposal dates')
        policy = policy.rename(columns={'date': 'signal_date', 'action': 'policy_action'})
        linked = linked.drop(columns='_merge').merge(policy, on='signal_date',
            how='left', validate='many_to_one')
        if not linked.policy_action.isin(ACTION_NAMES).all():
            raise ValueError('Unknown saved portfolio action')
        expected = linked.policy_action.map({0: 0, 1: 2, 2: 4, 3: 4}).to_numpy()
        unheld_skip = linked.policy_action.eq(3) & linked.pair_inventory_cohorts.gt(0)
        expected[unheld_skip.to_numpy()] = 0
        np.testing.assert_array_equal(expected, linked.level.to_numpy())
    return dict(unique_proposal_identities=True, scored_levels_match_decisions=True,
                requested_direction_matches=True, policy_actions_match_requests=actions is not None)


def verify_archived_threshold(hashes, budget):
    path = THRESHOLD / 'selections/selected_fixed.json'
    selected = json.loads(path.read_text())
    if (selected['policy'] != 'q05' or selected['status'] != 'selected'
            or pd.Timestamp(selected['asof']) != pd.Timestamp('2022-01-01')
            or pd.Timestamp(selected['latest_return']) != pd.Timestamp('2021-12-31')
            or pd.Timestamp(selected['latest_return']) >= pd.Timestamp(selected['asof'])):
        raise ValueError('Archived learned 5% threshold provenance differs')
    chosen = [c for c in selected['candidates'] if c['policy'] == 'q05']
    if len(chosen) != 1 or not chosen[0]['eligible'] or chosen[0]['q_low'] != .05:
        raise ValueError('Archived 5% selection is not an eligible earlier candidate')
    tracked_hash(path, hashes, budget)


def verify_new_case(case, folder, protocol, hashes, budget):
    status = json.loads((folder / 'status.json').read_text())
    if status['status'] != 'passed':
        raise ValueError(f'Replay did not pass: {case}')
    if status['protocol_sha256'] != run.base.digest(run.REPORT / 'protocol.json'):
        raise ValueError(f'Replay protocol changed: {case}')
    for name, digest in status['source_code'].items():
        if run.base.digest(run.ROOT / name) != digest:
            raise ValueError(f'Replay source changed: {case}: {name}')
    for name, digest in status['cache_signature']['sources'].items():
        if run.base.digest(run.ROOT / 'scripts/p25_rl' / name) != digest:
            raise ValueError(f'Cached replay source changed: {case}: {name}')
    tracked_hash(folder / 'status.json', hashes, budget)
    return status


def verify_models_and_actions(hashes, budget):
    audits, counts, importances, diagnostics = [], [], [], []
    for cutoff, phase in [('2020-01-01', 'validation'), ('2022-01-01', 'evaluation')]:
        for case in SPECS:
            model_path = run.OUT / 'models' / cutoff / f'{case}.joblib'
            model = joblib.load(model_path)
            audit = model.audit
            if (pd.Timestamp(audit['last_transition_end']) >= pd.Timestamp(cutoff)
                    or audit['spec'] != SPECS[case]):
                raise ValueError(f'Noncausal or changed model: {case}@{cutoff}')
            if audit['cutoff'] != cutoff:
                raise ValueError('Model fit cutoff mismatch')
            signature = audit['cache_signature']
            if (signature['case'] != case or signature['cutoff'] != cutoff
                    or signature['spec'] != SPECS[case]):
                raise ValueError('Model cache identity or training specification mismatch')
            if signature['protocol_sha256'] != run.base.digest(run.REPORT / 'protocol.json'):
                raise ValueError('Model protocol mismatch')
            if signature['policy_sha256'] != run.base.digest(run.ROOT / 'scripts/p25_rl/policy.py'):
                raise ValueError('Model policy source mismatch')
            if signature['provenance_sha256'] != run.base.digest(run.ROOT / 'scripts/p25_rl/provenance.py'):
                raise ValueError('Model provenance source mismatch')
            for name, digest in signature['sources'].items():
                if run.base.digest(run.ROOT / name) != digest:
                    raise ValueError(f'Model training data changed: {name}')
                hashes[name] = digest; budget.rest()
            folder = (run.OUT / 'validation' / case if phase == 'validation'
                      else run.OUT / 'replays' / case)
            actions = pd.read_parquet(folder / 'actions.parquet', use_threads=False)
            own = actions.loc[actions.date.ge(cutoff)]
            predicted = np.array([model.choose(np.asarray(state)) for state in own.state])
            np.testing.assert_array_equal(predicted, own.action.to_numpy())
            if len(own):
                values = model.values(np.stack(own.state.map(np.asarray)))
                spreads = np.ptp(values, axis=1)
                diagnostics.append(dict(case=case, phase=phase, decision_dates=len(own),
                    median_action_value_range=float(np.median(spreads)),
                    maximum_action_value_range=float(spreads.max()),
                    different_action_values_fraction=float((spreads > 1e-8).mean())))
            if phase == 'evaluation':
                names = [FEATURES[column] for column in model.columns]
                names += [f'action_{ACTION_NAMES[action]}' for action in range(4)]
                importance = model.model.feature_importances_
                if len(importance) != len(names):
                    raise ValueError('Feature importance columns do not match the model')
                importances.extend(dict(case=case, feature=name, importance=float(value),
                                        role='action' if name.startswith('action_') else 'state')
                                   for name, value in zip(names, importance))
            audits.append(dict(case=case, phase=phase, cutoff=cutoff,
                               **{k: v for k, v in audit.items()
                                  if k not in ['cutoff', 'spec', 'bellman_target_change', 'cache_signature']},
                               final_target_change=audit['bellman_target_change'][-1]))
            for action, number in own.action.value_counts().sort_index().items():
                counts.append(dict(case=case, phase=phase, action=int(action),
                                   action_name=ACTION_NAMES[int(action)],
                                   decision_dates=int(number),
                                   fraction=float(number / len(own))))
            tracked_hash(model_path, hashes, budget)
            tracked_hash(model_path.with_suffix('.json'), hashes, budget)
            tracked_hash(folder / 'actions.parquet', hashes, budget)
            del model, actions
            gc.collect()
            budget.rest()
    return audits, counts, importances, diagnostics


def verify_validation(validation, signal, calendar, protocol, hashes, budget):
    """Reconcile the earlier ledgers and reproduce every selection metric."""
    rows = []
    subset = signal.loc[signal.date.le('2021-10-31')].copy()
    history = calendar[calendar <= '2021-12-31']
    for case in ['fixed100', 'fixed50', 'diversified', 'congestion'] + list(SPECS):
        folder = run.OUT / 'validation' / case
        status = verify_new_case(case, folder, protocol, hashes, budget)
        model = (joblib.load(run.OUT / 'models/2020-01-01' / f'{case}.joblib')
                 if case in SPECS else None)
        expected = run.replay_signature(subset, history, status['kind'], model, '2020-01-01', 2026)
        if status['cache_signature'] != expected:
            raise ValueError(f'Validation replay provenance changed: {case}')
        nav = pd.read_parquet(folder / 'daily_nav.parquet', use_threads=False)
        trades = pd.read_csv(folder / 'trades.csv', parse_dates=['date'])
        decisions = pd.read_csv(folder / 'decisions.csv', parse_dates=['signal_date', 'entry_date'])
        signals = pd.read_parquet(folder / 'scored_allocations.parquet', use_threads=False)
        actions = pd.read_parquet(folder / 'actions.parquet', use_threads=False)
        links = verify_decision_links(signals, decisions, actions)
        audit, _ = run.reconcile(nav, trades, decisions, signals)
        metric = period_metrics(nav, decisions, '2020-01-01', '2021-12-31')
        metric.update({f'profit_{year}': period_metrics(nav, decisions, f'{year}-01-01',
                                                     f'{year}-12-31')['profit_above_cash']
                       for year in [2020, 2021]})
        saved = validation.loc[validation.case.eq(case)]
        if len(saved) != 1:
            raise ValueError(f'Missing or duplicated validation scores: {case}')
        for key, value in metric.items():
            actual = saved.iloc[0][key]
            if isinstance(value, str):
                if actual != value:
                    raise ValueError('Validation date mismatch')
            else:
                np.testing.assert_allclose(float(actual), value, atol=1e-7, rtol=1e-10)
        rows.append(dict(case=case, status='passed', **audit, **links))
        for filename in run.prior.FILES:
            tracked_hash(folder / filename, hashes, budget)
        tracked_hash(folder / 'actions.parquet', hashes, budget)
        del nav, trades, decisions, signals, actions, model
        gc.collect(); budget.rest()
    pd.DataFrame(rows).to_csv(run.REPORT / 'validation_ledger_verification.csv', index=False)


def read_quotes(trades, budget):
    records = []
    connection = sqlite3.connect(f'file:{run.prior.QUOTES.as_posix()}?mode=ro', uri=True)
    connection.execute('PRAGMA cache_size=-2048')
    try:
        for oid, group in trades.groupby('optionid', sort=False):
            result = connection.execute(
                'SELECT payload FROM quotes WHERE date>=? AND date<=? AND optionid=?',
                (str(group.date.min().date()), str(group.date.max().date()), int(oid)))
            for (payload,) in result:
                raw = json.loads(payload)
                records.append({key: raw.get(key) for key in QUOTE_FIELDS})
            budget.rest()
    finally:
        connection.close()
    quotes = pd.DataFrame(records, columns=QUOTE_FIELDS)
    for name in ['date', 'expiry']:
        quotes[name] = pd.to_datetime(quotes[name])
    for name in set(QUOTE_FIELDS) - {'date', 'expiry', 'cp_flag', 'bs_status'}:
        quotes[name] = pd.to_numeric(quotes[name], errors='coerce')
    return quotes


def pair_attribution(case, nav, trades, hashes, budget, destination=None):
    """Actual hedge accounting only; incomplete attribution is explicit."""
    saved = {
        'fixed100': THRESHOLD / 'pair_attribution/q20',
        'tree_cem': run.ROOT / 'artifacts/contracts/p25_strategy_audit/pairs',
        'learned5': THRESHOLD / 'pair_attribution/selected_fixed',
    }
    destination = (run.OUT / 'pair_attribution' / case if destination is None
                   else Path(destination))
    destination.mkdir(parents=True, exist_ok=True)
    if case in saved:
        source = saved[case]
        daily = pd.read_parquet(source / 'pair_daily.parquet', use_threads=False)
        financing = pd.read_parquet(source / 'shared_financing.parquet', use_threads=False)
        rows, concentration = pair_diagnostics(case, daily, financing, nav)
        for filename in ['pair_daily.parquet', 'shared_financing.parquet']:
            tracked_hash(source / filename, hashes, budget)
        return dict(case=case, status='complete', reused_saved_accounting=True,
                    pairs=len(daily.columns), reconciliation='daily pair profit plus financing matches NAV'), rows, concentration
    estimate = memory_estimate(nav, trades, budget)
    if estimate['estimated_peak_rss_mib'] > 360:
        return dict(case=case, status='unavailable_memory_preflight', **estimate), [], []
    quotes = read_quotes(trades, budget)
    stocks = pd.read_parquet(run.base.MARKET / 'stocks.parquet',
        columns=['date', 'secid', 'close', 'cfadj'],
        filters=[('secid', 'in', trades.secid.unique().tolist())], use_threads=False)
    dividends = pd.read_parquet(run.base.MARKET / 'dividends.parquet',
        columns=['secid', 'ex_date', 'amount'], use_threads=False)
    estimate = memory_estimate(nav, trades, budget, quotes)
    if estimate['estimated_peak_rss_mib'] > 360:
        return dict(case=case, status='unavailable_memory_preflight', **estimate), [], []
    original_allocate = attribution.allocate_cost

    def bounded_allocate(total, weights):
        budget.rest()
        return original_allocate(total, weights)

    try:
        attribution.allocate_cost = bounded_allocate
        legs, daily, financing, vegas, grossvegas, error, components = attribution.reconstruct_shared(
            nav, trades, quotes, stocks, dividends, json.loads(run.prior.CONFIG.read_text()))
    finally:
        attribution.allocate_cost = original_allocate
    del quotes, stocks, dividends, vegas, grossvegas, components, legs
    gc.collect()
    pair_index = pd.Series(daily.columns.astype(str), index=daily.columns).str.split('|').str[1]
    pair_daily = daily.T.groupby(pair_index).sum().T
    del daily
    gc.collect()
    pair_daily.to_parquet(destination / 'pair_daily.parquet')
    financing.to_frame('shared_financing_vs_cash').to_parquet(destination / 'shared_financing.parquet')
    rows, concentration = pair_diagnostics(case, pair_daily, financing, nav)
    status = dict(case=case, status='complete', **estimate,
                  pairs=len(pair_daily.columns), max_daily_reconciliation_error=error)
    run.base.write(destination / 'reconciliation.json', status)
    budget.rest()
    return status, rows, concentration


def describe_intervals(table, case, reference):
    own = table.loc[table.case.eq(case) & table.reference.eq(reference)]
    if len(own) != 2 or set(own.block_sessions) != {63, 126}:
        raise ValueError('Missing interval sensitivity check')
    result = '; '.join(f'{int(r.block_sessions)} sessions: {r.point:+.3f} '
                       f'[{r.lower95:+.3f}, {r.upper95:+.3f}]'
                       for r in own.sort_values('block_sessions').itertuples())
    conclusion = ('both above zero' if own.lower95.gt(0).all() else
                  'both below zero' if own.upper95.lt(0).all() else
                  'improvement not established across both block lengths')
    return result + '; ' + conclusion + '.'


def draw_plot(periods, intervals, books, selected, budget):
    os.environ['MPLCONFIGDIR'] = str(run.OUT / 'matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    later = periods.loc[periods.period.eq('evaluation')].set_index('case')
    cases = ['fixed100', 'q_capacity', 'q_risk', 'q_conservative', 'q_portfolio',
             'myopic_risk', 'q_no_inventory', 'diversified', 'tree_cem', 'learned5']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    ax = axes[0]
    colors = ['#247391' if c in SPECS else '#7c898e' for c in cases]
    bars = ax.barh(np.arange(len(cases)), later.loc[cases, 'annualized_excess_sharpe'], color=colors)
    ax.set_yticks(np.arange(len(cases)), [NAMES[c] for c in cases]); ax.invert_yaxis()
    ax.set_xlabel('2022–August 2025 excess Sharpe'); ax.axvline(0, color='#aab2b4', lw=.8)
    for bar, case in zip(bars, cases):
        ax.annotate(f'{later.loc[case, "annualized_excess_sharpe"]:.2f}',
                    (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0), textcoords='offset points', va='center', fontsize=8)
    ax = axes[1]
    for case in dict.fromkeys(['fixed100', selected, 'q_risk', 'tree_cem', 'learned5']):
        nav = books[case]
        part = nav.loc['2022-01-01':]
        anchor = nav.loc[:'2021-12-31'].iloc[-1]
        values = (part.nav - part.cash_benchmark) - (anchor.nav - anchor.cash_benchmark)
        ax.plot(values.index, values / 1000, label=NAMES[case], lw=1.5)
    ax.set_ylabel('Profit above cash ($ thousands)'); ax.tick_params(axis='x', rotation=35)
    ax.legend(fontsize=8); ax.set_title('Continuous funded books')
    ax = axes[2]
    comparisons = [(c, 'fixed100') for c in ['q_capacity', 'q_risk', 'q_conservative', 'q_portfolio']]
    comparisons += [('q_risk', 'myopic_risk'), ('q_risk', 'q_no_inventory')]
    for j, (case, reference) in enumerate(comparisons):
        own = intervals.loc[intervals.case.eq(case) & intervals.reference.eq(reference)]
        for offset, block, color in [(-.12, 63, '#247391'), (.12, 126, '#b07134')]:
            row = own.loc[own.block_sessions.eq(block)].iloc[0]
            ax.plot([row.lower95, row.upper95], [j + offset] * 2, color=color)
            ax.plot(row.point, j + offset, 'o', color=color,
                    label=f'{block}-session blocks' if j == 0 else None)
    labels = [NAMES[c].replace('Q: ', '') + ' minus ' + NAMES[r].replace('Original threshold, full size', 'baseline')
              for c, r in comparisons]
    ax.set_yticks(np.arange(len(comparisons)), labels); ax.invert_yaxis()
    ax.axvline(0, color='#aab2b4', lw=.8); ax.set_xlabel('Paired Sharpe difference, conditional 95% interval')
    ax.legend(fontsize=8)
    fig.suptitle('Inventory-aware Q learning: frozen earlier selection, exploratory later results', fontsize=13)
    fig.savefig(run.REPORT / 'comparison.png', dpi=160)
    plt.close(fig); budget.rest()


def main():
    started = time.monotonic()
    budget = run.Budget()
    protocol_path = run.REPORT / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    selection = json.loads((run.REPORT / 'selection.json').read_text())
    if selection['protocol_sha256'] != run.base.digest(protocol_path):
        raise ValueError('Selection and experiment protocols differ')
    for name in ['baseline_verification.json', 'handoff_verification.json']:
        if json.loads((run.REPORT / name).read_text())['status'] != 'passed':
            raise ValueError(f'Prerequisite verification failed: {name}')
    hashes = {}
    for path in [protocol_path, run.REPORT / 'selection.json', run.REPORT / 'validation.csv',
                 Path(__file__), Path(attribution.__file__)]:
        tracked_hash(path, hashes, budget)
    for name, digest in protocol['protected_hashes'].items():
        if run.base.digest(run.ROOT / name) != digest:
            raise ValueError(f'Protected source changed: {name}')
        hashes[name] = digest; budget.rest()
    validation = pd.read_csv(run.REPORT / 'validation.csv')
    selected = verify_selection(validation, selection)
    verify_archived_threshold(hashes, budget)
    input_signal, input_calendar, _, _ = run.inputs()
    verify_validation(validation, input_signal, input_calendar, protocol, hashes, budget)
    audits, action_counts, importances, diagnostics = verify_models_and_actions(hashes, budget)
    pd.DataFrame(audits).to_csv(run.REPORT / 'training_counts.csv', index=False)
    pd.DataFrame(action_counts).to_csv(run.REPORT / 'action_counts.csv', index=False)
    pd.DataFrame(importances).to_csv(run.REPORT / 'feature_importances.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(run.REPORT / 'action_value_diagnostics.csv', index=False)

    books, periods, annual, ledger_checks, pair_rows, concentration, pair_statuses = {}, [], [], [], [], [], []
    paths = replay_paths()
    for case, folder in paths.items():
        print(f'Checking and attributing {case}', flush=True)
        if case in NEW_CASES or case == 'fixed100':
            status = verify_new_case(case, folder, protocol, hashes, budget)
            model = (joblib.load(run.OUT / 'models/2022-01-01' / f'{case}.joblib')
                     if case in SPECS else None)
            active_from = '2018-01-01' if case == 'fixed100' else '2022-01-01'
            expected = run.replay_signature(input_signal, input_calendar, status['kind'],
                                            model, active_from, 2026)
            if status['cache_signature'] != expected:
                raise ValueError(f'Replay data/model provenance changed: {case}')
            del model
        nav = pd.read_parquet(folder / 'daily_nav.parquet', use_threads=False)
        trades = pd.read_csv(folder / 'trades.csv', parse_dates=['date'])
        decisions = pd.read_csv(folder / 'decisions.csv', parse_dates=['signal_date', 'entry_date'])
        signals = pd.read_parquet(folder / 'scored_allocations.parquet', use_threads=False)
        actions = (pd.read_parquet(folder / 'actions.parquet', use_threads=False)
                   if case in NEW_CASES or case == 'fixed100' else None)
        links = verify_decision_links(signals, decisions, actions)
        if not nav.index.is_unique or not nav.index.is_monotonic_increasing:
            raise ValueError(f'Invalid NAV calendar: {case}')
        if books and not nav.index.equals(books['fixed100'].index):
            raise ValueError(f'Nonmatching funded calendar: {case}')
        audit, trades = run.reconcile(nav, trades, decisions, signals)
        ledger_checks.append(dict(case=case, status='passed', **audit, **links))
        if case in NEW_CASES:
            pd.testing.assert_frame_equal(nav.loc[:'2021-12-31'], books['fixed100'].loc[:'2021-12-31'],
                                           check_exact=False, atol=1e-7, rtol=1e-11)
        books[case] = nav
        for period, start, stop in PERIODS:
            periods.append(dict(case=case, period=period,
                **period_metrics(nav, decisions, start, stop), **cost_metrics(nav, trades, start, stop)))
        for year in range(2018, 2026):
            start, stop = f'{year}-01-01', f'{year}-12-31'
            annual.append(dict(case=case, year=year,
                **period_metrics(nav, decisions, start, stop), **cost_metrics(nav, trades, start, stop)))
        try:
            status, own_pairs, own_concentration = pair_attribution(case, nav, trades, hashes, budget)
        except (AssertionError, ValueError, KeyError, FileNotFoundError) as error:
            status = dict(case=case, status='unavailable_reconstruction',
                          reason=f'{type(error).__name__}: {error}')
            own_pairs, own_concentration = [], []
        pair_statuses.append(status); pair_rows.extend(own_pairs); concentration.extend(own_concentration)
        for filename in run.prior.FILES:
            tracked_hash(folder / filename, hashes, budget)
        if actions is not None:
            tracked_hash(folder / 'actions.parquet', hashes, budget)
        del trades, decisions, signals, actions
        gc.collect(); budget.rest()
    periods = pd.DataFrame(periods); annual = pd.DataFrame(annual)
    periods.to_csv(run.REPORT / 'periods.csv', index=False)
    annual.to_csv(run.REPORT / 'annual.csv', index=False)
    pd.DataFrame(ledger_checks).to_csv(run.REPORT / 'ledger_verification.csv', index=False)
    pd.DataFrame(pair_rows).to_csv(run.REPORT / 'pair_periods.csv', index=False)
    pd.DataFrame(concentration).to_csv(run.REPORT / 'concentration.csv', index=False)
    run.base.write(run.REPORT / 'pair_attribution_status.json', dict(cases=pair_statuses,
        accounting='Actual option and stock hedge profit, costs and dividends; shared financing separate'))

    intervals = []
    comparisons = [(case, 'fixed100') for case in ORDER if case != 'fixed100']
    comparisons += [('q_risk', 'myopic_risk'), ('q_risk', 'q_no_inventory')]
    for case, reference in comparisons:
        for block in [63, 126]:
            intervals.append(dict(case=case, reference=reference, period='evaluation',
                **paired_sharpe_interval(period_returns(books[case], '2022-01-01', '2025-08-29'),
                    period_returns(books[reference], '2022-01-01', '2025-08-29'),
                    block=block, rest=budget.rest)))
    intervals = pd.DataFrame(intervals)
    later = periods.loc[periods.period.eq('evaluation')].set_index('case')
    descriptive_winner = later.loc[CANDIDATES].annualized_excess_sharpe.idxmax()
    for row in intervals.itertuples():
        expected = later.loc[row.case, 'annualized_excess_sharpe'] - later.loc[row.reference, 'annualized_excess_sharpe']
        np.testing.assert_allclose(row.point, expected, atol=1e-12, rtol=0)
    intervals.to_csv(run.REPORT / 'sharpe_intervals.csv', index=False)
    absolute = []
    for period, start, stop in PERIODS:
        for block in [63, 126]:
            absolute.append(dict(case='fixed100', period=period,
                **absolute_sharpe_interval(period_returns(books['fixed100'], start, stop),
                                           block, rest=budget.rest)))
    absolute = pd.DataFrame(absolute)
    absolute.to_csv(run.REPORT / 'baseline_sharpe_intervals.csv', index=False)

    lines = ['INVENTORY-AWARE Q LEARNING: BOUNDED EXPERIMENT', '',
        f'Policy selected using 2020-2021 only: {NAMES[selected]}.',
        f'Highest Q validation Sharpe: {NAMES[selection["best_q_by_validation"]]}.',
        f'Highest later Q point estimate: {NAMES[descriptive_winner]} (descriptive; not used for selection).',
        'All four candidate Q families and both ablations are reported; later performance does not change selection.',
        'Every new Q/control book follows fixed100 through 2021 and carries its actual inventory and NAV into 2022; archived tree/CEM retains its own history.', '',
        'LATER FUNDED RESULTS: JANUARY 2022-AUGUST 2025',
        'Policy                                  Sharpe    Above cash ($)  Entries  Pairs  Reserve (%)  Drawdown (%)']
    for case in ORDER:
        r = later.loc[case]
        lines.append(f'{NAMES[case]:39s} {r.annualized_excess_sharpe:7.3f} {r.profit_above_cash:17,.0f}'
                     f' {int(r.entries):8d} {int(r.active_entry_pairs):6d}'
                     f' {r.average_reserve_pct:12.3f} {r.max_cash_relative_drawdown_pct:13.3f}')
    candidate = selected if selected != 'fixed100' else selection['best_q_by_validation']
    lines += ['', 'FINDING',
        ('No Q policy met the earlier selection criteria; the retained policy is the original threshold.'
         if selected == 'fixed100' else f'The earlier selection criteria retain {NAMES[selected]}.'),
        f'{NAMES[candidate]} versus the original threshold: '
        + describe_intervals(intervals, candidate, 'fixed100'),
        'Sequential versus immediate-reward learning: ' + describe_intervals(intervals, 'q_risk', 'myopic_risk'),
        'Inventory versus signal-only Q: ' + describe_intervals(intervals, 'q_risk', 'q_no_inventory'), '',
        'INTERPRETATION',
        'This tests allocation of the existing 608 IV/RV proposals, including skip, half/full size and an unheld-pair action.',
        'Q learning learns delayed rewards through its own funded inventory and future admission capacity.',
        'All Q arms use ordinary maximum-value Bellman backups; conservative arms penalize model disagreement only when extracting actions.',
        'It does not generate a new broad stock universe, rehedge option exposure, or learn exit/roll decisions.',
        'The risk penalty optimizes a reward/second-moment utility, not Sharpe directly. Model dispersion is not a calibrated confidence bound.', '',
        'BASELINE SHARPE AND STATISTICAL LIMITS']
    for row in absolute.itertuples():
        lines.append(f'{row.period}, {row.block_sessions}-session blocks: baseline Sharpe {row.point:.3f}'
                     f' [{row.lower95:+.3f}, {row.upper95:+.3f}].')
    lines += ['These baseline intervals ask whether the saved strategy Sharpe is above zero; the paired intervals ask whether one strategy improves another.',
        'A paired interval containing zero does not establish that the baseline is uniquely significant or optimal.',
        'All intervals use 2,000 paired/circular calendar-block resamples and condition on fitted policies.',
        'All historical periods were previously inspected. Intervals do not correct for accumulated research, model selection, or multiple comparisons.',
        'These are exploratory historical results, not confirmation on an untouched holdout.', '',
        'EARLIER VALIDATION: 2020-2021 (COMMON OCTOBER 2021 PROPOSAL CUTOFF, DECEMBER RUNOFF)',
        validation[['case', 'annualized_excess_sharpe', 'entries', 'profit_2020', 'profit_2021']].to_string(index=False), '',
        'LATER ANNUAL EXCESS SHARPE (2025 IS PARTIAL)',
        annual.loc[annual.year.ge(2022)].pivot(index='case', columns='year', values='annualized_excess_sharpe').reindex(ORDER).round(3).to_string(), '',
        'LATER ACTUAL COSTS AND RISK',
        later[['option_commission', 'option_execution_penalty', 'stock_execution_cost',
               'stock_borrow_cost', 'peak_stress_pct', 'peak_reserve_pct', 'post_entry_risk_breach_days']].round(3).to_string(), '',
        'PAIR CONCENTRATION']
    for case in ORDER:
        matches = [r for r in concentration if r['case'] == case and r['period'] == 'evaluation']
        if matches:
            row = matches[0]
            lines.append(f'{NAMES[case]}: {row["positive_pairs"]} positive/{row["negative_pairs"]} negative pairs; '
                         f'top3 share of positive pair profit {100*row["top3_share_positive_profit"]:.1f}%; '
                         f'pair profit ${row["net_pair_pnl"]:,.0f}; shared financing ${row["shared_financing_vs_cash"]:,.0f}.')
        else:
            status = next(s for s in pair_statuses if s['case'] == case)
            lines.append(f'{NAMES[case]}: attribution unavailable ({status["status"]}); no proxy pair P&L substituted.')
    lines += ['Pairs overlap in names and dates; profitable-pair counts are not independent observations.', '',
        'LEARNED ACTIONS AND TRAINING DATA',
        pd.DataFrame(action_counts).query("phase == 'evaluation'")
            [['case', 'action_name', 'decision_dates', 'fraction']].round(3).to_string(index=False),
        pd.DataFrame(audits)[['case', 'phase', 'transitions', 'distinct_market_dates',
                              'last_transition_end', 'iterations']].to_string(index=False),
        'Behavior trajectories reuse the same market dates; transition counts are not independent sample counts.', '',
        'Final-model impurity importances are saved in feature_importances.csv; they describe fitted model use, not causal effects.', '',
        'VERIFICATION',
        f'All {len(paths)} funded ledgers independently reconciled; all saved Q actions reproduce from their frozen models.',
        'Protected sources and replay/model hashes were checked; original baseline accounting and pre-2022 Q histories match.',
        'No policy fitted, retuned, or selected using later returns in this report. Shared-financing attribution reconciles wherever available.',
        f'One numeric worker/thread, 20% CPU duty, 384 MiB RSS limit. Peak report RSS: {budget.peak:.1f} MiB.', '',
        'Reproduce: python -m scripts.p25_rl.run --phase all; python -m scripts.p25_rl.report.',
        'Detailed tables: periods.csv, annual.csv, validation.csv, selection.json, sharpe_intervals.csv, baseline_sharpe_intervals.csv,',
        'training_counts.csv, action_counts.csv, feature_importances.csv, concentration.csv, pair_periods.csv and pair_attribution_status.json.', '']
    (run.REPORT / 'Findings.txt').write_text('\n'.join(lines), encoding='utf-8')
    draw_plot(periods, intervals, books, selected, budget)
    for name, digest in hashes.items():
        if run.base.digest(run.ROOT / name) != digest:
            raise ValueError(f'Input changed during reporting: {name}')
        budget.rest()
    outputs = ['Findings.txt', 'comparison.png', 'periods.csv', 'annual.csv', 'ledger_verification.csv',
               'validation_ledger_verification.csv',
               'pair_periods.csv', 'concentration.csv', 'pair_attribution_status.json', 'sharpe_intervals.csv',
               'baseline_sharpe_intervals.csv', 'training_counts.csv', 'action_counts.csv', 'feature_importances.csv',
               'action_value_diagnostics.csv']
    run.base.write(run.REPORT / 'report_manifest.json', dict(status='passed',
        source_hashes=hashes, outputs={name: run.base.digest(run.REPORT / name) for name in outputs},
        elapsed_seconds=time.monotonic()-started, peak_rss_mib=budget.peak))
    run.base.write(run.REPORT / 'completion.json', dict(status='passed', selected=selected,
        ledgers_reconciled=len(paths), validation_selection_reproduced=True,
        saved_model_actions_reproduced=True, no_return_splicing=True,
        proposal_identities_and_requested_levels_verified=True,
        archived_threshold_selection_verified=True,
        interval_points_verified=True, protected_sources_unchanged=True,
        pair_attribution_available=[s['case'] for s in pair_statuses if s['status'] == 'complete'],
        peak_rss_mib=budget.peak))
    print('\n'.join(lines[:28]), flush=True)


if __name__ == '__main__':
    main()
