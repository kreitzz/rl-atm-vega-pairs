"""Frozen joint selection and concise analysis of bounded inventory-Q iterations.

Run --phase select after all four earlier validations, before later Q
evaluation. Run --phase report after the serial experiment worker stops.
"""
import os
for _key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'ARROW_NUM_THREADS']:
    os.environ[_key] = '1'

import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import joblib
import numpy as np
import pandas as pd

from scripts.p25_rl import run
from scripts.p25_rl import report as capacity_report
from scripts.analyze_p25_opportunity_filter import paired_sharpe_interval, period_metrics

ROOT = run.ROOT
REPORT = ROOT / 'reports/p25_rl_research'
MANAGEMENT_OUT = ROOT / 'artifacts/contracts/p25_inventory_q_management'
MANAGEMENT_REPORT = ROOT / 'reports/p25_inventory_q_management'
ACTION_OUT = ROOT / 'artifacts/contracts/p25_action_q'
ACTION_REPORT = ROOT / 'reports/p25_action_q'
TEMPORAL_OUT = ROOT / 'artifacts/contracts/p25_temporal_q'
TEMPORAL_REPORT = ROOT / 'reports/p25_temporal_q'
MANAGEMENT_CANDIDATES = ['manage_q', 'manage_q_risk']
MANAGEMENT_CASES = MANAGEMENT_CANDIDATES + ['manage_myopic_risk']
ACTION_CANDIDATES = ['action_q_capacity', 'action_q_risk', 'action_q_conservative']
ACTION_CASES = ACTION_CANDIDATES + ['action_myopic_risk']
TEMPORAL_CASES = ['entry_myopic_capacity', 'manage_myopic_capacity']
CANDIDATES = capacity_report.CANDIDATES + ACTION_CANDIDATES + MANAGEMENT_CANDIDATES
NAMES = dict(capacity_report.NAMES, manage_q='Q: hold/trim/exit',
             manage_q_risk='Q: risk-aware hold/trim/exit',
             manage_myopic_risk='Myopic hold/trim/exit',
             action_q_capacity='Action critics: capacity Q',
             action_q_risk='Action critics: risk Q',
             action_q_conservative='Action critics: conservative Q',
             action_myopic_risk='Action critics: myopic risk',
             entry_myopic_capacity='Myopic entry, same reward as capacity Q',
             manage_myopic_capacity='Myopic exits, same reward as hold/exit Q')


class Budget(run.Budget):
    def check(self):
        super().check()
        if any((path / 'STOP').exists() for path in [MANAGEMENT_OUT, ACTION_OUT, TEMPORAL_OUT, REPORT]):
            raise InterruptedError('Combined Q report STOP requested')


def earlier_tables():
    allocation = pd.read_csv(run.REPORT / 'validation.csv')
    management = pd.read_csv(MANAGEMENT_REPORT / 'validation.csv')
    action = pd.read_csv(ACTION_REPORT / 'validation.csv')
    temporal = pd.read_csv(TEMPORAL_REPORT / 'validation.csv')
    columns = ['annualized_excess_sharpe', 'profit_above_cash', 'entries',
               'active_entry_pairs', 'average_reserve_pct', 'peak_reserve_pct',
               'max_cash_relative_drawdown_pct', 'post_entry_risk_breach_days',
               'profit_2020', 'profit_2021']
    a = allocation.loc[allocation.case.eq('fixed100')]
    if len(a) != 1:
        raise ValueError('Missing common validation baseline')
    for other in [management, action, temporal]:
        b = other.loc[other.case.eq('fixed100')]
        if len(b) != 1:
            raise ValueError('Missing family validation baseline')
        np.testing.assert_allclose(a[columns].to_numpy(float), b[columns].to_numpy(float),
                                   atol=1e-7, rtol=1e-10)
        for column in ['start', 'end']:
            if a.iloc[0][column] != b.iloc[0][column]:
                raise ValueError('Validation baseline periods differ')
    management = management.loc[~management.case.eq('fixed100')]
    action = action.loc[~action.case.eq('fixed100')]
    temporal = temporal.loc[~temporal.case.eq('fixed100')]
    combined = pd.concat([allocation, action, management, temporal], ignore_index=True)
    if combined.case.duplicated().any() or not set(CANDIDATES).issubset(combined.case):
        raise ValueError('Missing or duplicated joint candidate identity')
    return combined


def choose_earlier(table):
    baseline = float(table.loc[table.case.eq('fixed100'), 'annualized_excess_sharpe'].iloc[0])
    own = table.loc[table.case.isin(CANDIDATES)].copy()
    own['eligible'] = (own.entries.ge(20) & own.profit_2020.ge(0) & own.profit_2021.ge(0)
                       & own.annualized_excess_sharpe.gt(baseline + .01))
    ranked = own.sort_values(['annualized_excess_sharpe', 'case'], ascending=[False, True])
    eligible = ranked.loc[ranked.eligible]
    selected = 'fixed100' if eligible.empty else eligible.case.iloc[0]
    return selected, ranked.case.iloc[0], own


def select(budget):
    """The only joint choice: nine declared candidates, earlier returns only."""
    REPORT.mkdir(parents=True, exist_ok=True)
    table = earlier_tables()
    selected, best, candidates = choose_earlier(table)
    source_paths = [run.REPORT / 'protocol.json', run.REPORT / 'validation.csv',
                    ACTION_REPORT / 'protocol.json', ACTION_REPORT / 'validation.csv',
                    TEMPORAL_REPORT / 'protocol.json', TEMPORAL_REPORT / 'validation.csv',
                    MANAGEMENT_REPORT / 'protocol.json', MANAGEMENT_REPORT / 'validation.csv']
    source_hashes = {}
    for path in source_paths:
        source_hashes[str(path.relative_to(ROOT))] = run.base.digest(path)
        budget.rest()
    result = dict(selected=selected, best_q_by_validation=best, candidates=CANDIDATES,
        criterion='At least20validation entries, nonnegative profit in both2020/2021, and Sharpe > common fixed100+.01; choose highest earlier Sharpe, case-name tie break. Otherwise retain fixed100.',
        validation_start='2020-01-01', validation_end='2021-12-31',
        proposal_cutoff='2021-10-31', validation_only=True,
        later_history_unseen=False, source_hashes=source_hashes)
    path = REPORT / 'selection.json'
    if path.exists() and json.loads(path.read_text()) != result:
        raise ValueError('Frozen joint validation selection changed')
    run.base.write(path, result)
    table.to_csv(REPORT / 'validation.csv', index=False)
    candidates.to_csv(REPORT / 'validation_candidates.csv', index=False)
    print(f'Joint earlier selection: {selected}; best earlier Q: {best}', flush=True)
    return result


def check_selection(selection, budget):
    for name, digest in selection['source_hashes'].items():
        if run.base.digest(ROOT / name) != digest:
            raise ValueError(f'Joint validation input changed: {name}')
        budget.rest()
    selected, best, _ = choose_earlier(earlier_tables())
    if (selected != selection['selected'] or best != selection['best_q_by_validation']
            or selection['validation_only'] is not True):
        raise ValueError('Joint earlier selection cannot be reproduced')


def verify_management(budget):
    """Independently verify managed books; no fitting or portfolio simulation."""
    from scripts.p25_rl import management
    protocol_path = MANAGEMENT_REPORT / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    hashes = {str(protocol_path.relative_to(ROOT)): run.base.digest(protocol_path)}
    for name, digest in protocol['protected_hashes'].items():
        if run.base.digest(ROOT / name) != digest:
            raise ValueError(f'Management protected source changed: {name}')
        hashes[name] = digest; budget.rest()
    signal, calendar, _, _ = management.entry.inputs()
    checks, model_rows, action_rows, periods, annual, books = [], [], [], [], [], {}
    pair_statuses, pair_rows, concentrations, importances, management_counts, diagnostics = [], [], [], [], [], []
    for phase, cutoff in [('validation', '2020-01-01'), ('evaluation', '2022-01-01')]:
        cases = ['fixed100'] + MANAGEMENT_CASES
        for case in cases:
            folder = (MANAGEMENT_OUT / 'validation' / case if phase == 'validation'
                      else MANAGEMENT_OUT / 'replays' / case)
            status = json.loads((folder / 'status.json').read_text())
            if status['status'] != 'passed':
                raise ValueError(f'Management replay failed: {case}/{phase}')
            if status['protocol_sha256'] != run.base.digest(protocol_path):
                raise ValueError('Management replay protocol mismatch')
            for name, digest in status['source_code'].items():
                if run.base.digest(ROOT / name) != digest:
                    raise ValueError(f'Management replay source changed: {name}')
                hashes[name] = digest; budget.rest()
            for name, digest in status['output_hashes'].items():
                if run.base.digest(folder / name) != digest:
                    raise ValueError(f'Management replay output changed: {case}/{name}')
            model_path = MANAGEMENT_OUT / 'models' / cutoff / f'{case}.joblib'
            model = joblib.load(model_path) if case != 'fixed100' else None
            used_signal = signal.loc[signal.date.le('2021-10-31')].copy() if phase == 'validation' else signal
            used_calendar = calendar[calendar <= '2021-12-31'] if phase == 'validation' else calendar
            expected_signature = management.replay_signature(used_signal, used_calendar,
                status['kind'], model, cutoff if case != 'fixed100' else status['cache_signature']['active_from'],
                status['cache_signature']['seed'])
            if expected_signature != status['cache_signature']:
                raise ValueError(f'Management replay input/model signature mismatch: {case}/{phase}')
            nav = pd.read_parquet(folder / 'daily_nav.parquet', use_threads=False)
            trades = pd.read_csv(folder / 'trades.csv', parse_dates=['date'])
            decisions = pd.read_csv(folder / 'decisions.csv', parse_dates=['signal_date', 'entry_date'])
            scored = pd.read_parquet(folder / 'scored_allocations.parquet', use_threads=False)
            links = capacity_report.verify_decision_links(scored, decisions)
            if not decisions.level.eq(4).all():
                raise ValueError('Management experiment changed fixed-full entries')
            audit, trades = run.reconcile(nav, trades, decisions, scored)
            charged = trades.reason.ne('expiry_settlement')
            np.testing.assert_allclose(trades.loc[charged, 'commission'],
                .03 * trades.loc[charged, 'quantity'].abs(), atol=1e-10, rtol=0)
            checks.append(dict(case=case, phase=phase, status='passed', **audit, **links))
            actions = pd.read_parquet(folder / 'actions.parquet', use_threads=False)
            log = pd.read_csv(folder / 'management_log.csv', parse_dates=['signal_date', 'date'])
            if actions.date.duplicated().any() or not pd.DatetimeIndex(actions.date).equals(nav.index):
                raise ValueError('Management actions differ from the daily calendar')
            if not actions.action.isin(management.ACTION_NAMES).all():
                raise ValueError('Illegal saved management action')
            expected_eligibility = actions.date.ge(cutoff) & actions.held_cohorts.gt(0)
            if case != 'fixed100':
                np.testing.assert_array_equal(actions.eligible_decision, expected_eligibility)
                if actions.loc[~expected_eligibility, 'action'].ne(0).any():
                    raise ValueError('Management policy acted before fit or while flat')
            if not log.empty:
                planned = log.merge(actions[['date', 'action']].rename(columns={'date': 'signal_date'}),
                    on='signal_date', how='left', validate='many_to_one')
                np.testing.assert_allclose(planned.target, planned.action.map({0: 1., 1: .5, 2: 0.}),
                                           atol=1e-12, rtol=0)
                if not planned.date.gt(planned.signal_date).all():
                    raise ValueError('Management instruction filled without causal next-session lag')
            management_counts.append(dict(case=case, phase=phase,
                filled_instructions=int(log.status.eq('filled').sum()),
                unavailable_exit_quote=int(log.status.eq('no_executable_exit_quote').sum()),
                integer_trim_unavailable=int(log.status.eq('integer_trim_unavailable').sum()),
                unbalanced_trim=int(log.status.eq('unbalanced_integer_trim').sum()),
                trim_fills=int(trades.reason.eq('learned_policy_trim').sum()),
                close_fills=int(trades.reason.eq('learned_policy_exit').sum())))
            if model is not None:
                model_audit = model.audit
                if (model_audit['cutoff'] != cutoff
                        or pd.Timestamp(model_audit['last_transition_end']) >= pd.Timestamp(cutoff)):
                    raise ValueError('Management Q training includes future transitions')
                if model_audit['features'] != management.FEATURES or model_audit['spec'] != management.SPECS[case]:
                    raise ValueError('Management Q model features or hyperparameters changed')
                cache = model_audit['cache_signature']
                if (cache['case'] != case or cache['cutoff'] != cutoff
                        or cache['spec'] != management.SPECS[case]):
                    raise ValueError('Management model cache identity mismatch')
                if cache['protocol_sha256'] != run.base.digest(protocol_path):
                    raise ValueError('Management model protocol mismatch')
                for name, digest in cache['sources'].items():
                    if run.base.digest(ROOT / name) != digest:
                        raise ValueError(f'Management model training data changed: {name}')
                    hashes[name] = digest; budget.rest()
                for name, digest in cache['training_files'].items():
                    if run.base.digest(ROOT / name) != digest:
                        raise ValueError(f'Management model training data changed: {name}')
                    hashes[name] = digest; budget.rest()
                # The management runner saves the exact causal eligibility flag.
                eligible = actions.date.ge(cutoff) & actions.eligible_decision
                own = actions.loc[eligible]
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
                    names = management.FEATURES + [f'action_{management.ACTION_NAMES[a]}' for a in range(3)]
                    if len(names) != len(model.model.feature_importances_):
                        raise ValueError('Management importance feature count mismatch')
                    importances.extend(dict(case=case, feature=name, importance=float(value))
                        for name, value in zip(names, model.model.feature_importances_))
                model_rows.append(dict(case=case, phase=phase, cutoff=cutoff,
                    transitions=model_audit['transitions'],
                    distinct_market_dates=model_audit['distinct_market_dates'],
                    last_transition_end=model_audit['last_transition_end'],
                    gamma=model_audit['spec']['gamma'], penalty=model_audit['spec']['penalty'],
                    daily_reward_scale_bps=model_audit['daily_reward_scale_bps'],
                    held_transitions=model_audit['held_transitions']))
                for action, count in own.action.value_counts().sort_index().items():
                    action_rows.append(dict(case=case, phase=phase, action=int(action),
                        action_name={0: 'hold', 1: 'half', 2: 'close'}[int(action)],
                        decision_dates=int(count)))
                for path in [model_path, model_path.with_suffix('.json')]:
                    capacity_report.tracked_hash(path, hashes, budget)
            if phase == 'validation':
                table = pd.read_csv(MANAGEMENT_REPORT / 'validation.csv')
                saved = table.loc[table.case.eq(case)].iloc[0]
                measured = period_metrics(nav, decisions, '2020-01-01', '2021-12-31')
                measured.update({f'profit_{year}': period_metrics(nav, decisions,
                    f'{year}-01-01', f'{year}-12-31')['profit_above_cash'] for year in [2020, 2021]})
                for name, value in measured.items():
                    if isinstance(value, str):
                        if saved[name] != value:
                            raise ValueError('Management validation period mismatch')
                    else:
                        np.testing.assert_allclose(float(saved[name]), value, atol=1e-7, rtol=1e-10)
            else:
                baseline = pd.read_parquet(run.OUT / 'replays/baseline_verification/daily_nav.parquet', use_threads=False)
                pd.testing.assert_frame_equal(nav.loc[:'2021-12-31'], baseline.loc[:'2021-12-31'],
                    check_exact=False, atol=1e-7, rtol=1e-11)
                if case == 'fixed100':
                    pd.testing.assert_frame_equal(nav, baseline, check_exact=False, atol=1e-7, rtol=1e-11)
                books[case] = nav
                if case != 'fixed100':
                    try:
                        pair_status, own_pairs, own_concentration = capacity_report.pair_attribution(
                            case, nav, trades, hashes, budget,
                            destination=MANAGEMENT_OUT / 'pair_attribution' / case)
                    except (AssertionError, ValueError, KeyError, FileNotFoundError) as error:
                        pair_status = dict(case=case, status='unavailable_reconstruction',
                                           reason=f'{type(error).__name__}: {error}')
                        own_pairs, own_concentration = [], []
                    pair_statuses.append(pair_status); pair_rows.extend(own_pairs)
                    concentrations.extend(own_concentration)
                for period, start, stop in capacity_report.PERIODS:
                    periods.append(dict(case=case, period=period,
                        **period_metrics(nav, decisions, start, stop),
                        **capacity_report.cost_metrics(nav, trades, start, stop)))
                for year in range(2018, 2026):
                    start, stop = f'{year}-01-01', f'{year}-12-31'
                    annual.append(dict(case=case, year=year,
                        **period_metrics(nav, decisions, start, stop),
                        **capacity_report.cost_metrics(nav, trades, start, stop)))
            for name in run.prior.FILES + ['status.json', 'actions.parquet', 'management_log.csv']:
                capacity_report.tracked_hash(folder / name, hashes, budget)
            del model, trades, decisions, scored, actions, log
            gc.collect(); budget.rest()
    pd.DataFrame(checks).to_csv(MANAGEMENT_REPORT / 'ledger_verification.csv', index=False)
    pd.DataFrame(model_rows).to_csv(MANAGEMENT_REPORT / 'training_counts.csv', index=False)
    pd.DataFrame(action_rows).to_csv(MANAGEMENT_REPORT / 'action_counts.csv', index=False)
    pd.DataFrame(management_counts).to_csv(MANAGEMENT_REPORT / 'management_counts.csv', index=False)
    pd.DataFrame(importances).to_csv(MANAGEMENT_REPORT / 'feature_importances.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(MANAGEMENT_REPORT / 'action_value_diagnostics.csv', index=False)
    pd.DataFrame(pair_rows).to_csv(MANAGEMENT_REPORT / 'pair_periods.csv', index=False)
    pd.DataFrame(concentrations).to_csv(MANAGEMENT_REPORT / 'concentration.csv', index=False)
    run.base.write(MANAGEMENT_REPORT / 'pair_attribution_status.json', dict(cases=pair_statuses))
    pd.DataFrame(periods).to_csv(MANAGEMENT_REPORT / 'periods.csv', index=False)
    pd.DataFrame(annual).to_csv(MANAGEMENT_REPORT / 'annual.csv', index=False)
    for name, digest in hashes.items():
        if run.base.digest(ROOT / name) != digest:
            raise ValueError(f'Management input changed during reporting: {name}')
        budget.rest()
    run.base.write(MANAGEMENT_REPORT / 'verification.json', dict(status='passed',
        ledgers_reconciled=len(checks), model_actions_reproduced=True,
        sources_unchanged=True, source_hashes=hashes))
    return books


def verify_action_family(budget):
    """Verify the separate-action critic iteration against its own provenance."""
    from scripts.p25_rl import action_q
    protocol_path = ACTION_REPORT / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    hashes = {str(protocol_path.relative_to(ROOT)): run.base.digest(protocol_path)}
    for name, digest in protocol['protected_hashes'].items():
        if run.base.digest(ROOT / name) != digest:
            raise ValueError(f'Action-Q protected source changed: {name}')
        hashes[name] = digest; budget.rest()
    signal, calendar, _, _ = run.inputs()
    checks, models, action_rows, periods, annual, books = [], [], [], [], [], {}
    pair_statuses, pair_rows, concentrations, importances, diagnostics = [], [], [], [], []
    for phase, cutoff in [('validation', '2020-01-01'), ('evaluation', '2022-01-01')]:
        for case in ACTION_CASES:
            folder = (ACTION_OUT / 'validation' / case if phase == 'validation'
                      else ACTION_OUT / 'replays' / case)
            status = json.loads((folder / 'status.json').read_text())
            if status['status'] != 'passed' or status['protocol_sha256'] != run.base.digest(protocol_path):
                raise ValueError('Action-Q replay/protocol mismatch')
            for name, digest in status['source_code'].items():
                if run.base.digest(ROOT / name) != digest:
                    raise ValueError(f'Action-Q source changed: {name}')
                hashes[name] = digest; budget.rest()
            for name, digest in status['output_hashes'].items():
                if run.base.digest(folder / name) != digest:
                    raise ValueError(f'Action-Q output changed: {case}/{name}')
            model_path = ACTION_OUT / 'models' / cutoff / f'{case}.joblib'
            model = joblib.load(model_path) if case != 'fixed100' else None
            used_signal = signal.loc[signal.date.le('2021-10-31')].copy() if phase == 'validation' else signal
            used_calendar = calendar[calendar <= '2021-12-31'] if phase == 'validation' else calendar
            expected = action_q.replay_signature(used_signal, used_calendar, status['kind'], model,
                cutoff if case != 'fixed100' else status['cache_signature']['active_from'],
                status['cache_signature']['seed'])
            if status['cache_signature'] != expected:
                raise ValueError(f'Action-Q replay signature mismatch: {case}/{phase}')
            nav = pd.read_parquet(folder / 'daily_nav.parquet', use_threads=False)
            trades = pd.read_csv(folder / 'trades.csv', parse_dates=['date'])
            decisions = pd.read_csv(folder / 'decisions.csv', parse_dates=['signal_date', 'entry_date'])
            scored = pd.read_parquet(folder / 'scored_allocations.parquet', use_threads=False)
            actions = pd.read_parquet(folder / 'actions.parquet', use_threads=False)
            links = capacity_report.verify_decision_links(scored, decisions, actions)
            audit, trades = run.reconcile(nav, trades, decisions, scored)
            checks.append(dict(case=case, phase=phase, status='passed', **audit, **links))
            if model is not None:
                audit = model.audit
                if audit['cutoff'] != cutoff or pd.Timestamp(audit['last_transition_end']) >= pd.Timestamp(cutoff):
                    raise ValueError('Action-Q model used future transitions')
                if audit['spec'] != action_q.SPECS[case]:
                    raise ValueError('Action-Q fitted specification changed')
                cache = audit['cache_signature']
                if (cache['case'] != case or cache['cutoff'] != cutoff
                        or cache['spec'] != action_q.SPECS[case]):
                    raise ValueError('Action-Q model cache identity mismatch')
                if cache['protocol_sha256'] != run.base.digest(protocol_path):
                    raise ValueError('Action-Q model protocol changed')
                for collection in ['sources', 'training_files']:
                    for name, digest in cache[collection].items():
                        if run.base.digest(ROOT / name) != digest:
                            raise ValueError(f'Action-Q training source changed: {name}')
                        hashes[name] = digest; budget.rest()
                own = actions.loc[actions.date.ge(cutoff)]
                predictions = np.array([model.choose(np.asarray(state)) for state in own.state])
                np.testing.assert_array_equal(predictions, own.action.to_numpy())
                if len(own):
                    values = model.values(np.stack(own.state.map(np.asarray)))
                    spreads = np.ptp(values, axis=1)
                    diagnostics.append(dict(case=case, phase=phase,
                        decision_dates=len(own), median_action_value_range=float(np.median(spreads)),
                        maximum_action_value_range=float(spreads.max()),
                        different_action_values_fraction=float((spreads > 1e-8).mean())))
                if phase == 'evaluation':
                    names = [action_q.FEATURES[column] for column in model.columns]
                    names += [f'action_{capacity_report.ACTION_NAMES[a]}' for a in range(4)]
                    if len(names) != len(model.model.feature_importances_):
                        raise ValueError('Action-Q importance feature count mismatch')
                    importances.extend(dict(case=case, feature=name, importance=float(value))
                        for name, value in zip(names, model.model.feature_importances_))
                models.append(dict(case=case, phase=phase, cutoff=cutoff,
                    transitions=audit['transitions'], distinct_market_dates=audit['distinct_market_dates'],
                    last_transition_end=audit['last_transition_end'],
                    gamma=audit['spec']['gamma'], penalty=audit['spec']['penalty'],
                    daily_reward_scale_bps=audit['daily_reward_scale_bps']))
                for action, count in own.action.value_counts().sort_index().items():
                    action_rows.append(dict(case=case, phase=phase, action=int(action),
                        action_name=capacity_report.ACTION_NAMES[int(action)], decision_dates=int(count)))
                for path in [model_path, model_path.with_suffix('.json')]:
                    capacity_report.tracked_hash(path, hashes, budget)
            if phase == 'validation':
                saved = pd.read_csv(ACTION_REPORT / 'validation.csv').set_index('case').loc[case]
                measured = period_metrics(nav, decisions, '2020-01-01', '2021-12-31')
                measured.update({f'profit_{year}': period_metrics(nav, decisions,
                    f'{year}-01-01', f'{year}-12-31')['profit_above_cash'] for year in [2020, 2021]})
                for name, value in measured.items():
                    if isinstance(value, str):
                        if saved[name] != value:
                            raise ValueError('Action-Q validation period mismatch')
                    else:
                        np.testing.assert_allclose(float(saved[name]), value, atol=1e-7, rtol=1e-10)
            else:
                baseline = pd.read_parquet(run.OUT / 'replays/baseline_verification/daily_nav.parquet', use_threads=False)
                pd.testing.assert_frame_equal(nav.loc[:'2021-12-31'], baseline.loc[:'2021-12-31'],
                                               check_exact=False, atol=1e-7, rtol=1e-11)
                if case == 'fixed100':
                    pd.testing.assert_frame_equal(nav, baseline, check_exact=False, atol=1e-7, rtol=1e-11)
                books[case] = nav
                if case != 'fixed100':
                    try:
                        pair_status, own_pairs, own_concentration = capacity_report.pair_attribution(
                            case, nav, trades, hashes, budget, destination=ACTION_OUT / 'pair_attribution' / case)
                    except (AssertionError, ValueError, KeyError, FileNotFoundError) as error:
                        pair_status = dict(case=case, status='unavailable_reconstruction',
                                           reason=f'{type(error).__name__}: {error}')
                        own_pairs, own_concentration = [], []
                    pair_statuses.append(pair_status); pair_rows.extend(own_pairs)
                    concentrations.extend(own_concentration)
                for period, start, stop in capacity_report.PERIODS:
                    periods.append(dict(case=case, period=period,
                        **period_metrics(nav, decisions, start, stop),
                        **capacity_report.cost_metrics(nav, trades, start, stop)))
                for year in range(2018, 2026):
                    start, stop = f'{year}-01-01', f'{year}-12-31'
                    annual.append(dict(case=case, year=year, **period_metrics(nav, decisions, start, stop),
                                       **capacity_report.cost_metrics(nav, trades, start, stop)))
            for name in run.prior.FILES + ['status.json', 'actions.parquet']:
                capacity_report.tracked_hash(folder / name, hashes, budget)
            del model, trades, decisions, scored, actions
            gc.collect(); budget.rest()
    pd.DataFrame(checks).to_csv(ACTION_REPORT / 'ledger_verification.csv', index=False)
    pd.DataFrame(models).to_csv(ACTION_REPORT / 'training_counts.csv', index=False)
    pd.DataFrame(action_rows).to_csv(ACTION_REPORT / 'action_counts.csv', index=False)
    pd.DataFrame(importances).to_csv(ACTION_REPORT / 'feature_importances.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(ACTION_REPORT / 'action_value_diagnostics.csv', index=False)
    pd.DataFrame(periods).to_csv(ACTION_REPORT / 'periods.csv', index=False)
    pd.DataFrame(annual).to_csv(ACTION_REPORT / 'annual.csv', index=False)
    pd.DataFrame(pair_rows).to_csv(ACTION_REPORT / 'pair_periods.csv', index=False)
    pd.DataFrame(concentrations).to_csv(ACTION_REPORT / 'concentration.csv', index=False)
    run.base.write(ACTION_REPORT / 'pair_attribution_status.json', dict(cases=pair_statuses))
    for name, digest in hashes.items():
        if run.base.digest(ROOT / name) != digest:
            raise ValueError(f'Action-Q source changed during reporting: {name}')
        budget.rest()
    run.base.write(ACTION_REPORT / 'verification.json', dict(status='passed',
        ledgers_reconciled=len(checks), model_actions_reproduced=True, source_hashes=hashes))
    return books


def verify_temporal_controls(budget):
    """Verify matched gamma-zero controls without promoting either control."""
    from scripts.p25_rl import temporal
    from scripts.p25_rl import management
    protocol_path = TEMPORAL_REPORT / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    hashes = {str(protocol_path.relative_to(ROOT)): run.base.digest(protocol_path)}
    for name, digest in protocol['source_protocols'].items():
        if run.base.digest(ROOT / name) != digest:
            raise ValueError(f'Temporal parent protocol changed: {name}')
        hashes[name] = digest; budget.rest()
    for name, digest in protocol['protected_hashes'].items():
        if run.base.digest(ROOT / name) != digest:
            raise ValueError(f'Temporal control protected source changed: {name}')
        hashes[name] = digest; budget.rest()
    signal, calendar, _, _ = run.inputs()
    checks, models, action_rows, periods, annual, books = [], [], [], [], [], {}
    pair_rows, concentrations, pair_statuses, importances, diagnostics = [], [], [], [], []
    for phase, cutoff in [('validation', '2020-01-01'), ('evaluation', '2022-01-01')]:
        for case in TEMPORAL_CASES:
            is_management = case == 'manage_myopic_capacity'
            folder = (TEMPORAL_OUT / 'validation' / case if phase == 'validation'
                      else TEMPORAL_OUT / 'replays' / case)
            status = json.loads((folder / 'status.json').read_text())
            if status['status'] != 'passed' or status['protocol_sha256'] != run.base.digest(protocol_path):
                raise ValueError('Temporal control replay/protocol mismatch')
            for name, digest in status['source_code'].items():
                if run.base.digest(ROOT / name) != digest:
                    raise ValueError(f'Temporal control source changed: {name}')
                hashes[name] = digest; budget.rest()
            for name, digest in status['output_hashes'].items():
                if run.base.digest(folder / name) != digest:
                    raise ValueError(f'Temporal control output changed: {case}/{name}')
            model_path = TEMPORAL_OUT / 'models' / cutoff / f'{case}.joblib'
            model = joblib.load(model_path)
            used_signal = signal.loc[signal.date.le('2021-10-31')].copy() if phase == 'validation' else signal
            used_calendar = calendar[calendar <= '2021-12-31'] if phase == 'validation' else calendar
            expected = temporal.replay_signature(used_signal, used_calendar, status['kind'], model,
                                                  cutoff, status['cache_signature']['seed'])
            if status['cache_signature'] != expected:
                raise ValueError(f'Temporal control replay signature mismatch: {case}/{phase}')
            nav = pd.read_parquet(folder / 'daily_nav.parquet', use_threads=False)
            trades = pd.read_csv(folder / 'trades.csv', parse_dates=['date'])
            decisions = pd.read_csv(folder / 'decisions.csv', parse_dates=['signal_date', 'entry_date'])
            scored = pd.read_parquet(folder / 'scored_allocations.parquet', use_threads=False)
            actions = pd.read_parquet(folder / 'actions.parquet', use_threads=False)
            links = capacity_report.verify_decision_links(scored, decisions,
                None if is_management else actions)
            audit, trades = run.reconcile(nav, trades, decisions, scored)
            charged = trades.reason.ne('expiry_settlement')
            np.testing.assert_allclose(trades.loc[charged, 'commission'],
                                       trades.loc[charged, 'quantity'].abs() * .03,
                                       atol=1e-10, rtol=0)
            checks.append(dict(case=case, phase=phase, status='passed', **audit, **links))
            audit = model.audit
            if audit['cutoff'] != cutoff or pd.Timestamp(audit['last_transition_end']) >= pd.Timestamp(cutoff):
                raise ValueError('Temporal control training includes future transitions')
            if audit['spec']['gamma'] != 0 or audit['spec']['penalty'] != 0:
                raise ValueError('Temporal control changes more than continuation')
            reference_name = 'manage_q' if is_management else 'action_q_capacity'
            reference_out = MANAGEMENT_OUT if is_management else ACTION_OUT
            reference = json.loads((reference_out / 'models' / cutoff / f'{reference_name}.json').read_text())
            if ({key: value for key, value in audit['spec'].items() if key != 'gamma'} !=
                    {key: value for key, value in reference['spec'].items() if key != 'gamma'}):
                raise ValueError('Temporal control changes a matched reward/action specification')
            for name in ['transitions', 'distinct_market_dates', 'last_transition_end']:
                if audit[name] != reference[name]:
                    raise ValueError(f'Temporal control has unmatched training rows: {name}')
            np.testing.assert_allclose(audit['daily_reward_scale_bps'], reference['daily_reward_scale_bps'],
                                       atol=1e-12, rtol=0)
            cache = audit['cache_signature']
            if (cache['case'] != case or cache['cutoff'] != cutoff
                    or cache['spec'] != temporal.SPECS[case] or audit['spec'] != temporal.SPECS[case]):
                raise ValueError('Temporal model cache identity mismatch')
            if cache['protocol_sha256'] != run.base.digest(protocol_path):
                raise ValueError('Temporal control model protocol mismatch')
            for collection in ['sources', 'training_files']:
                for name, digest in cache[collection].items():
                    if run.base.digest(ROOT / name) != digest:
                        raise ValueError(f'Temporal control training source changed: {name}')
                    hashes[name] = digest; budget.rest()
            eligible = actions.date.ge(cutoff)
            if is_management:
                eligible &= actions.held_cohorts.gt(0)
                np.testing.assert_array_equal(actions.eligible_decision, eligible)
                if actions.loc[~eligible, 'action'].ne(0).any() or not decisions.level.eq(4).all():
                    raise ValueError('Temporal management violated the hold/full-entry protocol')
                log = pd.read_csv(folder / 'management_log.csv', parse_dates=['signal_date', 'date'])
                if not log.empty:
                    planned = log.merge(actions[['date', 'action']].rename(columns={'date': 'signal_date'}),
                        on='signal_date', how='left', validate='many_to_one')
                    np.testing.assert_allclose(planned.target, planned.action.map({0: 1., 1: .5, 2: 0.}),
                                               atol=1e-12, rtol=0)
                    if not planned.date.gt(planned.signal_date).all():
                        raise ValueError('Temporal management execution uses same-session knowledge')
                capacity_report.tracked_hash(folder / 'management_log.csv', hashes, budget)
                del log
            own = actions.loc[eligible]
            prediction = np.array([model.choose(np.asarray(state)) for state in own.state])
            np.testing.assert_array_equal(prediction, own.action.to_numpy())
            if len(own):
                values = model.values(np.stack(own.state.map(np.asarray)))
                spreads = np.ptp(values, axis=1)
                diagnostics.append(dict(case=case, phase=phase, decision_dates=len(own),
                    median_action_value_range=float(np.median(spreads)),
                    maximum_action_value_range=float(spreads.max()),
                    different_action_values_fraction=float((spreads > 1e-8).mean())))
            names = management.ACTION_NAMES if is_management else capacity_report.ACTION_NAMES
            for action, count in own.action.value_counts().sort_index().items():
                action_rows.append(dict(case=case, phase=phase, action=int(action),
                    action_name=names[int(action)], decision_dates=int(count)))
            models.append(dict(case=case, phase=phase, cutoff=cutoff,
                transitions=audit['transitions'], distinct_market_dates=audit['distinct_market_dates'],
                last_transition_end=audit['last_transition_end'],
                gamma=audit['spec']['gamma'], penalty=audit['spec']['penalty'],
                daily_reward_scale_bps=audit['daily_reward_scale_bps']))
            if phase == 'evaluation':
                names = (management.FEATURES + [f'action_{management.ACTION_NAMES[a]}' for a in range(3)]
                         if is_management else run.FEATURES +
                         [f'action_{capacity_report.ACTION_NAMES[a]}' for a in range(4)])
                if len(names) != len(model.model.feature_importances_):
                    raise ValueError('Temporal control importance feature count mismatch')
                importances.extend(dict(case=case, feature=name, importance=float(value))
                    for name, value in zip(names, model.model.feature_importances_))
                baseline = pd.read_parquet(run.OUT / 'replays/baseline_verification/daily_nav.parquet', use_threads=False)
                pd.testing.assert_frame_equal(nav.loc[:'2021-12-31'], baseline.loc[:'2021-12-31'],
                                               check_exact=False, atol=1e-7, rtol=1e-11)
                books[case] = nav
                try:
                    pair_status, own_pairs, own_concentration = capacity_report.pair_attribution(
                        case, nav, trades, hashes, budget, destination=TEMPORAL_OUT / 'pair_attribution' / case)
                except (AssertionError, ValueError, KeyError, FileNotFoundError) as error:
                    pair_status = dict(case=case, status='unavailable_reconstruction',
                                       reason=f'{type(error).__name__}: {error}')
                    own_pairs, own_concentration = [], []
                pair_statuses.append(pair_status); pair_rows.extend(own_pairs)
                concentrations.extend(own_concentration)
                for period, start, stop in capacity_report.PERIODS:
                    periods.append(dict(case=case, period=period, **period_metrics(nav, decisions, start, stop),
                                       **capacity_report.cost_metrics(nav, trades, start, stop)))
                for year in range(2018, 2026):
                    start, stop = f'{year}-01-01', f'{year}-12-31'
                    annual.append(dict(case=case, year=year, **period_metrics(nav, decisions, start, stop),
                                       **capacity_report.cost_metrics(nav, trades, start, stop)))
            else:
                saved = pd.read_csv(TEMPORAL_REPORT / 'validation.csv').set_index('case').loc[case]
                measured = period_metrics(nav, decisions, '2020-01-01', '2021-12-31')
                measured.update({f'profit_{year}': period_metrics(nav, decisions,
                    f'{year}-01-01', f'{year}-12-31')['profit_above_cash'] for year in [2020, 2021]})
                for name, value in measured.items():
                    if isinstance(value, str):
                        if saved[name] != value:
                            raise ValueError('Temporal control validation date mismatch')
                    else:
                        np.testing.assert_allclose(float(saved[name]), value, atol=1e-7, rtol=1e-10)
            for name in run.prior.FILES + ['status.json', 'actions.parquet']:
                capacity_report.tracked_hash(folder / name, hashes, budget)
            for path in [model_path, model_path.with_suffix('.json')]:
                capacity_report.tracked_hash(path, hashes, budget)
            del model, trades, decisions, scored, actions
            gc.collect(); budget.rest()
    for filename, rows in [('ledger_verification', checks), ('training_counts', models),
                           ('action_counts', action_rows), ('periods', periods), ('annual', annual),
                           ('pair_periods', pair_rows), ('concentration', concentrations),
                           ('feature_importances', importances), ('action_value_diagnostics', diagnostics)]:
        pd.DataFrame(rows).to_csv(TEMPORAL_REPORT / f'{filename}.csv', index=False)
    run.base.write(TEMPORAL_REPORT / 'pair_attribution_status.json', dict(cases=pair_statuses))
    for name, digest in hashes.items():
        if run.base.digest(ROOT / name) != digest:
            raise ValueError(f'Temporal control source changed during reporting: {name}')
        budget.rest()
    run.base.write(TEMPORAL_REPORT / 'verification.json', dict(status='passed',
        ledgers_reconciled=len(checks), model_actions_reproduced=True, source_hashes=hashes))
    return books


def chart(table, intervals, books, selection, budget):
    os.environ['MPLCONFIGDIR'] = str(run.OUT / 'matplotlib_combined')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(20, 9), constrained_layout=True)
    colors = ['#2b7495' if case in CANDIDATES else '#7d888c' for case in table.case]
    axes[0].barh(np.arange(len(table)), table.later_sharpe, color=colors)
    axes[0].set_yticks(np.arange(len(table)), table.policy); axes[0].invert_yaxis()
    axes[0].axvline(0, color='#aeb7bb', lw=.8); axes[0].set_xlabel('2022–August 2025 excess Sharpe')
    chosen = list(dict.fromkeys(['fixed100', selection['selected'],
                                selection['best_q_by_validation'], 'tree_cem', 'learned5']))
    for case in chosen:
        nav = books[case]
        anchor = nav.loc[:'2021-12-31'].iloc[-1]
        pnl = (nav.nav - nav.cash_benchmark) - (anchor.nav - anchor.cash_benchmark)
        axes[1].plot(pnl.loc['2022-01-01':] / 1000, label=NAMES[case], lw=1.5)
    axes[1].set_ylabel('Profit above cash ($ thousands)'); axes[1].tick_params(axis='x', rotation=35)
    axes[1].legend(fontsize=8); axes[1].set_title('Continuous original portfolio accounting')
    contrasts = [(case, 'fixed100') for case in CANDIDATES]
    contrasts += [('action_q_capacity', 'entry_myopic_capacity'), ('manage_q', 'manage_myopic_capacity')]
    for j, (case, reference) in enumerate(contrasts):
        own = intervals.loc[intervals.case.eq(case) & intervals.reference.eq(reference)]
        for offset, block, color in [(-.12, 63, '#2b7495'), (.12, 126, '#b07639')]:
            row = own.loc[own.block_sessions.eq(block)].iloc[0]
            axes[2].plot([row.lower95, row.upper95], [j + offset] * 2, color=color)
            axes[2].plot(row.point, j + offset, 'o', color=color,
                         label=f'{block}-session blocks' if j == 0 else None)
    axes[2].set_yticks(np.arange(len(contrasts)),
        [NAMES[a].replace('Q: ', '') + ' minus ' + ('baseline' if b == 'fixed100' else 'myopic')
         for a, b in contrasts]); axes[2].invert_yaxis()
    axes[2].axvline(0, color='#aeb7bb', lw=.8); axes[2].legend(fontsize=8)
    axes[2].set_xlabel('Paired Sharpe difference, conditional 95% interval')
    fig.suptitle('Three Q-learning iterations: earlier selection, exploratory later evaluation', fontsize=13)
    fig.savefig(REPORT / 'comparison.png', dpi=160)
    plt.close(fig); budget.rest()


def model_root(case):
    return (MANAGEMENT_OUT if case in MANAGEMENT_CASES else
            ACTION_OUT if case in ACTION_CASES else
            TEMPORAL_OUT if case in TEMPORAL_CASES else run.OUT)


def read_optional_table(path):
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def environment_metadata():
    """Record the active reporting environment without installing packages."""
    versions = {}
    for name in ['numpy', 'pandas', 'scikit-learn', 'joblib', 'pyarrow', 'matplotlib', 'psutil']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                packages=versions,
                unavailable_versions=[name for name, version in versions.items() if version is None])


def improvement_supported(intervals, case, reference):
    """Conditional evidence flag, never a selection or deployment criterion."""
    own = intervals.loc[intervals.case.eq(case) & intervals.reference.eq(reference)
                        & intervals.period.eq('evaluation')]
    if len(own) != 2 or set(own.block_sessions) != {63, 126}:
        raise ValueError(f'Missing paired block sensitivity for {case} versus {reference}')
    if not np.isfinite(own[['point', 'lower95', 'upper95']].to_numpy(float)).all():
        raise ValueError('Nonfinite evidence interval')
    return bool(own.lower95.gt(0).all())


def verify_strategy_configuration(budget):
    """Explicitly verify the unchanged funded settings and event configuration."""
    protocol = json.loads((run.REPORT / 'protocol.json').read_text())
    verified = {}
    for path in [run.prior.CONFIG, ROOT / 'config/p25_submission.json']:
        name = str(path.relative_to(ROOT))
        expected = protocol['protected_hashes'].get(name)
        actual = run.base.digest(path)
        if expected is None or actual != expected:
            raise ValueError(f'Strategy configuration changed from the frozen protocol: {name}')
        verified[name] = actual
        budget.rest()
    return verified


def concise_summary(table, intervals, baseline_ci, selected, descriptive,
                    entry_dates, supported_candidates, supported_continuation):
    """Human-readable interpretation; never changes the frozen research choice."""
    scores = table.set_index('case')
    selected_score, baseline_score = scores.loc[selected], scores.loc['fixed100']
    best_later = scores.loc[descriptive]
    best_ci = intervals.loc[intervals.case.eq(descriptive) & intervals.reference.eq('fixed100')
                            & intervals.block_sessions.eq(126)].iloc[0]
    full_ci = baseline_ci.loc[baseline_ci.period.eq('full') & baseline_ci.block_sessions.eq(126)].iloc[0]
    candidate_note = ('None of the nine Q candidates establishes improvement at both block lengths.'
                      if not supported_candidates else
                      'Some Q comparisons have positive conditional bounds, requiring fresh confirmation.')
    continuation_note = ('Matched myopic controls do not establish a reliable benefit from learning delayed rewards.'
                         if not supported_continuation else
                         'Some matched comparisons support delayed rewards conditionally, requiring fresh confirmation.')
    paragraphs = [
        'We tested three Q-learning families: portfolio entry allocation with a shared critic, separate critics for each entry action, and daily hold/trim/exit management. Nine candidate models and their controls were evaluated.',
        f'Validation selected {NAMES[selected]}: Sharpe {selected_score.validation_sharpe:.3f} versus the baseline\'s {baseline_score.validation_sharpe:.3f}. In 2022–August 2025, it achieved {selected_score.later_sharpe:.3f} versus {baseline_score.later_sharpe:.3f}. Validation selection did not establish a better strategy.',
        f'The highest later Q point estimate was {NAMES[descriptive]} at {best_later.later_sharpe:.3f}; it was not promoted. Its Sharpe gain was {best_ci.point:+.3f}, with a conditional 95% interval [{best_ci.lower95:+.3f}, {best_ci.upper95:+.3f}] using 126-session blocks. {candidate_note} {continuation_note}',
        f'The entry Q refit used {entry_dates} distinct entry decision dates. Counterfactual trajectories reuse those dates; they are not new independent data.',
        f'The full-history baseline Sharpe was {full_ci.point:.3f}, with a positive conditional interval [{full_ci.lower95:.3f}, {full_ci.upper95:.3f}] at 126-session blocks; both block lengths support positivity. Later-period significance depends on block length. These historical intervals do not adjust for accumulated research.',
        'This study manages the inherited 608 proposals; it does not discover new portfolio IV/RV alpha. Strategy configuration remains unchanged. Retain the baseline benchmark. The next distinct hypothesis is cost-aware stock delta hedging against daily hedging and a fixed no-trade band at matched risk; it remains untested.'
    ]
    result = '\n\n'.join(paragraphs) + '\n'
    if len(result.split()) > 300:
        raise ValueError('Concise research summary exceeds 300 words')
    return result


def check_capacity_analysis(budget):
    completion = json.loads((run.REPORT / 'completion.json').read_text())
    manifest = json.loads((run.REPORT / 'report_manifest.json').read_text())
    if completion['status'] != 'passed' or manifest['status'] != 'passed':
        raise ValueError('Capacity family analysis must complete first')
    for name, digest in manifest['outputs'].items():
        if run.base.digest(run.REPORT / name) != digest:
            raise ValueError(f'Capacity analysis output changed: {name}')
        budget.rest()
    for name, digest in manifest['source_hashes'].items():
        if run.base.digest(ROOT / name) != digest:
            raise ValueError(f'Capacity analysis input changed: {name}')
        budget.rest()


def report(budget):
    context_path = ROOT / 'reports/p25_opportunity_filter/protocol.json'
    context = json.loads(context_path.read_text())['broad_funded_blocker']
    missing_contract_dates = int(context['missing_at_least_one_contract'])
    broad_contract_dates = missing_contract_dates + int(context['both_contract_ids'])
    analysis_sources = {str(path.relative_to(ROOT)): run.base.digest(path)
                        for path in [context_path, Path(__file__)]}
    config_hashes = verify_strategy_configuration(budget)
    analysis_sources.update(config_hashes)
    selection = json.loads((REPORT / 'selection.json').read_text())
    check_selection(selection, budget)
    check_capacity_analysis(budget)
    action_books = verify_action_family(budget)
    management_books = verify_management(budget)
    temporal_books = verify_temporal_controls(budget)
    capacity = pd.read_csv(run.REPORT / 'periods.csv')
    action = pd.read_csv(ACTION_REPORT / 'periods.csv')
    management = pd.read_csv(MANAGEMENT_REPORT / 'periods.csv')
    temporal = pd.read_csv(TEMPORAL_REPORT / 'periods.csv')
    combined = pd.concat([capacity, action, management.loc[~management.case.eq('fixed100')], temporal], ignore_index=True)
    combined.to_csv(REPORT / 'periods.csv', index=False)
    later = combined.loc[combined.period.eq('evaluation')].set_index('case')
    validation = earlier_tables().set_index('case')
    order = ['fixed100'] + capacity_report.NEW_CASES + ACTION_CASES + MANAGEMENT_CASES + TEMPORAL_CASES + ['tree_cem', 'learned5']
    rows = []
    for case in order:
        r = later.loc[case]
        rows.append(dict(case=case, policy=NAMES[case],
            family=('held-book management' if case in MANAGEMENT_CASES else
                    'per-action entry critics' if case in ACTION_CASES else
                    'matched temporal control' if case in TEMPORAL_CASES else
                    'rule control' if case in ['fixed100', 'fixed50', 'diversified', 'congestion'] else
                    'shared entry critic' if case in capacity_report.NEW_CASES else 'reference'),
            validation_sharpe=float(validation.loc[case, 'annualized_excess_sharpe']) if case in validation.index else np.nan,
            later_sharpe=float(r.annualized_excess_sharpe), later_profit_above_cash=float(r.profit_above_cash),
            later_entries=int(r.entries), later_pairs=int(r.active_entry_pairs),
            later_reserve_pct=float(r.average_reserve_pct), later_drawdown_pct=float(r.max_cash_relative_drawdown_pct),
            selected_by_validation=case == selection['selected']))
    table = pd.DataFrame(rows); table.to_csv(REPORT / 'results.csv', index=False)
    books = {case: pd.read_parquet(folder / 'daily_nav.parquet', use_threads=False)
             for case, folder in capacity_report.replay_paths().items()}
    books.update(action_books)
    books.update({case: book for case, book in management_books.items() if case != 'fixed100'})
    books.update(temporal_books)
    for filename in ['annual.csv', 'training_counts.csv', 'action_counts.csv',
                     'feature_importances.csv', 'concentration.csv', 'pair_periods.csv', 'action_value_diagnostics.csv']:
        pieces = []
        for folder in [run.REPORT, ACTION_REPORT, MANAGEMENT_REPORT, TEMPORAL_REPORT]:
            own = read_optional_table(folder / filename)
            if len(own):
                if folder == MANAGEMENT_REPORT and 'case' in own:
                    own = own.loc[~own.case.eq('fixed100')]
                pieces.append(own)
        (pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()).to_csv(REPORT / filename, index=False)
    pd.read_csv(MANAGEMENT_REPORT / 'management_counts.csv').to_csv(REPORT / 'management_counts.csv', index=False)
    pd.read_csv(run.REPORT / 'baseline_sharpe_intervals.csv').to_csv(REPORT / 'baseline_sharpe_intervals.csv', index=False)
    intervals = pd.read_csv(run.REPORT / 'sharpe_intervals.csv')
    extra = []
    contrasts = [(case, 'fixed100') for case in ACTION_CASES + MANAGEMENT_CASES + TEMPORAL_CASES]
    contrasts += [('action_q_risk', 'action_myopic_risk'), ('manage_q_risk', 'manage_myopic_risk'),
                  ('action_q_capacity', 'entry_myopic_capacity'), ('manage_q', 'manage_myopic_capacity')]
    for case, reference in contrasts:
        for block in [63, 126]:
            extra.append(dict(case=case, reference=reference, period='evaluation',
                **paired_sharpe_interval(capacity_report.period_returns(books[case], '2022-01-01', '2025-08-29'),
                    capacity_report.period_returns(books[reference], '2022-01-01', '2025-08-29'),
                    block=block, rest=budget.rest)))
    intervals = pd.concat([intervals, pd.DataFrame(extra)], ignore_index=True)
    for row in intervals.itertuples():
        expected = float(later.loc[row.case, 'annualized_excess_sharpe'] - later.loc[row.reference, 'annualized_excess_sharpe'])
        np.testing.assert_allclose(row.point, expected, atol=1e-12, rtol=0)
    intervals.to_csv(REPORT / 'sharpe_intervals.csv', index=False)
    selected, best = selection['selected'], selection['best_q_by_validation']
    _, _, candidate_table = choose_earlier(earlier_tables())
    best_eligible = bool(candidate_table.set_index('case').loc[best, 'eligible'])
    descriptive = table.loc[table.case.isin(CANDIDATES)].sort_values('later_sharpe', ascending=False).iloc[0].case
    incremental_supported = (None if selected == 'fixed100' else
                             improvement_supported(intervals, selected, 'fixed100'))
    continuation_reference = {'q_risk': 'myopic_risk',
        'action_q_risk': 'action_myopic_risk', 'manage_q_risk': 'manage_myopic_risk',
        'action_q_capacity': 'entry_myopic_capacity', 'manage_q': 'manage_myopic_capacity'}.get(selected)
    continuation_supported = (None if continuation_reference is None else
                              improvement_supported(intervals, selected, continuation_reference))
    supported_candidates = [case for case in CANDIDATES
                            if improvement_supported(intervals, case, 'fixed100')]
    matched_continuation = [('q_risk', 'myopic_risk'), ('action_q_risk', 'action_myopic_risk'),
        ('manage_q_risk', 'manage_myopic_risk'), ('action_q_capacity', 'entry_myopic_capacity'),
        ('manage_q', 'manage_myopic_capacity')]
    supported_continuation = [case for case, reference in matched_continuation
                             if improvement_supported(intervals, case, reference)]
    all_candidate_evidence = (
        'None of the nine candidate Q policies establishes a reliable Sharpe improvement over the baseline at both conditional block lengths.'
        if not supported_candidates else
        f'Candidates with positive conditional lower bounds at both block lengths: {", ".join(supported_candidates)}. Fresh confirmation remains necessary.')
    evidence_labels = {True: 'supported', False: 'not established', None: 'not applicable'}
    benchmark_evidence = (
        'The earlier selection retained the original benchmark; no incremental Q improvement is claimed.'
        if selected == 'fixed100' else
        'Selected research Q exceeds the benchmark in both conditional paired intervals; this remains exploratory evidence on previously researched dates.'
        if incremental_supported else
        'The selected research Q does not establish reliable benchmark improvement: its paired lower bound is not above zero at both block lengths.')
    approach = dict(selected=selected, policy=NAMES[selected], selected_on='2020-2021 only',
        role='Earlier-selected research approach; selection does not establish reliable benchmark improvement',
        reason=selection['criterion'], best_q_by_validation=best,
        best_q_meets_selection_criteria=best_eligible,
        best_q_model=str(model_root(best) / 'models/2022-01-01' / f'{best}.joblib'),
        later_descriptive_best_candidate=descriptive, later_used_for_selection=False,
        incremental_sharpe_supported_63_126=incremental_supported,
        continuation_supported_63_126=continuation_supported,
        continuation_reference=continuation_reference,
        evidence_definition='True only when both 63/126-session paired conditional 95% Sharpe-difference lower bounds are strictly positive. Null means no applicable comparison. No research-selection adjustment or later promotion.',
        validation_selected_baseline=selected == 'fixed100',
        strategy_configuration_unchanged=True, strategy_configuration_hashes=config_hashes,
        candidates_supported_63_126=supported_candidates,
        continuation_cases_supported_63_126=supported_continuation,
        untouched_holdout=False,
        selected_model=None if selected == 'fixed100' else str(model_root(selected) / 'models/2022-01-01' / f'{selected}.joblib'))
    run.base.write(REPORT / 'best_approach.json', approach)
    shared_diagnostic = pd.read_csv(run.REPORT / 'action_value_diagnostics.csv')
    shared_capacity = shared_diagnostic.loc[shared_diagnostic.case.eq('q_capacity')
                                           & shared_diagnostic.phase.eq('validation')]
    collapsed_shared = len(shared_capacity) == 1 and shared_capacity.iloc[0].different_action_values_fraction == 0
    initial_note = ('The initial capacity critic assigned identical values to all actions on validation states and reverted to full size; the separate-action follow-up addresses that representation failure.'
                    if collapsed_shared else 'The separate-action follow-up tests whether giving each action its own critic improves the original shared-critic allocation model.')
    baseline_ci = pd.read_csv(run.REPORT / 'baseline_sharpe_intervals.csv')
    baseline_later = baseline_ci.loc[baseline_ci.period.eq('evaluation') & baseline_ci.block_sessions.eq(126)].iloc[0]
    lines = ['PAIR IV/RV REINFORCEMENT LEARNING: THREE BOUNDED ITERATIONS', '',
        f'Earlier-selected research approach: {NAMES[selected]}. Best Q on earlier validation: {NAMES[best]}.',
        benchmark_evidence,
        f'Highest later candidate Q point estimate: {NAMES[descriptive]} (descriptive; not used for selection).', '',
        'Policy                                   Validation  Later Sharpe  Above cash ($)  Entries',
        '                                         2020-2021   2022-2025']
    for r in table.itertuples():
        v = '  —' if not np.isfinite(r.validation_sharpe) else f'{r.validation_sharpe:.3f}'
        lines.append(f'{r.policy:40s} {v:>9s} {r.later_sharpe:13.3f} {r.later_profit_above_cash:15,.0f} {r.later_entries:8d}')
    lines += ['', 'WHAT WE TESTED',
        '1. Entry allocation Q: skip/half/full proposals, with an unheld-pair action, actual inventory, admission limits and funded rewards.',
        '2. Action-specific entry Q: independent action critics ensure that allocations can receive different estimated values.',
        '3. Held-book Q: fixed-full entries, then hold/trim/close decisions using current RV-versus-IV carry, maturity, execution cost and inventory.',
        'All use small fitted-Q models, 2018-2019 transitions, 2020-2021 validation and a pre-2022 refit.',
        'Nine candidate policies were considered in joint selection before later Q evaluation; one research approach was selected. Myopic and missing-inventory arms are ablations.', '',
        'FINDING',
        ('None of the nine Q candidates met the earlier criteria. The original threshold remains the retained approach.'
         if selected == 'fixed100' else f'The earlier criteria selected {NAMES[selected]} for research evaluation, without establishing benchmark superiority.'),
        benchmark_evidence,
        all_candidate_evidence,
        f'Selected approach: incremental Sharpe improvement {evidence_labels[incremental_supported]}; matched continuation benefit {evidence_labels[continuation_supported]}. Both block lengths must have positive lower bounds. These conditional evidence flags do not change selection.',
        f'Best earlier Q versus baseline: {capacity_report.describe_intervals(intervals, best, "fixed100")}',
        'Entry Q versus myopic: ' + capacity_report.describe_intervals(intervals, 'q_risk', 'myopic_risk'),
        'Entry Q with versus without inventory: ' + capacity_report.describe_intervals(intervals, 'q_risk', 'q_no_inventory'),
        'Action-specific Q versus myopic: ' + capacity_report.describe_intervals(intervals, 'action_q_risk', 'action_myopic_risk'),
        'Management Q versus myopic: ' + capacity_report.describe_intervals(intervals, 'manage_q_risk', 'manage_myopic_risk'),
        'Unshaped capacity Q versus matched myopic: ' + capacity_report.describe_intervals(intervals, 'action_q_capacity', 'entry_myopic_capacity'),
        'Unshaped management Q versus matched myopic: ' + capacity_report.describe_intervals(intervals, 'manage_q', 'manage_myopic_capacity'), '',
        'HOW TO INTERPRET THIS',
        f'Original baseline later Sharpe: {baseline_later.point:.3f}, conditional 95% interval [{baseline_later.lower95:+.3f}, {baseline_later.upper95:+.3f}] using 126-session blocks.',
        'The threshold contains a useful simple opportunity rule. A confidence interval containing zero for CEM-minus-baseline does not prove that the baseline is uniquely significant.',
        'The benchmark also includes historical profit/count screens, pair formation, contract eligibility, risk admission and daily stock hedging; its result is not attributable to one number alone.',
        'Q learning has an economic role when decisions change later inventory/capacity or exit costs. These tests measure those mechanisms rather than relabeling a predictive score as RL.',
        'Risk-shaped Q optimizes a return/second-moment utility, not Sharpe. The fixed 0.5 penalty can favor cash; its failure would not establish that every risk-aware Q policy fails.',
        'All Q variants use ordinary maximum-value Bellman backups. Conservative variants penalize model disagreement only when choosing actions.',
        initial_note,
        'Sparse trades, repeated market histories and approximate book states limit learning. Higher historical Sharpe alone is insufficient to select a new method.',
        'These market dates had already been used in prior research; the new policies were frozen before their later replays. Intervals condition on fitted policies, use paired 63/126-session blocks and 2,000 resamples, and do not adjust for accumulated research.',
        'No later winner is promoted. The original strategy configuration is unchanged and its protected hashes were rechecked. A frozen future evaluation remains necessary.', '',
        'A DISTINCT FUTURE RL TEST (NOT RUN HERE)',
        'Keep the opportunity rule and option book fixed; learn when and how much stock delta to hedge, with transaction costs and explicit residual-delta risk. Compare with daily hedging and a validation-selected fixed no-trade band at matched risk.',
        'This gives actions a direct cost-versus-future-risk consequence and more decision dates; it is a control hypothesis, not a promise of higher Sharpe. [Deep Hedging](https://arxiv.org/abs/1802.03042) supports this control mechanism under trading frictions, not performance in this dataset.',
        f'A new broad-universe IV/RV alpha strategy first needs quote coverage repaired: the prior coverage audit found {missing_contract_dates:,} of {broad_contract_dates:,} candidate pair-dates missing at least one contract ID; ID presence alone does not establish complete daily quotes.', '',
        'OUTPUTS',
        'best_approach.json identifies the earlier-selected policy and best research Q model; results.csv includes every arm.',
        'Detailed family tables retain annual results, costs, risk, actions, training counts and independently reconciled ledgers.',
        'Feature importances describe model use, not causal effects. In separate-action critics, routing-action columns have zero split importance by construction; the action-specific models still differ.',
        'The portfolio experiments retain the inherited proposal universe; they do not demonstrate a new broad-universe strategy.', '']
    (REPORT / 'Findings.txt').write_text('\n'.join(lines), encoding='utf-8')
    training = pd.read_csv(REPORT / 'training_counts.csv')
    entry_training = training.loc[training.case.eq('action_q_capacity') & training.phase.eq('evaluation')]
    if len(entry_training) != 1:
        raise ValueError('Missing entry Q effective training-date count')
    (REPORT / 'Summary.txt').write_text(concise_summary(table, intervals, baseline_ci,
        selected, descriptive, int(entry_training.iloc[0].distinct_market_dates),
        supported_candidates, supported_continuation), encoding='utf-8')
    run.base.write(REPORT / 'environment.json', environment_metadata())
    chart(table, intervals, books, selection, budget)
    check_selection(selection, budget)
    check_capacity_analysis(budget)
    for folder in [ACTION_REPORT, MANAGEMENT_REPORT, TEMPORAL_REPORT]:
        verified = json.loads((folder / 'verification.json').read_text())
        for name, digest in verified['source_hashes'].items():
            if run.base.digest(ROOT / name) != digest:
                raise ValueError(f'Family source changed during combined analysis: {name}')
            budget.rest()
    for name, digest in analysis_sources.items():
        if run.base.digest(ROOT / name) != digest:
            raise ValueError(f'Combined analysis source changed: {name}')
        budget.rest()
    outputs = ['selection.json', 'validation.csv', 'validation_candidates.csv', 'periods.csv',
               'results.csv', 'sharpe_intervals.csv', 'best_approach.json', 'Findings.txt', 'comparison.png',
               'annual.csv', 'training_counts.csv', 'action_counts.csv', 'feature_importances.csv',
               'concentration.csv', 'pair_periods.csv', 'management_counts.csv', 'baseline_sharpe_intervals.csv',
               'action_value_diagnostics.csv', 'environment.json', 'Summary.txt']
    run.base.write(REPORT / 'completion.json', dict(status='passed', selected=selected,
        joint_earlier_selection_reproduced=True, management_verified=True,
        capacity_verified=True, action_family_verified=True, temporal_controls_verified=True,
        interval_points_verified=True, analysis_sources=analysis_sources,
        outputs={name: run.base.digest(REPORT / name) for name in outputs}, peak_rss_mib=budget.peak))
    print('\n'.join(lines[:26]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['select', 'report'], required=True)
    args = parser.parse_args(); budget = Budget()
    if args.phase == 'select':
        select(budget)
    else:
        report(budget)


if __name__ == '__main__':
    main()
