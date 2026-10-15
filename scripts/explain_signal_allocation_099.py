"""Frozen-policy explanations and timing controls for the corrected Q study.

No model is fitted and no candidate is selected. Interventions use saved models,
completed-history training centers, and the original allocation-cost paths.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import signal_capital_allocation as core

ROOT = Path('artifacts/contracts/signal_allocation_discount')
OUT = Path('reports/allocation_interpretability_099')
PERIODS = [('validation', '2018-01-01', '2021-12-31'),
           ('evaluation', '2022-01-01', '2025-08-29')]
FIELDS = core.FEATURES + ['previous_straddle_weight']
LABELS = ['Intercept', 'Put spread strength', 'Put variance spread',
          'Straddle IV minus RV', 'Completed straddle premium',
          'Relative strategy volatility', 'Strategy return correlation',
          'VIX', 'Relative recent Sharpe', 'Previous straddle weight']


def save(name, rows):
    pd.DataFrame(rows).to_csv(OUT / (name + '.csv'), index=False)


def book_for(data, actions, continuous=None):
    """Discrete schedules use the already verified action-transition return bank."""
    paths = []
    if continuous is None:
        previous = 3
        for m, action in zip(data['periods'], actions):
            paths.append(m['paths'][previous, action])
            previous = action
    else:
        previous = core.TARGETS[3].copy()
        for m, w in zip(data['periods'], continuous):
            path, previous, _ = core.engine.month_path(
                m['gross'], m['cash'], m['costs'], previous, np.array([1-w, w, 0.]))
            paths.append(path)
    excess = np.concatenate(paths)
    positions = np.concatenate([m['indices'] for m in data['periods']])
    cash = np.concatenate([m['cash'] for m in data['periods']])
    return pd.DataFrame({'excess_return': excess, 'cash_return': cash,
                         'nav': 1e7*np.cumprod(1+excess+cash)},
                        index=data['pair'].index[positions])


def replay_centered(data, models, feature):
    actions = []
    previous = core.TARGETS[3].copy()
    for m in data['periods']:
        model = models.get(str(m['cutoff'].to_period('M')))
        if model is not None and pd.Timestamp(model['cutoff']) > m['cutoff']:
            model = None
        action = 3
        if model is not None:
            state = np.r_[m['market'], previous[1]]
            state[feature] = model['center'][feature]
            scores = core.design(np.tile(state, (core.N, 1)), np.arange(core.N),
                                 np.asarray(model['center']), np.asarray(model['scale'])) @ model['beta']
            action = int(np.argmax(scores))
        actions.append(action)
        previous = m['after'][action]
    return np.asarray(actions)


def run_q():
    import joblib
    OUT.mkdir(parents=True, exist_ok=True)
    data = joblib.load('artifacts/contracts/signal_capital_allocation/prepared.joblib')
    dates = pd.DatetimeIndex([m['cutoff'] for m in data['periods']])
    records, gaps, sensitivity, controls, placebo, yearly, usage = [], [], [], [], [], [], []
    inputs = [Path(__file__), Path('scripts/signal_capital_allocation.py'),
              Path('artifacts/contracts/signal_capital_allocation/prepared.joblib')]
    for risk in (.1, .2):
        name = core.case(risk, .99)
        path = ROOT/'replays'/name
        decisions = pd.read_csv(path/'full_decisions.csv', parse_dates=['date'])
        saved = pd.read_parquet(path/'full.parquet', use_threads=False)
        inputs += [path/'full_decisions.csv', path/'full.parquet']
        actions = decisions.action.to_numpy(int)
        assert len(actions) == len(dates)
        np.testing.assert_array_equal(dates, decisions.date)
        replay = book_for(data, actions)
        np.testing.assert_allclose(replay.excess_return, saved.excess_return, atol=1e-14, rtol=0)
        np.testing.assert_allclose(replay.nav, saved.nav, atol=1e-6, rtol=0)
        models, immediate = {}, {}
        for p in (ROOT/'models'/name).glob('*.json'):
            models[p.stem] = json.loads(p.read_text()); inputs.append(p)
            other = ROOT/'models'/core.case(risk, 0.)/p.name
            immediate[p.stem] = json.loads(other.read_text()); inputs.append(other)
        previous = core.TARGETS[3].copy()
        for i, m in enumerate(data['periods']):
            key = str(m['cutoff'].to_period('M')); model = models.get(key)
            if model is not None and pd.Timestamp(model['cutoff']) > m['cutoff']:
                model = None
            period = 'validation' if m['cutoff'].year <= 2021 else 'evaluation'
            if model is not None:
                assert pd.Timestamp(model['last_training_end']) < pd.Timestamp(model['cutoff']) <= m['cutoff']
                state = np.r_[m['market'], previous[1]]
                center, scale = np.asarray(model['center']), np.asarray(model['scale'])
                z = np.r_[1., np.clip((state-center)/scale, -5, 5)]
                beta = np.asarray(model['beta']).reshape(3, -1)
                scores = core.design(np.tile(state, (core.N, 1)), np.arange(core.N), center, scale) @ model['beta']
                action = int(np.argmax(scores)); assert action == actions[i]
                runner = sorted(range(core.N), key=lambda j: (-scores[j], j))[1]
                a, b = core.SHARES[[action, runner]]
                terms = z * ((a-b)*beta[1] + (a*a-b*b)*beta[2])
                gap = float(scores[action]-scores[runner])
                np.testing.assert_allclose(terms.sum(), gap, atol=1e-10, rtol=1e-10)
                im = immediate[key]
                np.testing.assert_allclose(im['center'], center)
                np.testing.assert_allclose(im['scale'], scale)
                scores0 = core.design(np.tile(state, (core.N, 1)), np.arange(core.N), center, scale) @ im['beta']
                immediate_gap = float(scores0[action]-scores0[runner])
                record = dict(case=name, date=m['cutoff'], period=period, year=m['cutoff'].year,
                              straddle_weight=float(a), runner_straddle_weight=float(b),
                              immediate_action=float(core.SHARES[int(np.argmax(scores0))]),
                              score_gap=gap, immediate_gap=immediate_gap,
                              continuation_gap=gap-immediate_gap, previous_weight=previous[1])
                record.update(dict(zip(FIELDS, state)))
                records.append(record)
                for j, feature in enumerate(['intercept']+FIELDS):
                    gaps.append(dict(case=name, period=period, date=m['cutoff'], feature=feature, contribution=terms[j]))
                for j, feature in enumerate(FIELDS):
                    changed = state.copy(); changed[j] = center[j]
                    cf = core.design(np.tile(changed, (core.N, 1)), np.arange(core.N), center, scale) @ model['beta']
                    sensitivity.append(dict(case=name, period=period, date=m['cutoff'], feature=feature,
                                            action_changed=int(np.argmax(cf)) != action,
                                            weight_change=float(core.SHARES[int(np.argmax(cf))]-a)))
            previous = m['after'][actions[i]]
        for period, start, end in PERIODS:
            mask = (dates >= start) & (dates <= end)
            selected = np.flatnonzero(mask)
            average = float(core.SHARES[actions[mask]].mean())
            base_metrics = core.metrics(saved, start, end)
            controls.append(dict(case=name, period=period, control='actual_q', **base_metrics))
            weights = core.SHARES[actions].copy(); weights[mask] = average
            matched = book_for(data, actions, weights)
            controls.append(dict(case=name, period=period, control='same_mean_target_hindsight',
                                 mean_straddle_weight=average, **core.metrics(matched, start, end)))
            myopic_path = ROOT/'replays'/core.case(risk, 0.)/'full.parquet'
            inputs.append(myopic_path)
            myopic = pd.read_parquet(myopic_path, use_threads=False)
            controls.append(dict(case=name, period=period, control='saved_gamma_zero', **core.metrics(myopic, start, end)))
            # Rotate only fitted weeks within the reporting period. Keep the
            # genuine early fallback and the pre-period portfolio history.
            fitted = selected[decisions.fitted.to_numpy(bool)[selected]]
            shifts = np.unique(np.linspace(1, len(fitted)-1, 49, dtype=int))
            for shift in shifts:
                rotated = actions.copy(); rotated[fitted] = np.roll(actions[fitted], int(shift))
                metrics = core.metrics(book_for(data, rotated), start, end)
                placebo.append(dict(case=name, period=period, rotation=int(shift), **metrics))
            for action in range(core.N):
                usage.append(dict(case=name, period=period, straddle_weight=float(core.SHARES[action]),
                                  weeks=int((actions[mask] == action).sum()), total_weeks=int(mask.sum())))
        for feature_index, feature in enumerate(FIELDS):
            alternate_actions = replay_centered(data, models, feature_index)
            altered = book_for(data, alternate_actions)
            for period, start, end in PERIODS:
                controls.append(dict(case=name, period=period, control='centered_'+feature,
                                     **core.metrics(altered, start, end)))
        for year in range(2018, 2026):
            mask = dates.year == year
            yearly.append(dict(case=name, year=year,
                               average_straddle_weight=float(core.SHARES[actions[mask]].mean()),
                               **core.metrics(saved, f'{year}-01-01', min(f'{year}-12-31', '2025-08-29'))))
        print('Explained and replay-verified '+name, flush=True)
    save('q_decisions', records); save('q_exact_gap_terms', gaps)
    save('q_feature_sensitivity', sensitivity); save('q_controls', controls)
    save('q_timing_rotations', placebo); save('q_action_usage', usage); save('q_annual', yearly)
    table = pd.DataFrame(gaps)
    table['absolute_contribution'] = table.contribution.abs()
    table.groupby(['case','period','feature']).absolute_contribution.mean().reset_index().to_csv(OUT/'q_gap_importance.csv',index=False)
    pd.DataFrame(sensitivity).groupby(['case','period','feature']).action_changed.mean().reset_index().to_csv(OUT/'q_sensitivity_summary.csv',index=False)
    summary = []
    for (case, period), g in pd.DataFrame(placebo).groupby(['case','period']):
        actual = next(r for r in controls if r['case']==case and r['period']==period and r['control']=='actual_q')
        summary.append(dict(case=case, period=period, rotations=len(g),
                            actual_sharpe=actual['annualized_excess_sharpe'],
                            median_rotation_sharpe=g.annualized_excess_sharpe.median(),
                            fraction_at_least_actual=float((g.annualized_excess_sharpe>=actual['annualized_excess_sharpe']).mean()),
                            median_rotation_profit=g.profit_above_cash.median()))
    save('q_rotation_summary', summary)
    manifest = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(inputs))}
    (OUT/'q_protocol.json').write_text(json.dumps(dict(source_hashes=manifest,
        frozen_models=True, replay_exact=True, score_gaps_exact=True, rotations_per_period=49,
        interpretation='Conditional sensitivity, not causal attribution. Rotations and same-mean targets use realized actions retrospectively and are diagnostics, not investable benchmarks. No model is refit or selected.'), indent=2)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.parse_args(); run_q()
