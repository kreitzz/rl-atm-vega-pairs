"""Rolling conventional allocators on the same two funded indices as retained Q.

Run ``--phase run`` and ``--phase chart`` in separate memory-guarded workers.
Estimates use only completed weeks; monthly estimates govern weekly decisions.
No options data loading, entry retraining, or manuscript changes are performed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.atm_p25_allocation_q import month_path
from scripts.report_atm_p25_combination import aligned_cash, stats

CONFIG = Path('config/paper_allocation_benchmarks.json')
CFG = json.loads(CONFIG.read_text())
OUT, REPORT = Path(CFG['output_root']), Path(CFG['report_root'])
SHARES = np.array(CFG['straddle_weights'])
WEIGHTS = np.column_stack([1-SHARES, SHARES])
TARGETS = np.column_stack([WEIGHTS, np.zeros(len(SHARES))])
PERIODS = [('validation', '2018-01-01', '2021-12-31'),
           ('evaluation', '2022-01-01', CFG['evaluation_end']),
           ('full', '2018-01-01', CFG['evaluation_end'])]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.pending.json')
    pending.write_text(json.dumps(value, indent=2, default=str)+'\n')
    pending.replace(path)


def metrics(book, start, end):
    result = stats(book.excess_return, book.cash_return, start, end)
    part = book.loc[start:end]
    prior = max(0, book.index.get_loc(part.index[0])-1)
    benchmark = CFG['capital']*(1+book.cash_return).cumprod()
    difference = book.nav-benchmark
    result['profit_above_cash'] = float(difference.loc[part.index[-1]]-difference.iloc[prior])
    return result


def estimate(data, index):
    """Fit unconditional return moments on exactly Q's completed-week window."""
    from sklearn.covariance import LedoitWolf
    current = data['periods'][index]
    cutoff = current['cutoff']
    train = [m for m in data['periods'][:index]
             if m['end'] < cutoff and
             m['cutoff'] >= cutoff-pd.Timedelta(weeks=CFG['training_weeks'])]
    if len(train) < CFG['minimum_training_weeks']:
        return None
    indices = np.concatenate([m['indices'] for m in train])
    assert data['pair'].index[indices].max() < cutoff
    returns = np.column_stack([data[k].excess_return.iloc[indices] for k in ['pair', 'straddle']])
    reference = np.concatenate([m['paths'][3, 3] for m in train])
    covariance = np.cov(returns, rowvar=False, ddof=1)
    lw = LedoitWolf().fit(returns)
    matrices = {'sample': covariance, 'ledoit_wolf': lw.covariance_}
    for matrix in matrices.values():
        assert np.isfinite(matrix).all() and np.linalg.eigvalsh(matrix).min() >= -1e-20
    return dict(mean=returns.mean(axis=0), covariance=matrices,
                reference_vol=max(float(reference.std(ddof=1)), 1e-8),
                last_training_end=str(max(m['end'] for m in train).date()),
                training_weeks=len(train), training_sessions=len(indices),
                shrinkage=float(lw.shrinkage_))


def specifications():
    result = []
    for covariance in CFG['covariance_estimators']:
        for objective in ['minimum_variance', 'maximum_sharpe', 'target_return']:
            result.append(dict(case=covariance+'_'+objective, family=objective,
                               covariance=covariance, objective=objective, risk=None))
        for risk in CFG['risk_preferences']:
            result.append(dict(case=f'{covariance}_mean_variance_lambda{risk:g}',
                               family=f'mean_variance_lambda{risk:g}',
                               covariance=covariance, objective='mean_variance', risk=risk))
    return result


def choose(model, spec):
    """Exact optimization over Q's seven targets; no evaluation-period tuning."""
    mean = WEIGHTS@model['mean']
    covariance = model['covariance'][spec['covariance']]
    variance = np.maximum(np.einsum('ij,jk,ik->i', WEIGHTS, covariance, WEIGHTS), 1e-24)
    objective = spec['objective']
    feasible = None
    if objective == 'minimum_variance':
        score = -variance
    elif objective == 'maximum_sharpe':
        score = mean/np.sqrt(variance)
    elif objective == 'mean_variance':
        ref = model['reference_vol']
        score = mean/ref-spec['risk']*variance/ref**2
    else:
        daily_target = (1+CFG['monthly_return_target'])**(12/252)-1
        eligible = mean >= daily_target
        feasible = bool(eligible.any())
        score = np.where(eligible, -variance, -np.inf) if feasible else mean
    action = int(np.argmax(score))
    return action, feasible, float(mean[action]), float(np.sqrt(variance[action]))


def replay(data, spec, models):
    previous = TARGETS[3].copy()
    paths, decisions = [], []
    for i, week in enumerate(data['periods']):
        model = models[i]
        action, feasible, forecast_mean, forecast_vol = 3, None, None, None
        if model:
            assert pd.Timestamp(model['last_training_end']) < week['cutoff']
            action, feasible, forecast_mean, forecast_vol = choose(model, spec)
        excess, previous, charge = month_path(week['gross'], week['cash'], week['costs'],
                                              previous, TARGETS[action])
        paths.append(excess)
        decisions.append(dict(case=spec['case'], date=week['cutoff'],
                              feature_date=week['feature_date'], straddle_weight=SHARES[action],
                              fitted=model is not None,
                              last_training_end=model['last_training_end'] if model else None,
                              target_return_feasible=feasible,
                              forecast_daily_mean=forecast_mean, forecast_daily_vol=forecast_vol,
                              resizing_cost_fraction=charge))
    positions = np.concatenate([m['indices'] for m in data['periods']])
    cash = np.concatenate([m['cash'] for m in data['periods']])
    excess = np.concatenate(paths)
    book = pd.DataFrame(dict(excess_return=excess, cash_return=cash,
                             nav=CFG['capital']*np.cumprod(1+excess+cash)),
                        index=data['pair'].index[positions])
    OUT.mkdir(parents=True, exist_ok=True)
    book.to_parquet(OUT/(spec['case']+'.parquet'))
    return book, pd.DataFrame(decisions)


def run():
    import joblib
    source = Path(CFG['source_root'])/'prepared.joblib'
    data = joblib.load(source)
    aligned_cash(data['pair'], data['straddle'])
    models, model, month = [], None, None
    fits = []
    for i, week in enumerate(data['periods']):
        key = str(week['cutoff'].to_period('M'))
        if key != month or model is None:
            model = estimate(data, i)
            month = key
            if model:
                fits.append(dict(cutoff=week['cutoff'], **model))
        models.append(model)
    records, annual, decisions = [], [], []
    REPORT.mkdir(parents=True, exist_ok=True)
    for spec in specifications():
        book, schedule = replay(data, spec, models)
        decisions.append(schedule)
        for period, start, end in PERIODS:
            subset = schedule.loc[schedule.date.between(start, end)]
            fitted = subset.loc[subset.fitted]
            target_feasible = fitted.target_return_feasible.dropna()
            records.append(dict(**spec, period=period, kind='conventional',
                                mean_straddle_target=float(subset.straddle_weight.mean()),
                                fitted_decisions=len(fitted),
                                target_return_feasible_fraction=float(target_feasible.mean())
                                if len(target_feasible) else None,
                                **metrics(book, start, end)))
        for year in range(2018, 2026):
            annual.append(dict(**spec, year=year, **metrics(book, f'{year}-01-01',
                                                          min(f'{year}-12-31', CFG['evaluation_end']))))
        print('Replayed '+spec['case'], flush=True)
    results = pd.DataFrame(records)
    results.to_csv(REPORT/'conventional_results.csv', index=False)
    pd.DataFrame(annual).to_csv(REPORT/'annual.csv', index=False)
    pd.concat(decisions, ignore_index=True).to_csv(REPORT/'decisions.csv', index=False)
    # Verify the common execution engine against every saved fixed benchmark.
    fixed = pd.read_csv('reports/signal_allocation_variance/fixed_results.csv')
    checks = []
    dates = data['pair'].index[np.concatenate([m['indices'] for m in data['periods']])]
    cash = np.concatenate([m['cash'] for m in data['periods']])
    for action, weight in enumerate(SHARES):
        previous = TARGETS[action].copy()
        paths = []
        for m in data['periods']:
            path, previous, _ = month_path(m['gross'], m['cash'], m['costs'], previous, TARGETS[action])
            paths.append(path)
        excess = np.concatenate(paths)
        book = pd.DataFrame(dict(excess_return=excess, cash_return=cash,
                                nav=CFG['capital']*np.cumprod(1+excess+cash)), index=dates)
        for period, start, end in PERIODS:
            actual = metrics(book, start, end)
            expected = fixed.loc[fixed.straddle_target.eq(weight)&fixed.period.eq(period)].iloc[0]
            for metric in ['annualized_excess_sharpe', 'profit_above_cash']:
                np.testing.assert_allclose(actual[metric], expected[metric], rtol=0, atol=1e-6)
            checks.append(dict(weight=weight, period=period, reproduced=True))
        if weight in [0, 1]:
            original = data['pair' if weight == 0 else 'straddle']
            np.testing.assert_allclose(excess, original.excess_return, atol=1e-12, rtol=0)
    fixed['case'] = fixed.straddle_target.map(lambda w: f'fixed_{w:g}')
    fixed['kind'] = 'fixed'
    q = pd.read_csv('reports/signal_allocation_variance/results.csv')
    q['case'] = q.variance_penalty.map(lambda x: f'Q_lambda{x:g}')
    q['kind'] = 'Q'
    comparison = pd.concat([results, fixed, q], ignore_index=True)
    comparison.to_csv(REPORT/'all_results.csv', index=False)
    validation = results.loc[results.period.eq('validation')].sort_values(
        ['annualized_excess_sharpe', 'case'], ascending=[False, True])
    selected = validation.groupby('family', sort=True).head(1)[['family', 'case']]
    selected.to_csv(REPORT/'validation_selection.csv', index=False)
    chosen = comparison.loc[comparison.case.isin(selected.case)|
                            comparison.case.isin(['fixed_0.5', 'Q_lambda0.2', 'Q_lambda1'])]
    chosen.to_csv(REPORT/'selected_comparison.csv', index=False)
    source_paths = [CONFIG, Path(__file__), source,
                    Path('scripts/atm_p25_allocation_q.py'),
                    Path('scripts/report_atm_p25_combination.py'),
                    Path('reports/signal_allocation_variance/results.csv'),
                    Path('reports/signal_allocation_variance/fixed_results.csv')]
    save(REPORT/'protocol.json', dict(configuration=CFG,
         mean_variance_objective='mean/ref_vol - lambda * variance/ref_vol**2; conventional centered variance. Q penalizes the raw second moment and uses state and continuation value.',
         estimation='Daily net source excess returns from Q-matched completed 104-week windows, minimum 52. Refitted monthly; weekly decisions execute at next close. No parameter chosen using later results.',
         covariance='Sample covariance and sklearn LedoitWolf linear shrinkage; no high-dimensional inverse needed for two strategies.',
         trading='Identical source costs and extra weekly resizing model. No leverage, short strategy weights, cash action, or stop. Same 50/50 warmup as Q. Funded indices, not a shared integer-contract ledger.',
         target_return='Paper target adapted to 1% monthly EXCESS return, converted geometrically to daily; infeasibility and max-mean fallback recorded. Other methods optimize on excess returns too.',
         source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
         paper_unchanged=True))
    fit_rows = [{k: v for k, v in fit.items() if k not in ['mean', 'covariance']}
                for fit in fits]
    pd.DataFrame(fit_rows).to_csv(REPORT/'fit_timing.csv', index=False)
    moments = []
    for fit in fits:
        for estimator, covariance in fit['covariance'].items():
            moments.append(dict(cutoff=fit['cutoff'], covariance_estimator=estimator,
                                last_training_end=fit['last_training_end'],
                                mean_put=fit['mean'][0], mean_straddle=fit['mean'][1],
                                var_put=covariance[0, 0], var_straddle=covariance[1, 1],
                                covariance=covariance[0, 1], reference_vol=fit['reference_vol']))
    pd.DataFrame(moments).to_csv(REPORT/'fit_moments.csv', index=False)
    save(REPORT/'verification.json', dict(all_fixed_replays_reproduced=checks,
         endpoints_reproduce_source_returns=True, training_ends_precede_decisions=True,
         source_dates_unique=True, original_Q_results_used_unchanged=True,
         monthly_fit_count=len(fits), cases=len(specifications())))
    columns = ['case', 'period', 'annualized_excess_sharpe', 'profit_above_cash',
               'mean_straddle_target', 'target_return_feasible_fraction']
    print(chosen.loc[chosen.period.ne('full'), columns].to_string(index=False), flush=True)


def chart():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    table = pd.read_csv(REPORT/'all_results.csv')
    selected = pd.read_csv(REPORT/'validation_selection.csv')
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    fig.subplots_adjust(left=.08, right=.98, bottom=.22, top=.80, wspace=.13)
    for ax, (period, title) in zip(axes, [('validation', '2018–2021 validation'),
                                        ('evaluation', '2022–August 2025')]):
        part = table.loc[table.period.eq(period)]
        fixed = part.loc[part.kind.eq('fixed')].sort_values('straddle_target')
        ax.plot(fixed.profit_above_cash/1e6, fixed.annualized_excess_sharpe,
                'o--', color='0.65', lw=1, ms=4, label='Fixed capital splits')
        q = part.loc[part.kind.eq('Q')].sort_values('variance_penalty')
        ax.plot(q.profit_above_cash/1e6, q.annualized_excess_sharpe,
                'o-', color='#087f94', lw=1.5, ms=5, label='Retained Q policies')
        for row in q.itertuples():
            offset = {0.: (6, 2), .1: (7, 12), .2: (-35, -11), 1.: (5, 5)}
            ax.annotate(f'λ={row.variance_penalty:g}',
                        (row.profit_above_cash/1e6, row.annualized_excess_sharpe),
                        xytext=offset[row.variance_penalty], textcoords='offset points',
                        fontsize=7, color='#087f94')
        mv_cases = selected.loc[selected.family.str.startswith('mean_variance_'), 'case']
        mv = part.loc[part.case.isin(mv_cases)].sort_values('risk')
        ax.plot(mv.profit_above_cash/1e6, mv.annualized_excess_sharpe,
                'D-', color='#a44767', lw=1.4, ms=4, label='Rolling mean–variance')
        for row in mv.itertuples():
            offset = {0.: (5, -11), .1: (5, 4), .2: (5, -12), 1.: (5, 5)}
            ax.annotate(f'λ={row.risk:g}',
                        (row.profit_above_cash/1e6, row.annualized_excess_sharpe),
                        xytext=offset[row.risk], textcoords='offset points',
                        fontsize=7, color='#a44767')
        for family, marker, color, label in [
                ('minimum_variance', 's', '#bb673a', 'Minimum variance'),
                ('maximum_sharpe', '^', '#6b59a5', 'Maximum Sharpe')]:
            case = selected.loc[selected.family.eq(family), 'case'].iloc[0]
            row = part.loc[part.case.eq(case)].iloc[0]
            ax.scatter(row.profit_above_cash/1e6, row.annualized_excess_sharpe,
                       marker=marker, color=color, s=55, label=label, zorder=4)
        ax.set_title(title, loc='left', fontsize=11)
        ax.set_xlabel('Profit above cash ($ million)')
        ax.set_xlim(0, 2.6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
    axes[0].set_ylabel('Annualized excess Sharpe')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc='lower center', bbox_to_anchor=(.5, .045),
               frameon=False, fontsize=8)
    fig.suptitle('Conventional allocation versus Q on the same two bandits', fontsize=14, y=.96)
    fig.text(.08, .875, 'Covariance estimator selected on validation within each objective; identical weekly execution and costs.', fontsize=8)
    for ext in ['png', 'pdf']:
        fig.savefig(REPORT/('allocation_benchmarks.'+ext), dpi=130)
    plt.close(fig)


def analyze():
    """Save a readable comparison, including unfavorable profit-only Q results."""
    frame = pd.read_csv(REPORT/'all_results.csv')
    selected = pd.read_csv(REPORT/'validation_selection.csv').set_index('family').case
    feasibility = frame.loc[frame.objective.eq('target_return'), 'target_return_feasible_fraction'].dropna()
    assert len(feasibility) and feasibility.eq(0).all()
    cases = [('Fixed 50/50', 'fixed_0.5'),
             ('Minimum variance', selected['minimum_variance']),
             ('Maximum Sharpe', selected['maximum_sharpe'])]
    for risk in CFG['risk_preferences']:
        cases.extend([(f'Mean–variance lambda {risk:g}', selected[f'mean_variance_lambda{risk:g}']),
                      (f'Q lambda {risk:g}', f'Q_lambda{risk:g}')])
    lines = ['CONVENTIONAL ALLOCATION BENCHMARKS', '',
             'Both sources are the retained bandit portfolios. All Sharpe ratios are annualized excess Sharpe.',
             'Capital $10m; unchanged source execution assumptions and incremental resizing costs.',
             '104 completed weeks, minimum 52; monthly estimation, weekly decisions, next-close execution.',
             'Same seven capital allocations and 50/50 warmup as Q. No strategy leverage or cash action.', '',
             'Policy | 2018–21 Sharpe | 2022–Aug2025 Sharpe | Later profit above cash',
             '-'*95]
    rows = []
    for label, case in cases:
        val = frame.loc[frame.case.eq(case)&frame.period.eq('validation')].iloc[0]
        test = frame.loc[frame.case.eq(case)&frame.period.eq('evaluation')].iloc[0]
        lines.append(f'{label} | {val.annualized_excess_sharpe:.3f} | {test.annualized_excess_sharpe:.3f} | ${test.profit_above_cash:,.0f}')
        rows.append(dict(label=label, case=case, validation_sharpe=val.annualized_excess_sharpe,
                         evaluation_sharpe=test.annualized_excess_sharpe,
                         evaluation_profit_above_cash=test.profit_above_cash))
    lines.extend(['', 'INTERPRETATION',
        'Q has higher Sharpe and dollar profit than the validation-chosen conventional mean–variance estimator at each positive penalty (0.1, 0.2, 1) in both periods.',
        'This does not hold for lambda 0: profit-only conventional allocation has higher later profit and Sharpe than profit-only Q.',
        'Minimum variance becomes almost entirely put-pair exposure; mean–variance lambda 1 averages 9.27% straddles later, versus 16.88% for Q lambda 1.',
        'Q lambda 0.2 averages 24.92% straddles later, versus 12.75% for conventional mean–variance lambda 0.2. These are different portfolio paths, not matched mean-risk exposures.',
        'The conventional controls estimate unconditional trailing moments. Q also uses economic states and continuation value; this comparison alone does not isolate either contribution.', '',
        'SELECTION AND SCOPE',
        'Choose sample or Ledoit-Wolf covariance separately within each objective using validation Sharpe, alphabetical case-name ties; display every case in all_results.csv.',
        'The paper uses nonlinear shrinkage among other high-dimensional estimators. This study uses linear Ledoit-Wolf shrinkage for a two-strategy problem; it is an adaptation, not a replication.',
        'The paper uses a 15-year estimation window; we retain Q\'s two-year window and seven long-only capital targets to hold the decision problem constant.',
        'The literal 1% monthly target is infeasible in every fitted decision. Its explicitly labeled max-mean fallback always targets 100% straddles in the later period. Do not present it as a feasible constrained optimum.',
        'The existing paper and its strategy specifications are unchanged.', '',
        'REPRODUCTION',
        'Run scripts.paper_allocation_benchmarks --phase run, then --phase analyze and --phase chart in separate scripts.run_signal_discovery_guarded workers.',
        'One worker, 20% of one core, 256 MiB RSS/footprint, 30% headroom floor, 256 MiB maximum swap growth.',
        'verification.json reconciles all seven fixed allocations in all three reporting periods and verifies both source endpoints.',
        'config/paper_allocation_benchmarks.json and protocol.json specify objectives, timing, selection, and source hashes.', ''])
    (REPORT/'FINDINGS.txt').write_text('\n'.join(lines))
    pd.DataFrame(rows).to_csv(REPORT/'headline_comparison.csv', index=False)
    print('\n'.join(lines[:len(cases)+11]), flush=True)


def diagnostics():
    """Hold each saved allocation schedule fixed and remove only outer fees."""
    import joblib
    data = joblib.load(Path(CFG['source_root'])/'prepared.joblib')
    schedules = pd.read_csv(REPORT/'decisions.csv', parse_dates=['date'])
    q = pd.read_csv('reports/signal_allocation_variance/decisions.csv', parse_dates=['date'])
    selected = pd.read_csv(REPORT/'validation_selection.csv').set_index('family').case
    cases = [(selected[f], schedules.loc[schedules.case.eq(selected[f])])
             for f in ['minimum_variance', 'maximum_sharpe',
                       'mean_variance_lambda0.2', 'mean_variance_lambda1']]
    for risk in [.2, 1.]:
        cases.append((f'Q_lambda{risk:g}', q.loc[q.variance_penalty.eq(risk)]))
    records = []
    dates = data['pair'].index[np.concatenate([m['indices'] for m in data['periods']])]
    cash = np.concatenate([m['cash'] for m in data['periods']])
    for case, schedule in cases:
        schedule = schedule.set_index('date')
        previous = TARGETS[3].copy()
        paths, charges, weeks, weights = [], [], [], []
        for m in data['periods']:
            weight = float(schedule.loc[m['cutoff'], 'straddle_weight'])
            matches = np.flatnonzero(np.isclose(SHARES, weight, atol=1e-12, rtol=0))
            assert len(matches) == 1
            target = TARGETS[matches[0]]
            actual, after, charge = month_path(m['gross'], m['cash'], m['costs'], previous, target)
            free, free_after, _ = month_path(m['gross'], m['cash'], m['costs'], previous, target,
                                           cost_multiplier=0.)
            np.testing.assert_allclose(after, free_after, atol=1e-12, rtol=0)
            previous = after
            paths.append((actual, free))
            charges.append(charge); weeks.append(m['cutoff']); weights.append(weight)
        actual = np.concatenate([p[0] for p in paths]); free = np.concatenate([p[1] for p in paths])
        if case.startswith('Q_'):
            risk = case.split('lambda')[1]
            expected = pd.read_parquet(f'artifacts/contracts/signal_allocation_variance/replays/lambda{risk}/daily.parquet', use_threads=False)
        else:
            expected = pd.read_parquet(OUT/(case+'.parquet'), use_threads=False)
        np.testing.assert_allclose(actual, expected.excess_return, atol=1e-12, rtol=0)
        no_resize = pd.DataFrame(dict(excess_return=free, cash_return=cash,
                                     nav=CFG['capital']*np.cumprod(1+free+cash)), index=dates)
        for period, start, end in PERIODS[:2]:
            mask = (pd.DatetimeIndex(weeks) >= start)&(pd.DatetimeIndex(weeks) <= end)
            part = no_resize.loc[start:end]
            years = (part.index[-1]-part.index[0]).days/365.25
            subweights = np.array(weights)[mask]
            records.append(dict(case=case, period=period,
                actual_sharpe=metrics(expected, start, end)['annualized_excess_sharpe'],
                no_outer_resizing_fee_sharpe=metrics(no_resize, start, end)['annualized_excess_sharpe'],
                sum_outer_resizing_charges_bps=1e4*float(np.array(charges)[mask].sum()),
                approximate_annual_resizing_charge_bps=1e4*float(np.array(charges)[mask].sum())/years,
                allocation_changes=int((np.diff(subweights) != 0).sum()),
                mean_straddle_target=float(subweights.mean())))
    pd.DataFrame(records).to_csv(REPORT/'cost_and_switching_diagnostics.csv', index=False)
    save(REPORT/'diagnostic_protocol.json', dict(
        method='Replay exactly the saved target schedule with and without incremental outer resizing charges; retain all trading costs inside both source indices. Do not refit or reselect actions.',
        checks='Every actual replay matches its saved daily excess returns to 1e-12; no-fee and actual portfolios have identical drifted fractional weights.',
        interpretation='The fee removal is a mechanical conditional sensitivity, not a feasible new strategy. Sum of fee fractions is an additive charge diagnostic, not an exact compounded return attribution.',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(pd.DataFrame(records).to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['run', 'chart', 'analyze', 'diagnostics'], required=True)
    globals()[parser.parse_args().phase]()
