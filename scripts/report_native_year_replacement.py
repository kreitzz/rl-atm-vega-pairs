"""Replace the legacy one-year sleeve in the saved-return tenor blend."""
import argparse
import json
from pathlib import Path
import resource

import numpy as np
import pandas as pd

from scripts.report_long_tenor_search import optimal_nonnegative_weights, sharpe
from scripts.run_native_year_entry import OUT as NATIVE, REPORT, dump, old


def positive_sharpe_weights(scores):
    scores = np.asarray(scores, dtype=float)
    assert np.isfinite(scores).all()
    positive = np.maximum(scores, 0.)
    return positive / positive.sum() if positive.sum() > 0 else np.ones(len(scores)) / len(scores)


def past_only_allocation(panel, date, rule='optimizer'):
    """Use strictly earlier observations from a fixed 24-calendar-month window."""
    history = panel.loc[(panel.index < date) &
                        (panel.index >= date - pd.DateOffset(months=24))]
    assert len(history) >= 400 and history.index.max() < date
    if rule == 'sharpe_proportional':
        scores = np.array([sharpe(history[name]) for name in panel])
        return positive_sharpe_weights(scores), history
    assert rule == 'optimizer'
    # Keep valid corner portfolios even if every historical mean is negative.
    candidates = list(np.eye(panel.shape[1]))
    try:
        candidates.append(optimal_nonnegative_weights(history))
    except AssertionError:
        # The active-set helper requires at least one positive-mean solution.
        assert (history.mean() <= 0).all()
    scores = np.array([sharpe(history @ w) for w in candidates])
    assert np.isfinite(scores).all()
    weights = candidates[int(np.argmax(scores))]
    return weights, history


def historical_weights(panel, rule='optimizer'):
    """Equal-weight warmup, then quarterly past-only allocation; no window search."""
    weights = pd.DataFrame(index=panel.index, columns=panel.columns, dtype=float)
    decisions = []
    allocation = np.ones(panel.shape[1]) / panel.shape[1]
    first_optimization_year = panel.index[0].year + 2
    previous_quarter = None
    for date in panel.index:
        quarter = date.to_period('Q')
        if quarter != previous_quarter:
            previous_quarter = quarter
            if date.year >= first_optimization_year:
                allocation, history = past_only_allocation(panel, date, rule=rule)
                decisions.append(dict(date=date, history_start=history.index[0],
                    history_end=history.index[-1], observations=len(history),
                    historical_sharpe=sharpe(history @ allocation),
                    **{f'historical_sharpe_{name}': sharpe(history[name]) for name in panel},
                    **{f'weight_{name}': float(w) for name, w in zip(panel, allocation)}))
        weights.loc[date] = allocation
    assert np.isfinite(weights).all().all()
    np.testing.assert_allclose(weights.sum(axis=1), 1.)
    assert (weights >= 0).all().all()
    return weights, pd.DataFrame(decisions)


def walk_forward(two_tenors=False, sharpe_weighting=False):
    source = REPORT / 'replacement_blend/component_excess_returns.csv'
    saved = pd.read_csv(source, parse_dates=['date']).set_index('date')
    columns = ['30D', '180D'] if two_tenors else ['30D', '180D', 'native365D']
    panel = saved[columns].rename(columns={'native365D': '365D'})
    assert panel.index.is_unique and panel.index.is_monotonic_increasing
    assert np.isfinite(panel).all().all()
    rule = 'sharpe_proportional' if sharpe_weighting else 'optimizer'
    weights, decisions = historical_weights(panel, rule=rule)
    if sharpe_weighting:
        np.testing.assert_allclose(positive_sharpe_weights([1., .5, -.2]), [2/3, 1/3, 0.])
        np.testing.assert_allclose(positive_sharpe_weights([-1., 0., -.2]), [1/3]*3)
        for row in decisions.itertuples():
            scores = np.array([getattr(row, f'historical_sharpe_{name}') for name in panel])
            actual = np.array([getattr(row, f'weight_{name}') for name in panel])
            assert np.isfinite(scores).all()
            np.testing.assert_allclose(actual, positive_sharpe_weights(scores))
            if (scores > 0).any():
                assert (actual[scores <= 0] == 0).all()
    # Temporal invariance: alter all returns from decision day forward, and
    # independently truncate future data. Both must leave the decision unchanged.
    tested = []
    for date in decisions.date.iloc[[0, len(decisions)//2, -1]]:
        expected, history = past_only_allocation(panel, date, rule=rule)
        perturbed = panel.copy()
        perturbed.loc[perturbed.index >= date] = np.array([.1, -.2, .3])[:len(panel.columns)]
        actual, _ = past_only_allocation(perturbed, date, rule=rule)
        truncated, _ = past_only_allocation(panel.loc[panel.index < date], date, rule=rule)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
        np.testing.assert_allclose(truncated, expected, rtol=0, atol=0)
        assert history.index.max() < date
        tested.append(str(date.date()))
    returns = pd.DataFrame({
        ('sharpe_proportional24M_quarterly' if sharpe_weighting else 'rolling24M_quarterly'): (panel * weights).sum(axis=1),
        'equal_weights': panel.mean(axis=1),
        '30D_only': panel['30D'],
        '180D_only': panel['180D'],
        **({'new365D_only': panel['365D']} if '365D' in panel else {}),
    })
    if sharpe_weighting:
        optimizer_weights, _ = historical_weights(panel)
        returns['rolling24M_quarterly_optimizer'] = (panel * optimizer_weights).sum(axis=1)
    evaluation_start = decisions.date.iloc[0]
    rows, annual = [], []
    for case in returns:
        for period, r in [('full_including_equal_warmup', returns[case]),
                          ('after_allocation_warmup', returns.loc[evaluation_start:, case])]:
            rows.append(dict(case=case, period=period,
                annualized_excess_sharpe=sharpe(r), start=str(r.index[0].date()),
                end=str(r.index[-1].date()), sessions=len(r)))
        annual.extend(dict(case=case, year=int(year),
            annualized_excess_sharpe=sharpe(part))
            for year, part in returns[case].groupby(returns.index.year))
    folder = 'walk_forward_24M_30D_180D' if two_tenors else 'walk_forward_24M'
    if sharpe_weighting:
        folder += '_sharpe_proportional'
    output = REPORT / 'replacement_blend' / folder
    output.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output / 'comparison.csv', index=False)
    pd.DataFrame(annual).to_csv(output / 'annual.csv', index=False)
    returns.to_csv(output / 'daily_excess_returns.csv')
    weights.to_csv(output / 'daily_weights.csv')
    decisions.to_csv(output / 'allocation_decisions.csv', index=False)
    record = dict(components=list(panel.columns), allocation_rule=rule,
        sharpe_weight_formula='max(trailing_sharpe,0) / sum(max(trailing_sharpe,0)); equal if all nonpositive' if sharpe_weighting else None,
        latest_allocation={name: float(weights[name].iloc[-1]) for name in panel},
        latest_allocation_date=str(decisions.date.iloc[-1].date()),
        window_months=24, rebalance='quarterly',
        warmup='Equal weights during 2018-2019',
        first_historical_allocation=str(evaluation_start.date()),
        weights_use_strictly_prior_sessions=True,
        future_perturbation_and_truncation_audit_dates=tested,
        full_history_weight_optimization=False, window_search=False,
        retrospective_component_selection=True, no_fresh_holdout=True,
        after_component_costs=True, additional_reallocation_costs_included=False,
        shared_portfolio_replayed=False, one_year_replaces_old_component=not two_tenors,
        one_year_component_excluded=two_tenors,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    dump(output / 'audit.json', record)
    dump(output / 'source_manifest.json', {str(p): old.file_hash(p) for p in
        [Path(__file__), source, Path('scripts/report_long_tenor_search.py')]})
    (output / 'findings.txt').write_text(
        'PAST-ONLY TENOR ALLOCATION\n\n'
        'Freeze the components: ' + ', '.join(panel.columns) + '. Use equal weights in 2018-2019, '
        'then reestimate nonnegative weights summing to one on the first trading '
        'session of each quarter, using only the preceding 24 calendar months '
        'through the prior session. The quarter-start return is never in the fit. '
        'No window or rebalance-frequency search was performed. Weights multiply '
        'saved component returns from that session onward.\n\n'
        + ('Allocation weights are proportional to positive trailing annualized '
           'cash-excess Sharpe; negative and zero Sharpes get zero weight unless '
           'all are nonpositive, when equal weights apply. Higher positive Sharpe '
           'therefore receives higher weight. This rule does not use covariance.\n\n'
           if sharpe_weighting else '') +
        'Report all cases over matching calendars, both including the equal-weight '
        'warmup and after historical allocation begins. Daily cash-excess Sharpes '
        'annualize with sqrt252; 2025 is partial.\n\n'
        'This removes future-return use in allocation, not retrospective selection '
        'of the researched components. This is not a fresh holdout. Component costs '
        'are included; additional reallocation costs, shared inventory, integer '
        'scaling and joint risk admission require a joint portfolio replay. '
        'Daily weights are a return-stream allocation, not executable trading orders.\n\n'
        + comparison.to_string(index=False) + '\n')
    print(comparison.to_string(index=False), flush=True)
    print(json.dumps(record, indent=2), flush=True)


def main():
    output = REPORT / 'replacement_blend'
    output.mkdir(parents=True, exist_ok=True)
    snapshot_path = Path('reports/long_tenor_reversible/best_portfolio.json')
    snapshot = json.loads(snapshot_path.read_text())
    sources = {
        '30D': Path('artifacts/contracts/p25_long_management/replays/30D_manage_36M/daily_nav.parquet'),
        '180D': Path('artifacts/contracts/long_tenor_fitted_q/P25/180D/replays/fixed50_fitted_q_conservative_aligned_runoff/daily_nav.parquet'),
        '365D': Path('artifacts/contracts/long_tenor_reversible/P25/365D/replays/fixed50_hold_fork_control/daily_nav.parquet'),
        'native365D': NATIVE / 'replays/forecast_rl_sized/daily_nav.parquet',
    }
    references = json.loads(Path('artifacts/contracts/year_option_action_critics/P25/references.json').read_text())
    assert all(old.file_hash(Path(path)) == wanted for path, wanted in references.items())
    streams = {}
    for name, path in sources.items():
        streams[name] = pd.read_parquet(path).excess_return.iloc[1:]
        assert streams[name].index.is_unique and np.isfinite(streams[name]).all()
        pd.testing.assert_index_equal(streams[name].index, streams['30D'].index)
    components = pd.DataFrame(streams)
    original = components[['30D', '180D', '365D']]
    weights = np.array([snapshot['weights'][name] for name in original])
    original_return = original @ weights
    assert np.isclose(sharpe(original_return), snapshot['annualized_excess_sharpe'], atol=1e-12)
    saved_path = Path('reports/long_tenor_reversible/best_portfolio_daily.csv')
    saved = pd.read_csv(saved_path, parse_dates=['date']).set_index('date').excess_return
    pd.testing.assert_index_equal(saved.index, original_return.index, check_names=False)
    np.testing.assert_allclose(original_return, saved, rtol=1e-10, atol=1e-14)
    replacement = original.copy()
    replacement['365D'] = components.native365D
    optimized = optimal_nonnegative_weights(replacement)
    cases = {
        'prior_blend_reference': (original, weights),
        'replace365D_keep_weights': (replacement, weights),
        'replace365D_reoptimize_weights': (replacement, optimized),
    }
    rows, annual, returns = [], [], {}
    for case, (panel, allocation) in cases.items():
        assert np.isclose(allocation.sum(), 1.) and (allocation >= 0.).all()
        r = panel @ allocation
        score = sharpe(r)
        predicted = np.sqrt(252)*(panel.mean().to_numpy() @ allocation)/np.sqrt(allocation @ panel.cov().to_numpy() @ allocation)
        assert np.isclose(score, predicted, atol=1e-12)
        returns[case] = r
        rows.append(dict(case=case, annualized_excess_sharpe=score,
            weight_30D=float(allocation[0]), weight_180D=float(allocation[1]), weight_365D=float(allocation[2]),
            start=str(r.index[0].date()), end=str(r.index[-1].date()), sessions=len(r),
            replaces_one_year=case!='prior_blend_reference',
            full_history_weight_optimization=case=='replace365D_reoptimize_weights'))
        annual.extend(dict(case=case, year=int(year), annualized_excess_sharpe=sharpe(part))
                      for year, part in r.groupby(r.index.year))
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output / 'comparison.csv', index=False)
    pd.DataFrame(annual).to_csv(output / 'annual.csv', index=False)
    pd.DataFrame(returns).to_csv(output / 'daily_excess_returns.csv')
    components.to_csv(output / 'component_excess_returns.csv')
    components.corr().to_csv(output / 'correlations.csv')
    best = comparison.loc[comparison.replaces_one_year].sort_values('annualized_excess_sharpe', ascending=False).iloc[0]
    record = dict(best_replacement_case=best.case, annualized_excess_sharpe=float(best.annualized_excess_sharpe),
        weights=dict(zip(['30D', '180D', '365D'], [float(best[f'weight_{name}']) for name in ['30D', '180D', '365D']])),
        old_blend_sharpe=float(sharpe(original_return)),
        replacement_at_original_weights_sharpe=float(sharpe(returns['replace365D_keep_weights'])),
        native_one_year_policy='forecast_rl_sized', native_one_year_standalone_sharpe=float(sharpe(components.native365D)),
        replacement_not_additional_sleeve=True, old_reference_preserved=True,
        same_calendar=True, after_component_costs=True, shared_portfolio_replayed=False,
        full_history_optimized=True, no_fresh_holdout=True,
        one_worker=True, cpu_core_fraction=.5, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        sources={name: str(path) for name, path in sources.items()})
    dump(output / 'summary.json', record)
    paths = [Path(__file__), Path('scripts/report_long_tenor_search.py'),
             Path('scripts/run_signal_discovery_guarded.py'), snapshot_path, saved_path,
             NATIVE / 'replays/forecast_rl_sized/status.json', *sources.values()]
    dump(output / 'source_manifest.json', {str(path): old.file_hash(path) for path in paths})
    (output / 'findings.txt').write_text(
        'ONE-YEAR SLEEVE REPLACEMENT\n\n'
        'Replace the older one-year fixed50 hold sleeve with the new native one-year forecast direction plus learned sizing/skips. '
        'Retain the same managed30D and conservative fitted-Q180D streams. The new one-year sleeve replaces the old sleeve; it is not added as a fourth component.\n\n'
        'First reproduce the prior1.099449945825 blend from its saved components/weights, then substitute the new one-year stream at those same weights. '
        'Separately optimize three nonnegative weights summing to one across the full evaluation history. Every calendar must match exactly; initialization rows are excluded. '
        'All Sharpes annualize daily cash-excess returns with sqrt252. 2018–August2025,2025 partial.\n\n'
        'These are saved-return blends after component costs. Joint inventory, integer scaling, shared cash, hedge netting, rebalancing costs and portfolio risk admission have not been replayed. '
        'Reoptimized weights and the one-year policy choice use already-researched history; this is an exploratory comparison with no fresh holdout. '
        'The original checkpoint and blend remain unchanged.\n\n'+comparison.to_string(index=False)+'\n')
    print(comparison.to_string(index=False), flush=True)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--walk-forward', action='store_true')
    parser.add_argument('--two-tenors', action='store_true')
    parser.add_argument('--sharpe-weighting', action='store_true')
    args = parser.parse_args()
    if args.two_tenors and not args.walk_forward:
        parser.error('--two-tenors requires --walk-forward')
    if args.sharpe_weighting and not args.walk_forward:
        parser.error('--sharpe-weighting requires --walk-forward')
    if args.walk_forward:
        walk_forward(two_tenors=args.two_tenors, sharpe_weighting=args.sharpe_weighting)
    else:
        main()
