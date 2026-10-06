"""Consolidate all one-year follow-up outcomes without selecting a new model."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.report_long_tenor_search import sharpe

REPORT = Path('reports/year_option_followup')
SOURCES = [Path('reports') / ('year_option_' + name) for name in
           ['actual_policy', 'action_critics', 'entry_rl', 'sharpe_policy']]
ALIASES = {'fixed50_engine_control': 'fixed_50_hold',
           'fixed50_hold_control': 'fixed_50_hold'}
FIXED = {'fixed25_hold', 'fixed_50_hold', 'fixed75_hold', 'fixed_100_hold'}
OLD = {'fixed50_fitted_q_conservative_aligned_runoff',
       'enriched_reversible_aligned_warmup'}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def category(case):
    if case in FIXED:
        return 'Fixed allocation control'
    if case in OLD:
        return 'Previous fitted-Q reference'
    if case.startswith('hold_'):
        return 'Fixed holding-period rule'
    if case.endswith('_entry_hold'):
        return 'Learned-entry / no-management control'
    if case in ['actual_book_2state_48M', 'matched_shadow_2state_48M'] or case.startswith(('native_2_', 'native_3_', 'pair_native_')):
        return 'Contextual portfolio allocation search'
    return 'Fitted-Q entry or management'


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    records, annual, admission, returns, paths = [], [], [], {}, [Path(__file__), Path('scripts/run_signal_discovery_guarded.py')]
    checked = 0
    for source in SOURCES:
        manifest = json.loads((source / 'source_manifest.json').read_text())
        for path, expected in manifest.items():
            assert digest(path) == expected, f'Stale source manifest: {source}: {path}'
            checked += 1
        comparison = pd.read_csv(source / 'comparison.csv')
        paths += [source / 'comparison.csv', source / 'source_manifest.json', source / 'findings.txt']
        for record in comparison.to_dict('records'):
            case = ALIASES.get(record['case'], record['case'])
            folder = Path(record['source_folder'])
            nav = pd.read_parquet(folder / 'daily_nav.parquet')
            r = nav.excess_return.iloc[1:]
            assert np.isclose(sharpe(r), record['annualized_excess_sharpe'])
            if case in returns:
                pd.testing.assert_series_equal(r, returns[case], rtol=1e-12, atol=1e-12)
                continue
            returns[case] = r
            record.update(case=case, category=category(case),
                          evaluation_start=str(r.index.min().date()), evaluation_end=str(r.index.max().date()),
                          post_warmup_2020_partial2025_sharpe=sharpe(r.loc['2020-01-01':]),
                          recent_2024_partial2025_sharpe=sharpe(r.loc['2024-01-01':]))
            records.append(record)
            annual.append(pd.read_csv(folder / 'annual.csv').assign(case=case))
            paths += [folder / 'daily_nav.parquet', folder / 'status.json', folder / 'annual.csv']
            decisions = pd.read_csv(folder / 'decisions.csv')
            admission += [dict(case=case, status=status, proposals=int(count))
                          for status, count in decisions.status.value_counts().items()]
            paths.append(folder / 'decisions.csv')
    frame = pd.DataFrame(records).sort_values('annualized_excess_sharpe', ascending=False)
    frame.to_csv(REPORT / 'comparison.csv', index=False)
    pd.concat(annual, ignore_index=True).to_csv(REPORT / 'annual.csv', index=False)
    pd.DataFrame(admission).to_csv(REPORT / 'admissions.csv', index=False)
    fixed = frame.loc[frame.case.isin(FIXED)].iloc[0]
    learned = frame.loc[~frame.case.isin(FIXED) & ~frame.case.str.startswith('hold_') & ~frame.case.isin(OLD)]
    assert fixed.case == 'fixed25_hold'
    assert (learned.annualized_excess_sharpe < fixed.annualized_excess_sharpe).all()

    # No recomputed optimization or pooled-book claim: retain existing references.
    matched_path = SOURCES[0] / 'actual_vs_shadow.json'
    matched = json.loads(matched_path.read_text())
    paths.append(matched_path)
    stats = dict(best_fixed_case=fixed.case, best_fixed_sharpe=float(fixed.annualized_excess_sharpe),
                 original_fixed50_sharpe=float(frame.set_index('case').loc['fixed_50_hold', 'annualized_excess_sharpe']),
                 learned_variants_beat_best_fixed=False,
                 actual_vs_matched_shadow=matched, untouched_holdout=False,
                 verified_manifest_entries=checked,
                 selection='All outcomes shown; fixed25 and shorter holdings were adaptively explored after earlier learned failures')
    (REPORT / 'summary.json').write_text(json.dumps(stats, indent=2) + '\n')
    columns = ['case', 'category', 'annualized_excess_sharpe', 'entries', 'net_excess_pnl',
               'post_warmup_2020_partial2025_sharpe', 'recent_2024_partial2025_sharpe']
    (REPORT / 'findings.txt').write_text(
        'ONE-YEAR OPTION RESEARCH FOLLOW-UP\n\n'
        'Result: fixed 25% sizing improves annualized cash-excess Sharpe from 0.396012 at fixed 50% to 0.470775. '
        'None of the new learned variants beats the fixed 50% reference, much less fixed 25%. '
        'This improvement comes from a fixed sizing control; there is no demonstrated one-year RL advantage. Lower size produces less dollar profit.\n\n'
        'The same 529 frozen short-dated-policy pair/date/direction proposals are used throughout. '
        'Native one-year puts target absolute delta 0.25 and 300–400 days to expiry. '
        'There are 230 native training episodes; the old fixed 50% book admits 167 cohorts. '
        'Different sizes and management policies change integer contract eligibility and shared risk admissions. '
        'These are not proportional leverage transformations of an identical return series. '
        'Smaller sizing can fail the unchanged 10% integer-vega matching tolerance; admissions.csv records these exclusions separately from risk-limit exclusions. '
        'The fixed 50% book last enters 2024-05-28 and then runs off through 2025-08-29; recent-year results have limited new entries.\n\n'
        'Process changes tested: explicit action-output critics; 64 weekly Bellman iterations spanning a year; '
        'past-calibrated risk utility; greedy versus uncertainty-penalized inference; native-feature entry Q; '
        'quarterly contextual allocation search maximizing shadow-book daily Sharpe; '
        'and annual allocation search in the executable shared portfolio. '
        'The latter is episodic contextual policy optimization, not sequential fitted-Q/neural RL.\n\n'
        'The actual-book learner reaches 0.356002 versus 0.265155 for a matched shadow-trained control using exactly '
        'the same state trees, annual cadence, 48-month prefixes and 25/50/100% action menu. '
        'Its unadjusted block-bootstrap 95% Sharpe-gain interval contains zero. '
        'Actual-book training respects integer fills, shared hedges/cash and original risk admissions, retains true expiry dates, '
        'and marks open inventory at the last historical close. A runtime guard rejects future contract price requests. '
        'Its improved training objective does not produce an advantage over fixed sizing in evaluation.\n\n'
        'A separate fixed-rule holding scan exits the same one-year contracts after 30, 90 or 180 observed trading sessions; '
        'fills occur at the next close with observed quotes and normal execution costs. All three underperform.\n\n'
        'Evaluation: 2018-01-02 through 2025-08-29; 2025 partial. Sharpe is sqrt(252)*mean(daily cash-excess return)/sample standard deviation; '
        'exclude initialization row. $10m account. Actual option commission $0.03/contract, 0.1 vol point per leg execution penalty, '
        '1 bp stock execution, floating cash interest and 0.5% stock borrow. Training commissions $0.65. '
        'Original global/pair/wing risk limits retained. Entry vega matching tolerance 10%; daily delta hedging; '
        'no continuous vega rebalance; different leg expiries have natural runoff. European pricing/no early assignment.\n\n'
        'All history is repeatedly explored. Past-only walk-forward fits are enforced, but this is not a fresh holdout; '
        'reported intervals are not adjusted for adaptive research. No existing checkpoint is overwritten.\n\n'
        'Remaining research bottleneck: one-year entries and directions are inherited from short-dated signals. '
        'Build and validate native long-tenor opportunities/forecasts before additional action tuning. '
        'The existing isolated-label pair diagnostics show that positive carry forecasts can coincide with negative outcomes; '
        'those are small-sample retrospective diagnostics, not investable pair exclusions.\n\n'
        + frame[columns].to_string(index=False) + '\n\nMatched actual/shadow comparison:\n'
        + json.dumps(matched, indent=2) + '\n')
    plot(frame, returns)
    (REPORT / 'source_manifest.json').write_text(json.dumps({str(p): digest(p) for p in paths}, indent=2) + '\n')
    print(frame[columns].to_string(index=False), flush=True)


def plot(frame, returns):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = {'fixed25_hold': 'Fixed 25%', 'fixed_50_hold': 'Fixed 50% reference',
              'fixed75_hold': 'Fixed 75%', 'fixed_100_hold': 'Fixed 100%',
              'action_outputs_16': 'Best new fitted-Q management',
              'actual_book_2state_48M': 'Actual-book allocation learner',
              'matched_shadow_2state_48M': 'Matched shadow allocation learner',
              'native_mean_entry': 'Native entry Q + management',
              'native_3_48M': 'Best quarterly shadow allocation',
              'hold_180_sessions': 'Fixed exit after 180 sessions'}
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    values = frame.set_index('case').loc[list(labels), 'annualized_excess_sharpe']
    colors = ['#64748b' if c in FIXED else '#2563eb' for c in labels]
    axes[0].barh(list(labels.values()), values, color=colors)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, .57)
    axes[0].set_xlabel('Annualized daily cash-excess Sharpe')
    for i, value in enumerate(values):
        axes[0].text(value + .009, i, f'{value:.3f}', va='center')
    chosen = ['fixed25_hold', 'fixed_50_hold', 'action_outputs_16', 'actual_book_2state_48M', 'matched_shadow_2state_48M']
    annual = pd.read_csv(REPORT / 'annual.csv').pivot(index='year', columns='case', values='annualized_excess_sharpe')
    for case in chosen:
        axes[1].plot(annual.index, annual[case], marker='o', label=labels[case])
    axes[1].axhline(0, color='black', linewidth=.6)
    axes[1].set_ylabel('Annualized daily cash-excess Sharpe')
    axes[1].set_xticks(annual.index, [f'{y}*' if y == 2025 else str(y) for y in annual.index], rotation=45)
    axes[1].legend(fontsize=8)
    fig.suptitle('One-year 25-delta puts: no learned variant beats fixed allocation')
    fig.text(.5, .015, 'Explored 2018–2025 history; *2025 partial. Same costs and risk limits. No fresh holdout.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .94))
    fig.savefig(REPORT / 'comparison.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
