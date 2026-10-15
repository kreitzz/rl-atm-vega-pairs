"""Combine retained put-pair and performance60 straddle bandits without refitting.

Primary result sums the two unchanged $10m funded ledgers ($20m initial equity).
Constant-weight return blends are separate allocation diagnostics. They do not
simulate integer resizing, shared inventory limits, netting or reallocation fees.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from paper_code.metrics import sharpe
from scripts.report_atm_p25_combination import (
    aligned_cash, fixed_blend, initial_capital_blend, stats, validation_weight,
)

REPORT = Path('reports/put_straddle_bandit_combination')
PERIODS = [('validation', '2018-01-01', '2021-12-31'),
           ('evaluation', '2022-01-01', '2025-08-29'),
           ('full', '2018-01-01', '2025-08-29')]
PAIR = Path('artifacts/contracts/p25_validation_split/replays')
CAPITAL = Path('data/submission/artifacts/contracts/p25_capital_sensitivity/replays')
BOOKS = {
    'put_bandit': PAIR / 'screened_selected',
    'straddle_bandit': Path('artifacts/contracts/straddle_policy_refinement/cases/performance60/replays/tree18'),
    'put_fixed50': CAPITAL / '10m_fixed50',
    'put_fixed75': PAIR / 'screened_fixed75',
    'put_fixed100': CAPITAL / '10m_fixed100',
    'straddle_fixed50': Path('artifacts/contracts/atm_rv_straddle_fixed50/replays/tree18'),
    'straddle_fixed75': Path('artifacts/contracts/straddle_capped_books/fixed75/replays/tree18'),
    'straddle_fixed100': Path('artifacts/contracts/atm_rv_straddle_full/replays/straddle100'),
}


def run():
    REPORT.mkdir(parents=True, exist_ok=True)
    paths = {name: root / 'daily_nav.parquet' for name, root in BOOKS.items()}
    navs = {name: pd.read_parquet(path, columns=['nav', 'return_', 'excess_return', 'cash_benchmark'],
                                use_threads=False) for name, path in paths.items()}
    pair, other = navs['put_bandit'], navs['straddle_bandit']
    for nav in navs.values():
        assert nav.nav.iloc[0] == 10_000_000
        aligned_cash(pair, nav)
    cash = aligned_cash(pair, other)
    np.testing.assert_allclose(sharpe(pair.excess_return.iloc[1:]), .9604927224428859, atol=1e-12)
    np.testing.assert_allclose(sharpe(other.excess_return.iloc[1:]), .8993012708835477, atol=1e-12)
    weight = validation_weight(pair.excess_return, other.excess_return)
    cases = {
        'put_bandit': pair.excess_return,
        'straddle_bandit': other.excess_return,
        'combined_equal_initial_capital': initial_capital_blend(pair, other, cash),
        'combined_daily_50_50': fixed_blend(pair.excess_return, other.excess_return, .5),
        'combined_validation_risk_balance': fixed_blend(pair.excess_return, other.excess_return, weight),
    }
    # Fixed controls use the same capital convention as the primary combination.
    # All nine combinations are retained; no winner is selected on either period.
    for p in [50, 75, 100]:
        for s in [50, 75, 100]:
            cases[f'fixed_put{p}_straddle{s}'] = initial_capital_blend(
                navs[f'put_fixed{p}'], navs[f'straddle_fixed{s}'], cash)
    records, annual, correlations = [], [], []
    for period, start, stop in PERIODS:
        panel = pd.concat([pair.excess_return, other.excess_return], axis=1).loc[start:stop]
        if panel.index[0] == pair.index[0]:
            panel = panel.iloc[1:]
        correlations.append(dict(period=period, daily_excess_return_correlation=panel.iloc[:, 0].corr(panel.iloc[:, 1])))
        for name, values in cases.items():
            records.append(dict(case=name, period=period, **stats(values, cash, start, stop)))
    for name, values in cases.items():
        for year in range(2018, 2026):
            annual.append(dict(case=name, year=year, **stats(values, cash, f'{year}-01-01', min(f'{year}-12-31', '2025-08-29'))))
    table = pd.DataFrame(records)
    table.to_csv(REPORT / 'results.csv', index=False)
    pd.DataFrame(annual).to_csv(REPORT / 'annual.csv', index=False)
    pd.DataFrame(correlations).to_csv(REPORT / 'correlations.csv', index=False)
    daily = pd.DataFrame(cases)
    daily['cash_return'] = cash
    daily.rename_axis('date').to_parquet(REPORT / 'daily_returns.parquet')
    source_files = list(paths.values()) + [Path(__file__), Path('scripts/report_atm_p25_combination.py'),
        Path('paper_code/metrics.py'), Path('config/p25_validation_2018_2021.json'),
        Path('reports/straddle_policy_refinement/checkpoint.json')]
    protocol = dict(
        primary='Sum of unchanged $10m put-pair and $10m short-straddle funded books. $20m initial capital; weights drift with NAV. No additional leverage or hedge netting.',
        put='Retained screened 30D 25-delta put-pair contextual sizing policy.',
        straddle='Bottom-decile RV-minus-IV ATM straddles, performance60 contextual sizing.',
        risk_balance=dict(put_weight=weight, straddle_weight=1-weight,
            estimated_through='2021-12-31', method='Inverse standalone validation volatility; no Sharpe optimization.'),
        diagnostic='Daily 50/50 and risk-balanced saved-return blends do not replay whole contracts, shared account admission, net hedges or reallocation fees. Validation/full risk-balanced results use weights estimated from the entire validation period and are descriptive.',
        controls='All nine combinations of put and straddle fixed 50/75/100%, each summing two unchanged $10m books.',
        selection='No new model fits, thresholds or allocation searches. Source policies were selected using previously inspected historical dates; this is not untouched out-of-sample evidence.',
        checks='Identical calendars and cash returns, source Sharpe reproduction, covariance identity and summed-ledger return identity passed.',
        source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files},
    )
    (REPORT / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    primary = table.loc[~table.case.str.startswith('fixed_')]
    text = '\n\n'.join([__doc__, primary.to_string(index=False),
        pd.DataFrame(correlations).to_string(index=False), json.dumps(protocol, indent=2)])
    (REPORT / 'FINDINGS.txt').write_text(text + '\n')
    print(primary[['case', 'period', 'annualized_excess_sharpe', 'annualized_growth_relative_to_cash_pct']].to_string(index=False), flush=True)
    print(pd.DataFrame(correlations).to_string(index=False), flush=True)
    print('Validation risk-balanced put weight:', weight, flush=True)


def charts():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    table = pd.read_csv(REPORT / 'results.csv')
    names = ['put_bandit', 'straddle_bandit', 'combined_equal_initial_capital', 'combined_validation_risk_balance']
    labels = ['Put-pair bandit', 'Straddle bandit', 'Two funded books\n50/50 initial equity', 'Risk-balanced blend*']
    fig, ax = plt.subplots(figsize=(10, 5.3))
    for i, (period, label, color) in enumerate([('validation', '2018–2021', '#6483ab'), ('evaluation', '2022–Aug 2025', '#dc8843')]):
        values = table.loc[table.period.eq(period)].set_index('case').annualized_excess_sharpe.reindex(names)
        bars = ax.bar(np.arange(4) + (i-.5)*.36, values, .36, label=label, color=color)
        ax.bar_label(bars, fmt='%.3f', padding=3)
    ax.set_xticks(np.arange(4), labels)
    ax.set_ylabel('Annualized excess Sharpe')
    ax.set_title('Combining the retained put-pair and straddle sizing policies')
    ax.set_ylim(0, max(table.annualized_excess_sharpe)*1.18)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False)
    fig.text(.02, .02, '*Weights use 2018–2021 volatility; saved-return diagnostic, without a shared inventory replay.', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(REPORT / 'comparison.png', dpi=160)
    plt.close(fig)
    print('Saved combination chart', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['run', 'charts'], default='run')
    args = parser.parse_args()
    {'run': run, 'charts': charts}[args.phase]()
