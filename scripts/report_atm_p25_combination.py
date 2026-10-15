"""Combine frozen put-pair and ATM RV-minus-IV return histories, one small worker.

Equal-initial-capital sums two unchanged $10m ledgers. Constant-weight blends
are diversification diagnostics: no integer resizing or shared inventory replay.
Risk-balanced weights use validation volatility, then remain fixed during test.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from paper_code.metrics import paired_interval, sharpe

ROOT = Path('reports/atm_p25_combination')
CAPITAL = 10_000_000
PAIR = Path('artifacts/contracts/p25_validation_split/replays/screened_selected')
ATM = Path('artifacts/contracts/atm_rv_bandit_bottom10/replays')
PERIODS = [('validation', '2018-01-01', '2021-12-31'),
           ('test', '2022-01-01', '2025-08-29'),
           ('full', '2018-01-01', '2025-08-29')]


def aligned_cash(a, b):
    """Never silently drop a missing date or mix different cash benchmarks."""
    pd.testing.assert_index_equal(a.index, b.index)
    assert a.index.is_unique and a.index.is_monotonic_increasing
    cash = a.return_ - a.excess_return
    np.testing.assert_allclose(cash, b.return_ - b.excess_return, atol=1e-12, rtol=0)
    np.testing.assert_allclose(a.cash_benchmark, b.cash_benchmark, atol=1e-7, rtol=0)
    return cash


def validation_weight(a, b):
    """Pair weight giving equal standalone volatility contributions on validation."""
    part = pd.concat([a, b], axis=1).loc['2018-01-01':'2021-12-31'].iloc[1:]
    vol = part.std(ddof=1).to_numpy()
    if not np.isfinite(vol).all() or (vol <= 0).any():
        raise ValueError('Both validation volatilities must be positive')
    return float(vol[1] / vol.sum())


def fixed_blend(a, b, weight):
    result = weight * a + (1 - weight) * b
    # Check the covariance identity independently of summing daily paths.
    panel = pd.concat([a, b], axis=1).iloc[1:]
    w = np.array([weight, 1 - weight])
    expected = np.sqrt(252) * (w @ panel.mean()) / np.sqrt(w @ panel.cov() @ w)
    np.testing.assert_allclose(sharpe(result.iloc[1:]), expected, atol=1e-12)
    return result


def initial_capital_blend(a, b, cash):
    """Exact combined return of the two unchanged, separately funded ledgers."""
    wealth = a.nav + b.nav
    returns = wealth.pct_change().fillna(0) - cash
    lag_weight = a.nav.shift(1) / wealth.shift(1)
    expected = lag_weight * a.return_ + (1 - lag_weight) * b.return_ - cash
    np.testing.assert_allclose(returns.iloc[1:], expected.iloc[1:], atol=1e-12)
    return returns


def stats(returns, cash, start, stop):
    part = returns.loc[start:stop]
    first = returns.index.get_loc(part.index[0])
    valid = part.iloc[1:] if first == 0 else part
    gross = (1 + returns + cash).cumprod()
    benchmark = (1 + cash).cumprod()
    before = max(0, first - 1)
    years = (part.index[-1] - returns.index[before]).days / 365.25
    growth = gross.loc[part.index[-1]] / gross.iloc[before]
    relative = gross / benchmark
    relative_growth = relative.loc[part.index[-1]] / relative.iloc[before]
    drawdown = relative.iloc[before:returns.index.get_loc(part.index[-1]) + 1]
    return dict(annualized_excess_sharpe=sharpe(valid),
                annualized_total_account_return_pct=100 * (growth**(1 / years) - 1),
                annualized_growth_relative_to_cash_pct=100 * (relative_growth**(1 / years) - 1),
                annualized_excess_volatility_pct=100 * np.sqrt(252) * valid.std(ddof=1),
                max_cash_relative_drawdown_pct=100 * (drawdown / drawdown.cummax() - 1).min())


def run():
    ROOT.mkdir(parents=True, exist_ok=True)
    columns = ['nav', 'return_', 'excess_return', 'cash_benchmark']
    path = PAIR / 'daily_nav.parquet'
    pair = pd.read_parquet(path, columns=columns, use_threads=False)
    assert pair.nav.iloc[0] == CAPITAL
    expected = json.loads((PAIR / 'status.json').read_text())['annualized_excess_sharpe']
    np.testing.assert_allclose(sharpe(pair.excess_return.iloc[1:]), expected, atol=1e-12)
    sources = [path, PAIR / 'status.json', Path('config/p25_validation_2018_2021.json')]
    records, annual, correlations, weights, intervals, daily = [], [], [], [], [], []
    for variant in ['tree18', 'fixed200']:
        path = ATM / variant / 'test_nav.parquet'
        other = pd.read_parquet(path, columns=columns, use_threads=False)
        assert other.nav.iloc[0] == CAPITAL
        expected = json.loads((ATM / variant / 'test_status.json').read_text())['annualized_excess_sharpe']
        np.testing.assert_allclose(sharpe(other.excess_return.iloc[1:]), expected, atol=1e-12)
        cash = aligned_cash(pair, other)
        a, b = pair.excess_return, other.excess_return
        weight = validation_weight(a, b)
        weights.append(dict(atm_variant=variant, pair_weight=weight, atm_weight=1-weight,
                            estimated_through='2021-12-31', method='equal validation volatility contribution'))
        cases = dict(pair_only=a, atm_only=b,
                     equal_initial_capital=initial_capital_blend(pair, other, cash),
                     equal_capital_daily=fixed_blend(a, b, .5),
                     validation_risk_balanced=fixed_blend(a, b, weight))
        for period, start, stop in PERIODS:
            subset = pd.concat([a, b], axis=1).loc[start:stop]
            if period != 'test':subset = subset.iloc[1:]
            correlations.append(dict(atm_variant=variant, period=period,
                daily_excess_return_correlation=subset.iloc[:, 0].corr(subset.iloc[:, 1])))
            for case, values in cases.items():
                records.append(dict(atm_variant=variant, case=case, period=period,
                                    **stats(values, cash, start, stop)))
        for case, values in cases.items():
            daily.append(pd.DataFrame(dict(atm_variant=variant, case=case, excess_return=values,
                cash_return=cash, relative_wealth=(1+values+cash).cumprod()/(1+cash).cumprod())).rename_axis('date').reset_index())
            for year in range(2018, 2026):
                annual.append(dict(atm_variant=variant, case=case, year=year,
                    **stats(values, cash, f'{year}-01-01', min(f'{year}-12-31', '2025-08-29'))))
        for name, source in [('pair_only', a), ('atm_only', b)]:
            intervals.append(dict(atm_variant=variant, control=name, period='test',
                **paired_interval(cases['validation_risk_balanced'].loc['2022-01-01':],
                                  source.loc['2022-01-01':], block=63, draws=1000)))
        sources += [path, ATM / variant / 'test_status.json']
    results = pd.DataFrame(records)
    results.to_csv(ROOT / 'results.csv', index=False)
    pd.DataFrame(annual).to_csv(ROOT / 'annual.csv', index=False)
    pd.DataFrame(correlations).to_csv(ROOT / 'correlations.csv', index=False)
    pd.DataFrame(weights).to_csv(ROOT / 'weights.csv', index=False)
    pd.DataFrame(intervals).to_csv(ROOT / 'sharpe_intervals.csv', index=False)
    pd.concat(daily, ignore_index=True).to_parquet(ROOT / 'daily_returns.parquet', index=False)
    sources += [Path(__file__), Path('config/atm_rv_bandit_bottom10.json')]
    protocol = dict(pair='Current retained 27-input screened 30D 25-delta put-pair policy',
        atm='Bottom-10% trailing RV-minus-IV ATM short calls: validation-selected bandit and fixed200 control shown separately',
        selection='No new model fitting or test-period weight optimization',
        weights=weights, equal_initial_capital='Exact sum of two unchanged $10m books, $20m total initial equity',
        constant_weight_blends='Daily rebalanced saved-return diagnostics; no integer resizing, shared limits, hedge netting or reallocation costs simulated',
        accounting='Source execution costs retained. Both books subtract the identical cash return; weighted excess returns subtract cash once because weights sum to one.',
        limitation='Previously examined historical dates. Validation-fitted weights are known only after 2021; their earlier/full scores are descriptive. Fixed200 is a separate comparison, not a newly validation-selected replacement.',
        source_sharpes_reproduced=True, identical_calendar_and_cash_verified=True,
        covariance_identity_and_two_ledger_return_verified=True,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (ROOT / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (ROOT / 'FINDINGS.txt').write_text(__doc__+'\n\n'+results.to_string(index=False)+'\n\n'+json.dumps(protocol,indent=2)+'\n')
    print(results[['atm_variant','case','period','annualized_excess_sharpe','annualized_growth_relative_to_cash_pct']].to_string(index=False), flush=True)
    print(pd.DataFrame(correlations).to_string(index=False), flush=True)
    print(pd.DataFrame(weights).to_string(index=False), flush=True)


def charts():
    import os
    os.environ.setdefault('MPLCONFIGDIR', 'artifacts/contracts/atm_rv_persistence/matplotlib_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result = pd.read_csv(ROOT / 'results.csv')
    cases = ['pair_only', 'atm_only', 'equal_initial_capital', 'validation_risk_balanced']
    names = ['Put pair', 'ATM calls', '50/50 initial capital', 'Validation risk balance']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, variant in zip(axes, ['tree18', 'fixed200']):
        for i, case in enumerate(cases):
            part = result.loc[result.atm_variant.eq(variant) & result.case.eq(case)].set_index('period')
            ax.bar(np.arange(2)+(i-1.5)*.18, part.loc[['validation','test'],'annualized_excess_sharpe'], width=.18, label=names[i])
        ax.set_xticks([0,1], ['2018–2021 validation','2022–Aug 2025 test'], fontsize=9)
        ax.set_title('ATM bandit' if variant=='tree18' else 'ATM fixed 200%')
        ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Annualized excess Sharpe')
    axes[1].legend(fontsize=8)
    fig.suptitle('Frozen put-pair strategy combined with the ATM RV−IV signal')
    fig.tight_layout();fig.savefig(ROOT / 'comparison.png',dpi=140);plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['run','charts','test'], default='run')
    phase = parser.parse_args().phase
    if phase == 'test':
        import unittest
        suite = unittest.defaultTestLoader.loadTestsFromNames(['tests.test_atm_p25_combination','tests.test_research_memory_guard'])
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():raise SystemExit(1)
    else:{'run':run, 'charts':charts}[phase]()


if __name__ == '__main__':main()
