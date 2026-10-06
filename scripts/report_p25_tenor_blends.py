"""Small saved-return diversification diagnostic; no retraining or weight search."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPORT = Path('reports/p25_tenor_blends')
LONG = Path('artifacts/contracts/p25_long_expiries/P/replays')
BASELINES = {
    'frozen_30D': Path('artifacts/contracts/p25_capital_sensitivity/replays/10m_learned'),
    'managed_30D': Path('artifacts/contracts/p25_long_management/replays/30D_manage_36M'),
    'matched_cutoff_30D_hold': LONG / '30D_hold',
    'matched_cutoff_30D_managed': LONG / '30D_manage_36M',
}


def sharpe(returns):
    return float(np.sqrt(252) * returns.mean() / returns.std(ddof=1))


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    sources = {}
    books = {}
    folders = dict(BASELINES)
    folders.update({f'{tenor}D_{mode}': LONG / f'{tenor}D_{mode}'
                    for tenor in (180, 365) for mode in ('hold', 'manage_36M')})
    for label, folder in folders.items():
        path = folder / 'daily_nav.parquet'
        nav = pd.read_parquet(path, columns=['return_', 'excess_return', 'cash_benchmark'])
        assert nav.index.is_unique and nav.index.is_monotonic_increasing
        assert np.isfinite(nav.to_numpy()).all()
        status_path = folder / 'status.json'
        status = json.loads(status_path.read_text())
        assert np.isclose(sharpe(nav.excess_return.iloc[1:]),
                          status['annualized_excess_sharpe'], atol=1e-12)
        books[label] = nav
        for source in (path, status_path):
            sources[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()

    summaries, annual, daily = [], [], []
    triple_summaries, triple_annual, triple_daily = [], [], []
    for baseline in BASELINES:
        a = books[baseline]
        for tenor in (180, 365):
            for mode in ('hold', 'manage_36M'):
                other = f'{tenor}D_{mode}'
                b = books[other]
                pd.testing.assert_index_equal(a.index, b.index)
                np.testing.assert_allclose(a.cash_benchmark, b.cash_benchmark, rtol=1e-12)
                # Same cash return for both books: excess-return averaging is
                # exactly equivalent to averaging funded returns then subtracting cash.
                x, y = a.excess_return.iloc[1:], b.excess_return.iloc[1:]
                combined = .5 * x + .5 * y
                smaller_longer_allocation = .8 * x + .2 * y
                np.testing.assert_allclose(combined,
                    (.5*a.return_ + .5*b.return_ -
                     (a.return_ - a.excess_return)).iloc[1:], atol=1e-15)
                case = baseline + '+' + other
                summaries.append(dict(case=case, baseline=baseline, longer_strategy=other,
                    start=str(x.index[0].date()), end=str(x.index[-1].date()), sessions=len(x),
                    baseline_sharpe=sharpe(x), longer_sharpe=sharpe(y),
                    daily_excess_return_correlation=float(x.corr(y)),
                    equal_capital_combined_sharpe=sharpe(combined),
                    eighty_twenty_combined_sharpe=sharpe(smaller_longer_allocation),
                    baseline_weight=.5, longer_weight=.5))
                frame = pd.DataFrame(dict(baseline=x, longer=y, combined=combined,
                                          eighty_twenty=smaller_longer_allocation))
                daily.append(frame.assign(case=case).reset_index())
                for year, g in frame.groupby(frame.index.year):
                    annual.append(dict(case=case, year=int(year), sessions=len(g),
                        baseline_sharpe=sharpe(g.baseline), longer_sharpe=sharpe(g.longer),
                        combined_sharpe=sharpe(g.combined),
                        eighty_twenty_sharpe=sharpe(g.eighty_twenty),
                        daily_excess_return_correlation=float(g.baseline.corr(g.longer))))

        for six_month_mode in ('hold', 'manage_36M'):
            for one_year_mode in ('hold', 'manage_36M'):
                six_month = f'180D_{six_month_mode}'
                one_year = f'365D_{one_year_mode}'
                legs = pd.DataFrame({label: books[label].excess_return.iloc[1:]
                                     for label in (baseline, six_month, one_year)})
                assert np.isfinite(legs.to_numpy()).all()
                for allocation, weights in [('equal_thirds', np.full(3, 1/3)),
                                            ('80_10_10', np.array([.8, .1, .1]))]:
                    combined = legs @ weights
                    # Independent covariance identity confirms weighted-return statistics.
                    expected = np.sqrt(252) * (weights @ legs.mean().to_numpy()) / np.sqrt(
                        weights @ legs.cov().to_numpy() @ weights)
                    assert np.isclose(sharpe(combined), expected, atol=1e-12)
                    case = '+'.join(legs.columns) + ':' + allocation
                    triple_summaries.append(dict(case=case, baseline=baseline,
                        six_month_strategy=six_month, one_year_strategy=one_year,
                        allocation=allocation, baseline_weight=weights[0],
                        six_month_weight=weights[1], one_year_weight=weights[2],
                        baseline_sharpe=sharpe(legs[baseline]), combined_sharpe=sharpe(combined),
                        six_month_one_year_correlation=float(legs[six_month].corr(legs[one_year])),
                        annualized_excess_mean_pct=100*252*combined.mean(),
                        annualized_excess_volatility_pct=100*np.sqrt(252)*combined.std(ddof=1)))
                    triple_daily.append(pd.DataFrame(dict(case=case, excess_return=combined)).reset_index())
                    for year, returns in combined.groupby(combined.index.year):
                        triple_annual.append(dict(case=case, year=int(year),
                            combined_sharpe=sharpe(returns), sessions=len(returns)))

    result = pd.DataFrame(summaries)
    result.to_csv(REPORT / 'comparison.csv', index=False)
    pd.DataFrame(annual).to_csv(REPORT / 'annual.csv', index=False)
    pd.concat(daily, ignore_index=True).to_csv(REPORT / 'daily_excess_returns.csv', index=False)
    triples = pd.DataFrame(triple_summaries)
    triples.to_csv(REPORT / 'three_strategy_comparison.csv', index=False)
    pd.DataFrame(triple_annual).to_csv(REPORT / 'three_strategy_annual.csv', index=False)
    pd.concat(triple_daily, ignore_index=True).to_csv(REPORT / 'three_strategy_daily.csv', index=False)
    return_panel = pd.DataFrame({label: nav.excess_return.iloc[1:] for label, nav in books.items()})
    return_panel.corr().to_csv(REPORT / 'correlations.csv')
    diagnostics = pd.DataFrame([dict(strategy=label, sharpe=sharpe(returns),
        annualized_excess_mean_pct=100*252*returns.mean(),
        annualized_excess_volatility_pct=100*np.sqrt(252)*returns.std(ddof=1))
        for label, returns in return_panel.items()])
    diagnostics.to_csv(REPORT / 'individual_return_diagnostics.csv', index=False)
    sources[__file__] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (REPORT / 'audit.json').write_text(json.dumps(dict(
        method='50/50 and 80/20 two-strategy; equal-thirds and 80/10/10 three-strategy daily-rebalanced saved funded-equity return blends, less common cash return',
        annualization='sqrt(252), sample daily standard deviation, first initial-account row excluded',
        fifty_fifty_chosen_before_calculation=True,
        eighty_twenty='Additional descriptive allocation after initial equal-weight calculation; not a validated selected allocation',
        weight_optimization=False, retraining=False,
        three_strategy_weights_fixed_before_calculation=True,
        three_strategy_covariance_identity_verified=True,
        source_sharpes_reproduced=True, identical_calendar_and_cash_benchmark=True,
        shared_portfolio_replayed=False,
        limitations='Existing costs retained; no combined hedge netting, shared admission/risk limits, allocation trading costs, or integer sizing replay. Previously explored history, not a fresh holdout.',
        source_sha256=sources), indent=2))
    (REPORT / 'findings.txt').write_text(
        '25-delta put maturity diversification, 2018 through August 2025.\n'
        'Annualized daily cash-excess Sharpe. 50/50 and descriptive 80/20 funded-capital weights; no weight optimization. 80/20 evaluated after the initial equal-weight calculation.\n'
        'frozen_30D and managed_30D retain the full original entry calendar. matched_cutoff controls end entries in July 2024 like the longer-expiry study.\n'
        'This is a blend of existing after-cost returns, not a replay of shared cash, hedge netting, risk admission or integer inventories. No new model or independent holdout.\n\n'
        + result.to_string(index=False) + '\n\nTHREE-STRATEGY BLENDS\n'
        + triples.to_string(index=False) + '\n\nINDIVIDUAL RETURN DIAGNOSTICS\n'
        + diagnostics.to_string(index=False) + '\n')
    print(triples.loc[triples.baseline.isin(['frozen_30D', 'managed_30D'])].to_string(index=False))
    print(diagnostics.to_string(index=False))


if __name__ == '__main__':
    main()
