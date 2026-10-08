"""Analysis of frozen opportunity-filter replays; never fits or selects a policy.

CLI manifest: {"reference": "fixed100", "replays": {"case": "folder"},
"output": "folder", "predictions": "optional.parquet", "pair_daily":
{"case": "optional.parquet"}, "shared_financing": {"case": "optional.parquet"}}.
Predictions contain case,date,pair,prediction,target,selected and optionally end.
The target must use the same normalization as prediction. Prediction bins are
retrospective diagnostics, not thresholds used in the replay.
"""
import os
for _key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'ARROW_NUM_THREADS']:
    os.environ[_key] = '1'
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

PERIODS = [('validation', '2018-01-01', '2021-12-31'),
           ('evaluation', '2022-01-01', '2025-08-29'),
           ('full', '2018-01-01', '2025-08-29')]


def sharpe(values):
    a = np.asarray(values, float)
    if len(a) < 2 or not np.isfinite(a).all():
        raise ValueError('Sharpe requires at least two finite observations')
    sd = a.std(ddof=1)
    return float(np.sqrt(252) * a.mean() / sd) if sd > 1e-12 else 0.


def paired_sharpe_interval(candidate, reference, block=63, replications=2000,
                           seed=2026, rest=lambda: None):
    """Paired circular blocks preserve common calendar shocks in both books."""
    if not candidate.index.equals(reference.index):
        raise ValueError('Paired inference requires identical date indexes')
    if not candidate.index.is_unique or not candidate.index.is_monotonic_increasing:
        raise ValueError('Return dates must be unique and increasing')
    a, b = candidate.to_numpy(float), reference.to_numpy(float)
    if block < 1 or block > len(a) or replications < 2:
        raise ValueError('Invalid block length or replication count')
    point = sharpe(a) - sharpe(b)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replications)
    for j in range(replications):
        starts = rng.integers(0, len(a), size=int(np.ceil(len(a) / block)))
        ix = ((starts[:, None] + np.arange(block)) % len(a)).ravel()[:len(a)]
        estimates[j] = sharpe(a[ix]) - sharpe(b[ix])
        if j % 100 == 99:
            rest()
    return dict(point=point, lower95=float(np.quantile(estimates, .025)),
                upper95=float(np.quantile(estimates, .975)), block_sessions=block,
                replications=replications, seed=seed)


def period_metrics(nav, decisions, start, stop):
    """Same anchoring and cash-excess definitions as the previous P25 audits."""
    part = nav.loc[start:stop]
    if part.empty:
        raise ValueError('Requested period has no NAV observations')
    first = nav.index.get_loc(part.index[0])
    before = nav.iloc[max(0, first - 1)]
    returns = part.excess_return.iloc[1:] if first == 0 else part.excess_return
    ratio = np.r_[before.nav / before.cash_benchmark, (part.nav / part.cash_benchmark).to_numpy()]
    entered = decisions.status.eq('entered') & decisions.entry_date.between(part.index[0], part.index[-1])
    proposed = decisions.signal_date.between(part.index[0], part.index[-1])
    return dict(start=str(part.index[0].date()), end=str(part.index[-1].date()),
        annualized_excess_sharpe=sharpe(returns),
        profit_above_cash=float((part.nav.iloc[-1] - part.cash_benchmark.iloc[-1]) - (before.nav - before.cash_benchmark)),
        proposals=int(proposed.sum()), entries=int(entered.sum()),
        active_entry_pairs=int(decisions.loc[entered, 'pair'].nunique()),
        average_reserve_pct=float(100 * (part.capital_reserve / part.nav).mean()),
        peak_reserve_pct=float(100 * (part.capital_reserve / part.nav).max()),
        max_cash_relative_drawdown_pct=float(100 * (ratio / np.maximum.accumulate(ratio) - 1).min()),
        post_entry_risk_breach_days=int(part.post_trade_risk_breach.sum()))


def prediction_diagnostics(predictions, asof='2025-08-29', bins=5):
    """No imputing missing outcomes; unresolved outcomes explicitly counted."""
    required = {'case', 'date', 'pair', 'prediction', 'target', 'selected'}
    if not required.issubset(predictions.columns):
        raise ValueError(f'Missing prediction fields: {required-set(predictions.columns)}')
    p = predictions.copy()
    p['date'] = pd.to_datetime(p.date)
    if p.duplicated(['case', 'date', 'pair']).any():
        raise ValueError('Predictions must contain one chosen direction per case/date/pair')
    if not p.selected.isin([True, False, 0, 1]).all():
        raise ValueError('selected must be boolean')
    p['matured'] = np.isfinite(p.target)
    if 'end' in p:
        p['end'] = pd.to_datetime(p.end)
        p['matured'] &= p.end.notna() & p.end.le(pd.Timestamp(asof))
    rows, coverage = [], []
    for case, own in p.groupby('case'):
        for period, start, stop in PERIODS:
            full = own.loc[own.date.between(start, stop)]
            valid = full.loc[full.matured & np.isfinite(full.prediction)].copy()
            coverage.append(dict(case=case, period=period, candidates=len(full),
                finite_predictions=int(np.isfinite(full.prediction).sum()),
                matured_outcomes=int(full.matured.sum()), diagnostic_rows=len(valid),
                selected=int(full.selected.sum())))
            if valid.empty:
                continue
            # Cut only on scores, never outcomes; equal scores stay together.
            valid['score_bin'] = pd.qcut(valid.prediction, q=bins, labels=False, duplicates='drop')
            if valid.score_bin.isna().all():
                valid['score_bin'] = 0
            for label, group in [('all', valid), ('selected', valid.loc[valid.selected.astype(bool)]),
                                 ('rejected', valid.loc[~valid.selected.astype(bool)])] + [
                                     (f'bin_{int(k)+1}', g) for k, g in valid.groupby('score_bin')]:
                if group.empty:
                    continue
                error = group.prediction - group.target
                rows.append(dict(case=case, period=period, group=label, observations=len(group),
                    signal_dates=group.date.nunique(), pairs=group.pair.nunique(),
                    prediction_min=group.prediction.min(), prediction_max=group.prediction.max(),
                    mean_prediction=group.prediction.mean(), mean_target=group.target.mean(),
                    median_target=group.target.median(), target_p05=group.target.quantile(.05),
                    target_p95=group.target.quantile(.95), positive_fraction=group.target.gt(0).mean(),
                    mean_error=error.mean(), rmse=float(np.sqrt((error**2).mean())),
                    selected_fraction=group.selected.mean()))
    return pd.DataFrame(rows), pd.DataFrame(coverage)


def pair_diagnostics(case, daily, financing, nav):
    """Requires exact daily attribution, not sum of option fills as a P&L proxy."""
    if not daily.index.equals(nav.index) or not financing.index.equals(nav.index):
        raise ValueError('Attribution and NAV dates differ')
    if isinstance(financing, pd.DataFrame):
        if financing.shape[1] != 1:
            raise ValueError('Expected one shared-financing column')
        financing = financing.iloc[:, 0]
    expected = (nav.nav-nav.cash_benchmark).diff().fillna(0)
    np.testing.assert_allclose(daily.sum(axis=1)+financing, expected, atol=1e-5, rtol=0)
    rows, concentration = [], []
    for period, start, stop in PERIODS:
        total = daily.loc[start:stop].sum()
        positive = total.clip(lower=0).sort_values(ascending=False)
        positive_sum, net = float(positive.sum()), float(total.sum())
        concentration.append(dict(case=case, period=period, positive_pairs=int(total.gt(0).sum()),
            negative_pairs=int(total.lt(0).sum()), net_pair_pnl=net,
            shared_financing_vs_cash=float(financing.loc[start:stop].sum()),
            top3_share_positive_profit=float(positive.iloc[:3].sum()/positive_sum) if positive_sum else np.nan,
            top3_share_net_pair_pnl=float(positive.iloc[:3].sum()/net) if net > 0 else np.nan,
            positive_profit_hhi=float(((positive/positive_sum)**2).sum()) if positive_sum else np.nan))
        for pair, pnl in total.items():
            rows.append(dict(case=case, period=period, pair=pair, net_before_shared_financing=float(pnl)))
    return rows, concentration


def analyze(replays, output, reference='fixed100', predictions=None, pair_daily=None,
            shared_financing=None, replications=2000, rest=lambda: None):
    """Write comparisons without picking a winning policy from their results."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if reference not in replays:
        raise ValueError('Reference replay is missing')
    books, periods, annual, admissions, pair_rows, concentrations = {}, [], [], [], [], []
    pair_admissions = []
    for case, folder in replays.items():
        folder = Path(folder)
        nav = pd.read_parquet(folder/'daily_nav.parquet', use_threads=False)
        decisions = pd.read_csv(folder/'decisions.csv', parse_dates=['signal_date', 'entry_date'])
        if not nav.index.is_unique or not nav.index.is_monotonic_increasing:
            raise ValueError(f'Invalid NAV calendar: {case}')
        books[case] = nav
        for period, start, stop in PERIODS:
            periods.append(dict(case=case, period=period, **period_metrics(nav, decisions, start, stop)))
            own = decisions.loc[decisions.entry_date.between(start, stop)]
            admissions.extend(dict(case=case, period=period, status=k, count=int(v))
                              for k, v in own.status.value_counts().items())
            for pair, group in own.groupby('pair'):
                pair_admissions.append(dict(case=case, period=period, pair=pair,
                    submitted=len(group), entries=int(group.status.eq('entered').sum()),
                    distinct_signal_dates=group.signal_date.nunique()))
        for year in range(2018, 2026):
            annual.append(dict(case=case, year=year, **period_metrics(nav, decisions, f'{year}-01-01', f'{year}-12-31')))
        if pair_daily and case in pair_daily:
            if not shared_financing or case not in shared_financing:
                raise ValueError('Pair attribution needs shared-financing reconciliation')
            r, c = pair_diagnostics(case, pd.read_parquet(pair_daily[case], use_threads=False),
                pd.read_parquet(shared_financing[case], use_threads=False), nav)
            pair_rows.extend(r); concentrations.extend(c)
        rest()
    intervals = []
    for case, nav in books.items():
        if not nav.index.equals(books[reference].index):
            raise ValueError(f'Calendar differs from reference: {case}')
        if case == reference:
            continue
        for period, start, stop in PERIODS[:2]:
            a = nav.loc[start:stop, 'excess_return']
            b = books[reference].loc[start:stop, 'excess_return']
            if a.index[0] == nav.index[0]:
                a, b = a.iloc[1:], b.iloc[1:]
            for block in [63, 126]:
                intervals.append(dict(case=case, reference=reference, period=period,
                    **paired_sharpe_interval(a, b, block=block, replications=replications, rest=rest)))
    outputs = dict(periods=periods, annual=annual, admissions=admissions,
                   pair_admissions=pair_admissions, sharpe_intervals=intervals)
    if pair_rows:
        outputs.update(pair_periods=pair_rows, concentration=concentrations)
    for name, rows in outputs.items():
        pd.DataFrame(rows).to_csv(output/f'{name}.csv', index=False)
    if predictions is not None:
        pred = predictions if isinstance(predictions, pd.DataFrame) else pd.read_parquet(predictions, use_threads=False)
        bins, coverage = prediction_diagnostics(pred)
        bins.to_csv(output/'prediction_bins.csv', index=False)
        coverage.to_csv(output/'prediction_coverage.csv', index=False)
    notes = dict(reference=reference, cases=list(replays), replications=replications,
        blocks=[63, 126], paired_common_calendar=True, interval_type='circular-block percentile',
        uncertainty_scope='Conditional on frozen fitted policies; no correction for research search or multiple comparisons.',
        historical_status='Exploratory: evaluation history was inspected before this experiment.',
        pair_attribution_cases=list((pair_daily or {}).keys()),
        prediction_bins='Retrospective descriptive score quantiles; not fitted entry thresholds; unresolved targets excluded.')
    (output/'analysis_protocol.json').write_text(json.dumps(notes, indent=2)+'\n', encoding='utf-8')
    return pd.DataFrame(periods)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    config = json.loads(args.manifest.read_text())
    # Reuse the worker CPU/RSS limiter only at execution, keeping helper imports light.
    from scripts.run_p25_representation import Budget
    budget = Budget()
    print(analyze(**config, rest=budget.rest).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
