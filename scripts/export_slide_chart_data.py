"""Export compact, reconciled chart data for the slide 8–15 specification."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import shutil
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'slides/chart_data'
ZIP = ROOT / 'slides/slide_chart_inputs.zip'


def annualized_sharpe(returns):
    return float(np.sqrt(252) * returns.mean() / returns.std(ddof=1))


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'sources').mkdir(exist_ok=True)
    sources = {}

    def track(relative):
        path = ROOT / relative
        sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    track('scripts/export_slide_chart_data.py')
    original_comparison = 'reports/p25_capital_sensitivity/comparison.csv'
    original_training = 'reports/p25_variance_validation/initial_train_walk_forward.csv'
    shutil.copy2(track(original_comparison), OUT / 'sources/p25_capital_sensitivity_comparison.csv')
    shutil.copy2(track(original_training), OUT / 'sources/initial_train_walk_forward.csv')
    comparison = pd.read_csv(ROOT / original_comparison)
    cases = {'10m_fixed50': 'Fixed 50%', '10m_fixed100': 'Fixed 100%',
             '10m_learned': 'Learned 50/75/100%'}
    selected = comparison.set_index('case').loc[list(cases)].reset_index()
    assert len(selected) == 3 and selected.funded_equity.eq(10_000_000).all()
    selected.insert(1, 'label', selected.case.map(cases))
    fields = ['case', 'label', 'funded_equity', 'annualized_excess_sharpe', 'entries',
              'active_pairs', 'net_excess_pnl', 'funded_cagr_pct',
              'relative_to_cash_cagr_pct', 'maximum_nav_drawdown_pct',
              'maximum_relative_to_cash_drawdown_pct']
    selected[fields].to_csv(OUT / 'main_sizing_comparison_10m.csv', index=False)
    annual_source = pd.read_csv(track('reports/p25_capital_sensitivity/annual.csv'))
    official_annual = annual_source.loc[annual_source.case.isin(cases)].copy()
    navs, daily, annual, decisions, example_events = {}, [], [], [], []
    wealth = None
    example_pair = '101121__108321'
    for case, label in cases.items():
        base = f'artifacts/contracts/p25_capital_sensitivity/replays/{case}'
        nav = pd.read_parquet(track(base + '/daily_nav.parquet'))
        assert nav.index.is_unique and nav.index.is_monotonic_increasing
        assert float(nav.nav.iloc[0]) == 10_000_000
        assert float(nav.cash_benchmark.iloc[0]) == 10_000_000
        if navs:
            pd.testing.assert_index_equal(nav.index, next(iter(navs.values())).index)
        navs[case] = nav
        expected = selected.loc[selected.case.eq(case)].iloc[0]
        score = annualized_sharpe(nav.excess_return.iloc[1:])
        assert np.isclose(score, expected.annualized_excess_sharpe, atol=1e-12)
        excess_dollars = nav.nav - nav.cash_benchmark
        cash_daily = nav.cash_benchmark.diff().fillna(0.)
        net_daily = nav.pnl - cash_daily
        np.testing.assert_allclose(net_daily.cumsum(), excess_dollars, atol=1e-7)
        assert np.isclose(excess_dollars.iloc[-1], expected.net_excess_pnl, atol=1e-7)
        data = pd.DataFrame(index=nav.index)
        data['case'], data['label'] = case, label
        for col in ['nav', 'cash_benchmark', 'return_', 'excess_return', 'pnl']:
            data[col] = nav[col]
        data.rename(columns={'nav': 'nav_usd', 'cash_benchmark': 'cash_benchmark_usd',
                             'return_': 'funded_daily_return', 'excess_return': 'daily_cash_excess_return',
                             'pnl': 'daily_funded_pnl_usd'}, inplace=True)
        data['daily_cash_benchmark_pnl_usd'] = cash_daily
        data['daily_pnl_above_cash_usd'] = net_daily
        data['cumulative_funded_pnl_usd'] = nav.nav - 10_000_000
        data['cumulative_pnl_above_cash_usd'] = excess_dollars
        data['cash_relative_wealth_pct'] = 100 * (nav.nav / nav.cash_benchmark - 1)
        data['is_initialization_row'] = data.index == data.index[0]
        daily.append(data.reset_index())
        if wealth is None:
            wealth = pd.DataFrame(index=nav.index)
        wealth[case + '_cumulative_excess_pnl_usd'] = excess_dollars
        wealth[case + '_cash_relative_wealth_pct'] = data.cash_relative_wealth_pct
        prior_nav = nav.nav.shift(1).fillna(10_000_000)
        prior_cash = nav.cash_benchmark.shift(1).fillna(10_000_000)
        for year, g in nav.groupby(nav.index.year):
            r = g.excess_return.iloc[1:] if g.index[0] == nav.index[0] else g.excess_return
            funded = 100 * (g.nav.iloc[-1] / prior_nav.loc[g.index[0]] - 1)
            cash = 100 * (g.cash_benchmark.iloc[-1] / prior_cash.loc[g.index[0]] - 1)
            recorded = official_annual.loc[official_annual.case.eq(case) & official_annual.year.eq(year)]
            assert len(recorded) == 1
            np.testing.assert_allclose([funded, funded-cash, annualized_sharpe(r)],
                recorded[['funded_return_pct', 'return_above_cash_pp', 'annualized_excess_sharpe']].iloc[0], atol=1e-10)
            annual.append(dict(case=case, label=label, year=int(year),
                funded_return_pct=funded, cash_benchmark_return_pct=cash,
                return_above_cash_pp=funded-cash, annualized_excess_sharpe=annualized_sharpe(r),
                net_pnl_above_cash_usd=float(net_daily.loc[g.index].sum()),
                partial_year=int(year)==2025))
        d = pd.read_csv(track(base + '/decisions.csv'), parse_dates=['signal_date', 'entry_date'])
        assert len(d) == 608 and not d.duplicated(['signal_date', 'pair', 'wing']).any()
        if decisions:
            pd.testing.assert_frame_equal(d[['signal_date', 'pair', 'wing']],
                decisions[0][['signal_date', 'pair', 'wing']])
        assert d.status.eq('entered').sum() == expected.entries
        d.insert(0, 'case', case)
        decisions.append(d)
        example_events.append(d.loc[d.pair.eq(example_pair)].copy())
    pd.concat(daily, ignore_index=True).to_csv(OUT / 'daily_pnl_and_returns_10m.csv', index=False)
    wealth.to_csv(OUT / 'cumulative_performance_10m.csv')
    pd.DataFrame(annual).to_csv(OUT / 'yearly_results_10m.csv', index=False)
    pd.concat(decisions, ignore_index=True).to_csv(OUT / 'common_proposals_and_decisions_10m.csv', index=False)

    # Explain the original seed pair without selecting a pair by its profits.
    rates = pd.read_parquet(track('artifacts/contracts/uniform_smile/market/rates.parquet'))
    cal = pd.DatetimeIndex(rates.index)
    series = pd.DataFrame(index=cal)
    columns = ['date', 'cp_flag', 'bucket', 'tenor', 'optionid', 'expiry', 'bs_iv', 'bs_delta']
    for name, sid in [('amd', 101121), ('nvda', 108321)]:
        selections = pd.read_parquet(track(f'artifacts/contracts/uniform_smile/market/assets/{sid}_selection.parquet'), columns=columns)
        selections = selections.loc[selections.cp_flag.eq('P') & selections.bucket.eq('25') & selections.tenor.eq(30)].set_index('date')
        assert selections.index.is_unique
        for source_col, output_col in [('bs_iv', 'iv_decimal'), ('bs_delta', 'put_delta'),
                                       ('optionid', 'selected_optionid'), ('expiry', 'selected_expiry')]:
            series[f'{name}_{output_col}'] = selections[source_col].reindex(cal)
        series[f'{name}_iv_decimal'] = series[f'{name}_iv_decimal'].where(series[f'{name}_iv_decimal'].gt(0))
    series['iv_spread_decimal'] = series.amd_iv_decimal - series.nvda_iv_decimal
    series['iv_spread_vol_points'] = 100 * series.iv_spread_decimal
    series['spread_rank_252'] = series.iv_spread_decimal.rolling(252, min_periods=63).rank(pct=True)
    # These rolling levels are for visual context; rank is the actual signal.
    series['spread_20th_percentile_vol_points'] = series.iv_spread_vol_points.rolling(252, min_periods=63).quantile(.2)
    series['spread_80th_percentile_vol_points'] = series.iv_spread_vol_points.rolling(252, min_periods=63).quantile(.8)
    series['rank_rule_direction'] = np.where(series.spread_rank_252.le(.2), 1,
                                           np.where(series.spread_rank_252.gt(.8), -1, 0))
    observed = pd.read_parquet(track(f'artifacts/contracts/uniform_smile/pairs/{example_pair}/P25_observed.parquet')).set_index('date')
    checks = series.reindex(observed.index)
    good = checks.spread_rank_252.notna()
    np.testing.assert_allclose(checks.loc[good, 'spread_rank_252'], observed.loc[good, 'iv_rank'], atol=1e-12)
    np.testing.assert_allclose((checks.amd_iv_decimal+checks.nvda_iv_decimal)/2, observed.iv_mean, atol=1e-12)
    for name in ['beta', 'correlation', 'eligible', 'context']:
        if name in observed:
            series[name] = observed[name].reindex(cal)
    series['pair'] = example_pair
    series['pair_label'] = 'AMD/NVDA'
    signals = pd.read_parquet(track('artifacts/contracts/p25_feature_search/replays/variance_premium/scored_allocations.parquet'))
    example_signals = signals.loc[signals.pair.eq(example_pair)].set_index('date')
    assert example_signals.index.is_unique
    example = pd.concat(example_events, ignore_index=True)
    assert example.loc[example.case.eq('10m_learned'), 'status'].eq('entered').any()
    series['is_qualified_proposal_date'] = series.index.isin(example_signals.index)
    for name in ['direction', 'learned_level']:
        series[f'proposal_{name}'] = example_signals[name].reindex(cal)
    np.testing.assert_array_equal(series.loc[example_signals.index, 'rank_rule_direction'], example_signals.direction)
    example['direction'] = example.signal_date.map(example_signals.direction)
    example['signal_iv_spread_vol_points'] = example.signal_date.map(series.iv_spread_vol_points)
    example['signal_spread_rank_252'] = example.signal_date.map(series.spread_rank_252)
    for case in cases:
        subset = example.loc[example.case.eq(case)]
        series[case + '_executed_entry'] = series.index.isin(subset.loc[subset.status.eq('entered'), 'entry_date'])
    series.loc['2018-01-01':].to_csv(OUT / 'example_amd_nvda_iv_spread_daily.csv', index_label='date')
    example.to_csv(OUT / 'example_amd_nvda_proposals_and_entries.csv', index=False)

    recency = pd.read_csv(track('reports/p25_recency/comparison.csv')).set_index('model')
    reference_cfg = json.loads(track('config/p25_variance_premium_candidate.json').read_text())
    development = [dict(configuration='Five-year learned sizing', training_months=60,
                        sharpe=float(recency.loc['5Y_reference', 'annualized_excess_sharpe'])),
                   dict(configuration='18-month learned sizing', training_months=18,
                        sharpe=float(recency.loc['18M_floor', 'annualized_excess_sharpe'])),
                   dict(configuration='18-month sizing + variance-premium features', training_months=18,
                        sharpe=reference_cfg['annualized_cash_excess_sharpe'])]
    for row in development:
        row.update(funded_equity=345_000_000, controlled_sizing_only_comparison=False,
                   previously_researched_history=True)
    pd.DataFrame(development).to_csv(OUT / 'development_progression_original_reference.csv', index=False)
    shutil.copy2(ROOT / 'config/p25_variance_premium_candidate.json', OUT / 'sources/retained_strategy_configuration.json')
    teammate = ROOT / 'slides/teammate_only/teammate_paper_slides.pdf'
    if teammate.exists():
        shutil.copy2(track('slides/teammate_only/teammate_paper_slides.pdf'), OUT / 'teammate_paper_slides.pdf')
    guide = '''SLIDE CHART INPUTS — WORKING SPECIFICATION FOR SLIDES 8–15

Use the retained approximately 30-day 25-delta PUT strategy. Main comparison:
$10m equity, own-expiry holding, daily delta hedge, initial integer entry-vega
matching within 10%, same 608 qualified proposals, same actual costs/risk rules.
Actual option commission $0.03; training commission $0.65; option penalty
0.1 vol point per leg; stock hedge execution 1 bp. No new fit or replay here.

FILE MAP
sources/p25_capital_sensitivity_comparison.csv: first cited CSV, copied exactly.
sources/initial_train_walk_forward.csv: second cited CSV, copied exactly.
main_sizing_comparison_10m.csv: curated fixed50/fixed100/learned rows for slide12.
daily_pnl_and_returns_10m.csv: daily NAV, cash benchmark, funded and cash-excess
returns, daily/cumulative P&L in USD; one row per case/date.
cumulative_performance_10m.csv: convenient wide-form curves for all three cases.
yearly_results_10m.csv: funded percentage returns, above-cash percentage points,
annualized excess Sharpe and net above-cash dollars for each year/case.
common_proposals_and_decisions_10m.csv: all 608 proposals per case, including
risk/quote rejections, chosen size and execution status. Executed fills differ.
example_amd_nvda_iv_spread_daily.csv: daily signal-day selected P25 implied
volatility for AMD/NVDA, raw spread in vol points, trailing252 rank and context
levels, qualified proposal dates, directions and actual entry flags.
example_amd_nvda_proposals_and_entries.csv: exact signal/entry dates, sizes,
direction and entered/rejected status for each of the three policies.
development_progression_original_reference.csv: slide13 development path;
original $345m funding reference, not the controlled $10m sizing comparison.
sources/retained_strategy_configuration.json: entry rules, state, actions,
costs, algorithm and research-selection caveats.
teammate_paper_slides.pdf: original slides1–7, if present in the local export.

CHARTS
Slide9/10: AMD/NVDA IV spread or rank; mark qualified SIGNAL dates separately
from ACTUAL entry dates. Direction +1 means long AMD puts/short NVDA puts;
-1 means short AMD puts/long NVDA puts. Negative size is not the convention.
Suggested compact illustration: calendar2019, including Aug12/13 signals
and Aug13/14 executions; AMD/NVDA was the original project seed pair and
was not selected here by its P&L. Full2018–Aug2025 history is supplied.
The IV curve rolls signal-day selected contracts; it is not the held-cohort
daily mark series. Do not fill missing quotes with zero or invent entries.
Rolling quantile spread levels are visual context; the exact rule uses rank
<=.2 or >.8 with at least63 observations in a252-session window. Other
pair/memory/formation/quote/risk checks also apply. Not every extreme trades.

Slide12: Sharpe bars plus cumulative ABOVE-CASH dollars or cash-relative
wealth percentage for all three $10m policies. These are existing results
after execution/hedging/financing, not a newly simulated comparison.
Slide13: original development progression on its original funding reference.
Slide14: compare policies WITHIN training/evaluation rows; training uses
aggregated P&L paths and $0.65 commissions, evaluation the actual shared ledger
and $0.03 commissions. The initial training fit has74 examples, July2016–Dec2017.
Slide15: yearly excess Sharpe or yearly funded/above-cash returns; show losing
years. 2025 is partial through Aug29, with June30 final signals and runoff.

METRIC DEFINITIONS
annualized excess Sharpe = sqrt(252)*mean(daily_cash_excess_return)/sampleSD.
Exclude the explicitly flagged initialization row (Jan2,2018) from Sharpe.
daily_cash_excess_return is a DECIMAL, not a percentage; use the supplied
series. Daily cash-excess return is not cumulative dollar P&L divided by NAV.
cumulative_pnl_above_cash_usd = funded NAV - same-equity cash benchmark.
cash_relative_wealth_pct = 100*(funded NAV / cash benchmark -1).
Yearly funded_return_pct is a calendar-period percentage, not annualized.
return_above_cash_pp = funded calendar return minus cash calendar return,
in percentage points. It differs from cash-relative wealth percentage.

INTERPRETATION
Our main method is contextual tree states + direct-Sharpe CEM allocation,
not Q-learning, and not a new matched supervised-NN comparison. Main learned
sizing Sharpe is1.002988. The1.029 managed checkpoint is a DIFFERENT experiment
and is deliberately excluded from these main curves and yearly tables.
Retrospective component/universe and repeated strategy research remain;
walk-forward fitting is causal but this is not an untouched holdout.
Closing sentence: Our strongest finding is that adaptive sizing improved the
historical risk-adjusted performance of the retained options strategy relative
to constant exposure.
Vendor-derived data remain subject to the existing data-access terms.
'''
    (OUT / 'CHART_GUIDE.txt').write_text(guide)
    audit = dict(main_cases=list(cases), common_proposals_per_case=608,
        funded_equity=10_000_000, daily_returns_and_yearly_metrics_reconciled=True,
        pnl_reconciled_to_cash_benchmark=True, example='AMD/NVDA',
        example_rank_reproduces_saved_states=True, actual_entry_dates_included=True,
        models_or_backtests_rerun=False, managed_checkpoint_in_main_data=False,
        development_funding_reference=345_000_000,
        one_worker=True, cpu_core_fraction=.5,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256=sources)
    (OUT / 'audit.json').write_text(json.dumps(audit, indent=2))
    payloads = {str(path.relative_to(OUT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(OUT.rglob('*')) if path.is_file() and path.name != 'file_hashes.json'}
    (OUT / 'file_hashes.json').write_text(json.dumps(payloads, indent=2))
    for relative, expected in sources.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    with zipfile.ZipFile(ZIP, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT.rglob('*')):
            if path.is_file():
                archive.write(path, arcname=str(Path('slide_chart_inputs') / path.relative_to(OUT)))
    with zipfile.ZipFile(ZIP) as archive:
        assert archive.testzip() is None
    print(json.dumps(dict(zip=str(ZIP), size_bytes=ZIP.stat().st_size,
        files=len(payloads)+1, main_sharpes=dict(zip(selected.case, selected.annualized_excess_sharpe)),
        example_entered_rows=int(example.status.eq('entered').sum()),
        peak_rss_bytes=audit['peak_rss_bytes']), indent=2), flush=True)


def inspect():
    nav = pd.read_parquet(ROOT / 'artifacts/contracts/p25_capital_sensitivity/replays/10m_learned/daily_nav.parquet')
    print('NAV columns:', list(nav.columns), flush=True)
    print('NAV endpoints:', nav.iloc[[0, -1]][['nav', 'cash_benchmark', 'excess_return']].to_string(), flush=True)
    definitions = json.loads((ROOT / 'artifacts/contracts/tail_put25_pooled/inputs.json').read_text())['definitions']
    for d in definitions:
        if 'AMD' in d['label'] and 'NVDA' in d['label']:
            print('Example definition:', json.dumps(d), flush=True)
            path = ROOT / 'artifacts/contracts/uniform_smile/market/assets' / f'{d["secids"][0]}_selection.parquet'
            sel = pd.read_parquet(path)
            print('Selection columns:', list(sel.columns), flush=True)
            print('Selection buckets:', sel[['cp_flag', 'bucket', 'tenor']].drop_duplicates().to_string(index=False), flush=True)
            entered = pd.read_csv(ROOT / 'artifacts/contracts/p25_capital_sensitivity/replays/10m_learned/decisions.csv')
            print('Example decisions:', entered.loc[entered.pair.eq(d['pair'])].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inspect', action='store_true')
    args = parser.parse_args()
    if args.inspect:
        inspect()
    else:
        build()


if __name__ == '__main__':
    main()
