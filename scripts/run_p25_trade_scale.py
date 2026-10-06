"""Frozen P25 sizing at 10m equity with declared trade/budget scale cases.

Separate size-only changes from changes to the inherited per-pair budget.
Account-wide constraints remain fixed. This is not leverage retraining, broker
margin simulation, forced-liquidation modeling, or a liquidity capacity test.
"""
from dataclasses import asdict, replace
import gc
from pathlib import Path
import resource
import time

import pandas as pd

from scripts import run_p25_action_windows as base
from scripts.run_p25_capital_sensitivity import metrics, annual_metrics

OUT = Path('artifacts/contracts/p25_trade_scale')
REPORT = Path('reports/p25_trade_scale')
REFERENCE = Path('artifacts/contracts/p25_capital_sensitivity/replays/10m_learned')
ALLOCATIONS = Path('artifacts/contracts/p25_feature_search/replays/variance_premium/scored_allocations.parquet')
CASES = [('baseline', 1, 1), ('size2_budget1', 2, 1), ('size4_budget1', 4, 1),
         ('size2_budget2', 2, 2), ('size4_budget4', 4, 4)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    inputs = [Path(__file__), Path('scripts/run_p25_capital_sensitivity.py'),
              Path('scripts/uniform_smile_engine.py'), Path('scripts/uniform_smile_risk.py'),
              ALLOCATIONS, REFERENCE / 'daily_nav.parquet']
    inputs += sorted((base.TRANSFER / 'market').glob('*.parquet'))
    hashes = {str(p): base.digest(p) for p in inputs}
    base.write_json(OUT / 'inputs.json', dict(
        source_hashes=hashes, funded_equity=10_000_000, cases=CASES,
        unchanged='608 proposals, directions, contracts, frozen monthly allocations, fees, hedges, own expiry',
        full_vega_per_leg='450 times size_multiplier dollars per volatility point',
        pair_and_wing_reference_capital='1m times budget_multiplier; reference is not funded cash',
        global_limits='5% stress, 80% reserve, 2x stock hedge, 200k gross vega',
        no_retraining=True, original_candidate_unchanged=True,
        broker_margin_model=False, forced_liquidation=False, market_impact_modeled=False,
        no_fresh_holdout=True))
    base.cached.OUT = base.TRANSFER
    market = base.Market()
    cal = market.rates.index[market.rates.index >= '2018-01-01']
    original = pd.read_parquet(ALLOCATIONS)
    assert len(original) == 608 and original.learned_level.isin([2, 3, 4]).all()
    comparisons, annual, entries, dispositions = [], [], [], []
    for case, size_multiplier, budget_multiplier in CASES:
        folder = OUT / 'replays' / case
        folder.mkdir(parents=True, exist_ok=True)
        cfg = replace(base.config(), option_commission=.03, capital=10_000_000,
                      max_gross_vega=200_000, entry_vega_per_leg=450 * size_multiplier)
        pair_capital = 1_000_000 * budget_multiplier
        base.write_json(folder / 'config.json', dict(account=asdict(cfg), pair_capital=pair_capital,
                                                   wing_capital=pair_capital))
        market.variant = case
        market.started = time.monotonic()
        market.reported_year = None
        saved = base.engine.book_risk
        try:
            base.engine.book_risk = base.book_risk
            nav, trades, decisions, ledger = base.engine.simulate(
                market, original, cal, cfg, pair_capital,
                policy=lambda row, context: int(row['learned_level']),
                corporate_events=base.UNIFORM_CORPORATE_EVENTS, wing_capital=pair_capital)
        finally:
            base.engine.book_risk = saved
        base.mark_book(nav, market.rates, cfg.capital)
        audit = base.verify(nav, trades, decisions, original, cfg)
        if case == 'baseline':
            pd.testing.assert_frame_equal(nav, pd.read_parquet(REFERENCE / 'daily_nav.parquet'),
                                          rtol=1e-12, atol=1e-6)
        entered = decisions.loc[decisions.status.eq('entered')]
        dispositions.extend(dict(case=case, status=status, count=int(n))
                            for status, n in decisions.status.value_counts().items())
        # A shared-account risk constraint and a per-pair constraint are different.
        # Expose both without inferring a broker margin call from either one.
        record = dict(case=case, size_multiplier=size_multiplier,
                      pair_budget_multiplier=budget_multiplier, **metrics(nav, cfg.capital),
                      entries=audit['entries'], active_pairs=audit['active_pairs'],
                      average_reserve_fraction=float((nav.capital_reserve / nav.nav).mean()),
                      active_days_average_reserve_fraction=float(
                          (nav.capital_reserve / nav.nav).loc[nav.active_cohorts.gt(0)].mean()),
                      peak_reserved_dollars=float(nav.capital_reserve.max()),
                      average_entry_vega_per_leg=float(entered[['vega_a', 'vega_b']].mean(axis=1).mean()),
                      risk_rejected_proposals=int(decisions.status.str.contains('risk').sum()),
                      independent_ledger_error=audit['independent_ledger_error'], **ledger)
        comparisons.append(record)
        annual.extend(annual_metrics(nav, case, cfg.capital))
        entries.extend(dict(case=case, signal_date=str(row.signal_date.date()), pair=row.pair,
                            level=int(row.level)) for row in entered.itertuples())
        nav.to_parquet(folder / 'daily_nav.parquet')
        trades.to_csv(folder / 'trades.csv', index=False)
        decisions.to_csv(folder / 'decisions.csv', index=False)
        base.write_json(folder / 'status.json', dict(status='complete', **record))
        pd.DataFrame(comparisons).to_csv(REPORT / 'comparison.csv', index=False)
        pd.DataFrame(annual).to_csv(REPORT / 'annual.csv', index=False)
        pd.DataFrame(dispositions).to_csv(REPORT / 'decisions.csv', index=False)
        pd.DataFrame(entries).to_csv(REPORT / 'entered_proposals.csv', index=False)
        print(f'{case}: {record["entries"]} entries; '
              f'Sharpe {record["annualized_excess_sharpe"]:.6f}; '
              f'cash-relative CAGR {record["relative_to_cash_cagr_pct"]:.4f}%; '
              f'reserve mean/peak {record["average_reserve_fraction"]:.2%}/'
              f'{record["peak_reserve_fraction"]:.2%}; '
              f'{record["post_entry_risk_breach_days"]} account breach days', flush=True)
        base.write_json(OUT / 'progress.json', dict(completed=len(comparisons), total=len(CASES),
                        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        del nav, trades, decisions, entered
        gc.collect()
    assert len(comparisons) == 5
    for path, expected in hashes.items():
        assert base.digest(Path(path)) == expected, path
    base.write_json(REPORT / 'audit.json', dict(complete=True, cases=5,
        baseline_10m_exactly_reproduced=True, all_608_proposals_retained=True,
        independent_cash_ledgers_verified=True, frozen_allocations=True,
        no_retraining=True, global_risk_limits_unchanged=True,
        pair_budget_changes_explicit=True, broker_margin_model=False,
        forced_liquidation=False, market_impact_modeled=False,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    findings = (
        'Five declared size/budget sensitivities at the same 10m funded equity. '
        'Original 10m learned cash book reproduced exactly. All 608 proposals, '
        'directions, option IDs and frozen monthly allocations retained; no retraining. '
        'Full entry vega per leg is 450/900/1800 dollars per volatility point, before '
        'the learned 50/75/100 ladder and account NAV compounding.\n'
        'Size-only cases retain 1m pair/wing reference budgets. Matched-budget '
        'cases increase those references to 2m/4m while retaining the same 10m '
        'account and global 5% stress, 80% reserve, 2x hedge, 200k gross-vega limits. '
        'Risk references are not allocations of actual funded cash.\n'
        'Actual commission .03 per option contract, option mid plus .1 volatility '
        'point cash penalty per leg, stock hedge 1bp, original financing, borrow, '
        'dividends and daily delta hedging unchanged. Execution costs scale '
        'linearly; quote depth, volume participation and nonlinear market impact '
        'are unmodeled. Larger-size returns do not establish executable capacity.\n'
        'Reserve is gross option premium plus 50% of gross netted stock hedge '
        'notional, not broker margin. Limits gate entries; held inventory can '
        'breach limits after market moves. There is no forced liquidation. '
        'Report breach days and drawdowns, not only profit and Sharpe.\n'
        'All history has been previously inspected, with inherited retrospective '
        'roster/formation conditioning. No new holdout or learned leverage result.\n\n'
    )
    (REPORT / 'findings.txt').write_text(findings + pd.DataFrame(comparisons).to_string(index=False) + '\n')


if __name__ == '__main__':
    main()
