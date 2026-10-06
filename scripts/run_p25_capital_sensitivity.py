"""Replay frozen sizing at explicit account equities; no new leverage learner.

Capital scenarios are sensitivities, not selected capital recommendations.
Original research reserves and entry limits are retained, not broker margin.
Held positions are not forcibly liquidated; post-entry breaches are reported.
"""
import argparse
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd

from scripts import run_p25_action_windows as base

OUT = Path('artifacts/contracts/p25_capital_sensitivity')
REPORT = Path('reports/p25_capital_sensitivity')
REFERENCE = Path('artifacts/contracts/p25_feature_search/replays/variance_premium')
ACCOUNT_EQUITIES = [1_000_000, 3_000_000, 10_000_000, 25_000_000]


def metrics(nav, capital):
    """Use funded equity returns and a same-equity cash benchmark."""
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    relative = nav.nav / nav.cash_benchmark
    elapsed = nav.index.to_series().diff().dt.days.fillna(0) / 365
    borrowing_spread = -nav.cash.shift(1).clip(upper=0).fillna(0) * .01 * elapsed
    return dict(
        funded_equity=capital,
        annualized_excess_sharpe=base.sharpe(nav.excess_return.iloc[1:]),
        funded_cagr_pct=100 * ((nav.nav.iloc[-1] / capital) ** (1 / years) - 1),
        relative_to_cash_cagr_pct=100 * (relative.iloc[-1] ** (1 / years) - 1),
        net_excess_pnl=float(nav.nav.iloc[-1] - nav.cash_benchmark.iloc[-1]),
        maximum_nav_drawdown_pct=100 * float((nav.nav / nav.nav.cummax() - 1).min()),
        maximum_relative_to_cash_drawdown_pct=100 * float((relative / relative.cummax() - 1).min()),
        peak_reserve_fraction=float((nav.capital_reserve / nav.nav).max()),
        peak_stress_fraction=float((nav.stress_loss / nav.nav).max()),
        peak_gross_hedge_multiple=float((nav.gross_hedge_notional / nav.nav).max()),
        post_entry_risk_breach_days=int(nav.post_trade_risk_breach.sum()),
        pair_risk_breach_days=int(nav.pair_risk_breaches.gt(0).sum()),
        negative_cash_days=int(nav.cash.lt(0).sum()),
        borrowing_spread_cost=float(borrowing_spread.sum()),
    )


def annual_metrics(nav, case, capital):
    rows = []
    previous_nav = pd.Series(nav.nav.shift(1).fillna(capital), index=nav.index)
    previous_cash = nav.cash_benchmark.shift(1).fillna(capital)
    for year, g in nav.groupby(nav.index.year):
        funded = float(g.nav.iloc[-1] / previous_nav.loc[g.index[0]] - 1)
        cash = float(g.cash_benchmark.iloc[-1] / previous_cash.loc[g.index[0]] - 1)
        returns = g.excess_return.iloc[1:] if g.index[0] == nav.index[0] else g.excess_return
        rows.append(dict(case=case, year=int(year), funded_return_pct=100 * funded,
                         return_above_cash_pp=100 * (funded - cash),
                         annualized_excess_sharpe=base.sharpe(returns)))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--anchor-only', action='store_true')
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    inputs = [Path(__file__), Path('scripts/uniform_smile_engine.py'),
              Path('scripts/uniform_smile_risk.py'),
              REFERENCE / 'scored_allocations.parquet', REFERENCE / 'daily_nav.parquet',
              base.TRANSFER / 'market/rates.parquet']
    hashes = {str(path): base.digest(path) for path in inputs}
    base.write_json(OUT / 'inputs.json', dict(
        source_hashes=hashes, account_equities=ACCOUNT_EQUITIES,
        pair_reference_capital=1_000_000, entry_vega_per_leg=450,
        training='Frozen previously fitted monthly models; no retraining',
        leverage_learning=False, broker_margin_model=False,
        risk_limits='Inherited 5% stress/80% reserve/2x hedge; research assumptions',
        forced_liquidation=False, previously_inspected_history=True))
    base.cached.OUT = base.TRANSFER
    market = base.Market()
    cal = market.rates.index[market.rates.index >= '2018-01-01']
    original = pd.read_parquet(REFERENCE / 'scored_allocations.parquet')
    assert len(original) == 608
    cases = [('anchor_345m', 345_000_000, 'learned')]
    if not args.anchor_only:
        cases += [(f'{capital // 1_000_000}m_{policy}', capital, policy)
                  for capital in ACCOUNT_EQUITIES for policy in ['learned', 'fixed50', 'fixed100']]
    comparison, annual = [], []
    for case, capital, policy in cases:
        folder = OUT / 'replays' / case
        folder.mkdir(parents=True, exist_ok=True)
        status_path = folder / 'status.json'
        if status_path.exists():
            status = json.loads(status_path.read_text())
            nav = pd.read_parquet(folder / 'daily_nav.parquet')
        else:
            rows = original.copy()
            if policy != 'learned':
                rows['learned_level'] = 2 if policy == 'fixed50' else 4
            pd.testing.assert_frame_equal(rows.drop(columns='learned_level'), original.drop(columns='learned_level'))
            cfg = replace(base.config(), option_commission=.03, capital=capital,
                          max_gross_vega=20_000 * capital / 1_000_000)
            base.write_json(folder / 'config.json', asdict(cfg))
            market.variant = case
            market.started = time.monotonic()
            market.reported_year = None
            saved = base.engine.book_risk
            try:
                base.engine.book_risk = base.book_risk
                nav, trades, decisions, ledger = base.engine.simulate(
                    market, rows, cal, cfg, 1_000_000,
                    policy=lambda row, context: int(row['learned_level']),
                    corporate_events=base.UNIFORM_CORPORATE_EVENTS, wing_capital=1_000_000)
            finally:
                base.engine.book_risk = saved
            base.mark_book(nav, market.rates, capital)
            audit = base.verify(nav, trades, decisions, rows, cfg)
            if case == 'anchor_345m':
                pd.testing.assert_frame_equal(nav, pd.read_parquet(REFERENCE / 'daily_nav.parquet'),
                                              rtol=1e-12, atol=1e-6)
            nav.to_parquet(folder / 'daily_nav.parquet')
            trades.to_csv(folder / 'trades.csv', index=False)
            decisions.to_csv(folder / 'decisions.csv', index=False)
            status = dict(status='complete', case=case, policy=policy, **metrics(nav, capital),
                          entries=audit['entries'], active_pairs=audit['active_pairs'],
                          independent_ledger_error=audit['independent_ledger_error'], **ledger)
            base.write_json(status_path, status)
            del rows, trades, decisions
        comparison.append(status)
        annual.extend(annual_metrics(nav, case, capital))
        print(f'{case}: Sharpe {status["annualized_excess_sharpe"]:.6f}; '
              f'{status["entries"]} entries; cash-relative CAGR '
              f'{status["relative_to_cash_cagr_pct"]:.4f}%; '
              f'{status["post_entry_risk_breach_days"]} portfolio breach days', flush=True)
        pd.DataFrame(comparison).to_csv(REPORT / 'comparison.csv', index=False)
        pd.DataFrame(annual).to_csv(REPORT / 'annual.csv', index=False)
        base.write_json(OUT / 'progress.json', dict(completed=len(comparison), total=len(cases),
                        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        del nav
        gc.collect()
    for path, expected in hashes.items():
        assert base.digest(Path(path)) == expected, path
    base.write_json(REPORT / 'audit.json', dict(
        complete=True, cases=len(comparison), anchor_reproduced=True,
        all_608_proposals_retained=True, all_ledgers_verified=True,
        frozen_learning_only=True, broker_margin_model=False,
        forced_liquidation=False, new_leverage_learner=False,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__ == '__main__':
    main()
