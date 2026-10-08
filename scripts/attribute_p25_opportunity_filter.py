"""Exact funded-pair attribution for completed opportunity-filter replays.

Reuses the audited shared-book reconstruction without changing its accounting.
Loads only actually traded contract histories. A conservative dense-array
preflight skips books too large for the 384 MiB worker budget. Unavailable
attribution is reported explicitly; option-fill cash flows are never substituted
for hedged pair profits. Run serially after the experiment worker stops.
"""
import os
for _key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'ARROW_NUM_THREADS']:
    os.environ[_key] = '1'

import argparse
import gc
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
import pandas as pd

from scripts.analyze_p25_opportunity_filter import pair_diagnostics
from scripts.diagnose_p25_pca import Budget as MemoryBudget, digest
from scripts import shared_trade_ledger as ledger

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/contracts/p25_opportunity_filter'
REPORT = ROOT / 'reports/p25_opportunity_filter/funded'
MARKET = ROOT / 'data/submission/artifacts/contracts/p25_signal_transfer/market'
CONFIG = ROOT / 'data/submission/artifacts/contracts/p25_capital_sensitivity/replays/10m_learned/config.json'
CASES = ['rank_ridge', 'rank_spline', 'current_fixed100', 'rank_all_fixed100']
QUOTE_FIELDS = ['date', 'optionid', 'expiry', 'cp_flag', 'strike', 'bid', 'ask',
                'delta', 'vega', 'bs_iv', 'bs_status', 'bs_rate', 'known_dividend_pv']


class Budget(MemoryBudget):
    def __init__(self):
        super().__init__(duty_cycle=.2, rss_limit_mib=384)
        self.started = time.monotonic()
        self.cpu_started = time.process_time()

    def rest(self):
        self.check()
        if (OUT / 'STOP').exists():
            raise InterruptedError('Opportunity experiment STOP requested')
        while True:
            delay = (time.process_time() - self.cpu_started) / self.duty - (time.monotonic() - self.started)
            if delay <= 0:
                return
            time.sleep(min(delay, .25))
            self.check()


def read_traded_quotes(trades, budget):
    records = []
    connection = sqlite3.connect(f'file:{(OUT / "quotes.sqlite").as_posix()}?mode=ro', uri=True)
    connection.execute('PRAGMA cache_size=-2048')
    try:
        for oid, group in trades.groupby('optionid', sort=False):
            first, last = group.date.min(), group.date.max()
            result = connection.execute('SELECT payload FROM quotes WHERE date>=? AND date<=? AND optionid=?',
                                        (str(first.date()), str(last.date()), int(oid)))
            for (payload,) in result:
                raw = json.loads(payload)
                records.append({key: raw.get(key) for key in QUOTE_FIELDS})
            budget.rest()
    finally:
        connection.close()
    quotes = pd.DataFrame(records, columns=QUOTE_FIELDS)
    for name in ['date', 'expiry']:
        quotes[name] = pd.to_datetime(quotes[name])
    for name in set(QUOTE_FIELDS) - {'date', 'expiry', 'cp_flag', 'bs_status'}:
        quotes[name] = pd.to_numeric(quotes[name], errors='coerce')
    return quotes


def memory_estimate(nav, trades, budget, quotes=None):
    """Conservative bound for the unchanged reconstruction's dense matrices."""
    cells = len(nav) * trades.cohort.nunique()
    dense_mib = cells * 8 * 36 / 1024**2
    rss = budget.process.memory_info().rss / 1024**2
    # Python quote-lookup dictionaries are more expensive than a numeric frame.
    lookup_mib = 32. if quotes is None else len(quotes) * 1600 / 1024**2
    return dict(cohorts=int(trades.cohort.nunique()), calendar_rows=len(nav),
                current_rss_mib=rss, dense_array_allowance_mib=dense_mib,
                quote_lookup_allowance_mib=lookup_mib,
                estimated_peak_rss_mib=rss + dense_mib + lookup_mib + 16.)


def attribute_case(case, budget):
    folder = OUT / 'replays' / case
    destination = OUT / 'pair_attribution' / case
    destination.mkdir(parents=True, exist_ok=True)
    sources = {str(p.relative_to(ROOT)): digest(p) for p in
               [folder / 'daily_nav.parquet', folder / 'trades.csv', CONFIG, Path(__file__),
                Path(ledger.__file__), OUT / 'quotes.sqlite', MARKET / 'stocks.parquet',
                MARKET / 'dividends.parquet']}
    nav = pd.read_parquet(folder / 'daily_nav.parquet', use_threads=False)
    trades = pd.read_csv(folder / 'trades.csv', parse_dates=['date'])
    estimate = memory_estimate(nav, trades, budget)
    if estimate['estimated_peak_rss_mib'] > 360:
        return dict(case=case, status='unavailable_memory_preflight', **estimate, sources=sources), [], []
    quotes = read_traded_quotes(trades, budget)
    stocks = pd.read_parquet(MARKET / 'stocks.parquet', columns=['date', 'secid', 'close', 'cfadj'],
                             filters=[('secid', 'in', trades.secid.unique().tolist())], use_threads=False)
    dividends = pd.read_parquet(MARKET / 'dividends.parquet', columns=['secid', 'ex_date', 'amount'],
                                use_threads=False)
    estimate = memory_estimate(nav, trades, budget, quotes)
    if estimate['estimated_peak_rss_mib'] > 360:
        return dict(case=case, status='unavailable_memory_preflight', **estimate, sources=sources), [], []
    cfg = json.loads(CONFIG.read_text())
    original_allocate = ledger.allocate_cost

    def bounded_allocate(total, weights):
        budget.rest()
        return original_allocate(total, weights)

    try:
        ledger.allocate_cost = bounded_allocate
        legs, daily, financing, vegas, grossvegas, error, components = ledger.reconstruct_shared(
            nav, trades, quotes, stocks, dividends, cfg)
    finally:
        ledger.allocate_cost = original_allocate
    del quotes, stocks, dividends, vegas, grossvegas, components
    gc.collect()
    pair_index = pd.Series(daily.columns.astype(str), index=daily.columns).str.split('|').str[1]
    pair_daily = daily.T.groupby(pair_index).sum().T
    del daily
    gc.collect()
    pair_daily.to_parquet(destination / 'pair_daily.parquet')
    financing.to_frame('shared_financing_vs_cash').to_parquet(destination / 'shared_financing.parquet')
    legs.to_csv(destination / 'legs.csv', index=False)
    rows, concentrations = pair_diagnostics(case, pair_daily, financing, nav)
    status = dict(case=case, status='complete', **estimate,
                  quotes_used=int(legs.optionid.nunique()), pairs=len(pair_daily.columns),
                  max_daily_reconciliation_error=error, sources=sources,
                  financing='Shared financing remains separate from pair profit',
                  costs='Shared stock execution and borrow costs allocated using audited hedge-weight convention')
    (destination / 'reconciliation.json').write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')
    budget.rest()
    return status, rows, concentrations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', nargs='+', choices=CASES, default=CASES)
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    budget = Budget()
    statuses, pairs, concentration = [], [], []
    for case in args.cases:
        try:
            status, rows, own = attribute_case(case, budget)
            pairs.extend(rows)
            concentration.extend(own)
        except (AssertionError, ValueError, KeyError, FileNotFoundError, MemoryError) as error:
            status = dict(case=case, status='unavailable_reconstruction', reason=f'{type(error).__name__}: {error}')
        statuses.append(status)
        print(json.dumps(status), flush=True)
        gc.collect()
        budget.rest()
    if pairs:
        pd.DataFrame(pairs).to_csv(REPORT / 'pair_periods.csv', index=False)
        pd.DataFrame(concentration).to_csv(REPORT / 'concentration.csv', index=False)
    result = dict(cases=statuses, peak_rss_mib=budget.peak,
                  successful_cases=[s['case'] for s in statuses if s['status'] == 'complete'],
                  accounting='Actual funded fills, delta hedges, costs, dividends; financing separate; exact daily NAV reconciliation')
    (REPORT / 'pair_attribution_status.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
