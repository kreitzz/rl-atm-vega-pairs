"""Step 1 of the index-tracking project: daily candidate contracts and the 30-day index.

The target is OptionMetrics' volatility-surface point (vsurfd, 30 days, 50 delta, call),
already saved in data/wrds_q/iv.csv. It is a smoothed surface value, not a contract.
The candidates are the real listed calls in data/q_contracts/<PAIR>/quotes (all strikes,
expiries up to 60 days), cleaned so that each (stock, day) has a tradable call chain.

Units used throughout the tracking scripts
  iv           implied vol in vol points (e.g. 45.2)
  vega         dollars per ONE vol point per CONTRACT (OptionMetrics vega per share per
               1.00 of sigma, times 100 shares, divided by 100 vol points)
  half_spread  dollars per contract ((ask - bid) / 2 * 100)

Usage:  python -m scripts.tracking_data --pair AMD_NVDA
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SECIDS = {101121: 'AMD', 108321: 'NVDA'}
COLUMNS = ['secid', 'date', 'optionid', 'exdate', 'strike_price', 'best_bid', 'best_offer',
           'open_interest', 'impl_volatility', 'delta', 'vega', 'ss_flag', 'contract_size']


class Filters:
    min_dte = 5            # calendar days; shorter options are never candidates
    max_dte = 60           # the download itself stops at 60
    min_delta = 0.20
    max_delta = 0.80
    max_relative_spread = 0.50


def load_quotes(root, csv=None):
    """Raw call quotes. Reads the partitioned parquet files (or a CSV export of them)."""
    if csv is not None:
        q = pd.read_csv(csv, usecols=COLUMNS, parse_dates=['date', 'exdate'], dtype={'ss_flag': str})
    else:
        files = sorted(Path(root).glob('quotes/year=*/pair.parquet'))
        if not files:
            raise FileNotFoundError(f'No quote partitions under {root}')
        q = pd.concat([pd.read_parquet(f, columns=COLUMNS) for f in files], ignore_index=True)
        q['date'] = pd.to_datetime(q.date)
        q['exdate'] = pd.to_datetime(q.exdate)
    return q


def clean_chain(q, f=Filters):
    """Two tiers of quotes.

    marks:      every standard two-sided quote with usable IV/vega (any delta, any maturity).
                Used to value and trade out of contracts already held.
    candidate:  marks that also sit in the entry window (delta, maturity, relative spread).
                Only candidates can be bought into.
    A held contract with no mark on a later day is handled by the simulator (flagged and
    closed at its last quote), never silently repaired.
    """
    q = q.copy()
    q['ticker'] = q.secid.map(SECIDS)
    q['dte'] = (q.exdate - q.date).dt.days
    q['mid'] = (q.best_bid + q.best_offer) / 2
    ok = (q.ticker.notna() & q.ss_flag.astype(str).eq('0') & q.contract_size.eq(100)
          & (q.best_bid >= 0) & (q.best_offer > q.best_bid)
          & np.isfinite(q.impl_volatility) & (q.impl_volatility > 0)
          & np.isfinite(q.vega) & (q.vega > 0) & np.isfinite(q.delta)
          & (q.dte >= 1))
    q = q[ok]
    candidate = ((q.best_bid > 0)
                 & ((q.best_offer - q.best_bid) / q.mid <= f.max_relative_spread)
                 & q.delta.between(f.min_delta, f.max_delta)
                 & q.dte.between(f.min_dte, f.max_dte))
    out = pd.DataFrame({
        'ticker': q.ticker, 'date': q.date, 'optionid': q.optionid.astype('int64'),
        'exdate': q.exdate, 'dte': q.dte.astype(int), 'strike': q.strike_price / 1000,
        'iv': q.impl_volatility * 100, 'delta': q.delta, 'vega': q.vega,
        'half_spread': (q.best_offer - q.best_bid) / 2 * 100, 'mid': q.mid,
        'open_interest': q.open_interest,
        'monthly': is_monthly(q.exdate), 'candidate': candidate.to_numpy(),
    })
    dup = out.duplicated(['ticker', 'date', 'optionid'], keep=False)
    if dup.any():
        raise ValueError(f'{int(dup.sum())} duplicate (ticker, date, optionid) rows')
    return out.sort_values(['ticker', 'date', 'exdate', 'strike']).reset_index(drop=True)


def is_monthly(exdate):
    """Standard monthly expiry: third Friday (or the Saturday after it, older OptionMetrics convention)."""
    d = pd.DatetimeIndex(exdate)
    fri = (d.weekday == 4) & (d.day >= 15) & (d.day <= 21)
    sat = (d.weekday == 5) & (d.day >= 16) & (d.day <= 22)
    return np.asarray(fri | sat)


def load_index(path):
    """30-day 50-delta call IV (vol points) per ticker and date, from vsurfd."""
    iv = pd.read_csv(path, parse_dates=['date'])
    iv = iv[iv.ticker.isin(SECIDS.values()) & (iv.iv > 0)]
    return iv.pivot(index='date', columns='ticker', values='iv').sort_index()


def build(root, index_path, csv=None):
    chain = clean_chain(load_quotes(root, csv))
    index = load_index(index_path)
    days = sorted(set(chain.date.unique()) & set(index.index))
    chain = chain[chain.date.isin(days)]
    return chain, index.loc[days]


def summary(chain, index):
    per_day = chain[chain.candidate].groupby(['ticker', 'date']).agg(contracts=('optionid', 'size'),
                                                    expiries=('exdate', 'nunique'))
    return {
        'days': int(index.shape[0]),
        'first': str(index.index.min().date()), 'last': str(index.index.max().date()),
        'mark_rows': int(len(chain)), 'candidate_rows': int(chain.candidate.sum()),
        'median_contracts_per_stock_day': float(per_day.contracts.median()),
        'median_expiries_per_stock_day': float(per_day.expiries.median()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pair', default='AMD_NVDA')
    p.add_argument('--index', type=Path, default=Path('data/wrds_q/iv.csv'))
    p.add_argument('--quotes-csv', type=Path, default=None, help='optional CSV export instead of parquet')
    p.add_argument('--out', type=Path, default=Path('artifacts/tracking'))
    a = p.parse_args()
    chain, index = build(Path('data/q_contracts')/a.pair, a.index, a.quotes_csv)
    a.out.mkdir(parents=True, exist_ok=True)
    chain.to_pickle(a.out/'chain.pkl')
    index.to_pickle(a.out/'index.pkl')
    print(summary(chain, index))


if __name__ == '__main__':
    main()
