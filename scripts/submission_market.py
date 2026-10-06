"""Selected-contract replay lookups without importing training pipelines.

Lookup and ledger conventions match uniform_smile_data.ReplayMarket and
run_uniform_smile.verify. The frozen replay checks their resulting NAV and
execution records against the original reference.
"""
from collections import OrderedDict

import numpy as np
import pandas as pd

from scripts.option_flex.contract_engine import MarketRow


class FrozenMarket:
    def __init__(self, folder):
        self.folder = folder
        stock = pd.read_parquet(folder / 'stocks.parquet', columns=['date', 'secid', 'close', 'cfadj'])
        self.spots = stock.pivot(index='date', columns='secid', values='close')
        self.adjustments = stock.pivot(index='date', columns='secid', values='cfadj')
        self.rates = pd.read_parquet(folder / 'rates.parquet').rate
        self.divs = pd.read_parquet(folder / 'dividends.parquet')
        self.years = OrderedDict()
        self.days = OrderedDict()
        self.reported_year = None

    def stock(self, date, sid):
        if sid == 101322 and date == pd.Timestamp('2025-07-18'):
            return 1.025 * self.spots.at[date, 102968], self.adjustments.at[pd.Timestamp('2025-07-17'), sid]
        return self.spots.at[date, sid], self.adjustments.at[date, sid]

    def rate(self, date):
        if self.reported_year != date.year:
            self.reported_year = date.year
            print(f'Frozen portfolio replay: {date.date()}', flush=True)
        return float(self.rates.loc[date])

    def quote(self, date, oid):
        if date not in self.days:
            year = date.year
            if year not in self.years:
                q = pd.read_parquet(self.folder / f'{year}_quotes.parquet').set_index('date', drop=False).sort_index()
                self.years[year] = q
                while len(self.years) > 2:
                    self.years.popitem(last=False)
            self.years.move_to_end(year)
            q = self.years[year]
            daily = q.loc[[date]] if date in q.index else q.iloc[:0]
            self.days[date] = {int(r['optionid']): MarketRow(r) for r in daily.to_dict('records')}
            while len(self.days) > 2:
                self.days.popitem(last=False)
        self.days.move_to_end(date)
        return self.days[date].get(int(oid))

    def dividend_pv(self, date, sid, expiry, rate):
        d = self.divs.loc[self.divs.secid.eq(sid) & self.divs.declare_date.le(date)
                          & self.divs.ex_date.gt(date) & self.divs.ex_date.le(expiry)]
        return float((d.amount * np.exp(-rate * (d.ex_date - date).dt.days / 365)).sum())


def verify_ledger(nav, trades, decisions, signals):
    if len(decisions) != len(signals):
        raise ValueError('Every proposal must have a recorded decision')
    entered = decisions.loc[decisions.status.eq('entered')]
    imbalance = (entered.vega_a - entered.vega_b).abs() / entered[['vega_a', 'vega_b']].max(axis=1)
    if not imbalance.le(.1 + 1e-12).all():
        raise ValueError('Entry vega imbalance exceeds the reference limit')
    paid = trades.reason.isin(['entry', 'announced_corporate_action_exit'])
    np.testing.assert_allclose(trades.loc[paid, 'commission'], trades.loc[paid, 'quantity'].abs() * .03, rtol=0, atol=1e-10)
    if not trades.groupby(['cohort', 'optionid']).quantity.sum().eq(0).all():
        raise ValueError('Open option inventory remains')
    cashflow = (-trades.quantity * 100 * trades.price - trades.commission - trades.execution_penalty).groupby(trades.date).sum().reindex(nav.index, fill_value=0)
    expected = cashflow + nav.option_value.diff().fillna(nav.option_value) + nav.hedge_gross_pnl - nav.hedge_cost + nav.interest - nav.stock_borrow_cost + nav.dividend_cash
    error = float((expected - nav.pnl).abs().max())
    if error >= 1e-5 or abs(nav.option_value.iloc[-1]) + abs(nav.stock_value.iloc[-1]) >= 1e-7:
        raise ValueError('Independent cash ledger failed reconciliation')
    return dict(entries=len(entered), active_pairs=int(entered.pair.nunique()),
                independent_ledger_error=error, all_inventory_closed=True)
