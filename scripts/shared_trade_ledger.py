"""Reconstruct shared-book cohorts from fills and observed contract Greeks.

Option and hedge P&L are additive. Actual netted stock execution/borrow costs
are allocated pro rata to absolute hedge changes/short hedges respectively.
Shared sleeve financing stays separate. No simulated fills or new trades.
"""
import numpy as np
import pandas as pd
from scripts.option_flex.black_scholes import option_values
from scripts.option_flex.contract_engine import valid_greeks

FIELDS = ['option_mid_pnl', 'stock_hedge_gross_pnl', 'dividends', 'option_commission',
          'option_penalty', 'stock_execution', 'stock_borrow']


def allocate_cost(total, weights):
    total = np.asarray(total, float); weights = np.asarray(weights, float)
    assert np.isfinite(total).all() and np.isfinite(weights).all() and (weights >= 0).all()
    denominator = weights.sum(axis=1)
    assert not ((denominator == 0) & (abs(total) > 1e-8)).any()
    allocation = weights * np.divide(total, denominator, out=np.zeros_like(total), where=denominator > 0)[:, None]
    np.testing.assert_allclose(allocation.sum(axis=1), total, atol=1e-8, rtol=0)
    return allocation


def connected_episodes(frame):
    result = pd.Series(index=frame.index, dtype=object)
    for pair, g in frame.groupby('pair', sort=False):
        number = 0; end = None
        for row in g.sort_values(['entry_date', 'exit_date']).itertuples():
            if end is None or row.entry_date > end:
                number += 1; end = row.exit_date
            else: end = max(end, row.exit_date)
            result.loc[row.Index] = f'{pair}#{number}'
    return result


def reconstruct_shared(nav, trades, quotes, stocks, dividends, cfg):
    cal = nav.index; cohorts = sorted(trades.cohort.unique())
    n, m = len(cal), len(cohorts); dates = {date: k for k, date in enumerate(cal)}
    q = quotes.copy(); q['date'] = pd.to_datetime(q.date); q['expiry'] = pd.to_datetime(q.expiry)
    assert not q.duplicated(['date', 'optionid']).any()
    lookup = {(r.date, int(r.optionid)): r._asdict() for r in q.itertuples(index=False)}
    all_daily = {f: np.zeros((n, m)) for f in FIELDS}
    all_marks = np.zeros((n, m)); all_vega = np.zeros((n, m)); all_gross_vega = np.zeros((n, m))
    legs = []; errors = []
    aggregate_stock_value = np.zeros(n); aggregate_gross_hedge = np.zeros(n)
    for sid, own in trades.groupby('secid'):
        s = stocks[stocks.secid.eq(sid)].set_index('date').reindex(cal)
        spot = s.close.to_numpy(float)
        factor = (s.cfadj / s.cfadj.shift(1)).fillna(1).to_numpy(float)
        holdings = own.pivot_table(index='date', columns='cohort', values='quantity', aggfunc='sum', fill_value=0)
        holdings = holdings.reindex(index=cal, columns=cohorts, fill_value=0).fillna(0).cumsum().to_numpy(float)
        cash = own.assign(flow=-own.quantity*100*own.price).pivot_table(index='date', columns='cohort', values='flow', aggfunc='sum', fill_value=0)
        cash = cash.reindex(index=cal, columns=cohorts, fill_value=0).fillna(0).to_numpy(float)
        fees = {}
        for field, source in [('option_commission', 'commission'), ('option_penalty', 'execution_penalty')]:
            table = own.pivot_table(index='date', columns='cohort', values=source, aggfunc='sum', fill_value=0)
            fees[field] = table.reindex(index=cal, columns=cohorts, fill_value=0).fillna(0).to_numpy(float)
        hedge = np.zeros((n, m)); mark = np.zeros((n, m)); vega = np.zeros((n, m))
        for j, cohort in enumerate(cohorts):
            t = own[own.cohort.eq(cohort)]
            if t.empty: continue
            assert t.optionid.nunique() == 1 and abs(t.quantity.sum()) < 1e-8
            oid = int(t.optionid.iloc[0]); last_iv = np.nan
            for k in np.flatnonzero(holdings[:, j]):
                row = lookup.get((cal[k], oid)); assert row is not None, f'Missing actual held quote {cal[k]} {oid}'
                row = dict(row)
                assert row.get('bs_status') != 'unsupported_distribution'
                sigma = float(row.get('bs_iv', np.nan))
                if valid_greeks(row) and np.isfinite(sigma) and sigma > 0: last_iv = sigma
                else:
                    rate = float(row['bs_rate']); pv = float(row['known_dividend_pv'])
                    assert np.isfinite([last_iv, rate, pv]).all() and last_iv > 0
                    assert spot[k]-pv > 0 and row['expiry'] > cal[k]
                    _, delta, _, vv = option_values(spot[k]-pv, row['strike'], (row['expiry']-cal[k]).days/365,
                        rate, last_iv, row['cp_flag'])
                    row.update(delta=float(delta), vega=float(vv))
                assert np.isfinite(spot[k]) and spot[k] > 0
                hedge[k, j] = -holdings[k, j]*100*row['delta']
                mark[k, j] = holdings[k, j]*100*(row['bid']+row['ask'])/2
                vega[k, j] = holdings[k, j]*100*row['vega']
        target = hedge.sum(axis=1)
        aggregate_stock_value += target * np.nan_to_num(spot)
        aggregate_gross_hedge += abs(target) * np.nan_to_num(spot)
        previous = np.vstack([np.zeros((1, m)), hedge[:-1]])
        old_spot = np.r_[np.nan, spot[:-1]]
        change = spot*factor-old_spot
        assert not ((previous != 0) & ~np.isfinite(change[:, None])).any()
        gross = previous*np.nan_to_num(change)[:, None]
        changes = hedge-previous*factor[:, None]
        total_cost = abs(changes.sum(axis=1))*np.nan_to_num(spot)*cfg['stock_cost_bps']/10000
        stock_cost = allocate_cost(total_cost, abs(changes))
        elapsed = cal.to_series().diff().dt.days.fillna(0).to_numpy()/365
        borrow_total = np.maximum(-previous.sum(axis=1), 0)*np.nan_to_num(old_spot)*cfg['stock_borrow_rate']*elapsed
        borrow = allocate_cost(borrow_total, np.maximum(-previous, 0))
        div = np.zeros(n)
        for event in dividends[dividends.secid.eq(sid)].itertuples():
            k = cal.searchsorted(event.ex_date)
            if 0 < k < n: div[k] += event.amount
        dividend = previous*(factor*div)[:, None]
        option = cash+np.vstack([mark[:1], np.diff(mark, axis=0)])
        data = dict(option_mid_pnl=option, stock_hedge_gross_pnl=gross, dividends=dividend,
            stock_execution=stock_cost, stock_borrow=borrow, **fees)
        for field in FIELDS: all_daily[field] += data[field]
        all_marks += mark; all_vega += vega; all_gross_vega += abs(vega)
        for j, cohort in enumerate(cohorts):
            t = own[own.cohort.eq(cohort)]
            if t.empty: continue
            entry = t[t.reason.eq('entry')]; assert len(entry) == 1
            first = entry.iloc[0]; initial = lookup[(first.date, int(first.optionid))]
            totals = {field: float(value[:, j].sum()) for field, value in data.items()}
            totals['gross_hedged_pnl'] = totals['option_mid_pnl']+totals['stock_hedge_gross_pnl']+totals['dividends']
            totals['net_before_shared_financing'] = totals['gross_hedged_pnl']-sum(totals[f] for f in FIELDS[3:])
            legs.append(dict(cohort=cohort, secid=int(sid), optionid=int(first.optionid), entry_date=first.date,
                exit_date=t.date.max(), role='long' if first.quantity > 0 else 'short', contracts=abs(first.quantity),
                initial_vega=abs(first.quantity)*100*initial['vega'], entry_iv=initial['bs_iv']*100,
                entry_dte=(initial['expiry']-first.date).days, **totals))
    np.testing.assert_allclose(aggregate_stock_value, nav.stock_value, atol=1e-7, rtol=0)
    np.testing.assert_allclose(aggregate_gross_hedge, nav.gross_hedge_notional, atol=1e-7, rtol=0)
    np.testing.assert_allclose(all_daily['stock_hedge_gross_pnl'].sum(axis=1), nav.hedge_gross_pnl, atol=1e-7, rtol=0)
    np.testing.assert_allclose(all_marks.sum(axis=1), nav.option_value, atol=1e-7, rtol=0)
    np.testing.assert_allclose(all_vega.sum(axis=1), nav.net_vega, atol=1e-7, rtol=0)
    np.testing.assert_allclose(all_gross_vega.sum(axis=1), nav.gross_vega, atol=1e-7, rtol=0)
    np.testing.assert_allclose(all_daily['option_commission'].sum(axis=1)+all_daily['option_penalty'].sum(axis=1), nav.option_execution_cost, atol=1e-7, rtol=0)
    for field, source in [('stock_execution', 'hedge_cost'), ('stock_borrow', 'stock_borrow_cost'), ('dividends', 'dividend_cash')]:
        np.testing.assert_allclose(all_daily[field].sum(axis=1), nav[source], atol=1e-7, rtol=0)
    daily_net = sum(all_daily[f] for f in FIELDS[:3])-sum(all_daily[f] for f in FIELDS[3:])
    benchmark_interest = nav.cash_benchmark.diff().fillna(0)
    financing = nav.interest-benchmark_interest
    expected = nav.nav.diff().fillna(0)-benchmark_interest
    error = float(np.max(abs(daily_net.sum(axis=1)+financing.to_numpy()-expected.to_numpy())))
    np.testing.assert_allclose(daily_net.sum(axis=1)+financing, expected, atol=1e-6, rtol=0)
    components = {field: pd.DataFrame(value,index=cal,columns=cohorts) for field,value in all_daily.items()}
    return pd.DataFrame(legs), pd.DataFrame(daily_net, index=cal, columns=cohorts), financing, all_vega, all_gross_vega, error, components
