"""Run saved, unmodified policy actions against actual daily option contracts."""
import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.contract_engine import Settings, choose_contract, simulate


def normalize_quotes(raw):
    q = raw.rename(columns={'exdate': 'expiry', 'best_bid': 'bid',
                            'best_offer': 'ask', 'contract_size': 'multiplier'}).copy()
    for col in ['date', 'expiry']:
        q[col] = pd.to_datetime(q[col])
    for col in ['secid', 'optionid']:
        q[col] = q[col].astype('int64')
    q['strike'] = q.strike_price / 1000
    # IvyDB vega is cents of premium per 1 vol point, per share.
    # Convert to dollars per share; 100-share contract vega equals raw vega.
    q['vega'] = q.vega / 100
    q['standard'] = q.ss_flag.astype(str).eq('0')
    return q


def load_inputs(cfg, source_root, quotes_root):
    start, end = pd.Timestamp(cfg['data_start']), pd.Timestamp(cfg['data_end'])
    files = [quotes_root / f'year={year}.parquet' for year in range(start.year, end.year+1)]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError('Download actual contract quotes first: '+', '.join(missing))
    q = normalize_quotes(pd.concat([pd.read_parquet(f) for f in files], ignore_index=True))
    q = q[q.secid.isin(cfg['secids']) & q.date.between(start, end)]
    calendar = pd.DatetimeIndex(pd.read_csv('artifacts/broad_pairs/calendar.csv', parse_dates=['date']).date)
    calendar = calendar[(calendar >= start) & (calendar <= end)]
    stock_files = [source_root/'underlying_prices'/f'year={year}.parquet'
                   for year in range(start.year, end.year+1)]
    stocks = pd.concat([pd.read_parquet(f, filters=[('secid', 'in', cfg['secids'])])
                        for f in stock_files], ignore_index=True)
    stocks = stocks[stocks.date.between(start, end)].copy()
    stocks['secid'] = stocks.secid.astype(int)
    zero = pd.read_parquet(source_root/'zero_curve.parquet')
    zero['distance'] = abs(zero.days-30)
    zero = zero.sort_values(['date', 'distance', 'days']).drop_duplicates('date').sort_values('date')
    # A rate observed at a close funds the following interval, never the past one.
    zero['rate'] = zero.rate / 100
    stocks = pd.merge_asof(stocks.sort_values('date'), zero[['date', 'rate']], on='date',
                           direction='backward', tolerance=pd.Timedelta(days=7))
    dividends = pd.read_parquet(source_root/'actual_distributions.parquet')
    dividends = dividends[dividends.secid.isin(cfg['secids']) &
                          dividends.ex_date.between(start, end) &
                          dividends.cancel_flag.astype(str).eq('0')].copy()
    if len(dividends):
        ordinary = (dividends.distr_type.astype(str).eq('1') &
                    dividends.currency.eq('USD') & dividends.adj_factor.eq(1) &
                    dividends.approx_flag.astype(str).eq('0') & dividends.liquid_flag.astype(str).eq('0'))
        if not ordinary.all():
            raise ValueError('Pilot does not support nonordinary/corporate-action distributions')
        if (dividends.declare_date > dividends.ex_date).any() or (dividends.declare_date < '1950-01-01').any():
            raise ValueError('Unknown dividend announcement timing')
    raw_signals = pd.read_csv(cfg['signal_file'], index_col=0, parse_dates=[0])
    signals = raw_signals.loc[cfg['signal_start']:cfg['signal_end'], cfg['action_column']]
    expected = calendar[(calendar >= cfg['signal_start']) & (calendar <= cfg['signal_end'])]
    if not signals.index.equals(expected):
        raise ValueError('Missing/extra saved policy actions within the specified signal calendar')
    # Require both quote chains on all requested dates, independently of actions.
    expected_keys = pd.MultiIndex.from_product([calendar, cfg['secids']])
    present = pd.MultiIndex.from_frame(q[['date', 'secid']].drop_duplicates())
    if len(expected_keys.difference(present)):
        raise ValueError('Incomplete contract-chain date coverage')
    return q, stocks, signals, calendar, dividends, files+stock_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('config/contract_pilot.json'))
    parser.add_argument('--source-root', type=Path, required=True,
                        help='Local OptionMetrics directory containing underlying_prices and zero_curve.parquet')
    parser.add_argument('--quotes', type=Path, default=Path('artifacts/contracts/quotes'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/contracts/pilot'))
    parser.add_argument('--repair-crossed-held-quotes', action='store_true',
                        help='Explicit sensitivity only: swap crossed held bid/ask into a conservative envelope')
    parser.add_argument('--direction-control', choices=['saved', 'higher_iv'], default='saved')
    parser.add_argument('--greeks-source', choices=['vendor', 'black_scholes'], default='vendor')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    settings = Settings(**{f.name: cfg[f.name] for f in fields(Settings) if f.name in cfg})
    if args.repair_crossed_held_quotes:
        settings.repair_crossed_held_quotes = True
    cfg['repair_crossed_held_quotes'] = settings.repair_crossed_held_quotes
    q, stocks, signals, calendar, dividends, files = load_inputs(cfg, args.source_root, args.quotes)
    cfg['greeks_source'] = args.greeks_source
    if args.greeks_source == 'black_scholes':
        from scripts.black_scholes import fit_quotes
        zero = pd.read_parquet(args.source_root/'zero_curve.parquet')
        distributions = pd.read_parquet(args.source_root/'actual_distributions.parquet')
        # Fit all available marks, including contracts after they cease to be ATM.
        # Bound temporary pricing arrays; preserve observed quotes for executions.
        fitted = []
        for offset in range(0, len(q), 5000):
            part = fit_quotes(q.iloc[offset:offset+5000], stocks, zero, distributions)
            fitted.append(part[['bs_iv', 'bs_delta', 'bs_vega', 'bs_status']])
        marks = pd.concat(fitted, ignore_index=True)
        for name in ['delta', 'vega', 'impl_volatility']:
            q['vendor_'+name] = q[name]
        for name, fitted_name in [('delta','bs_delta'), ('vega','bs_vega'), ('impl_volatility','bs_iv')]:
            q[name] = marks[fitted_name].to_numpy()
        cfg['bs_fit_statuses'] = marks.bs_status.value_counts().to_dict()
    cfg['direction_control'] = args.direction_control
    if args.direction_control == 'higher_iv':
        control = signals.copy()
        for date, action in signals.items():
            if action == 0:
                continue
            next_date = calendar[calendar.get_loc(date)+1]
            selected = [choose_contract(q[q.date.eq(date) & q.secid.eq(s)],
                        float(stocks.loc[stocks.date.eq(date) & stocks.secid.eq(s), 'close'].iloc[0]),
                        next_date, settings) for s in cfg['secids']]
            if all(x is not None and np.isfinite(x.impl_volatility) for x in selected):
                control.loc[date] = abs(action)*np.sign(selected[1].impl_volatility-selected[0].impl_volatility)
            else:
                control.loc[date] = 0.
        signals = control
    nav, trades, decisions = simulate(q, stocks, signals, calendar, cfg['secids'], settings, dividends)
    rates = stocks.groupby('date').rate.mean().reindex(nav.index).shift(1).fillna(0)
    dt = nav.index.to_series().diff().dt.days.fillna(0)/365
    cash_returns = rates*dt
    nav['cash_benchmark'] = settings.capital*(1+cash_returns).cumprod()
    nav['excess_return'] = nav.return_ - cash_returns
    nav['excess_pnl'] = nav.nav-nav.cash_benchmark
    elapsed = (calendar[-1]-calendar[0]).days
    excess = nav.excess_return.iloc[1:]
    peak = nav.nav.cummax().clip(lower=settings.capital)
    summary = dict(pair=cfg['pair'], action_column=cfg['action_column'],
                   greeks_source=args.greeks_source,
                   direction_control=args.direction_control, sessions=len(nav),
                   start=str(calendar[0].date()), end=str(calendar[-1].date()),
                   entered_cohorts=int(decisions.status.eq('entered').sum()),
                   total_return=float(nav.nav.iloc[-1]/settings.capital-1),
                   annualized_return=float((nav.nav.iloc[-1]/settings.capital)**(365/elapsed)-1),
                   excess_pnl=float(nav.excess_pnl.iloc[-1]),
                   excess_return_on_initial_capital=float(nav.excess_pnl.iloc[-1]/settings.capital),
                   daily_excess_sharpe=float(np.sqrt(252)*excess.mean()/excess.std()) if excess.std() > 0 else None,
                   max_drawdown=float((nav.nav/peak-1).min()),
                   option_execution_cost=float(nav.option_execution_cost.sum()),
                   hedge_cost=float(nav.hedge_cost.sum()),
                   stock_borrow_cost=float(nav.stock_borrow_cost.sum()),
                   interest=float(nav.interest.sum()), dividend_cash=float(nav.dividend_cash.sum()),
                   max_gross_vega=float(nav.gross_vega.max()),
                   max_abs_net_vega=float(nav.net_vega.abs().max()),
                   min_cash=float(nav.cash.min()),
                   gross_vega_limit_breach_days=int(nav.gross_vega_limit_breached.sum()),
                   repaired_held_quote_days=int(nav.repaired_held_quotes.gt(0).sum()),
                   repaired_held_quotes=int(nav.repaired_held_quotes.sum()),
                   dispositions={str(k): int(v) for k,v in decisions.status.value_counts().items()})
    args.output.mkdir(parents=True, exist_ok=True)
    nav.to_csv(args.output/'daily_nav.csv')
    trades.to_csv(args.output/'option_trades.csv', index=False)
    decisions.to_csv(args.output/'decisions.csv', index=False)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    provenance = dict(config=cfg, source_files=[dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest())
                       for f in files+[Path(cfg['signal_file']), args.source_root/'zero_curve.parquet',
                                       args.source_root/'actual_distributions.parquet',
                                       Path('artifacts/broad_pairs/calendar.csv')]])
    (args.output/'manifest.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
