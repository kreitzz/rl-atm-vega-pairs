"""Read frozen ledgers; attribute wing volatility without training or replay."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path('reports/smile_volatility')
REPORT = Path('reports/smile_sizing')
METRICS = ['gross_vega', 'active_cohorts', 'gross_hedge_notional',
           'capital_reserve', 'stress_loss', 'hedge_turnover',
           'option_execution_cost', 'hedge_cost', 'stock_borrow_cost',
           'repaired_held_quotes', 'greek_fallbacks', 'causal_held_mark_fallbacks',
           'post_trade_risk_breach']


def check_reversing_marks(source):
    """Exact option/stock attribution for three previously identified outliers."""
    from scripts.option_flex.black_scholes import option_values
    full = Path('artifacts/contracts/smile_sizing/pairs/109072__213108/P75/floor/full')
    nav = pd.read_parquet(source(full / 'daily_nav.parquet'))
    trades = pd.read_csv(source(full / 'trades.csv'), parse_dates=['date'])
    events = pd.to_datetime(['2024-03-01','2024-03-18','2024-03-19'])
    dates = sorted(set(events) | {nav.index[nav.index.get_loc(d)-1] for d in events})
    stocks = pd.read_parquet(source(Path('data/option_contracts/underlying/year=2024.parquet')),
                             columns=['date','secid','close'],
                             filters=[('date','in',dates),('secid','in',[109072,213108])])
    stocks['date'] = pd.to_datetime(stocks.date)
    stocks = stocks.set_index(['date','secid'])
    quotes = []
    for sid in [109072,213108]:
        ids = trades.loc[trades.secid.eq(sid),'optionid'].unique().tolist()
        path = Path(f'data/curve_trading/history/P/2024/{sid}.parquet')
        quotes.append(pd.read_parquet(source(path), filters=[('date','in',dates),('optionid','in',ids)]))
    quotes = pd.concat(quotes)
    quotes['date'] = pd.to_datetime(quotes.date)
    quotes['expiry'] = pd.to_datetime(quotes.expiry)
    quotes = quotes.set_index(['date','optionid'])
    rows, event_rows = [], []
    for date in events:
        prev = nav.index[nav.index.get_loc(date)-1]
        assert trades.loc[trades.date.eq(date)].empty, 'Use only unchanged-inventory events'
        held = trades.loc[trades.date.le(prev)].groupby(['secid','optionid']).quantity.sum()
        held = held.loc[held.ne(0)]
        option_pnl = stock_pnl = prior_value = current_value = 0.
        for sid in [109072,213108]:
            old_spot, new_spot = stocks.loc[(prev,sid),'close'], stocks.loc[(date,sid),'close']
            assert nav.loc[date,f'split_factor_{sid}'] == 1
            hedge_pnl = nav.loc[prev,f'hedge_{sid}']*(new_spot-old_spot)
            stock_pnl += hedge_pnl
            leg_option_pnl = 0.
            for (s,oid),qty in held.items():
                if s != sid:
                    continue
                a,b = quotes.loc[(prev,oid)],quotes.loc[(date,oid)]
                old_mid,new_mid = (a.bid+a.ask)/2,(b.bid+b.ask)/2
                prior_value += qty*100*old_mid
                current_value += qty*100*new_mid
                change = qty*100*(new_mid-old_mid)
                option_pnl += change;leg_option_pnl += change
                old_time = (a.expiry-prev).days/365
                new_time = (b.expiry-date).days/365
                old_lower = max(a.strike*np.exp(-a.bs_rate*old_time)-(old_spot-a.known_dividend_pv),0)
                new_lower = max(b.strike*np.exp(-b.bs_rate*new_time)-(new_spot-b.known_dividend_pv),0)
                constant_iv = float(option_values(new_spot-b.known_dividend_pv,b.strike,new_time,b.bs_rate,a.bs_iv,'P')[0]) if np.isfinite(a.bs_iv) else np.nan
                rows.append(dict(date=str(date.date()), previous_date=str(prev.date()), secid=sid, optionid=int(oid),
                                 quantity=int(qty), strike=b.strike, expiry=str(b.expiry.date()),
                                 prior_spot=old_spot,current_spot=new_spot,
                                 prior_bid=a.bid,prior_ask=a.ask,current_bid=b.bid,current_ask=b.ask,
                                 prior_mid=old_mid,current_mid=new_mid,
                                 prior_iv=a.bs_iv,current_iv=b.bs_iv,prior_bs_status=a.bs_status,current_bs_status=b.bs_status,
                                 prior_european_lower_bound=old_lower,current_european_lower_bound=new_lower,
                                 option_mid_pnl=change,
                                 current_price_at_prior_iv=constant_iv,
                                 current_mark_minus_constant_iv_pnl=qty*100*(new_mid-constant_iv)))
            event_rows.append(dict(date=str(date.date()), secid=sid, option_mid_pnl=leg_option_pnl,
                                   stock_hedge_pnl=hedge_pnl, gross_hedged_pnl=leg_option_pnl+hedge_pnl))
        np.testing.assert_allclose([prior_value,current_value],[nav.loc[prev,'option_value'],nav.loc[date,'option_value']],atol=1e-7,rtol=0)
        expected = option_pnl+stock_pnl+nav.loc[date,'interest']-nav.loc[date,'stock_borrow_cost']+nav.loc[date,'dividend_cash']-nav.loc[date,'hedge_cost']
        np.testing.assert_allclose(expected,nav.loc[date,'pnl'],atol=1e-7,rtol=0)
    pd.DataFrame(rows).to_csv(OUT/'reversing_contract_marks.csv',index=False)
    pd.DataFrame(event_rows).to_csv(OUT/'reversing_leg_pnl.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = {}

    def source(path):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    defs = json.loads(source(Path('artifacts/contracts/tail_put25_pooled/inputs.json')).read_text())['definitions']
    comparison = pd.read_csv(source(REPORT / 'comparison.csv'), float_precision='round_trip').set_index('model')
    roots = {
        'put25_frozen': lambda p: Path('artifacts/contracts/tail_put25_residual_rl') / p / 'context_residual_rl/full',
        'P75_floor': lambda p: Path('artifacts/contracts/smile_sizing/pairs') / p / 'P75/floor/full',
        'P75_flex': lambda p: Path('artifacts/contracts/smile_sizing/pairs') / p / 'P75/flex/full',
    }
    summary, concentration, extreme, entries, all_pnl = [], [], [], [], {}
    for model, locate in roots.items():
        book = pd.read_csv(source(REPORT / f'{model}_full_portfolio.csv'), index_col=0, parse_dates=[0], float_precision='round_trip')
        cal = book.index
        total = (book.nav.diff() - book.cash_benchmark.diff()).fillna(0)
        exposures = pd.DataFrame(0., index=cal, columns=METRICS + ['abs_pair_net_vega'])
        parts, pair_info, entry_vegas, entry_contracts = {}, {}, [], []
        n_entries = 0
        for d in defs:
            full = locate(d['pair'])
            status = json.loads(source(full / 'status.json').read_text())
            assert status['status'] == 'complete'
            n_entries += status['entries']
            if status.get('cash_only'):
                continue
            nav = pd.read_parquet(source(full / 'daily_nav.parquet'))
            assert nav.index.equals(cal)
            assert set(METRICS).issubset(nav.columns)
            exposures[METRICS] += nav[METRICS].astype(float)
            exposures['abs_pair_net_vega'] += nav.net_vega.abs()
            parts[d['pair']] = (nav.nav.diff() - nav.cash_benchmark.diff()).fillna(0)
            pair_info[d['pair']] = d['label']
            decisions = pd.read_csv(source(full / 'decisions.csv'))
            entered = decisions.loc[decisions.status.eq('entered')].copy()
            assert len(entered) == status['entries']
            entry_vegas.extend((entered.vega_a.abs() + entered.vega_b.abs()).div(2).tolist())
            trades = pd.read_csv(source(full / 'trades.csv'))
            contracts = trades.loc[trades.reason.eq('entry'), 'quantity'].abs()
            entry_contracts.extend(contracts.tolist())
            for r in entered.itertuples():
                entries.append(dict(model=model, pair=d['pair'], label=d['label'],
                                    date=r.entry_date, action=r.action,
                                    mean_leg_vega=(abs(r.vega_a)+abs(r.vega_b))/2))
        pnl = pd.DataFrame(parts, index=cal)
        np.testing.assert_allclose(pnl.sum(axis=1), total, atol=1e-5, rtol=0)
        all_pnl[model] = pnl
        x = total.iloc[1:]
        centered = x - x.mean()
        ss = centered.pow(2).sum()
        var = x.var()
        sharpe = np.sqrt(252) * book.excess_return.iloc[1:].mean() / book.excess_return.iloc[1:].std()
        # Match the frozen score function: exclude the initial zero-return row.
        frozen_sharpe = sharpe
        np.testing.assert_allclose(frozen_sharpe, comparison.loc[model, 'annualized_excess_sharpe'], atol=1e-10, rtol=0)
        row = dict(model=model, entries=n_entries, active_pairs=len(parts),
                   daily_excess_pnl_sd=float(x.std()),
                   daily_excess_return_sd=float(book.excess_return.iloc[1:].std()),
                   annualized_excess_sharpe=float(frozen_sharpe),
                   mean_entry_leg_vega=float(np.mean(entry_vegas)),
                   median_entry_leg_vega=float(np.median(entry_vegas)),
                   mean_entry_contracts_per_leg=float(np.mean(entry_contracts)),
                   largest_10_days_variance_share=float(centered.pow(2).nlargest(10).sum()/ss),
                   all_calendar_sessions=len(cal),
                   inventory_active_sessions=int(exposures.active_cohorts.gt(0).sum()))
        for metric in exposures:
            row[f'mean_{metric}'] = float(exposures[metric].iloc[1:].mean())
            row[f'max_{metric}'] = float(exposures[metric].max())
            row[f'sum_{metric}'] = float(exposures[metric].sum())
        row['daily_pnl_sd_per_mean_gross_vega'] = row['daily_excess_pnl_sd']/row['mean_gross_vega']
        summary.append(row)
        for pair in pnl:
            z = pnl[pair].iloc[1:]
            concentration.append(dict(model=model, pair=pair, label=pair_info[pair],
                                      variance_contribution=float(z.cov(x)/var),
                                      daily_pnl_sd=float(z.std()), net_excess_pnl=float(z.sum())))
        for date in centered.abs().nlargest(10).index:
            contributors = pnl.loc[date].sort_values(key=lambda s: s.abs(), ascending=False).iloc[:5]
            for rank, (pair, value) in enumerate(contributors.items(), 1):
                extreme.append(dict(model=model, date=str(date.date()),
                                    portfolio_daily_pnl=float(total.loc[date]),
                                    pair=pair, label=pair_info[pair], rank=rank,
                                    pair_daily_pnl=float(value)))
        exposures['daily_excess_pnl'] = total
        exposures.to_csv(OUT / f'{model}_exposures.csv')
        print(f'{model}: {n_entries} entries, daily SD ${x.std():,.2f}, mean gross vega {exposures.gross_vega.iloc[1:].mean():,.2f}', flush=True)
    pd.DataFrame(summary).to_csv(OUT / 'summary.csv', index=False)
    pd.DataFrame(concentration).sort_values(['model','variance_contribution'], ascending=[True,False]).to_csv(OUT / 'variance_by_pair.csv', index=False)
    pd.DataFrame(extreme).to_csv(OUT / 'largest_daily_moves.csv', index=False)
    pd.DataFrame(entries).to_csv(OUT / 'entry_sizes.csv', index=False)
    common = sorted(set(all_pnl['put25_frozen']) & set(all_pnl['P75_floor']))
    matched = {m: float(p[common].sum(axis=1).iloc[1:].std()) for m,p in all_pnl.items() if m != 'P75_flex'}
    check_reversing_marks(source)
    combinations = {'ATM_plus_put25':['atm_frozen','put25_frozen'],
                    'whole_smile_P75_floor':['atm_frozen','put25_frozen','P75_floor'],
                    'whole_smile_P75_flex':['atm_frozen','put25_frozen','P75_flex']}
    stack_rows = []
    for name,members in combinations.items():
        book = pd.read_csv(source(REPORT/f'{name}_full_portfolio.csv'),index_col=0,parse_dates=[0],float_precision='round_trip')
        constituent_books = [pd.read_csv(source(REPORT/f'{m}_full_portfolio.csv'),index_col=0,parse_dates=[0],float_precision='round_trip')[['nav','cash_benchmark']] for m in members]
        expected = sum(constituent_books)
        assert expected.index.equals(book.index)
        np.testing.assert_allclose(book[['nav','cash_benchmark']],expected,atol=1e-5,rtol=0)
        returns = expected.nav.pct_change()-expected.cash_benchmark.pct_change()
        sharpe = np.sqrt(252)*returns.iloc[1:].mean()/returns.iloc[1:].std()
        np.testing.assert_allclose(sharpe,comparison.loc[name,'annualized_excess_sharpe'],atol=1e-10,rtol=0)
        stack_rows.append(dict(model=name,annualized_excess_sharpe=sharpe,funded_sleeve_sets=len(members)))
    pd.DataFrame(stack_rows).to_csv(OUT/'verified_atm_combinations.csv',index=False)
    verified_sources = 0
    for manifest in [Path('reports/smile_sizing/output_hashes.json'),Path('reports/tail_put25_residual_rl/output_hashes.json')]:
        expected_hashes = json.loads(manifest.read_text())
        for path,value in hashes.items():
            if path in expected_hashes:
                assert value == expected_hashes[path], f'Frozen source changed: {path}'
                verified_sources += 1
    assert verified_sources > 0
    audit = dict(status='complete', read_only_saved_ledger_analysis=True,
                 retrained=False, shared_inventory=False,
                 common_active_pairs=len(common), common_pair_daily_sd=matched,
                 combined_ledgers_reproduced=True,frozen_sources_verified=verified_sources,
                 resource_limits='one worker, 50% of one core, 3 GiB RSS stop',
                 source_hashes=hashes)
    (OUT / 'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='source_hashes'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
