"""Attribute calendar-year outcomes of saved books; no retraining or exclusions."""
from dataclasses import asdict
import gc
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd

from scripts.shared_trade_ledger import reconstruct_shared, FIELDS, connected_episodes
from scripts.run_contract_walkforward import digest
from scripts.run_atm65_ablation import write_json
from scripts.curve_trading.config import config
from scripts.run_aligned_contract_learning import asset_paths
from scripts.tail_put25_pooled import FEATURES

BASE = Path('artifacts/contracts/p25_feature_search')
MARKET = Path('artifacts/contracts/p25_signal_transfer/market')
OUT = Path('reports/p25_variance_years')
MODELS = ['variance_premium', 'control', 'fixed2']
FOCUS = [2019, 2021, 2022, 2024]


def forward_stats(stock, start, end):
    close = (stock.close * stock.cfadj).loc[start:end]
    r = np.log(close / close.shift(1)).iloc[1:]
    if not len(r) or not np.isfinite(r).all():
        return dict(forward_rv=np.nan, largest_move_pct=np.nan, largest_move_date=pd.NaT)
    biggest = r.abs().idxmax()
    return dict(forward_rv=float(100 * np.sqrt(252 * r.pow(2).mean())), largest_move_pct=float(100 * np.expm1(r.loc[biggest])), largest_move_date=biggest)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    sources = {}
    def track(path):
        path = Path(path)
        if str(path) not in sources:
            sources[str(path)] = digest(path)
        return path
    def read(path, **kwargs):
        return pd.read_parquet(track(path), **kwargs)
    def csv(path, **kwargs):
        return pd.read_csv(track(path), **kwargs)
    definitions = json.loads(track(Path('artifacts/contracts/tail_put25_pooled/inputs.json')).read_text())['definitions']
    definitions = {d['pair']: d for d in definitions}
    labels = {pair: d['label'] for pair, d in definitions.items()}
    symbols = {}
    for d in definitions.values():
        for sid, symbol in zip(d['secids'], d['label'].split('/')):
            symbols[int(sid)] = symbol
    stocks = read(MARKET / 'stocks.parquet')
    divs = read(MARKET / 'dividends.parquet')
    bystock = {int(sid): g.set_index('date').sort_index() for sid, g in stocks.groupby('secid')}
    earnings = {}
    annual = []
    pair_years = []
    cohort_years = []
    all_legs = []
    extreme_days = []
    quote_audits = []
    entry_profiles = []
    cases = []
    for model in MODELS:
        folder = BASE / 'replays' / model
        nav = read(folder / 'daily_nav.parquet')
        trades = csv(folder / 'trades.csv', parse_dates=['date'])
        decisions = csv(folder / 'decisions.csv', parse_dates=['signal_date', 'entry_date'])
        proposals = read(folder / 'scored_allocations.parquet')
        ids = trades.optionid.unique().tolist()
        quotes = pd.concat([read(MARKET / f'{year}_quotes.parquet', filters=[('optionid', 'in', ids)]) for year in range(2018, 2026)], ignore_index=True)
        legs, daily, financing, vegas, gross_vegas, error, components = reconstruct_shared(nav, trades, quotes, stocks, divs, asdict(config()))
        legs['model'] = model
        legs['pair'] = legs.cohort.str.split('|').str[1]
        legs['label'] = legs.pair.map(labels)
        legs['symbol'] = legs.secid.map(symbols)
        for idx, leg in legs.iterrows():
            for field, value in forward_stats(bystock[int(leg.secid)], leg.entry_date, leg.exit_date).items():
                legs.loc[idx, field] = value
            pair_def = definitions[leg.pair]
            for sid, path in zip(pair_def['secids'], asset_paths(pair_def)):
                sid = int(sid)
                if sid not in earnings and path.exists():
                    earnings[sid] = read(path, columns=['earnings']).earnings
            schedule = earnings.get(int(leg.secid))
            legs.loc[idx, 'historical_earnings_days_at_entry'] = schedule.get(leg.entry_date, np.nan) if schedule is not None else np.nan
            legs.loc[idx, 'historical_earnings_days_at_largest_move'] = schedule.get(legs.loc[idx, 'largest_move_date'], np.nan) if schedule is not None else np.nan
        group = legs.groupby('cohort')
        meta = pd.DataFrame({'entry_date': group.entry_date.min(), 'exit_date': group.exit_date.max()})
        meta['signal_date'] = pd.to_datetime(meta.index.str.split('|').str[0])
        meta['pair'] = meta.index.str.split('|').str[1]
        meta['wing'] = meta.index.str.split('|').str[2]
        meta['label'] = meta.pair.map(labels)
        meta['model'] = model
        meta['lifecycle_net_before_financing'] = daily.sum()
        meta['episode'] = connected_episodes(meta)
        selected = ['date', 'pair', 'direction', 'learned_level'] + FEATURES + ['ewma_vrp', 'ewma_vrp_available', 'ewma_vrp_z', 'ewma_vrp_z_available', 'variance_ratio', 'variance_ratio_available']
        proposal_lookup = proposals[selected].set_index(['date', 'pair'])
        keys = pd.MultiIndex.from_frame(meta[['signal_date', 'pair']])
        for field in selected[2:]:
            meta[field] = proposal_lookup[field].reindex(keys).to_numpy()
        for role in ['long', 'short']:
            own = legs[legs.role.eq(role)].set_index('cohort')
            for field in ['symbol', 'contracts', 'initial_vega', 'entry_iv', 'entry_dte', 'net_before_shared_financing', 'forward_rv', 'largest_move_pct', 'largest_move_date', 'historical_earnings_days_at_entry', 'historical_earnings_days_at_largest_move']:
                meta[role + '_' + field] = own[field].reindex(meta.index)
        all_legs.append(legs)
        benchmark = nav.cash_benchmark.diff().fillna(0)
        for year, year_nav in nav.groupby(nav.index.year):
            use = nav.index.year == year
            values = {field: float(table.loc[use].to_numpy().sum()) for field, table in components.items()}
            gross = sum(values[f] for f in FIELDS[:3])
            charges = sum(values[f] for f in FIELDS[3:])
            fund = float(financing.loc[use].sum())
            pnl = daily.loc[use].sum()
            entered = decisions.status.eq('entered') & decisions.entry_date.dt.year.eq(year)
            returns = year_nav.excess_return.iloc[1:] if year == 2018 else year_nav.excess_return
            excess_daily = (nav.pnl - benchmark).loc[use]
            bypair = pnl.groupby(meta.pair).sum()
            negative = bypair[bypair.lt(0)].sort_values()
            positive = bypair[bypair.gt(0)].sort_values(ascending=False)
            record = dict(model=model, year=int(year), entries=int(entered.sum()), active_pairs=int(bypair.ne(0).sum()), winning_pairs=int(bypair.gt(0).sum()), losing_pairs=int(bypair.lt(0).sum()), annualized_cash_excess_sharpe=float(np.sqrt(252) * returns.mean() / returns.std(ddof=1)), gross_delta_hedged_pnl=gross, execution_and_borrow=charges, financing_vs_cash=fund, net_excess_pnl=float(excess_daily.sum()), carried_cohort_pnl=float(pnl.loc[meta.entry_date.dt.year.lt(year)].sum()), worst_day=str(excess_daily.idxmin().date()), worst_day_pnl=float(excess_daily.min()), worst_3_days_pnl=float(excess_daily.nsmallest(3).sum()), other_days_pnl=float(excess_daily.sum() - excess_daily.nsmallest(3).sum()), top_3_losing_pairs=','.join(labels[p] for p in negative.index[:3]), top_3_losing_pair_pnl=float(negative.iloc[:3].sum()), top_3_winning_pairs=','.join(labels[p] for p in positive.index[:3]), top_3_winning_pair_pnl=float(positive.iloc[:3].sum()), active_sessions=int(year_nav.active_cohorts.gt(0).sum()), mean_gross_vega=float(year_nav.gross_vega.mean()), greek_fallbacks=int(year_nav.greek_fallbacks.sum()), modeled_marks=int(year_nav.causal_held_mark_fallbacks.sum()), crossed_quote_repairs=int(year_nav.repaired_held_quotes.sum()), **values)
            np.testing.assert_allclose(gross - charges + fund, record['net_excess_pnl'], atol=1e-5, rtol=0)
            annual.append(record)
            for pair, amount in bypair[bypair.ne(0)].items():
                cohorts = meta.index[meta.pair.eq(pair)]
                pair_years.append(dict(model=model, year=int(year), pair=pair, label=labels[pair], calendar_net_before_shared_financing=float(amount), entries=int((entered & decisions.pair.eq(pair)).sum()), carried_pnl=float(pnl.loc[cohorts[meta.loc[cohorts, 'entry_date'].dt.year.lt(year)]].sum())))
            active = pnl.ne(0)
            detail = meta.loc[active].copy()
            detail['year'] = int(year)
            detail['calendar_net_before_shared_financing'] = pnl.loc[active]
            detail['entered_before_year'] = detail.entry_date.dt.year.lt(year)
            cohort_years.append(detail.reset_index())
            new = meta.loc[meta.entry_date.dt.year.eq(year)]
            entry_profiles.append(dict(model=model, year=int(year), entries=len(new), mean_size=float(new.learned_level.mean() / 4), median_beta=float(new.beta.median()), median_correlation=float(new.correlation.median()), median_iv=float(new.iv_mean.median() * 100), median_long_iv=float(new.long_entry_iv.median()), median_short_iv=float(new.short_entry_iv.median()), median_long_forward_rv=float(new.long_forward_rv.median()), median_short_forward_rv=float(new.short_forward_rv.median()), mean_memory91=float(new.profit_91d.mean()), mean_memory365=float(new.profit_365d.mean()), mean_ewma_vrp=float(new.ewma_vrp.mean()), mean_ewma_vrp_z=float(new.ewma_vrp_z.mean()), median_earnings_days_long=float(new.long_historical_earnings_days_at_entry.median()), median_earnings_days_short=float(new.short_historical_earnings_days_at_entry.median()), lifecycle_net_before_shared_financing=float(new.lifecycle_net_before_financing.sum()), note='Entry-year cohorts and lifecycle outcomes; distinct from calendar-year attribution; forward RV and historical earnings fields are diagnosis only'))
            if model == 'variance_premium' and year in FOCUS:
                dates = list(excess_daily.nsmallest(3).index) + list(excess_daily.nlargest(3).index)
                for date in dates:
                    previous = nav.index[nav.index.get_loc(date) - 1]
                    values_at_date = daily.loc[date]
                    contributors = list(values_at_date.nsmallest(3).index) if excess_daily.loc[date] < 0 else list(values_at_date.nlargest(3).index)
                    for cohort in contributors:
                        if abs(values_at_date.loc[cohort]) < 1e-8:
                            continue
                        row = dict(model=model, year=int(year), date=date, cohort=cohort, label=meta.loc[cohort, 'label'], cohort_net_before_financing=float(values_at_date.loc[cohort]), portfolio_excess_pnl=float(excess_daily.loc[date]), **{field: float(table.at[date, cohort]) for field, table in components.items()})
                        extreme_days.append(row)
                        own = trades.loc[trades.cohort.eq(cohort)]
                        held = own.loc[own.date.le(previous)].groupby(['secid', 'optionid']).quantity.sum()
                        lookup = quotes.set_index(['date', 'optionid'])
                        for (sid, oid), qty in held[held.ne(0)].items():
                            s = bystock[int(sid)]
                            first_spot, last_spot = float(s.at[previous, 'close']), float(s.at[date, 'close'])
                            adjustment = float(s.at[date, 'cfadj'] / s.at[previous, 'cfadj'])
                            audit = dict(date=date, previous_date=previous, cohort=cohort, pair=meta.loc[cohort, 'label'], symbol=symbols.get(int(sid)), secid=int(sid), optionid=int(oid), quantity=float(qty), prior_stock=first_spot, current_stock=last_spot, split_adjusted_stock_move_pct=100 * (last_spot * adjustment / first_spot - 1))
                            schedule = earnings.get(int(sid))
                            audit['historical_earnings_days_at_event'] = schedule.get(date, np.nan) if schedule is not None else np.nan
                            for prefix, d in [('prior', previous), ('current', date)]:
                                q = lookup.loc[(d, oid)] if (d, oid) in lookup.index else None
                                audit[prefix + '_observed_quote'] = q is not None
                                if q is not None:
                                    for field in ['bid', 'ask', 'bs_iv', 'bs_status', 'delta', 'vega', 'strike', 'expiry']:
                                        audit[prefix + '_' + field] = q[field]
                                    audit[prefix + '_mid'] = (q.bid + q.ask) / 2
                            quote_audits.append(audit)
        daily.to_parquet(OUT / f'{model}_cohort_daily.parquet')
        financing.to_frame('financing_vs_cash').to_parquet(OUT / f'{model}_financing.parquet')
        meta.to_csv(OUT / f'{model}_cohort_profiles.csv')
        cases.append(dict(model=model, cohorts=len(meta), maximum_daily_ledger_error=error))
        pd.DataFrame(annual).to_csv(OUT / 'annual_components.csv', index=False)
        pd.DataFrame(pair_years).to_csv(OUT / 'annual_pairs.csv', index=False)
        pd.concat(cohort_years, ignore_index=True).to_csv(OUT / 'calendar_cohorts.csv', index=False)
        pd.concat(all_legs, ignore_index=True).to_csv(OUT / 'legs.csv', index=False)
        pd.DataFrame(extreme_days).to_csv(OUT / 'extreme_days.csv', index=False)
        pd.DataFrame(quote_audits).to_csv(OUT / 'event_quotes.csv', index=False)
        pd.DataFrame(entry_profiles).to_csv(OUT / 'entry_profiles.csv', index=False)
        write_json(OUT / 'progress.json', dict(model=model, books_complete=len(cases), maximum_daily_error=error, elapsed_seconds=time.monotonic() - started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        print(f'{model}: {len(meta)} cohorts, daily ledger error ${error:.8f}', flush=True)
        print(pd.DataFrame(annual).query('model == @model')[['year', 'entries', 'net_excess_pnl', 'gross_delta_hedged_pnl', 'execution_and_borrow', 'financing_vs_cash', 'top_3_losing_pairs', 'top_3_winning_pairs']].to_string(index=False), flush=True)
        del nav, trades, decisions, quotes, legs, daily, components, vegas, gross_vegas, meta
        gc.collect()
    for code in [Path(__file__), Path('scripts/shared_trade_ledger.py'), Path('scripts/option_flex/black_scholes.py')]:
        track(code)
    for path, expected in sources.items():
        assert digest(Path(path)) == expected, path
    write_json(OUT / 'audit.json', dict(status='complete', models=cases, maximum_daily_ledger_error=max(c['maximum_daily_ledger_error'] for c in cases), attribution_only=True, no_new_trades_or_retraining=True, exact_observed_marks=True, stock_netting_and_cost_allocation_preserved=True, financing_separate_from_cohorts=True, calendar_year_pnl_not_entry_year_pnl=True, forward_rv_and_earnings_diagnostic_only=True, source_hashes=sources, elapsed_seconds=time.monotonic() - started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    write_json(OUT / 'progress.json', dict(stage='complete', books_complete=len(cases)))
    write_json(OUT / 'output_hashes.json', {str(p): digest(p) for p in OUT.glob('*') if p.is_file() and p.name != 'output_hashes.json'})


if __name__ == '__main__':
    main()
