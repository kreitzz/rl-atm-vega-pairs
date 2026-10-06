"""Restore only the memory filter on broad signals and audit actual exposures."""
from dataclasses import replace
import gc
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd

from scripts.repair_tail_put25_pooled import AccountingReplay, save
from scripts.run_tail_put25_rule_sizing import PutMarket
from scripts.run_broad_contract_replay import Market, ROOT
from scripts.curve_trading.config import config
from scripts.run_atm65_ablation import read_actions, write_json
from scripts.select_atm_contracts import load_calendar
from scripts.option_flex.black_scholes import fit_quotes
from scripts.option_flex.contract_engine import UnresolvedPosition
from scripts.replay_frozen_atm_commission import cash_book
from scripts.run_aligned_contract_learning import mark_book
from scripts.run_selected_pair_backtest import scores
from scripts.run_selected_pair_backtest import audit
from scripts.run_contract_walkforward import digest

BASE = Path('artifacts/contracts/tail_put25_pooled')
COMPLETE = Path('artifacts/contracts/tail_put25_pooled_complete')
OUT = Path('artifacts/contracts/tail_universe_change')
REPORT = Path('reports/tail_universe_change')


def memory_filter(panel):
    return (panel.count_91d.ge(21) & panel.count_365d.ge(63) &
            panel.profit_91d.gt(0) & panel.profit_365d.gt(0))


def working_inputs(assets, secids):
    q, s, sel, divs = [pd.concat([a[c] for a in assets], ignore_index=True) for c in ['q', 's', 'sel', 'divs']]
    s.loc[s.secid.eq(106899) & s.date.ge('2019-05-31'), 'cfadj'] /= 1.02
    if 108969 in secids:
        distributions = pd.read_parquet(ROOT / 'distributions.parquet')
        supplement = (distributions.secid.eq(108969) & distributions.distr_type.eq('5') &
            distributions.currency.eq('USD') & distributions.adj_factor.eq(1) & distributions.approx_flag.eq('0') &
            distributions.liquid_flag.eq('0') & distributions.cancel_flag.eq('0') & distributions.amount.ge(0))
        distributions.loc[supplement, 'distr_type'] = '1'
        extra = distributions.loc[supplement].copy()
        for col in ['ex_date', 'declare_date']: extra[col] = pd.to_datetime(extra[col])
        divs = pd.concat([divs, extra], ignore_index=True)
        rows = q.secid.eq(108969) & q.bs_status.eq('unsupported_distribution')
        if rows.any():
            fitted = fit_quotes(q.loc[rows].drop(columns=['close', 'dte'], errors='ignore'), s,
                pd.read_parquet(ROOT / 'zero_curve.parquet'), distributions)
            for col in ['bs_iv', 'bs_rate', 'known_dividend_pv', 'bs_status']:
                q.loc[rows, col] = fitted[col].to_numpy()
            q.loc[rows, 'delta'] = fitted.bs_delta.to_numpy()
            q.loc[rows, 'vega'] = fitted.bs_vega.to_numpy()
    return q, s, sel, divs


def gated_control(defs, market, cfg):
    loader = PutMarket(market, cfg, {})
    cal = market.calendar[market.calendar >= '2018-01-01']
    cash = cash_book(cal, market.rates, cfg.capital)
    book = cash[['nav', 'cash_benchmark']] * len(defs)
    entries = active = 0; failures = []; costs = dict(option_cost=0., hedge_cost=0.)
    for n, d in enumerate(defs, 1):
        path = BASE / d['pair'] / 'states.parquet'
        dest = OUT / d['pair'] / 'full'; dest.mkdir(parents=True, exist_ok=True)
        if not (dest / 'status.json').exists():
            source = COMPLETE / d['pair'] / 'ungated_spread/actions.csv'
            if path.exists() and source.exists():
                panel = pd.read_parquet(path); codes = read_actions(source)
                codes.loc[~memory_filter(panel.loc[codes.index])] = 0
                codes.to_csv(dest.parent / 'actions.csv')
            else:
                codes = pd.Series(dtype=int)
            if not codes.ne(0).any():
                write_json(dest / 'status.json', dict(status='complete', cash_only=True, entries=0,
                    pair=d['pair'], label=d['label']))
            else:
                loader.retain(d['secids']); assets = [loader.asset(sid) for sid in d['secids']]
                q, s, sel, divs = working_inputs(assets, d['secids'])
                engine = AccountingReplay(q, s, sel, d['secids'], divs, cfg)
                try:
                    nav, trades, decisions = engine.run(codes, cal)
                    mark_book(nav, market.rates, cfg.capital)
                    if trades.empty:
                        checks = audit(nav, trades, s, cfg, d['secids'])
                        np.testing.assert_allclose(nav.nav, cash.nav, atol=1e-7, rtol=0)
                        assert not decisions.status.eq('entered').any()
                        decisions.to_csv(dest / 'decisions.csv', index=False)
                        write_json(dest / 'status.json', dict(status='complete', cash_only=True, entries=0,
                            pair=d['pair'], label=d['label'], reason='all_proposals_rejected_at_entry', **checks))
                    else:
                        save(dest, nav, trades, decisions, cfg, s, d)
                except UnresolvedPosition as exc:
                    write_json(dest / 'status.json', dict(status='incomplete', pair=d['pair'], error=str(exc)))
                del engine, assets, q, s, sel, divs; gc.collect()
                print(f'Memory-only control {n}/{len(defs)} {d["label"]}', flush=True)
        status = json.loads((dest / 'status.json').read_text())
        if status['status'] != 'complete':
            failures.append(status); continue
        nav = cash if status.get('cash_only') else pd.read_parquet(dest / 'daily_nav.parquet')
        book += nav[['nav', 'cash_benchmark']] - cash[['nav', 'cash_benchmark']]
        entries += status['entries']; active += int(status['entries'] > 0)
        for key in costs: costs[key] += status.get(key, 0.)
    loader.cache.clear(); gc.collect()
    if failures:
        raise RuntimeError(str(failures))
    book['return_'] = book.nav.pct_change().fillna(0)
    book['excess_return'] = book.return_ - book.cash_benchmark.pct_change().fillna(0)
    book.to_csv(REPORT / 'restored_memory_full_portfolio.csv')
    return dict(model='broad_rule_with_memory_only', entries=entries, active_pairs=active,
        net_excess_pnl=float(book.nav.iloc[-1] - book.cash_benchmark.iloc[-1]),
        **costs, **scores(book, cfg.capital * len(defs)))


def exposure_summary(name, root, model, portfolio_path, capital, sources):
    book = pd.read_csv(portfolio_path, index_col=0, parse_dates=[0])
    sources[str(portfolio_path)] = digest(portfolio_path)
    totals = pd.DataFrame(0., index=book.index, columns=['net', 'gross', 'absolute_pair_net'])
    entries = mismatched = 0; start_imbalances = []; unpaired_pnl = other_pnl = 0.
    unpaired_pair_sessions = 0; pair_rows = []
    for path in root.glob(f'*/{model}/full/daily_nav.parquet'):
        sources[str(path)] = digest(path)
        nav = pd.read_parquet(path)
        assert nav.index.equals(book.index)
        totals['net'] += nav.net_vega
        totals['gross'] += nav.gross_vega
        totals['absolute_pair_net'] += nav.net_vega.abs()
        trades_path = path.parent / 'trades.csv'; decision_path = path.parent / 'decisions.csv'
        sources[str(trades_path)], sources[str(decision_path)] = digest(trades_path), digest(decision_path)
        try:
            trades = pd.read_csv(trades_path, parse_dates=['date'])
        except pd.errors.EmptyDataError:
            assert nav.gross_vega.eq(0).all() and nav.stock_value.eq(0).all()
            continue
        decisions = pd.read_csv(decision_path)
        entered = decisions[decisions.status.eq('entered')]
        entries += len(entered)
        start_imbalances.extend((abs(entered.vega_a-entered.vega_b) / entered[['vega_a','vega_b']].max(axis=1)).to_list())
        closed = trades[trades.reason.isin(['expiry_settlement', 'announced_corporate_action_exit'])]
        intervals = closed.groupby('cohort').date.agg(['min', 'max'])
        unequal = intervals[intervals['min'].lt(intervals['max'])]
        mismatched += len(unequal)
        unpaired = np.zeros(len(nav), bool)
        for r in unequal.itertuples():
            # P&L over prior close to current close; include the final leg's exit day.
            unpaired |= (nav.index > r.min) & (nav.index <= r.max)
        dollar_pnl = nav.excess_return * nav.nav.shift(1).fillna(capital)
        unpaired_pnl += float(dollar_pnl.loc[unpaired].sum())
        other_pnl += float(dollar_pnl.loc[~unpaired].sum())
        unpaired_pair_sessions += int(unpaired.sum())
        pair_rows.append(dict(strategy=name, pair=path.parents[2].name, entries=len(entered),
            unequal_exit_cohorts=len(unequal), unpaired_pair_sessions=int(unpaired.sum()),
            excess_daily_pnl_on_unpaired_sessions=float(dollar_pnl.loc[unpaired].sum()),
            excess_daily_pnl_on_other_sessions=float(dollar_pnl.loc[~unpaired].sum())))
    active = totals.gross.gt(1e-9)
    ratio = totals.net.abs()[active] / totals.gross[active]
    local = totals.absolute_pair_net[active] / totals.gross[active]
    excess = book.excess_return.iloc[1:]
    totals.to_csv(REPORT / f'{name}_daily_vega.csv')
    metrics = dict(model=name, entries=entries, unequal_exit_cohorts=mismatched,
        unequal_exit_fraction=mismatched / entries if entries else 0.,
        mean_entry_leg_vega_imbalance=float(np.mean(start_imbalances)),
        max_entry_leg_vega_imbalance=float(max(start_imbalances)),
        active_vega_sessions=int(active.sum()), mean_absolute_book_net_over_gross=float(ratio.mean()),
        p95_absolute_book_net_over_gross=float(ratio.quantile(.95)),
        mean_sum_absolute_pair_net_over_gross=float(local.mean()),
        unpaired_pair_sessions=unpaired_pair_sessions,
        excess_daily_pnl_on_unpaired_pair_sessions=unpaired_pnl,
        excess_daily_pnl_on_other_pair_sessions=other_pnl,
        annualized_excess_return=float(excess.mean() * 252), annualized_excess_volatility=float(excess.std() * np.sqrt(252)),
        annualized_excess_sharpe=float(excess.mean() / excess.std() * np.sqrt(252)))
    return metrics, pair_rows


def main():
    started = time.monotonic(); OUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    defs = json.loads((BASE / 'inputs.json').read_text())['definitions']
    protected = {}
    for study in ['tail_put25_strategy', 'tail_put25_pooled', 'tail_put25_pooled_complete']:
        protected.update(json.loads(Path(f'reports/{study}/output_hashes.json').read_text()))
    sources = {}; summaries = []; pairs = []
    for name, root, model, path in [
        ('frozen_rule', Path('artifacts/contracts/tail_put25_strategy'), 'rule', Path('reports/tail_put25_strategy/rule_full_portfolio.csv')),
        ('ungated_spread', COMPLETE, 'ungated_spread', Path('reports/tail_put25_pooled_complete/ungated_spread_full_portfolio.csv')),
        ('pooled_rl_sized', COMPLETE, 'pooled_rl_sized', Path('reports/tail_put25_pooled_complete/pooled_rl_sized_full_portfolio.csv'))]:
        row, perpair = exposure_summary(name, root, model, path, 1_000_000, sources)
        summaries.append(row); pairs.extend(perpair)
    pd.DataFrame(summaries).to_csv(REPORT / 'exposures.csv', index=False)
    pd.DataFrame(pairs).to_csv(REPORT / 'holding_intervals_by_pair.csv', index=False)
    cal = load_calendar(ROOT); cal = cal[(cal >= '2010-01-01') & (cal <= '2025-08-29')]
    row = gated_control(defs, Market(cal), replace(config(), option_commission=.03))
    old = pd.read_csv('reports/tail_put25_strategy/comparison.csv').query("model == 'rule' and period == 'full'").iloc[0].to_dict()
    old['model'] = 'frozen_rule'
    new = pd.read_csv('reports/tail_put25_pooled_complete/comparison.csv').query("period == 'full'").to_dict('records')
    pd.DataFrame([old, *new, row]).to_csv(REPORT / 'comparison.csv', index=False)
    for name, table in [('protected', protected), ('sources', sources)]:
        for path, sha in table.items(): assert digest(path) == sha, f'{name} input changed: {path}'
    write_json(REPORT / 'audit.json', dict(status='complete', sources_verified=len(sources), protected_files_verified=len(protected),
        elapsed_seconds=time.monotonic()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        attribution_limit='Unpaired-session P&L is a diagnostic bucket at pair-book level, not isolated single-leg or Greek P&L; paired cohorts can overlap those dates. Entry neutrality and daily net-vega statistics do not establish market-vol-factor neutrality. No retraining; the new replay restores only the historical completed-profit/count filter on the same broad signal calendar, quotes, initial vega sizing, costs and lifecycle handling.'))
    (REPORT / 'findings.txt').write_text(pd.DataFrame(summaries).to_string(index=False)+'\n\n'+
        pd.read_csv(REPORT / 'comparison.csv').to_string(index=False)+'\n\n'+(REPORT / 'audit.json').read_text())
    hashes = {str(p): digest(p) for root in [OUT, REPORT] for p in root.rglob('*') if p.is_file() and p.name != 'output_hashes.json'}
    hashes[str(Path(__file__))] = digest(Path(__file__))
    write_json(REPORT / 'output_hashes.json', hashes)
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)
    print(pd.read_csv(REPORT / 'comparison.csv').to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
