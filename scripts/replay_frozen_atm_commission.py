"""Replay frozen ATM30 decisions; change only the option commission.

One worker, per-pair checkpoints, streaming ATM quote filtering. Run through
run_signal_discovery_guarded to enforce the CPU duty cycle and RSS stop.
"""
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd

from scripts.curve_trading.config import config, ACTIONS, COMBOS, DATA
from scripts.curve_trading.policy import Replay
from scripts.run_broad_contract_replay import Market, ROOT
from scripts.run_aligned_contract_learning import mark_book
from scripts.run_selected_pair_backtest import audit, scores
from scripts.run_contract_walkforward import digest
from scripts.select_atm_contracts import load_calendar

SOURCE = Path('artifacts/contracts/curve_trading')
OUTPUT = Path('artifacts/contracts/frozen_atm_commission_003')
REPORT = Path('reports/frozen_atm_commission')
VARIANTS = [('full_065', '2018-01-01', .65),
            ('full_003', '2018-01-01', .03),
            ('recent_065', '2022-01-01', .65),
            ('recent_003', '2022-01-01', .03)]


def cash_book(calendar, rates, capital):
    nav = pd.DataFrame(index=calendar)
    elapsed = calendar.to_series().diff().dt.days.fillna(0) / 365
    nav['return_'] = rates.reindex(calendar).shift(1).fillna(0) * elapsed
    nav['nav'] = capital * (1 + nav.return_).cumprod()
    mark_book(nav, rates, capital)
    return nav


def load_pair(market, secids):
    quotes, selections, stocks = [], [], []
    years = sorted(set(market.calendar.year))
    for sid in secids:
        parts = []
        for year in years:
            path = DATA / 'selection/C' / str(year) / f'{sid}.parquet'
            if path.exists():
                piece = pd.read_parquet(path)
                parts.append(piece[piece.tenor.eq(30) & piece.bucket.eq('atm') & piece.cp_flag.eq('C')])
        selected = pd.concat(parts, ignore_index=True)
        ids = selected.optionid.unique().tolist()
        # Predicate filtering reads only relevant held contracts from each partition.
        parts = []
        for year in years:
            path = DATA / 'history/C' / str(year) / f'{sid}.parquet'
            if path.exists():
                parts.append(pd.read_parquet(path, filters=[('optionid', 'in', ids)]))
        quotes.append(pd.concat(parts, ignore_index=True))
        selections.append(selected)
        stock = market.stocks[market.stocks.secid.eq(sid)].set_index('date').reindex(market.calendar)
        stock['secid'] = sid
        stock['rate'] = market.rates
        stock.index.name = 'date'
        stocks.append(stock.reset_index())
    q, s, selection = [pd.concat(x, ignore_index=True) for x in [quotes, stocks, selections]]
    assert not q.duplicated(['date', 'optionid']).any()
    return q, s, selection, market.divs[market.divs.secid.isin(secids)]


def read_trades(path):
    try:
        return pd.read_csv(path, parse_dates=['date', 'cohort'])
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def save_result(dest, nav, trades, decisions, cfg, stocks, d):
    checks = audit(nav, trades, stocks, cfg, d['secids'])
    assert abs(nav.option_value.iloc[-1]) + abs(nav.stock_value.iloc[-1]) < 1e-7
    if not trades.empty:
        assert set(trades.reason).issubset({'entry', 'split_adjustment', 'expiry_settlement'})
        paid = trades.reason.eq('entry')
        np.testing.assert_allclose(trades.loc[paid, 'commission'],
                                   abs(trades.loc[paid, 'quantity']) * cfg.option_commission)
        assert trades.loc[~paid, 'commission'].eq(0).all()
    entered = decisions[decisions.status.eq('entered')] if len(decisions) else decisions
    if len(entered):
        assert (abs(entered.vega_a - entered.vega_b) / entered[['vega_a', 'vega_b']].max(axis=1)).le(cfg.max_entry_vega_imbalance + 1e-12).all()
    nav.to_parquet(dest / 'daily_nav.parquet')
    trades.to_csv(dest / 'trades.csv', index=False)
    decisions.to_csv(dest / 'decisions.csv', index=False)
    row = dict(status='complete', pair=d['pair'], label=d['label'], entries=len(entered),
               commission=float(trades.commission.sum()) if len(trades) else 0.,
               option_cost=float(nav.option_execution_cost.sum()),
               hedge_cost=float(nav.hedge_cost.sum()), **scores(nav, cfg.capital), **checks)
    (dest / 'status.json').write_text(json.dumps(row, indent=2) + '\n')
    return row


def summary(defs, market, cfg):
    rows, annual, pairs = [], [], []
    for variant, start, fee in VARIANTS:
        cal = market.calendar[market.calendar >= start]
        cash = cash_book(cal, market.rates, cfg.capital)
        book = cash[['nav', 'cash_benchmark']] * len(defs)
        entries = active = 0
        commissions = option_cost = hedge_cost = 0.
        pnl_paths = {}
        for d in defs:
            dest = OUTPUT / d['pair'] / variant
            row = json.loads((dest / 'status.json').read_text())
            assert row['status'] == 'complete'
            if row.get('cash_only'):
                nav = cash
            else:
                nav = pd.read_parquet(row.get('nav_source', dest / 'daily_nav.parquet'))
                assert nav.index.equals(cal)
                book += nav[['nav', 'cash_benchmark']] - cash[['nav', 'cash_benchmark']]
            pnl = float(nav.nav.iloc[-1] - nav.cash_benchmark.iloc[-1])
            if row['entries']:
                pnl_paths[d['pair']] = nav.nav - nav.cash_benchmark
            pairs.append(dict(variant=variant, pair=d['pair'], label=d['label'],
                              entries=row['entries'], net_excess_pnl=pnl))
            entries += row['entries']; active += int(row['entries'] > 0)
            commissions += row.get('commission', 0.)
            option_cost += row.get('option_cost', 0.)
            hedge_cost += row.get('hedge_cost', 0.)
        book['return_'] = book.nav.pct_change().fillna(0)
        book['excess_return'] = book.return_ - book.cash_benchmark.pct_change().fillna(0)
        book.to_csv(REPORT / f'{variant}_portfolio.csv')
        capital = cfg.capital * len(defs)
        result = dict(variant=variant, start=str(cal[0].date()), end=str(cal[-1].date()),
                      commission_per_contract=fee, pairs=len(defs), active_pairs=active,
                      entries=entries, option_commission=commissions, option_cost=option_cost,
                      hedge_cost=hedge_cost, net_excess_pnl=float(book.nav.iloc[-1] - book.cash_benchmark.iloc[-1]),
                      cash_return=float(book.cash_benchmark.iloc[-1] / capital - 1),
                      **scores(book, capital))
        winners = sorted([r for r in pairs if r['variant'] == variant and r['net_excess_pnl'] > 1e-7],
                         key=lambda r: -r['net_excess_pnl'])
        if winners:
            top = winners[0]
            without = book.nav - pnl_paths[top['pair']]
            excess = without.pct_change().fillna(0) - book.cash_benchmark.pct_change().fillna(0)
            result.update(largest_winner=top['label'], largest_winner_pnl=top['net_excess_pnl'],
                          top_three_share_positive_pnl=sum(r['net_excess_pnl'] for r in winners[:3]) / sum(r['net_excess_pnl'] for r in winners),
                          sharpe_largest_winner_replaced_by_cash=float(np.sqrt(252) * excess.iloc[1:].mean() / excess.iloc[1:].std()))
        rows.append(result)
        for year, g in book.groupby(book.index.year):
            annual.append(dict(variant=variant, year=year, total_return_percent=100 * ((1 + g.return_).prod() - 1),
                               cash_return_percent=100 * ((1 + book.cash_benchmark.pct_change().fillna(0).loc[g.index]).prod() - 1),
                               annualized_excess_sharpe=float(np.sqrt(252) * g.excess_return.mean() / g.excess_return.std()) if g.excess_return.std() > 1e-12 else None))
    pd.DataFrame(rows).to_csv(REPORT / 'comparison.csv', index=False)
    pd.DataFrame(annual).to_csv(REPORT / 'annual.csv', index=False)
    pd.DataFrame(pairs).to_csv(REPORT / 'pairs.csv', index=False)
    lines = ['# Frozen ATM strategy: commission replay', '',
             'Frozen saved monthly ATM30 call/call decisions. Full history starts flat in 2018; the matched recent test starts flat in 2022. Both use all 346 original pair sleeves, $1m each, including cash sleeves. No refitting or retrospective pair removal. Only commission changes; NAV-based sizing, integer rounding, daily delta hedges and the original risk limits are reapplied.', '',
             '| Period | Commission | Annualized excess Sharpe | Entries | P&L above cash |',
             '|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['start']} to {r['end']} | ${r['commission_per_contract']:.2f} | {r['annualized_excess_sharpe']:.3f} | {r['entries']} | ${r['net_excess_pnl']:,.0f} |")
    lines += ['', 'Sharpe = daily cash-excess mean / sample standard deviation × √252, excluding the initial session. Midpoint plus a 0.10-vol-point entry penalty per leg and 1 bp stock execution are unchanged. Initial integer vega mismatch ≤10%; option quantities then remain fixed apart from splits and each leg settles at its own expiry. No daily option-vega rebalancing. Settlement carries no modeled commission.', '',
              'The full-history $0.65 column uses the saved original ledger. Its active-pair replay is independently checked against that ledger before the lower-fee run. Recent-period columns are fresh simulator replays, so they need not equal a slice of the older continuously funded portfolio. The policy still contains its original $0.65-trained cost inputs and validation/memory gates: this isolates execution fees rather than retraining for cheaper execution.', '',
              'This is previously explored research history, not an untouched holdout. Replacing the largest winner with cash is an ex-post concentration diagnostic only.', '',
              '[Annual results](frozen_atm_commission/annual.csv) · [Pair results](frozen_atm_commission/pairs.csv) · [Comparison](frozen_atm_commission/comparison.csv)', '']
    REPORT.with_suffix('.md').write_text('\n'.join(lines))
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    source_inputs = SOURCE / 'control_inputs.json'
    defs = json.loads(source_inputs.read_text())['definitions']
    cfg = config(); assert cfg.option_commission == .65 and cfg.hold_to_expiry
    paths = [source_inputs, Path(__file__), Path('scripts/option_flex/contract_engine.py'),
             Path('scripts/curve_trading/policy.py'), Path('scripts/curve_trading/config.py')]
    for d in defs:
        folder = SOURCE / d['pair'] / 'atm_control'
        paths += [p for p in [folder / 'status.json', folder / 'actions.csv', folder / 'daily_nav.parquet', folder / 'trades.csv'] if p.exists()]
    signature = dict(settings=asdict(cfg), new_commission=.03, definitions=defs,
                     frozen_inputs={str(p): digest(p) for p in paths})
    manifest = OUTPUT / 'inputs.json'
    if manifest.exists():
        assert json.loads(manifest.read_text()) == json.loads(json.dumps(signature)), 'Replay inputs changed'
    else:
        manifest.write_text(json.dumps(signature, indent=2) + '\n')
    cal = load_calendar(ROOT); cal = cal[(cal >= '2018-01-01') & (cal <= '2025-08-29')]
    market = Market(cal); started = time.monotonic()
    for n, d in enumerate(defs, 1):
        folder = SOURCE / d['pair'] / 'atm_control'
        old = json.loads((folder / 'status.json').read_text()); assert old['status'] == 'complete'
        actions = (pd.read_csv(folder / 'actions.csv', index_col=0, parse_dates=[0]).code.astype(int)
                   if (folder / 'actions.csv').exists() else pd.Series(dtype=int, index=pd.DatetimeIndex([])))
        assert old.get('structural_cash') or (folder / 'actions.csv').exists()
        assert all(COMBOS[ACTIONS[int(c)][0]] == (30, 'C', 'atm', 'atm') for c in actions[actions.ne(0)])
        replay = None
        for variant, start, fee in VARIANTS:
            dest = OUTPUT / d['pair'] / variant; dest.mkdir(parents=True, exist_ok=True)
            status = dest / 'status.json'
            if status.exists():
                assert json.loads(status.read_text())['status'] == 'complete'; continue
            own = actions[actions.index >= start]; period = cal[cal >= start]
            if variant == 'full_065':
                original = pd.read_parquet(old.get('nav_source', folder / 'daily_nav.parquet'))
                row = dict(status='complete', pair=d['pair'], label=d['label'], entries=old['entries'],
                           nav_source=str(old.get('nav_source', folder / 'daily_nav.parquet')),
                           commission=float(read_trades(folder / 'trades.csv').commission.sum()) if old['entries'] else 0.,
                           option_cost=float(original.option_execution_cost.sum()), hedge_cost=float(original.hedge_cost.sum()))
                status.write_text(json.dumps(row, indent=2) + '\n'); continue
            if not own.ne(0).any():
                assert variant != 'full_003' or old['entries'] == 0
                status.write_text(json.dumps(dict(status='complete', pair=d['pair'], label=d['label'], entries=0, cash_only=True)) + '\n')
                continue
            if replay is None:
                q, s, selected, divs = load_pair(market, d['secids'])
                replay = Replay(q, s, selected, d['secids'], divs, cfg)
                # Every nonzero frozen control must reproduce before repricing.
                check, t, dec = replay.run(actions, cal); mark_book(check, market.rates, cfg.capital)
                original = pd.read_parquet(folder / 'daily_nav.parquet')
                np.testing.assert_allclose(check.nav, original.nav, atol=1e-6, rtol=0)
                checks = audit(check, t, s, cfg, d['secids'])
                assert int(dec.status.eq('entered').sum()) == old['entries']
                (OUTPUT / d['pair'] / 'baseline_reconciliation.json').write_text(json.dumps(dict(max_nav_error=float(abs(check.nav-original.nav).max()), **checks)) + '\n')
                del check, t, dec, original
            replay.cfg = replace(cfg, option_commission=fee)
            nav, t, dec = replay.run(own, period); mark_book(nav, market.rates, cfg.capital)
            row = save_result(dest, nav, t, dec, replay.cfg, replay.s, d)
            print(f'{n}/{len(defs)} {d["label"]} {variant}: {row["entries"]} entries, {time.monotonic()-started:.0f}s', flush=True)
            del nav, t, dec
        if replay is not None:
            del replay, q, s, selected, divs
        gc.collect()
        (OUTPUT / 'progress.json').write_text(json.dumps(dict(completed_pairs=n, pairs=len(defs),
             last_pair=d['label'], elapsed_seconds=time.monotonic()-started,
             peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)) + '\n')
    summary(defs, market, cfg)
    (OUTPUT / 'task_status.json').write_text(json.dumps(dict(status='complete', pairs=len(defs),
         peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         elapsed_seconds=time.monotonic()-started)) + '\n')


if __name__ == '__main__':
    main()
