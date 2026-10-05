"""Finish unchanged NN fits; preserve unresolved books and match the roster.

The original NN source and input hashes remain intact. This continuation only
catches unresolved evaluation paths and aggregates a common complete subset.
No missing sleeve is assigned zero P&L or a synthetic cash portfolio.
"""
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scripts.run_atm_nn_cost_hurdles as nn
from scripts.curve_trading.policy import Replay
from scripts.run_contract_walkforward import digest
from scripts.replay_frozen_atm_commission import cash_book
from scripts.run_selected_pair_backtest import scores
from scripts.retrain_atm_commission import check_fits

OUTPUT, REPORT = nn.OUTPUT, nn.REPORT
ACTIVE_FEE = None
FAILURE = None
ORIGINAL_PREPARE = nn.runner.prepare_hurdle


def prepare(*args):
    global ACTIVE_FEE
    ACTIVE_FEE = args[-1]
    return ORIGINAL_PREPARE(*args)


class TrackedReplay(Replay):
    def run(self, codes, calendar):
        global FAILURE
        try:
            return super().run(codes, calendar)
        except nn.UnresolvedPosition as exc:
            FAILURE = dict(secids=self.secids, training_hurdle=ACTIVE_FEE,
                           start=str(calendar[0].date()), error=str(exc))
            raise


def rl_folder(pair, fee, period):
    if fee == .03:
        return Path('artifacts/contracts/atm_retrained_commission_003') / pair / period
    if fee == .65:
        return Path('artifacts/contracts/frozen_atm_commission_003') / pair / (period + '_003')
    root = 'atm_cost_hurdles' if fee in [.15, .30] else 'atm_higher_cost_hurdles'
    return Path('artifacts/contracts') / root / pair / nn.fee_name(fee) / period


def common_roster(defs, statuses, hurdles):
    return [d for d in defs if all(statuses[(model, d['pair'], fee, period)]['status'] == 'complete'
            for model in ['nn_fixed', 'rl_sized'] for fee in hurdles for period, _ in nn.PERIODS)]


def summary(defs, market, cfg):
    statuses, folders, universe, fit_tables = {}, {}, [], []
    for d in defs:
        for fee in nn.HURDLES:
            own = OUTPUT / d['pair'] / nn.fee_name(fee)
            if (own / 'fits.csv').exists():
                fits = nn.read_fits(own / 'fits.csv'); check_fits(fits)
                if len(fits):
                    fits['pair'] = d['pair']; fits['training_hurdle'] = fee
                    fits['model'] = 'nn_fixed'; fit_tables.append(fits)
            for model in ['nn_fixed', 'rl_sized']:
                for period, _ in nn.PERIODS:
                    folder = own / period if model == 'nn_fixed' else rl_folder(d['pair'], fee, period)
                    status = json.loads((folder / 'status.json').read_text())
                    assert status['status'] in ['complete', 'incomplete']
                    key = model, d['pair'], fee, period
                    statuses[key] = status; folders[key] = folder
                    universe.append(dict(model=model, pair=d['pair'], label=d['label'],
                        training_hurdle=fee, period=period, status=status['status'], error=status.get('error')))
    complete = common_roster(defs, statuses, nn.HURDLES)
    assert complete
    included = {d['pair'] for d in complete}
    for row in universe: row['included_in_all_matched_comparisons'] = row['pair'] in included
    coverage = pd.DataFrame(universe)
    coverage.to_csv(REPORT / 'universe.csv', index=False)
    coverage[coverage.status.ne('complete')].to_csv(REPORT / 'unresolved.csv', index=False)
    pd.DataFrame([dict(pair=d['pair'], label=d['label'], included=d['pair'] in included)
                  for d in defs]).to_csv(REPORT / 'matched_roster.csv', index=False)
    rows, annual, pairs = [], [], []
    for model in ['nn_fixed', 'rl_sized']:
        for fee in nn.HURDLES:
            for period, start in nn.PERIODS:
                cal = market.calendar[market.calendar >= start]
                cash = cash_book(cal, market.rates, cfg.capital)
                book = cash[['nav', 'cash_benchmark']] * len(complete)
                entries = active = 0; pnl_paths = {}
                for d in complete:
                    key = model, d['pair'], fee, period
                    status, folder = statuses[key], folders[key]
                    nav = cash if status.get('cash_only') else pd.read_parquet(status.get('nav_source', folder / 'daily_nav.parquet'))
                    assert nav.index.equals(cal)
                    book += nav[['nav', 'cash_benchmark']] - cash[['nav', 'cash_benchmark']]
                    pnl = float(nav.nav.iloc[-1] - nav.cash_benchmark.iloc[-1])
                    pairs.append(dict(model=model, training_hurdle=fee, period=period, pair=d['pair'],
                                      label=d['label'], entries=status['entries'], net_excess_pnl=pnl))
                    if status['entries']: pnl_paths[d['pair']] = nav.nav - nav.cash_benchmark
                    entries += status['entries']; active += int(status['entries'] > 0)
                book['return_'] = book.nav.pct_change().fillna(0)
                book['excess_return'] = book.return_ - book.cash_benchmark.pct_change().fillna(0)
                book.to_csv(REPORT / f'{model}_{nn.fee_name(fee)}_{period}_portfolio.csv')
                row = dict(model=model, training_hurdle=fee, period=period, scope='common_complete_subset',
                    execution_commission=.03, pairs=len(complete), attempted_pairs=len(defs), entries=entries,
                    active_pairs=active, net_excess_pnl=float(book.nav.iloc[-1]-book.cash_benchmark.iloc[-1]),
                    **scores(book, cfg.capital * len(complete)))
                winners = sorted([r for r in pairs if r['model']==model and r['training_hurdle']==fee
                                  and r['period']==period and r['net_excess_pnl']>1e-7], key=lambda r:-r['net_excess_pnl'])
                if winners:
                    top = winners[0]; without = book.nav - pnl_paths[top['pair']]
                    excess = without.pct_change().fillna(0)-book.cash_benchmark.pct_change().fillna(0)
                    row.update(largest_winner=top['label'], largest_winner_pnl=top['net_excess_pnl'],
                        top_three_share_positive_pnl=sum(r['net_excess_pnl'] for r in winners[:3])/sum(r['net_excess_pnl'] for r in winners),
                        sharpe_largest_winner_replaced_by_cash=float(np.sqrt(252)*excess.iloc[1:].mean()/excess.iloc[1:].std()))
                rows.append(row)
                for year, g in book.groupby(book.index.year):
                    annual.append(dict(model=model, training_hurdle=fee, period=period, year=year,
                        scope='common_complete_subset', pairs=len(complete),
                        total_return_percent=100*((1+g.return_).prod()-1),
                        cash_return_percent=100*((1+book.cash_benchmark.pct_change().fillna(0).loc[g.index]).prod()-1),
                        annualized_excess_sharpe=float(np.sqrt(252)*g.excess_return.mean()/g.excess_return.std()) if g.excess_return.std()>1e-12 else None))
    comparison = pd.DataFrame(rows).sort_values(['period', 'training_hurdle', 'model'])
    comparison.to_csv(REPORT / 'comparison.csv', index=False)
    pd.DataFrame(annual).to_csv(REPORT / 'annual.csv', index=False)
    pd.DataFrame(pairs).to_csv(REPORT / 'pairs.csv', index=False)
    pd.concat(fit_tables, ignore_index=True).to_csv(REPORT / 'fits.csv', index=False)
    meta = dict(attempted_pairs=len(defs), common_complete_pairs=len(complete),
                incomplete_windows=int(coverage.status.ne('complete').sum()),
                excluded_pairs=[dict(pair=d['pair'], label=d['label']) for d in defs if d['pair'] not in included],
                scope='common_complete_subset', incomplete_books_assigned_cash=False)
    (REPORT / 'coverage.json').write_text(json.dumps(meta, indent=2)+'\n')
    excluded = ', '.join(d['label'] for d in defs if d['pair'] not in included)
    lines = ['# NN fixed size versus RL sizing at matched training hurdles', '',
        f'Attempted all six hurdles across all {len(defs)} original pairs. The comparison uses the same {len(complete)} complete pairs for both models, all costs and both periods. Excluded pairs: {excluded}. Any unresolved NN book removes that pair from every matched comparison, including RL; it is never assigned cash or zero P&L. This is an ex-post data-completeness subset and can be biased. The full-roster NN portfolio remains unavailable.', '',
        '| Training hurdle | NN full Sharpe | RL full Sharpe | NN recent Sharpe | RL recent Sharpe | NN full entries | RL full entries |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for fee in nn.HURDLES:
        def own(model, period):
            return comparison[comparison.model.eq(model)&comparison.training_hurdle.eq(fee)&comparison.period.eq(period)].iloc[0]
        nf, rf, nr, rr = own('nn_fixed','full'), own('rl_sized','full'), own('nn_fixed','recent'), own('rl_sized','recent')
        lines.append(f'| ${fee:.2f} | {nf.annualized_excess_sharpe:.3f} | {rf.annualized_excess_sharpe:.3f} | {nr.annualized_excess_sharpe:.3f} | {rr.annualized_excess_sharpe:.3f} | {int(nf.entries)} | {int(rf.entries)} |')
    lines += ['', 'Full: January 2018–August 2025. Recent: January 2022–August 2025, starting flat. Daily cash-excess Sharpes are annualized with √252. Same $1m per retained pair, including complete cash sleeves. Actual execution is $0.03 per contract for every model and hurdle; option entry penalty, stock hedging, borrowing, financing, lifecycle and risk limits match.', '',
        'NN: one 16-unit tanh hidden layer, L-BFGS, alpha=1, maximum 1000 iterations, train-only scaling. It predicts the two net full-size outcomes and chooses the larger positive prediction or cash. RL: the saved eight-leaf direct Sharpe policy search with the signed 25/50/75/100% ladder or cash. Architecture, learning objective and sizing change together, so this is not a pure RL-effect ablation.', '',
        'Identical feature columns, cost-rebuilt completed-outcome memory, 91/365-day directional gates, beta/correlation eligibility, monthly schedule, five-year fitting windows, outcome purging and prior-year positive-Sharpe validation. Validation enables the proposal, followed by the inherited recent-five-year refit; the newly refitted weights do not have an independent validation test. NN convergence warnings remain in the fit audit. This is exploratory research on previously examined history.', '',
        'The original NN source and fitting hashes remain intact. A separately recorded continuation catches unresolved evaluation paths, attempts the other windows, and reports matching without changing prices, trades, rewards, fees or Greeks. The DVN/MRO $0.30 NN path first failed on 2023-02-14 with unsupported distribution for option 152049246.', '',
        '[Comparison](atm_nn_cost_hurdles/comparison.csv) · [Annual results](atm_nn_cost_hurdles/annual.csv) · [Pair results](atm_nn_cost_hurdles/pairs.csv) · [Fit audit](atm_nn_cost_hurdles/fits.csv) · [Unresolved cases](atm_nn_cost_hurdles/unresolved.csv) · [Matched roster](atm_nn_cost_hurdles/matched_roster.csv)', '']
    REPORT.with_suffix('.md').write_text('\n'.join(lines))
    print(comparison[['model','training_hurdle','period','pairs','annualized_excess_sharpe','entries']].to_string(index=False), flush=True)


def main():
    import sys
    if '--check' in sys.argv:
        import unittest
        suite=unittest.defaultTestLoader.loadTestsFromName('tests.test_atm_nn_common_roster')
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful(): raise SystemExit(1)
        return
    paths=[Path(__file__), Path('scripts/audit_atm_nn_cost_hurdles_complete.py')]
    protocol=dict(reason='Continue unsupported held-distribution failures without inventing valuation',
        unchanged_training=True, common_roster_across_all_models_hurdles_periods=True,
        code_and_sources={str(p):digest(p) for p in paths})
    path=OUTPUT/'continuation_protocol.json'
    if path.exists(): assert json.loads(path.read_text())==protocol, 'Continuation changed'
    else: path.write_text(json.dumps(protocol, indent=2)+'\n')
    nn.runner.prepare_hurdle=prepare; nn.runner.Replay=TrackedReplay; nn.summary=summary
    while True:
        try:
            nn.main()
            meta=json.loads((REPORT/'coverage.json').read_text())
            task=json.loads((OUTPUT/'task_status.json').read_text()); task.update(meta)
            (OUTPUT/'task_status.json').write_text(json.dumps(task, indent=2)+'\n')
            return
        except nn.UnresolvedPosition:
            assert FAILURE is not None
            defs=json.loads((OUTPUT/'inputs.json').read_text())['definitions']
            d=next(d for d in defs if set(d['secids'])==set(FAILURE['secids']))
            year=pd.Timestamp(FAILURE['start']).year
            assert year in [2018,2022], FAILURE
            period='full' if year==2018 else 'recent'
            folder=OUTPUT/d['pair']/nn.fee_name(FAILURE['training_hurdle'])/period
            assert not (folder/'status.json').exists()
            folder.mkdir(parents=True,exist_ok=True)
            status=dict(status='incomplete',pair=d['pair'],label=d['label'],period=period,
                training_hurdle=FAILURE['training_hurdle'],execution_commission=.03,error=FAILURE['error'])
            (folder/'status.json').write_text(json.dumps(status,indent=2)+'\n')
            print(f'Flagged unresolved {d["label"]} ${FAILURE["training_hurdle"]:.2f} {period}: {FAILURE["error"]}',flush=True)
            gc.collect()


if __name__=='__main__': main()
