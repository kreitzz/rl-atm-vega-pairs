"""Follow-up $1/$1.50 training hurdles; actual execution stays at $0.03.

Reuse the prior checkpointed single-pair runner without changing its source or
provenance. Write new artifacts/reports and retain all 346 original sleeves.
"""
from scripts.run_atm_cost_hurdles import *
import scripts.run_atm_cost_hurdles as runner

OUTPUT = Path('artifacts/contracts/atm_higher_cost_hurdles')
REPORT = Path('reports/atm_higher_cost_hurdles')
HURDLES = [1., 1.5]


def summary(defs, market, cfg):
    rows, annual, pairs, fits_all = [], [], [], []
    previous = pd.read_csv('reports/atm_cost_hurdles/comparison.csv')
    previous['source'] = 'saved_policy'
    rows.extend(previous.to_dict('records'))
    annual.extend(pd.read_csv('reports/atm_cost_hurdles/annual.csv').to_dict('records'))
    for fee in HURDLES:
        name = fee_name(fee)
        for period, start in PERIODS:
            cal = market.calendar[market.calendar >= start]
            cash = cash_book(cal, market.rates, cfg.capital)
            book = cash[['nav', 'cash_benchmark']] * len(defs)
            entries = active = 0; commission = option_cost = hedge_cost = 0.
            pnl_paths = {}
            for d in defs:
                folder = OUTPUT / d['pair'] / name
                status = json.loads((folder / period / 'status.json').read_text())
                assert status['status'] == 'complete'
                assert status['execution_commission'] == EXECUTION_FEE and status['training_hurdle'] == fee
                nav = cash if status.get('cash_only') else pd.read_parquet(folder / period / 'daily_nav.parquet')
                assert nav.index.equals(cal)
                book += nav[['nav', 'cash_benchmark']] - cash[['nav', 'cash_benchmark']]
                pnl = float(nav.nav.iloc[-1] - nav.cash_benchmark.iloc[-1])
                pairs.append(dict(training_hurdle=fee, period=period, pair=d['pair'], label=d['label'], entries=status['entries'], net_excess_pnl=pnl))
                if status['entries']: pnl_paths[d['pair']] = nav.nav - nav.cash_benchmark
                entries += status['entries']; active += int(status['entries'] > 0)
                commission += status.get('commission', 0.); option_cost += status.get('option_cost', 0.)
                hedge_cost += status.get('hedge_cost', 0.)
                if period == 'full' and (folder / 'fits.csv').exists():
                    fits = read_fits(folder / 'fits.csv'); check_fits(fits)
                    if len(fits):
                        fits['pair'] = d['pair']; fits['training_hurdle'] = fee; fits_all.append(fits)
            book['return_'] = book.nav.pct_change().fillna(0)
            book['excess_return'] = book.return_ - book.cash_benchmark.pct_change().fillna(0)
            book.to_csv(REPORT / f'{name}_{period}_portfolio.csv')
            row = dict(training_hurdle=fee, execution_commission=EXECUTION_FEE, period=period, source='new_fit',
                       start=str(cal[0].date()), end=str(cal[-1].date()), pairs=len(defs), entries=entries,
                       active_pairs=active, option_commission=commission, option_cost=option_cost, hedge_cost=hedge_cost,
                       net_excess_pnl=float(book.nav.iloc[-1] - book.cash_benchmark.iloc[-1]),
                       cash_return=float(book.cash_benchmark.iloc[-1] / (cfg.capital * len(defs)) - 1),
                       **scores(book, cfg.capital * len(defs)))
            winners = sorted([r for r in pairs if r['training_hurdle'] == fee and r['period'] == period and r['net_excess_pnl'] > 1e-7], key=lambda r: -r['net_excess_pnl'])
            if winners:
                top = winners[0]; without = book.nav - pnl_paths[top['pair']]
                excess = without.pct_change().fillna(0) - book.cash_benchmark.pct_change().fillna(0)
                row.update(largest_winner=top['label'], largest_winner_pnl=top['net_excess_pnl'],
                           top_three_share_positive_pnl=sum(r['net_excess_pnl'] for r in winners[:3]) / sum(r['net_excess_pnl'] for r in winners),
                           sharpe_largest_winner_replaced_by_cash=float(np.sqrt(252) * excess.iloc[1:].mean() / excess.iloc[1:].std()))
            rows.append(row)
            for year, g in book.groupby(book.index.year):
                annual.append(dict(training_hurdle=fee, period=period, year=year,
                                   total_return_percent=100 * ((1 + g.return_).prod() - 1),
                                   cash_return_percent=100 * ((1 + book.cash_benchmark.pct_change().fillna(0).loc[g.index]).prod() - 1),
                                   annualized_excess_sharpe=float(np.sqrt(252) * g.excess_return.mean() / g.excess_return.std()) if g.excess_return.std() > 1e-12 else None))
    comparison = pd.DataFrame(rows).sort_values(['period', 'training_hurdle'])
    comparison.to_csv(REPORT / 'comparison.csv', index=False)
    pd.DataFrame(annual).sort_values(['period', 'training_hurdle', 'year']).to_csv(REPORT / 'annual.csv', index=False)
    pd.DataFrame(pairs).to_csv(REPORT / 'pairs.csv', index=False)
    pd.concat(fits_all, ignore_index=True).to_csv(REPORT / 'fits.csv', index=False)
    lines = ['# Training cost as a profitability hurdle', '',
             'Follow-up grid: $1.00 and $1.50 per option contract, compared with the saved $0.03/$0.15/$0.30/$0.65 policies. All evaluation books pay $0.03. The two new policies rebuild rewards, commission features, completed-outcome profitability memory, directional gates and prior-year validation using the higher training cost. The extra cost is a conservative profitability hurdle, not a calibrated probability of profit.', '',
             '| Training hurdle | Full-history Sharpe | Recent Sharpe | Full entries | Active pairs | Full P&L above cash |',
             '|---|---:|---:|---:|---:|---:|']
    for fee in [.03, .15, .30, .65, *HURDLES]:
        full = comparison[comparison.training_hurdle.eq(fee) & comparison.period.eq('full')].iloc[0]
        recent = comparison[comparison.training_hurdle.eq(fee) & comparison.period.eq('recent')].iloc[0]
        lines.append(f'| ${fee:.2f} | {full.annualized_excess_sharpe:.3f} | {recent.annualized_excess_sharpe:.3f} | {int(full.entries)} | {int(full.active_pairs)} | ${full.net_excess_pnl:,.0f} |')
    lines += ['', 'Full history: January 2018–August 2025. Recent test: January 2022–August 2025, starting flat with the same $1m per pair. All 346 original pairs remain, including cash sleeves. Same ATM30 call/call contracts, signed 25/50/75/100% ladder, monthly beta 0.85–1.15 and correlation ≥0.5 screen, actual-contract rewards, monthly eight-leaf direct Sharpe search, completed-outcome purging, prior-year validation and recent-five-year refit.', '',
              'Training validation pays the selected hurdle; evaluation resets commission to $0.03. The 0.10-vol-point option entry penalty, 1 bp daily stock hedge execution, financing, borrowing and risk limits are identical everywhere. Option quantities are fixed after initially matching vega within 10%, apart from supported splits, and each leg settles at its own expiry. No daily option-vega rebalancing. Training-cost changes can alter directions, dates and sizes, so this is more than filtering a frozen prediction by a confidence threshold.', '',
              'The $0.03/$0.15/$0.30/$0.65 policies reuse saved results; $1.00 and $1.50 are fresh fits of the same algorithm. This follow-up was motivated by the preceding cost sweep. Original $0.65 memory and curve gates are checked before fitting. Accounting, expiry and commission checks run on every simulated book. This is exploratory sensitivity on previously examined history; selecting the highest reported Sharpe would not constitute untouched validation. Largest-winner replacement is an ex-post concentration diagnostic.', '',
              '[Annual results](atm_higher_cost_hurdles/annual.csv) · [Comparison](atm_higher_cost_hurdles/comparison.csv) · [New pair results](atm_higher_cost_hurdles/pairs.csv)', '']
    REPORT.with_suffix('.md').write_text('\n'.join(lines))
    print(comparison[['training_hurdle', 'period', 'annualized_excess_sharpe', 'entries', 'active_pairs', 'net_excess_pnl']].to_string(index=False), flush=True)


def main():
    import sys
    if '--check' in sys.argv:
        import unittest
        suite = unittest.defaultTestLoader.loadTestsFromNames([
            'tests.test_atm_higher_cost_hurdles', 'tests.test_atm_commission_training'])
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__), Path('scripts/audit_atm_higher_cost_hurdles.py'),
             Path('reports/atm_cost_hurdles/comparison.csv'),
             Path('reports/atm_cost_hurdles/annual.csv')]
    protocol = dict(new_hurdles=HURDLES, execution_commission=EXECUTION_FEE,
                    code_and_sources={str(p): digest(p) for p in paths})
    path = OUTPUT / 'followup_protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == protocol, 'Follow-up inputs changed'
    else:
        path.write_text(json.dumps(protocol, indent=2) + '\n')
    runner.OUTPUT = OUTPUT; runner.REPORT = REPORT; runner.HURDLES = HURDLES
    runner.summary = summary
    runner.main()


if __name__ == '__main__': main()
