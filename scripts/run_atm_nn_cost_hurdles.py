"""Matched fixed-size supervised NN benchmark for the ATM RL cost sweep."""
from scripts.run_atm_cost_hurdles import *
import scripts.run_atm_cost_hurdles as runner
from scripts.run_contract_walkforward import fit_nn, predict as predict_direction
from scripts.run_curve_trading import RUNOFF, screen_dates, audit, UnresolvedPosition, load_subset

OUTPUT = Path('artifacts/contracts/atm_nn_cost_hurdles')
REPORT = Path('reports/atm_nn_cost_hurdles')
HURDLES = [.03, .15, .30, .65, 1., 1.5]


def fit_fixed_nn(x, labels, positions, ends, codes):
    # load_subset prepends cash. fit_nn expects first/last noncash columns.
    assert codes == [0, *range(1 + ATM * 8, 1 + (ATM + 1) * 8)]
    model, details = fit_nn(x, labels[:, 1:])
    details['training_shadow_daily_sharpe'] = None
    return model, details


def predict_fixed_nn(model, x):
    direction = predict_direction(model, x, neural=True)
    return np.where(direction > 0, 1 + ATM * 8 + 7,
                    np.where(direction < 0, 1 + ATM * 8, 0)).astype(int)


def train_nn(d,panel,labels,ends,gates,available,cal,replay,rates,folder,features,variant):
    start=pd.Timestamp('2018-01-01');stop=cal[-1]-pd.Timedelta(days=RUNOFF)
    wanted=(panel.index>=start)&(panel.index<=stop)&screen_dates(d,panel.index)
    actions=pd.Series(0,index=panel.index[wanted],dtype=int,name='code');logs=[];ranking=[];year=None
    checkpoint=folder/'month_checkpoint.json'
    if checkpoint.exists():
        prior=json.loads(checkpoint.read_text())
        assert prior['dates']==[str(x) for x in actions.index]
        actions.iloc[:]=prior['codes'];logs=prior['fits']
    completed={row['fit'] for row in logs}
    positions=cal.get_indexer(panel.index);cfg=replay.cfg
    for month in pd.period_range(start,stop,freq='M'):
        cutoff=month.start_time;test=(actions.index>=cutoff)&(actions.index<(month+1).start_time)
        if not test.any():continue
        if str(cutoff.date()) in completed:continue
        if year!=month.year:
            year=month.year
            combos,rank=([ATM],[]) if variant=='atm_control' else annual_shortlist(year,panel,labels,ends,gates,cal)
            ranking.extend(rank);daily,finish,codes=load_subset(labels,ends,combos)
            finite=np.isfinite(daily).all(axis=(1,2));completion=cal[finish]
            pd.DataFrame(ranking).to_csv(folder/'shortlists.csv',index=False)
        validation_start=cutoff-pd.DateOffset(years=1)
        use=(panel.index>=validation_start-pd.DateOffset(years=5))&(completion<validation_start)&finite
        val_cal=cal[(cal>=validation_start)&(cal<cutoff)]
        if use.sum()<504 or len(val_cal)<200:
            logs.append(dict(fit=str(cutoff.date()),status='insufficient_training',training_rows=int(use.sum())));continue
        val=(panel.index>=val_cal[0])&(panel.index<=val_cal[-1]-pd.Timedelta(days=RUNOFF))
        has_opportunity=any((panel.eligible.to_numpy()&available[c]&gates[c].any(axis=1)&val).any() for c in combos)
        info=dict(fit=str(cutoff.date()),training_rows=int(use.sum()),combos=json.dumps(combos),
                  last_training_outcome=str(completion[use].max().date()),validation_start=str(val_cal[0].date()),
                  validation_end=str(val_cal[-1].date()))
        if not has_opportunity:
            logs.append(dict(info,status='flat_no_validation_opportunities'));continue
        train_labels=daily[use].copy();train_labels[~panel.loc[use,'eligible'].to_numpy(bool)]=0
        model,details=fit_fixed_nn(panel.loc[use,features],train_labels,positions[use],finish[use],codes)
        proposal=gate(predict_fixed_nn(model,panel.loc[val,features]),panel.index[val],panel,gates,available)
        error=None;score=None;permitted=False
        try:
            nav,t,_=replay.run(proposal,val_cal);mark_book(nav,rates,cfg.capital);audit(nav,t,replay.s,cfg,d['secids'])
            score=scores(nav,cfg.capital)['annualized_excess_sharpe'];permitted=score is not None and score>0
        except (UnresolvedPosition,ValueError) as exc:error=str(exc)
        if permitted:
            recent=(panel.index>=cutoff-pd.DateOffset(years=5))&(completion<cutoff)&finite
            if recent.sum()<504:permitted=False;error='insufficient recent training'
            else:
                own=daily[recent].copy();own[~panel.loc[recent,'eligible'].to_numpy(bool)]=0
                model,refit=fit_fixed_nn(panel.loc[recent,features],own,positions[recent],finish[recent],codes)
                details.update(recent_training_rows=int(recent.sum()),last_recent_outcome=str(completion[recent].max().date()),
                               recent_converged=refit['converged'],recent_iterations=refit['iterations'])
                dates=actions.index[test]
                actions.loc[dates]=gate(predict_fixed_nn(model,panel.loc[dates,features]),dates,panel,gates,available)
        logs.append(dict(info,status='enabled' if permitted else 'flat_validation_gate',validation_sharpe=score,validation_error=error,**details))
        pd.DataFrame(logs).to_csv(folder/'fits.csv',index=False);actions.to_csv(folder/'actions.partial.csv')
        snapshot=dict(dates=[str(x) for x in actions.index],codes=actions.tolist(),
                      fits=json.loads(pd.DataFrame(logs).to_json(orient='records')))
        temporary=checkpoint.with_suffix('.tmp')
        temporary.write_text(json.dumps(snapshot)+'\n');temporary.replace(checkpoint)
        if month.month in [1,7]:print(f'  {d["label"]} nn_fixed {month}: {logs[-1]["status"]}',flush=True)
    return actions,pd.DataFrame(logs)


def summary(defs, market, cfg):
    rows, annual, pairs, fits_all = [], [], [], []
    previous = pd.read_csv('reports/atm_higher_cost_hurdles/comparison.csv')
    previous = previous[previous.training_hurdle.isin(HURDLES)].copy()
    previous['model'] = 'rl_sized'
    previous['source'] = 'saved_policy'
    rows.extend(previous.to_dict('records'))
    prior_annual = pd.read_csv('reports/atm_higher_cost_hurdles/annual.csv')
    prior_annual = prior_annual[prior_annual.training_hurdle.isin(HURDLES)].copy()
    prior_annual['model'] = 'rl_sized'
    annual.extend(prior_annual.to_dict('records'))
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
                pairs.append(dict(model='nn_fixed', training_hurdle=fee, period=period, pair=d['pair'], label=d['label'], entries=status['entries'], net_excess_pnl=pnl))
                if status['entries']: pnl_paths[d['pair']] = nav.nav - nav.cash_benchmark
                entries += status['entries']; active += int(status['entries'] > 0)
                commission += status.get('commission', 0.); option_cost += status.get('option_cost', 0.)
                hedge_cost += status.get('hedge_cost', 0.)
                if period == 'full' and (folder / 'fits.csv').exists():
                    fits = read_fits(folder / 'fits.csv'); check_fits(fits)
                    if len(fits):
                        fits['model'] = 'nn_fixed'; fits['pair'] = d['pair']; fits['training_hurdle'] = fee; fits_all.append(fits)
            book['return_'] = book.nav.pct_change().fillna(0)
            book['excess_return'] = book.return_ - book.cash_benchmark.pct_change().fillna(0)
            book.to_csv(REPORT / f'{name}_{period}_portfolio.csv')
            row = dict(model='nn_fixed', training_hurdle=fee, execution_commission=EXECUTION_FEE, period=period, source='new_fit',
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
                annual.append(dict(model='nn_fixed', training_hurdle=fee, period=period, year=year,
                                   total_return_percent=100 * ((1 + g.return_).prod() - 1),
                                   cash_return_percent=100 * ((1 + book.cash_benchmark.pct_change().fillna(0).loc[g.index]).prod() - 1),
                                   annualized_excess_sharpe=float(np.sqrt(252) * g.excess_return.mean() / g.excess_return.std()) if g.excess_return.std() > 1e-12 else None))
    comparison = pd.DataFrame(rows).sort_values(['period', 'training_hurdle'])
    comparison.to_csv(REPORT / 'comparison.csv', index=False)
    pd.DataFrame(annual).sort_values(['period', 'training_hurdle', 'year']).to_csv(REPORT / 'annual.csv', index=False)
    pd.DataFrame(pairs).to_csv(REPORT / 'pairs.csv', index=False)
    pd.concat(fits_all, ignore_index=True).to_csv(REPORT / 'fits.csv', index=False)
    lines = ['# NN fixed size versus RL sizing at matched training hurdles', '',
             'NN: one 16-unit tanh hidden layer, L-BFGS, alpha=1, maximum 1000 iterations; train-only input/target scaling. It predicts net full-size outcomes in each direction, chooses the larger positive prediction, or holds cash. RL: the existing eight-leaf direct Sharpe policy search with a signed 25/50/75/100% ladder or cash. This compares modeling/objective and sizing together; it does not isolate a pure RL effect.', '',
             'Both models use identical feature columns, cost-rebuilt completed-outcome memory, directional 91/365-day gates, beta/correlation screen, monthly timing, five-year training window and separate preceding-year positive-Sharpe validation. Each enabled proposal is refitted on the most recent five years. Validation still enables the refit based on the proposal rather than validating the refitted weights, as in the original RL runner. Actual execution is $0.03 per contract for both; all other execution costs, lifecycle and risk limits match. Original RL actions are reused, while NN is freshly fitted. All 346 original pairs remain in the capital denominator.', '',
             '| Training hurdle | Model | Full Sharpe | Recent Sharpe | Full entries | Active pairs | Full P&L above cash |',
             '|---|---|---:|---:|---:|---:|---:|']
    for fee in HURDLES:
        for model in ['nn_fixed', 'rl_sized']:
            full = comparison[comparison.training_hurdle.eq(fee) & comparison.period.eq('full') & comparison.model.eq(model)].iloc[0]
            recent = comparison[comparison.training_hurdle.eq(fee) & comparison.period.eq('recent') & comparison.model.eq(model)].iloc[0]
            lines.append(f'| ${fee:.2f} | {model} | {full.annualized_excess_sharpe:.3f} | {recent.annualized_excess_sharpe:.3f} | {int(full.entries)} | {int(full.active_pairs)} | ${full.net_excess_pnl:,.0f} |')
    lines += ['', 'Daily cash-excess Sharpes are annualized with √252. Full: January 2018–August 2025. Recent: January 2022–August 2025 starting flat. $1m capital per pair, cash sleeves retained. This is exploratory analysis of previously inspected history, including a follow-up cost sweep.', '',
              '[Comparison](atm_nn_cost_hurdles/comparison.csv) · [Annual results](atm_nn_cost_hurdles/annual.csv) · [NN pair results](atm_nn_cost_hurdles/pairs.csv) · [NN monthly fit audit](atm_nn_cost_hurdles/fits.csv)', '']
    REPORT.with_suffix('.md').write_text('\n'.join(lines))
    print(comparison[['model', 'training_hurdle', 'period', 'annualized_excess_sharpe', 'entries', 'active_pairs']].to_string(index=False), flush=True)

def main():
    global HURDLES
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--hurdles', nargs='+', type=float, default=HURDLES)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    if args.check:
        import unittest
        suite = unittest.defaultTestLoader.loadTestsFromName('tests.test_atm_nn_cost_hurdles')
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful(): raise SystemExit(1)
        return
    HURDLES = args.hurdles
    assert HURDLES == sorted(set(HURDLES))
    assert all(fee in [.03, .15, .30, .65, 1., 1.5] for fee in HURDLES)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__), Path('scripts/run_contract_walkforward.py'),
             Path('scripts/audit_atm_nn_cost_hurdles.py'),
             Path('reports/atm_higher_cost_hurdles/comparison.csv'),
             Path('reports/atm_higher_cost_hurdles/annual.csv')]
    protocol = dict(new_hurdles=HURDLES, execution_commission=EXECUTION_FEE,
                    model='nn_fixed', code_and_sources={str(p): digest(p) for p in paths})
    path = OUTPUT / 'followup_protocol.json'
    if path.exists(): assert json.loads(path.read_text()) == protocol, 'NN inputs changed'
    else: path.write_text(json.dumps(protocol, indent=2) + '\n')
    runner.train = train_nn
    runner.OUTPUT = OUTPUT; runner.REPORT = REPORT; runner.HURDLES = HURDLES
    runner.summary = summary
    sys.argv = [sys.argv[0]] + (['--report-only'] if args.report_only else [])
    runner.main()


if __name__ == '__main__': main()
