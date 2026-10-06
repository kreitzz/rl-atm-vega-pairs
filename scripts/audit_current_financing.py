"""Funding sensitivity for the frozen best: saved holdings, no new fitting."""
import json
from pathlib import Path
import resource

import numpy as np
import pandas as pd

from scripts.run_selected_pair_backtest import scores
from scripts.run_contract_walkforward import digest

REPORT = Path('reports/current_financing')
REFERENCE = Path('config/current_contract_reference.json')


def reprice_funding(nav, capital, base_borrow, borrow_rate, loan_spread, rebate_discount=0., segregate=False):
    """Separate short collateral from freely lendable cash; fixed noncash book.

    Recover preceding short notional from saved borrow expense. Collateral is
    assumed 100% of notional here, with no broker rounding/settlement lag.
    Keep the saved cash benchmark fixed. Recompute the entire interest path.
    """
    elapsed = nav.index.to_series().diff().dt.days.fillna(0).to_numpy()/365
    rate = np.divide(nav.cash_benchmark.pct_change().fillna(0).to_numpy(), elapsed,
                     out=np.zeros(len(nav)), where=elapsed > 0)
    # Expense already uses the preceding holdings and preceding marks.
    short_years = nav.stock_borrow_cost.to_numpy()/base_borrow
    short = np.divide(short_years, elapsed, out=np.zeros(len(nav)), where=elapsed > 0)
    flow = (nav.cash.diff().fillna(nav.cash.iloc[0]-capital)-nav.interest+nav.stock_borrow_cost).to_numpy()
    cash = capital; path=[]; interest=[]; borrowing=[]; loan_years=0.
    for delta, r, dt, collateral, exposure in zip(flow, rate, elapsed, short, short_years):
        free = cash-collateral if segregate else cash
        own_interest = cash*r*dt-max(-free,0)*loan_spread*dt-collateral*rebate_discount*dt
        expense = exposure*borrow_rate
        loan_years += max(-free, 0)*dt
        cash += delta+own_interest-expense
        path.append(cash); interest.append(own_interest); borrowing.append(expense)
    result = nav.copy()
    result['cash'] = path; result['interest'] = interest; result['stock_borrow_cost'] = borrowing
    result['nav'] = result.cash+result.stock_value+result.option_value
    result['return_'] = result.nav.pct_change().fillna(result.nav.iloc[0]/capital-1)
    result['excess_return'] = result.return_-result.cash_benchmark.pct_change().fillna(0)
    return result, dict(short_dollar_years=float(short_years.sum()),
                        segregated_cash_loan_dollar_years=float(loan_years))


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    ref = json.loads(REFERENCE.read_text()); cfg = ref['execution_settings']; capital=cfg['capital']
    mapping = Path(ref['policy_sources']); policies=json.loads(mapping.read_text())['policies']
    # One assumption at a time, plus combinations; no outcome-selected setting.
    scenarios = [('saved',.003,.01,0.,False), ('segregated_collateral',.003,.01,0.,True),
                 ('borrow_1pct',.01,.01,0.,True), ('borrow_3pct',.03,.01,0.,True),
                 ('rebate_minus_50bp',.003,.01,.005,True),
                 ('borrow_1pct_rebate_minus_50bp',.01,.01,.005,True),
                 ('borrow_3pct_rebate_minus_50bp',.03,.01,.005,True),
                 ('loan_plus_2pct',.003,.02,0.,True)]
    rows=[]; pairs=[]; annual=[]; files={str(REFERENCE):digest(REFERENCE),str(mapping):digest(mapping)}
    for period in ['full','recent']:
        saved = REPORT.parent/'atm65_employee_sentiment'/f'all_rl_sentiment_{period}_portfolio.csv'
        original=pd.read_csv(saved,index_col=0,parse_dates=[0]); files[str(saved)]=digest(saved)
        books={name:original[['nav','cash_benchmark']].copy() for name,*_ in scenarios}
        for policy in policies:
            folder=Path(policy['ledgers']['rl_sized_'+period]); status_path=folder/'status.json'
            status=json.loads(status_path.read_text()); assert status['status']=='complete'
            files[str(status_path)]=digest(status_path)
            if status['entries']==0: continue
            path=Path(status.get('nav_source',folder/'daily_nav.parquet')); files[str(path)]=digest(path)
            nav=pd.read_parquet(path); assert nav.index.equals(original.index)
            elapsed=nav.index.to_series().diff().dt.days.fillna(0)/365
            benchmark_interest=nav.cash_benchmark.diff().fillna(0)
            rate=nav.cash_benchmark.pct_change().fillna(0).div(elapsed.where(elapsed>0)).fillna(0)
            # Independently recover the original net-cash interest convention.
            previous=nav.cash.shift(1).fillna(capital)
            expected=previous*(rate+cfg['borrowing_spread']*(previous<0))*elapsed
            np.testing.assert_allclose(expected,nav.interest,atol=1e-7,rtol=0)
            position_funding=-(nav.stock_value+nav.option_value).shift(1).fillna(0)*rate*elapsed
            prior_pnl_interest=(nav.nav-nav.cash_benchmark).shift(1).fillna(0)*rate*elapsed
            net_cash_surcharge=previous.clip(upper=0)*cfg['borrowing_spread']*elapsed
            np.testing.assert_allclose(position_funding+prior_pnl_interest+net_cash_surcharge,
                                       nav.interest-benchmark_interest,atol=1e-7,rtol=0)
            for name,borrow,loan,rebate,segregate in scenarios:
                priced,exposure=reprice_funding(nav,capital,cfg['stock_borrow_rate'],borrow,loan,rebate,segregate)
                if name=='saved':
                    # Identical policies must reproduce the saved funding path.
                    np.testing.assert_allclose(priced.nav,nav.nav,atol=1e-6,rtol=0)
                books[name]['nav']+=priced.nav-nav.nav
                pairs.append(dict(period=period,scenario=name,pair=policy['pair'],label=policy['label'],
                    entries=status['entries'],stock_borrow=float(priced.stock_borrow_cost.sum()),
                    financing_vs_cash=float((priced.interest-benchmark_interest).sum()),
                    saved_position_funding=float(position_funding.sum()),
                    saved_interest_on_prior_excess_pnl=float(prior_pnl_interest.sum()),
                    negative_net_cash_days=int((previous<0).sum()),
                    net_excess_pnl=float(priced.nav.iloc[-1]-priced.cash_benchmark.iloc[-1]),**exposure))
        for name,borrow,loan,rebate,segregate in scenarios:
            book=books[name];book['return_']=book.nav.pct_change().fillna(0)
            book['excess_return']=book.return_-book.cash_benchmark.pct_change().fillna(0)
            s=scores(book,capital*len(policies))
            if name=='saved':
                expected=next(r for r in ref['matched_results'] if r['model']=='rl_sentiment' and r['period']==period)
                np.testing.assert_allclose(s['annualized_excess_sharpe'],expected['annualized_excess_sharpe'],atol=1e-10)
            rows.append(dict(period=period,scenario=name,borrow_rate=borrow,loan_spread=loan,
                             short_collateral_rebate_discount=rebate,segregate_short_collateral=segregate,**s,
                             net_excess_pnl=float(book.nav.iloc[-1]-book.cash_benchmark.iloc[-1])))
            for year,g in book.groupby(book.index.year):
                annual.append(dict(period=period,scenario=name,year=int(year),
                    annualized_excess_sharpe=float(np.sqrt(252)*g.excess_return.mean()/g.excess_return.std())))
    comparison=pd.DataFrame(rows);comparison.to_csv(REPORT/'comparison.csv',index=False)
    pd.DataFrame(pairs).to_csv(REPORT/'pairs.csv',index=False)
    pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False)
    lines=['# Current best: financing audit','',
        'Frozen 0.368 employee-leadership RL checkpoint. Saved contracts, quantities, hedge inventories and dates are unchanged. No policy fits, resizing or renewed risk intervention. These are funding sensitivities, not retrained strategy results.','',
        'The saved model earns the closest-to-30-day OptionMetrics zero rate on net positive cash, pays that rate plus 1% on negative cash and charges 0.30% annual stock borrow on each short hedge. Both strategy and cash benchmark use the same lagged rate with actual calendar days / 365. Cash is not separated into short collateral and free cash in the original engine.','',
        '| Assumption | Full annualized excess Sharpe | Since 2022 |',
        '|---|---:|---:|']
    for name,*_ in scenarios:
        full=comparison[(comparison.period=='full')&(comparison.scenario==name)].iloc[0]
        recent=comparison[(comparison.period=='recent')&(comparison.scenario==name)].iloc[0]
        lines.append(f'| {name} | {full.annualized_excess_sharpe:.3f} | {recent.annualized_excess_sharpe:.3f} |')
    lines+=['','All scenarios retain the original idle-cash benchmark and unencumbered-cash rate. Revised scenarios segregate 100% of preceding short-stock notional and accrue any loan spread on negative free cash. The 50 bp rebate discount applies only to short collateral. Actual collateral rounding/haircuts, settlement timing, broker credit/debit schedules and historically available stock-specific loan fees are not modeled. Each pair retains its separate cash sleeve; cross-pair hedge/stock-loan netting is not introduced.','',
        'Base replay reproduced the saved Sharpe and NAV paths. The independent interest identity reconciled position funding, interest on earlier excess P&L and any negative-cash surcharge. Sources were checked unchanged. Margin interest and short-sale borrow/rebate terms are distinct in [broker documentation](https://www.interactivebrokers.com/en/pricing/short-sale-cost.php). The cited current broker schedule is illustrative, not assumed historical institutional pricing.','',
        'The flat 0.30% fee and full benchmark-rate short collateral rebate are simplifying assumptions, not validated prime-broker terms. The most useful next refinement is historical, stock-specific stock-loan pricing and an explicit collateral rebate. A 30-day zero rate is also a proxy for actual overnight cash/debit funding.','',
        '[Sensitivity table](current_financing/comparison.csv) · [Pair funding and exposures](current_financing/pairs.csv) · [Annual Sharpe](current_financing/annual.csv) · [Source manifest](current_financing/manifest.json)','']
    REPORT.with_suffix('.md').write_text('\n'.join(lines))
    assert all(digest(Path(p))==h for p,h in files.items())
    (REPORT/'manifest.json').write_text(json.dumps(dict(checkpoint=ref['checkpoint'],sources=files,
        method='Fixed contracts, quantities, hedges and dates. Cash interest recomputed; no retraining or renewed sizing/risk interventions.',
        short_collateral='100% of preceding short hedge notional; no settlement lag or collateral rounding.',
        cash_benchmark='Original 30-day rate proxy unchanged; no haircut on unencumbered cash.',
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),indent=2)+'\n')
    print(comparison[['period','scenario','annualized_excess_sharpe','net_excess_pnl']].to_string(index=False),flush=True)
    print(pd.DataFrame(pairs).query("scenario=='saved'").groupby('period')[
        ['stock_borrow','financing_vs_cash','saved_position_funding','saved_interest_on_prior_excess_pnl',
         'short_dollar_years','segregated_cash_loan_dollar_years','negative_net_cash_days']].sum().to_string(),flush=True)


if __name__=='__main__': main()
