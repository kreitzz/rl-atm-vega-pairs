"""Independently reconcile the cash ledger and publish aggregate pilot results."""
import json
from pathlib import Path
import argparse

import numpy as np
import pandas as pd


def reconcile(folder, source_root):
    manifest = json.loads((folder/'manifest.json').read_text())
    cfg = manifest['config']
    nav = pd.read_csv(folder/'daily_nav.csv', index_col=0, parse_dates=[0])
    trades = pd.read_csv(folder/'option_trades.csv', parse_dates=['date'])
    stocks = pd.concat([pd.read_parquet(source_root/'underlying_prices'/f'year={year}.parquet',
                        filters=[('secid', 'in', cfg['secids'])]) for year in nav.index.year.unique()])
    spots = stocks.pivot(index='date', columns='secid', values='close').reindex(nav.index)
    holdings = pd.DataFrame({s: nav[f'hedge_{s}'] for s in cfg['secids']})
    turnover = (holdings.diff().fillna(holdings)*spots).abs().sum(axis=1)
    np.testing.assert_allclose(turnover*cfg['stock_cost_bps']/10_000, nav.hedge_cost, atol=1e-8)
    stock_pnl = (holdings.shift(1).fillna(0)*spots.diff().fillna(0)).sum(axis=1)-nav.hedge_cost
    option_cash = (-trades.quantity*100*trades.price-trades.commission).groupby(trades.date).sum().reindex(nav.index,fill_value=0)
    option_pnl = option_cash+nav.option_value.diff().fillna(nav.option_value)
    expected = option_pnl+stock_pnl+nav.interest-nav.stock_borrow_cost+nav.dividend_cash
    np.testing.assert_allclose(expected,nav.pnl,atol=1e-7)
    np.testing.assert_allclose(nav.nav,nav.cash+nav.option_value+nav.stock_value,atol=1e-7)
    np.testing.assert_allclose(nav.nav, cfg['capital']+nav.pnl.cumsum(),atol=1e-7)
    if not trades.groupby(['cohort','optionid']).quantity.sum().eq(0).all():
        raise AssertionError('Some contract quantities did not roundtrip')
    summary = json.loads((folder/'summary.json').read_text())
    summary.update(option_pnl_after_cost=float(option_pnl.sum()),
                   stock_hedge_pnl_after_cost=float(stock_pnl.sum()),
                   ledger_max_error=float(abs(expected-nav.pnl).max()),
                   missing_greeks_exit_cohorts=trades.loc[trades.reason.eq('missing_greeks_exit'),'cohort'].nunique(),
                   pre_dividend_exit_cohorts=trades.loc[trades.reason.eq('pre_ex_dividend_exit'),'cohort'].nunique())
    return summary, nav


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args()
    output=Path('reports/contract_pilot');output.mkdir(exist_ok=True)
    rows=[]
    for label,folder in [('Saved RL','pilot_envelope'),('Higher-IV direction','pilot_higher_iv')]:
        summary,nav=reconcile(Path('artifacts/contracts')/folder,args.source_root)
        summary['model']=label
        rows.append(summary)
        nav[['nav','excess_pnl','gross_vega','net_vega']].to_csv(output/f'{folder}_nav.csv')
    report=pd.DataFrame(rows).drop(columns='dispositions')
    report.to_csv(output/'summary.csv',index=False)
    lines=['# Actual-contract pilot: AMD/NVDA','',
           '**Engineering pilot, not a fresh holdout or a validated executable performance claim.** '
           'Saved RL actions are unchanged; no model is retrained. The strict quote run stops on crossed held quotes on April 4, 2025. '
           'The completed results below explicitly use a conservative bid/ask-envelope sensitivity on that date.','',
           '| Policy | Return including cash | Above cash, initial-capital basis | Annualized daily excess Sharpe | Max NAV drawdown | Entries |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f"| {r['model']} | {r['total_return']:.2%} | {r['excess_return_on_initial_capital']:.2%} | {r['daily_excess_sharpe']:.2f} | {r['max_drawdown']:.2%} | {r['entered_cohorts']} |")
    lines += ['',
        'Signals: January 2–July 25, 2025. NAV: January 2–August 29, including runoff. Returns above are period returns, not annualized; Sharpe uses sqrt(252) on daily NAV returns minus the daily cash benchmark return. This convention differs from the earlier overlapping 30-day proxy Sharpe. The short sample does not support a stable Sharpe estimate. The higher-IV control keeps the original absolute signal sizes and active signal dates but shorts the richer of the two selected contracts, using prior-session contract IV. It is not an independently trained strategy.','',
        '## Trading convention','',
        '- One call on each underlying: standard 100-share contracts, nearest 30-day expiry within 21–45 days at entry, then nearest spot ATM strike within 10%. Select contract IDs using the previous session’s chain and stock close. Entry occurs the next session; retain those IDs until exit. Each leg chooses its expiry separately.',
        '- Previous-session open interest must be at least 100. Entry bid must be positive and bid/ask width no more than 25% of midpoint. The live entry quote is rechecked; there is no substitution with a different contract when a selected quote fails.',
        '- Initial capital $1 million. Maximum entry dollar vega is $450 per leg times the saved action magnitude. Floor contract quantities to integers and reject entry imbalance above 10%. Block new entries above $20,000 gross outstanding dollar vega. This gate does not force liquidation if existing vega subsequently grows beyond the limit; any breach is reported.',
        '- Buy at ask, sell at bid at both entry and exit, plus $0.65 per contract per transaction. Mark open positions at midpoint. Aggregate stock delta hedges by underlying and rebalance daily at stock close, charging 1 basis point of traded notional. Fractional stock shares are allowed; option contracts are integer.',
        '- Exit both legs after at most 30 calendar days or on the last session strictly before the earlier expiry. Close the pair before a known cash ex-dividend date. If observed contract prices are usable but Greeks disappear, liquidate both legs at quoted sides instead of imputing Greeks. Missing prices and unsupported held corporate actions fail the run.',
        '- Cash earns the previous close’s nearest-30-day zero-curve rate, using ACT/365 simple accrual over each interval. Negative cash pays that rate plus 1%; short stock costs 0.3% annually. Short-stock dividends are debited. Dividend entitlement is recognized on ex-date; this is an accrual approximation, not a payment-date financing model.',
        '- The full cash balance, including short-sale proceeds, earns the modeled rate. Broker collateral haircuts, stock-loan rebates, option margin, liquidity impact and arbitrary early assignment are not modeled. No deployable leverage or capacity conclusion follows from these returns.',
        '- Daily quotes and closing stock prices are not perfectly synchronized. Selection uses lagged information, but same-session quote-based sizing and close-based hedging remain an EOD execution approximation. Contract size participation and quote depth are not enforced.','',
        '## Data quality and accounting','',
        'Downloaded 566,156 call-option rows for AMD/NVDA, January–August 2025. OptionMetrics strike is divided by 1,000; its vega is cents per share per one volatility point, so dollar vega per standard 100-share contract equals the raw vega field. '
        '[OptionMetrics reference manual, pp. 18–20](https://lib.tsinghua.edu.cn/PDF/OptionMetricsshiyongzhinan.pdf).','',
        'The default strict simulator rejects crossed held quotes. For the explicitly labeled sensitivity, set bid=min(raw bid, raw ask) and ask=max(raw bid, raw ask) only when valuing/exiting existing positions. New entries still reject crossed quotes. This is not evidence those envelope prices were executable. Four unique held contracts on April 4 require this treatment; other dates are unchanged. This handling and missing-Greek liquidation were added after observing data failures during engineering, not selected through performance optimization.','',
        '| Policy | Option P&L after costs | Stock hedge P&L after costs | Cash interest | Stock borrow | Missing-Greek exits | Pre-dividend exits |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f"| {r['model']} | ${r['option_pnl_after_cost']:,.0f} | ${r['stock_hedge_pnl_after_cost']:,.0f} | ${r['interest']:,.0f} | ${r['stock_borrow_cost']:,.0f} | {r['missing_greeks_exit_cohorts']} | {r['pre_dividend_exit_cohorts']} |")
    lines += ['',
        'A separate reconciliation reconstructs option cash flows and stock-hedge price changes, verifies every daily P&L and cash/position NAV identity, and checks that every contract position closes. There are no omitted held-position outcomes in the completed sensitivity run. Raw licensed quotes and detailed ledgers stay in ignored artifacts; aggregate research results are linked below.','',
        'The next extensions are a verified source for the crossed quotes, corporate-action and assignment handling, broker-consistent financing/margin, additional pairs and years, then a frozen contract-based training/evaluation experiment. Existing 2025 signals and the already-inspected sample are a transfer diagnostic, not a new out-of-sample claim.','',
        '## Reproduce','',
        'Install `requirements-contracts.txt` for the downloader dependencies. WRDS credentials remain in the local libpq credential store. Reproduction also requires the licensed underlying, rates and distribution cache, the broad-universe trading calendar, and saved policy actions generated by the preceding research pipeline; these are not bundled with the public code.','',
        '```sh',
        'python -m scripts.download_contract_quotes',
        '# Strict run (currently stops at a crossed held quote):',
        'python -m scripts.run_contract_pilot --source-root /path/to/optionmetrics',
        '# Explicit quote-envelope sensitivities:',
        'python -m scripts.run_contract_pilot --source-root /path/to/optionmetrics --repair-crossed-held-quotes --output artifacts/contracts/pilot_envelope',
        'python -m scripts.run_contract_pilot --source-root /path/to/optionmetrics --repair-crossed-held-quotes --direction-control higher_iv --output artifacts/contracts/pilot_higher_iv',
        'python -m scripts.report_contract_pilot --source-root /path/to/optionmetrics',
        'python -m unittest discover -s tests -q',
        '```','',
        '[Configuration](../config/contract_pilot.json) · [Aggregate results](contract_pilot/summary.csv) · [RL daily NAV](contract_pilot/pilot_envelope_nav.csv) · [Higher-IV daily NAV](contract_pilot/pilot_higher_iv_nav.csv)']
    Path('reports/contract_pilot.md').write_text('\n'.join(lines)+'\n')
    print(report.to_string(index=False))


if __name__=='__main__':main()
