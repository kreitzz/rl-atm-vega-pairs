"""Summarize only completed, reconciled daily portfolio ledgers."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.run_vega_portfolio import metrics,CAPITAL

OUT=Path('artifacts/contracts/vega_portfolio');REPORT=Path('reports/vega_portfolio')

def main():
    status=json.loads((OUT/'status.json').read_text())
    if status['status']!='complete':raise ValueError('Incomplete portfolio test')
    results=[];annual=[];attribution=[];risk=[]
    for mode in ['forecast','equal_rank']:
        f=pd.read_csv(OUT/mode/'nav.csv',index_col=0,parse_dates=[0]);tr=pd.read_csv(OUT/mode/'trades.csv');h=pd.read_csv(OUT/mode/'holdings.csv')
        assert f.names.iloc[-1]==0 and (f.nav>0).all()
        assert (f.net_vega.abs()<=np.maximum(10.,.01*f.gross_vega)+1e-6).all()
        assert f.ledger_residual.abs().max()<1e-5
        commission=float(tr.quantity.abs().sum()*status['option_commission_per_contract'])
        components={c:float(f[c].sum()) for c in ['option_pnl','hedge_pnl','dividends','interest','borrow','option_cost','stock_cost']}
        net=sum(components[c] for c in ['option_pnl','hedge_pnl','dividends','interest'])-sum(components[c] for c in ['borrow','option_cost','stock_cost'])
        assert np.isclose(net,f.nav.iloc[-1]-CAPITAL,rtol=0,atol=1e-5)
        attribution.append(dict(model=mode,**components,option_commission=commission,option_vol_penalty=components['option_cost']-commission,net_pnl=net,net_excess_pnl=f.nav.iloc[-1]-f.cash_benchmark.iloc[-1]))
        r=metrics(f);r.update(model=mode,option_fills=len(tr),monthly_position_entries=int(tr.reason.eq('open').sum()),active_names=h.secid.nunique(),
                             cagr_percent=100*((f.nav.iloc[-1]/CAPITAL)**(365.25/((f.index[-1]-f.index[0]).days+1))-1),cash_return_percent=100*(f.cash_benchmark.iloc[-1]/CAPITAL-1))
        results.append(r)
        prior_nav=prior_cash=CAPITAL
        for year,g in f.groupby(f.index.year):
            row=metrics(g);row.update(model=mode,year=year,return_percent=100*(g.nav.iloc[-1]/prior_nav-1),
                                      cash_return_percent=100*(g.cash_benchmark.iloc[-1]/prior_cash-1),
                                      excess_pnl=(g.nav.iloc[-1]-prior_nav)-(g.cash_benchmark.iloc[-1]-prior_cash))
            annual.append(row);prior_nav=g.nav.iloc[-1];prior_cash=g.cash_benchmark.iloc[-1]
        risk.append(dict(model=mode,max_names=int(f.names.max()),mean_names=float(f.names.mean()),mean_gross_vega=float(f.gross_vega.mean()),
                         max_gross_vega=float(f.gross_vega.max()),max_abs_net_vega=float(f.net_vega.abs().max()),
                         greek_fallbacks=int(f.greek_fallbacks.sum()),max_abs_ledger_residual=float(f.ledger_residual.abs().max())))
    pd.DataFrame(results).to_csv(REPORT/'summary.csv',index=False);pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False)
    pd.DataFrame(attribution).to_csv(REPORT/'attribution.csv',index=False);pd.DataFrame(risk).to_csv(REPORT/'risk_audit.csv',index=False)
    lines=['# Portfolio-level vega-neutral test','',
      'Completed January 2, 2020–August 29, 2025, with $1 million initial capital for each strategy. These are actual marked-option and stock-hedge cash ledgers. Sharpe is annualized from daily cash-excess returns using √252. Total returns include interest on cash.','',
      '| Allocator | Annualized excess Sharpe | Total return | CAGR | Max drawdown | Excess P&L |', '|---|---:|---:|---:|---:|---:|']
    for r in results:lines.append(f"| {r['model']} | {r['annualized_excess_sharpe']:.3f} | {r['return_percent']:.2f}% | {r['cagr_percent']:.2f}% | {r['max_drawdown_percent']:.2f}% | ${r['excess_pnl']:,.0f} |")
    lines.append(f"| Cash | Undefined (zero excess volatility) | {results[0]['cash_return_percent']:.2f}% | — | 0.00% | $0 |")
    lines += ['', '## Annual results', '', '| Year | Forecast Sharpe | Forecast return | Rank Sharpe | Rank return | Cash return |', '|---|---:|---:|---:|---:|---:|']
    a=pd.DataFrame(annual)
    for year in sorted(a.year.unique()):
        x=a[(a.year==year)&a.model.eq('forecast')].iloc[0];y=a[(a.year==year)&a.model.eq('equal_rank')].iloc[0]
        lines.append(f'| {year} | {x.annualized_excess_sharpe:.3f} | {x.return_percent:.2f}% | {y.annualized_excess_sharpe:.3f} | {y.return_percent:.2f}% | {x.cash_return_percent:.2f}% |')
    lines += ['', '2025 is partial through August 29.', '', '## Specification', '',
      f"- Universe: {status['universe_names']} cached underlying IDs in the original 346-pair roster. Monthly eligibility takes the union of names in historically eligible pairs. Positions are not coupled into pairs and there is no additional pairwise-beta gate. Upstream universe/screen construction and data availability remain inherited limitations.",
      '- Signal: the prior HAR+quoted-IV variance forecast (5/21/63-session variance, DTE, quoted IV), pooled across securities. Each monthly decision uses the preceding close, with models fitted on outcomes completed before that signal quarter. No RL/NN policy was trained in this first construction test. Forecast training still comes from the prior screened/resolved history; it is not a new broad-universe label set.',
      '- At most the 15 highest and 15 lowest demeaned forecast-value scores enter the candidate set. Forecast allocation maximizes approximate relative variance value less two option-fill penalties/commissions. Rank allocation targets equal long/short vega per candidate, then scales to the same risk limits; it trades only if aggregate estimated edge exceeds those option costs. Both are long/short relative-value books. Stock hedging, financing and borrow costs are charged in the ledger, but not fully forecast by this entry objective.',
      '- Contracts: approximately 30-day ATM calls, 21–45 DTE and 0.25–0.75 delta at entry, positive uncrossed quotes, at least 100 OI, relative bid/ask width ≤25%, stock ≥$5 at signal and execution. Contract IDs stay fixed between monthly rebuilds. All holdings close one trading session before the earliest expiry, or at the next monthly rebuild. This changes holding schedules versus the previous expiry experiments.',
      '- Daily balancing preserves initial relative vega targets where feasible and rescales for current risks. Integer contracts only; net vega must be within the larger of $10 per volatility point or 1% of gross vega after each close. Delta is hedged per name at each close. Neutrality is an end-of-day constraint, not an intraday guarantee.',
      '- Limits per $1 million NAV: $20,000 gross vega, $2,000 per-name vega, 2× gross stock-hedge notional, 80% capital reserve, 5% independent-name stress loss for ±20% spot and ±10 volatility-point shocks, 1% absolute gamma P&L for a 1% spot move. A 1% pretrade capital cushion accommodates fees. These stress scenarios are not guarantees against larger moves.',
      '- Industry buckets use cached SIC prefixes, not GICS sectors. Each has a gross-vega cap of 35% of the overall gross-vega budget and a signed cap of 10% of that budget. These are budget-based limits, not caps on the fraction of actual utilized exposure. Missing/conflicting industries share an unknown bucket.',
      '- Names with earnings within seven days (or unknown countdowns) have an aggregate gross-vega cap of 25% of the total budget, updated daily. Countdown data uses final historical event dates assumed known, not an archived announcement schedule. The forecast itself excludes earnings.',
      '- No more than 100 contracts per name or 10% of signal-date option open interest, whichever is smaller; verified forward splits adjust this inventory cap. No position increases when delta is outside 0.15–0.85 or the current entry-quality filter fails. Whole-contract rounding may reduce the book further.',
      '- Execution costs on every option adjustment: midpoint plus a separate 0.10-vol-point penalty and $0.03 per contract per fill (user-specified institutional assumption). Stock hedge execution is 1 bp, stock borrow is 0.3% annualized, and negative cash costs the financing rate plus 1%. Ordinary stock dividends flow through the hedge. This midpoint assumption does not demonstrate executable fills.',
      '- Actual held midpoints are retained, including wide quotes. Crossed held bid/ask quotes are swapped, consistent with the inherited engine. Missing fitted Greeks use the last observed IV with current spot/time/rate/dividend PV. Missing held marks, unsupported corporate adjustments, insolvency or ledger failures stop the test rather than deleting the affected dates. American early assignment is not modeled.', '',
      '## Audit and interpretation', '',
      'This is an initial allocation experiment on repeatedly explored history, not an untouched test or a clean fixed-pair ablation: universe use, rebalancing and holding schedules also changed. It does not establish that RL improves portfolio construction. Both complete ledgers reconcile and end flat; all daily net-vega checks pass. No pair is removed because of subsequent profitability.', '',
      'The first development draft omitted contract-count/liquidity caps. It accumulated tens of thousands of low-vega penny-stock contracts and was invalid as a tradable sizing proposal; a held quote widening also caused a large marked loss. That draft is preserved locally under `unbounded_contract_draft`. The general price, OI, contract-count and off-ATM increase limits above were added after that failure and apply identically to both final allocators. Therefore these controls are a disclosed research revision, not a pre-registered untouched specification.', '',
      f"Peak worker RSS: {status['peak_rss_bytes']/1024**3:.2f} GiB, with one worker and a 50% duty-cycle cap.", '',
      '[Annual data](vega_portfolio/annual.csv) · [P&L attribution](vega_portfolio/attribution.csv) · [Exposure audit](vega_portfolio/risk_audit.csv) · [Fit timing](vega_portfolio/fit_audit.csv)', '']
    REPORT.with_suffix('.md').write_text('\n'.join(lines))
    print(pd.DataFrame(results).to_string(index=False),flush=True)

if __name__=='__main__':main()
