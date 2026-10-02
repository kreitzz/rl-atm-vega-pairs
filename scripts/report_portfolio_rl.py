"""Matched-period comparison of portfolio policy search and benchmarks."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scripts.run_vega_portfolio import CAPITAL

OUT=Path('artifacts/contracts/portfolio_rl');REPORT=Path('reports/portfolio_rl')
MODES={'forecast':'forecast','equal_rank':'equal_rank','nn_fixed':'forecast','rl_sized':'forecast'}


def performance(f,start_nav,start_cash):
    e=f.excess_return;sd=e.std()
    peak=np.maximum.accumulate(np.r_[start_nav,f.nav.to_numpy()])[1:]
    return dict(annualized_excess_sharpe=float(np.sqrt(252)*e.mean()/sd) if sd>1e-12 else np.nan,
                return_percent=100*(f.nav.iloc[-1]/start_nav-1),cash_return_percent=100*(f.cash_benchmark.iloc[-1]/start_cash-1),
                excess_pnl=(f.nav.iloc[-1]-start_nav)-(f.cash_benchmark.iloc[-1]-start_cash),
                max_drawdown_percent=100*(f.nav.to_numpy()/peak-1).min(),mean_gross_vega=f.gross_vega.mean(),active_days=int(f.names.gt(0).sum()))


def main():
    status=json.loads((OUT/'status.json').read_text());assert status['status']=='complete'
    decisions=pd.read_csv(REPORT/'decisions.csv',parse_dates=['date','signal_date','last_training_end'])
    assert (decisions.last_training_end<decisions.signal_date).all()
    summaries=[];annual=[];costs=[]
    for model,mode in MODES.items():
        root=OUT/model/mode
        f=pd.read_csv(root/'nav.csv',index_col=0,parse_dates=[0]);t=pd.read_csv(root/'trades.csv')
        assert f.names.iloc[-1]==0 and (f.nav>0).all()
        assert f.ledger_residual.abs().max()<1e-5
        assert (f.net_vega.abs()<=np.maximum(10.,.01*f.gross_vega)+1e-6).all()
        components={c:f[c].sum() for c in ['option_pnl','hedge_pnl','dividends','interest','borrow','option_cost','stock_cost']}
        net=sum(components[c] for c in ['option_pnl','hedge_pnl','dividends','interest'])-sum(components[c] for c in ['borrow','option_cost','stock_cost'])
        assert np.isclose(net,f.nav.iloc[-1]-CAPITAL,rtol=0,atol=1e-5)
        comm=.03*t.quantity.abs().sum()
        costs.append(dict(model=model,**components,option_commission=comm,option_vol_penalty=components['option_cost']-comm))
        summary=dict(model=model,**performance(f,CAPITAL,CAPITAL),funded_months=t.loc[t.reason.eq('open'),'date'].nunique(),option_fills=len(t))
        summaries.append(summary)
        nav=cash=CAPITAL
        for year,g in f.groupby(f.index.year):
            annual.append(dict(model=model,year=year,**performance(g,nav,cash)))
            nav=g.nav.iloc[-1];cash=g.cash_benchmark.iloc[-1]
    result=pd.DataFrame(summaries);result.to_csv(REPORT/'comparison.csv',index=False)
    yearly=pd.DataFrame(annual);yearly.to_csv(REPORT/'annual.csv',index=False)
    pd.DataFrame(costs).to_csv(REPORT/'costs.csv',index=False)
    ladder=pd.concat([decisions.groupby('rl_scale').size().rename('months').reset_index().rename(columns={'rl_scale':'scale'}).assign(model='rl_sized'),
                      decisions.groupby('nn_scale').size().rename('months').reset_index().rename(columns={'nn_scale':'scale'}).assign(model='nn_fixed')])
    ladder.to_csv(REPORT/'sizing.csv',index=False)
    lines=['# RL portfolio sizing versus matched benchmarks','',
      'Evaluation: January 3, 2022–August 29, 2025. All four policies start with $1 million, use the same monthly candidate portfolios and signals, $0.03 per option contract per fill, 0.10-vol-point execution penalty, daily delta/vega balancing, and identical liquidity/risk limits. Results use continuous actual-option cash-ledger replays. Total returns include cash interest; Sharpe uses daily cash-excess returns annualized with √252.','',
      '| Policy | Annualized excess Sharpe | Total return | Max drawdown | Excess P&L | Funded months | Mean gross vega |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for r in result.itertuples():lines.append(f'| {r.model} | {r.annualized_excess_sharpe:.3f} | {r.return_percent:.2f}% | {r.max_drawdown_percent:.2f}% | ${r.excess_pnl:,.0f} | {r.funded_months} | {r.mean_gross_vega:,.0f} |')
    lines.append(f'| Cash | Undefined | {result.cash_return_percent.iloc[0]:.2f}% | 0.00% | $0 | 0 | 0 |')
    lines+=['','## Annual results','','| Year | Forecast Sharpe | NN Sharpe | RL Sharpe | NN return | RL return |','|---|---:|---:|---:|---:|---:|']
    for year,g in yearly.groupby('year'):
        g=g.set_index('model');a=g.loc['forecast'];b=g.loc['nn_fixed'];c=g.loc['rl_sized']
        lines.append(f'| {year} | {a.annualized_excess_sharpe:.3f} | {b.annualized_excess_sharpe:.3f} | {c.annualized_excess_sharpe:.3f} | {b.return_percent:.2f}% | {c.return_percent:.2f}% |')
    lines += ['', '2025 is partial through August 29. These results must not be compared directly with the earlier 2020–2025 full-period Sharpe: the matched evaluation starts in 2022 because 2020–2021 supplies initial policy training.', '',
      '## What learns', '',
      '- RL is the existing direct Sharpe policy-search approach: a shallow regression-tree state partition (at most three leaves, minimum six training months per leaf) and cross-entropy search over state-dependent sizing actions. The ladder is cash, 25%, 50%, 75%, or 100% of the forecast allocator’s initial portfolio. It learns portfolio timing/sizing; individual-name selection and relative weights still come from the constrained forecast allocator.',
      '- The NN benchmark has one 16-unit tanh hidden layer. It predicts full-size monthly net excess payoff and chooses full-size or cash. It receives exactly the same monthly state features and completed training months as RL. Architecture, regularization, seed, thresholds and search budget were fixed before inspecting evaluation results; there is no selection of a winning test seed.',
      '- State features aggregate the existing candidate signals: median IV, mean absolute forecast value, RV/IV variance ratio, value dispersion, earnings share, relative quote width and DTE, plus trailing six-/twelve-completed-month full-size portfolio profitability. Memory comes from the fixed forecast portfolio, not the learned policy’s future or unobserved outcome.',
      '- Both policies refit monthly on at most five years of completed episodes. The last outcome date must precede the preceding-close signal date. At least 18 training months are required. States, scalers, imputation and tree partitioning use training data only. Neither policy sees current-month reward during fitting.',
      '- Training rewards come from independent monthly portfolio episodes at each ladder size, with exact contract quantities, daily rebalancing and full costs. Each episode starts with $1 million and ends flat by the next monthly rebuild. The RL objective sums those daily excess-dollar paths. Final evaluation replays the chosen actions continuously, with actual evolving NAV and integer constraints; it is not a splice of standalone episode returns.',
      '- The sizing hook leaves the full-size baseline unchanged. Lower sizing is applied before integer rounding and daily risk projection. The same forecast-model training inputs and constrained allocator are shared across every comparison. No new turnover-reduction rule was introduced in this experiment.', '',
      '## Limits and checks', '',
      'This is a small monthly training sample on already explored research history, not a fresh holdout or proof of live profitability. Monthly shadow rewards and the continuous NAV-dependent book need not match exactly. The NN optimizes a payoff forecast while RL optimizes Sharpe; this comparison does not isolate reinforcement learning from differences in objective or policy family. A gross-sizing layer is narrower than unrestricted RL portfolio construction. Different policies can use different amounts of risk under identical limits.', '',
      'Inherited data assumptions include historical earnings dates treated as known, EOD midpoint execution with a penalty, daily rather than intraday neutrality, existing quote/Greek fallbacks, and no American early assignment. The prior general contract-count/liquidity controls were developed after an earlier failed sizing draft. No evaluation period or name is excluded for poor performance.', '',
      f"All four ledgers reconcile, finish flat, and pass daily net-vega checks. All {len(decisions)} policy fits pass strict completed-outcome timing checks. NN convergence warnings: {int((~decisions.nn_converged).sum())}. Peak worker RSS: {status['peak_rss_bytes']/1024**3:.2f} GiB, with one worker and the 50% duty-cycle cap.", '',
      '[Annual data](portfolio_rl/annual.csv) · [Monthly decisions and fit audit](portfolio_rl/decisions.csv) · [Sizing frequencies](portfolio_rl/sizing.csv) · [Costs](portfolio_rl/costs.csv) · [Base portfolio specification](vega_portfolio.md)', '']
    REPORT.with_suffix('.md').write_text('\n'.join(lines));print(result.to_string(index=False),flush=True)

if __name__=='__main__':main()
