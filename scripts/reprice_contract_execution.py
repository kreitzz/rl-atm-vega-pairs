"""Execution-cost sensitivity on fixed realized trade/hedge inventories.

This isolates costs; NAV sizing, trade selection and risk exits do not re-run.
The separate walk-forward experiment retrains and reapplies all risk limits.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_contract_walkforward import DEFINITIONS, MODELS, load_pair, digest
from scripts.run_selected_pair_backtest import scores


def reprice_cash(nav, rates, old_slippage, new_penalty, capital, borrowing_spread):
    """Recompute financing exactly for fixed holdings and non-interest flows."""
    flow=nav.cash.diff().fillna(nav.cash.iloc[0]-capital)-nav.interest
    elapsed=nav.index.to_series().diff().dt.days.fillna(0).to_numpy()/365
    previous_rates=rates.reindex(nav.index).shift(1).fillna(0).to_numpy()
    cash=capital;cash_path=[];interest_path=[]
    for delta,rate,years,saving,charge in zip(flow,previous_rates,elapsed,old_slippage,new_penalty):
        interest=cash*(rate+(borrowing_spread if cash<0 else 0))*years
        cash+=interest+delta+saving-charge
        cash_path.append(cash);interest_path.append(interest)
    result=nav.copy()
    result['cash']=cash_path;result['interest']=interest_path
    result['nav']=result.cash+result.stock_value+result.option_value
    result['return_']=result.nav.pct_change().fillna(result.nav.iloc[0]/capital-1)
    result['excess_return']=result.return_-result.cash_benchmark.pct_change().fillna(0)
    result['penalty']=np.asarray(new_penalty)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('artifacts/contracts/walkforward'))
    parser.add_argument('--source-report',type=Path,default=Path('reports/contract_walkforward'))
    parser.add_argument('--output',type=Path,default=Path('artifacts/contracts/mid_sensitivity'))
    parser.add_argument('--report',type=Path,default=Path('reports/mid_execution_sensitivity'))
    parser.add_argument('--penalties',type=float,nargs='+',default=[0,.05,.1,.25])
    args=parser.parse_args()
    if not np.isfinite(args.penalties).all() or min(args.penalties)<0:raise ValueError('Penalties must be finite and nonnegative')
    args.output.mkdir(parents=True,exist_ok=True);args.report.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((args.source_report/'manifest.json').read_text());cfg=manifest['settings']
    source_status=pd.read_csv(args.source_report/'status.csv')
    rows=[];books={};hashes={}
    for definition in DEFINITIONS:
        pair=definition['pair']
        if pair not in manifest['pairs']:continue
        q,stocks,*_=load_pair(definition)
        q=q.sort_values(['optionid','date']).copy()
        # Only the most recent observed Greek, adjusted for any integer split.
        q['scaled_vega']=(q.vega*q.cfadj).where(q.vega.ge(0))
        q['cost_vega']=q.groupby('optionid').scaled_vega.ffill()/q.cfadj
        q['mid']=(q.bid+q.ask)/2
        quotes=q.set_index(['date','optionid'])[['mid','cost_vega']]
        rates=stocks.groupby('date').rate.mean()
        for model in MODELS:
            state=source_status[(source_status.pair==pair)&(source_status.model==model)]
            if len(state)!=1 or state.status.iloc[0]!='complete':
                raise ValueError(f'Cannot reprice incomplete source {pair}/{model}')
            folder=args.source/pair/model
            nav=pd.read_csv(folder/'daily_nav.csv',index_col=0,parse_dates=[0])
            trades=pd.read_csv(folder/'trades.csv',parse_dates=['date'])
            for name in ['daily_nav.csv','trades.csv']:hashes[str(folder/name)]=digest(folder/name)
            fills=trades[trades.reason.ne('split_adjustment')].copy()
            fills=fills.join(quotes,on=['date','optionid'])
            if not np.isfinite(fills[['mid','cost_vega']]).all().all():raise ValueError('Unpriceable execution penalty')
            flow=(fills.quantity.abs()*100*fills.cost_vega).groupby(fills.date).sum().reindex(nav.index,fill_value=0)
            commission=trades.groupby('date').commission.sum().reindex(nav.index,fill_value=0)
            old_slippage=nav.option_execution_cost-commission
            observed=(fills.quantity*100*(fills.price-fills.mid)+fills.get('execution_penalty',0.)).groupby(fills.date).sum().reindex(nav.index,fill_value=0)
            np.testing.assert_allclose(observed,old_slippage,atol=1e-7,rtol=0)
            check=reprice_cash(nav,rates,old_slippage,old_slippage,cfg['capital'],cfg['borrowing_spread'])
            np.testing.assert_allclose(check.nav,nav.nav,atol=1e-7,rtol=0)
            for penalty in args.penalties:
                result=reprice_cash(nav,rates,old_slippage,penalty*flow,cfg['capital'],cfg['borrowing_spread'])
                if result.nav.le(0).any():raise ValueError(f'Nonpositive fixed-inventory NAV: {pair}/{model}/{penalty}')
                breaches=np.zeros(len(nav),bool)
                for field,limit in [('gross_vega',cfg['max_gross_vega']/cfg['capital']),
                                     ('gross_hedge_notional',cfg['max_hedge_leverage']),
                                     ('capital_reserve',cfg['max_reserve_fraction']),('stress_loss',cfg['max_stress_loss_fraction'])]:
                    breaches|=(nav[field]/result.nav>limit+1e-10).to_numpy()
                rows.append(dict(pair=pair,model=model,penalty_vol_points=penalty,**scores(result,cfg['capital']),
                                 old_option_slippage=float(old_slippage.sum()),new_option_penalty=float(result.penalty.sum()),
                                 option_commission=float(commission.sum()),stock_execution=float(nav.hedge_cost.sum()),
                                 stock_borrow=float(nav.stock_borrow_cost.sum()),interest=float(result.interest.sum()),
                                 fixed_inventory_risk_breach_days=int(breaches.sum())))
                books.setdefault((model,penalty),[]).append(result[['nav','cash_benchmark']])
        print(f'Repriced {pair}',flush=True)
        del q,stocks,quotes
    portfolios=[];annual=[]
    for (model,penalty),members in books.items():
        if len(members)!=len(manifest['pairs']):raise ValueError('Missing portfolio member')
        assert all(n.index.equals(members[0].index) for n in members)
        book=sum(members);book['return_']=book.nav.pct_change().fillna(0)
        book['excess_return']=book.return_-book.cash_benchmark.pct_change().fillna(0)
        capital=len(members)*cfg['capital'];years=(book.index[-1]-book.index[0]).days/365.25
        portfolios.append(dict(model=model,penalty_vol_points=penalty,**scores(book,capital),cagr=(book.nav.iloc[-1]/capital)**(1/years)-1))
        book.to_csv(args.output/f'{model}_{penalty:g}_portfolio.csv')
        for year,g in book.groupby(book.index.year):
            previous=capital if g.index[0]==book.index[0] else book.loc[:g.index[0]].nav.iloc[-2]
            annual.append(dict(model=model,penalty_vol_points=penalty,year=int(year),total_return=g.nav.iloc[-1]/previous-1,
                               annualized_excess_sharpe=np.sqrt(252)*g.excess_return.mean()/g.excess_return.std() if g.excess_return.std()>0 else np.nan))
    pd.DataFrame(rows).to_csv(args.report/'pairs.csv',index=False)
    pd.DataFrame(portfolios).to_csv(args.report/'portfolio.csv',index=False)
    pd.DataFrame(annual).to_csv(args.report/'annual.csv',index=False)
    (args.report/'manifest.json').write_text(json.dumps(dict(source=str(args.source),source_report=str(args.source_report),
        source_sha256=hashes,penalties=args.penalties,settings=cfg,method='fixed trades and hedges; financing recomputed; no retraining, resizing or renewed risk intervention',
        code_sha256=digest(Path(__file__))),indent=2)+'\n')
    print(pd.DataFrame(portfolios).to_string(index=False),flush=True)


if __name__=='__main__':main()
