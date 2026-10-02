"""Fixed-inventory cost bridge and matched proxy entry-filter diagnostics."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.reprice_contract_execution import reprice_cash
from scripts.run_selected_pair_backtest import scores
from scripts.run_first_pass import payoff_sharpe

ROOT=Path('artifacts/contracts/broad_replay')
REPORT=Path('reports/broad_contract_replay')


def main():
    defs=json.loads((REPORT/'universe.json').read_text())
    membership=pd.read_csv(REPORT/'diagnostic_membership.csv')
    selected=set(membership.loc[membership.group.eq('broad')&membership.common_complete,'pair'])
    defs=[d for d in defs if d['pair'] in selected]
    cfg=json.loads((REPORT/'manifest.json').read_text())['settings']
    zero=pd.read_parquet('data/option_contracts/zero_curve.parquet')
    zero['date']=pd.to_datetime(zero.date);zero['distance']=abs(zero.days-30)
    zero=zero.sort_values(['date','distance','days']).drop_duplicates('date').sort_values('date')
    zero['rate']/=100
    summaries=[];cost_rows=[];decision_rows=[];proxy_rows=[]
    for model in ['rl_reference','rl_sized','nn_fixed']:
        books={};proxy=None;unknown=None
        for d in defs:
            folder=ROOT/d['pair']/model
            nav=pd.read_parquet(folder/'daily_nav.parquet')
            rates=pd.merge_asof(pd.DataFrame({'date':nav.index}),zero[['date','rate']],on='date',direction='backward').set_index('date').rate
            commission=nav.option_execution_cost-nav.option_model_penalty
            stages={'net':np.zeros(len(nav)),
                    'no_option_penalty':nav.option_model_penalty,
                    'no_option_commission':commission,
                    'no_option_costs':nav.option_execution_cost,
                    'no_execution_costs':nav.option_execution_cost+nav.hedge_cost,
                    'no_execution_or_stock_borrow':nav.option_execution_cost+nav.hedge_cost+nav.stock_borrow_cost}
            for stage,saving in stages.items():
                out=reprice_cash(nav,rates,saving,np.zeros(len(nav)),cfg['capital'],cfg['borrowing_spread'])
                if stage=='net':np.testing.assert_allclose(out.nav,nav.nav,atol=1e-7,rtol=0)
                if stage not in books:books[stage]=out[['nav','cash_benchmark']].copy()
                else:books[stage]+=out[['nav','cash_benchmark']]
            try:
                dec=pd.read_csv(folder/'decisions.csv',parse_dates=['signal_date','entry_date'])
            except pd.errors.EmptyDataError:
                dec=pd.DataFrame(columns=['signal_date','entry_date','status'])
            entered=dec[dec.status.eq('entered')]
            total_vega=((entered.vega_a+entered.vega_b)/2).sum() if len(entered) else 0.
            cost_rows.append(dict(pair=d['pair'],label=d['label'],model=model,entries=len(entered),
                                  entry_vega_sum=total_vega,commission=commission.sum(),
                                  penalty=nav.option_model_penalty.sum(),hedge_cost=nav.hedge_cost.sum(),
                                  stock_borrow=nav.stock_borrow_cost.sum(),
                                  excess_dollars=nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1]))
            for reason,count in dec.status.value_counts().items():
                decision_rows.append(dict(model=model,reason=reason,count=count))
            if model!='rl_sized':continue
            if proxy is None:
                proxy=pd.DataFrame(0.,index=nav.index,columns=['all_saved_entries','accepted_entries','accepted_actual_vega'])
                unknown=pd.Series(False,index=nav.index)
            pieces=[]
            for path in d['paths']:
                f=pd.read_csv(path,index_col=0,parse_dates=[0]);year=int(Path(path).stem)
                pieces.append(f[(f.index.year>=year)&(f.index.year<year+2)])
            if not pieces:continue
            p=pd.concat(pieces).sort_index()
            p=p[p.index.isin(nav.index)&(p.index<=nav.index[-1]-pd.Timedelta(days=35))]
            rewards=p.rl_memory_reward
            unknown.loc[p.index]|=rewards.isna()
            proxy.loc[p.index,'all_saved_entries']+=rewards.fillna(0)/len(defs)
            days=pd.DatetimeIndex(entered.signal_date)
            r=rewards.reindex(days)
            assert days.isin(p.index).all()
            proxy.loc[days,'accepted_entries']+=r.fillna(0).to_numpy()/len(defs)
            if len(entered):
                weight=((entered.vega_a+entered.vega_b)/2/(cfg['entry_vega_per_leg']*entered.action.abs())).to_numpy()
                proxy.loc[days,'accepted_actual_vega']+=r.fillna(0).to_numpy()*weight/len(defs)
        for stage,book in books.items():
            book['return_']=book.nav.pct_change().fillna(0)
            book['excess_return']=book.return_-book.cash_benchmark.pct_change().fillna(0)
            summaries.append(dict(model=model,stage=stage,**scores(book,len(defs)*cfg['capital'])))
        if model=='rl_sized':
            # Exclude runoff and use one common complete-label calendar for all variants.
            good=proxy[(~unknown)&(proxy.index<=proxy.index[-1]-pd.Timedelta(days=35))]
            for col in good:
                proxy_rows.append(dict(variant=col,dates=len(good),excluded_unknown_dates=int(unknown.sum()),
                                       payoff_sharpe=payoff_sharpe(good[col]),net_proxy_sum=good[col].sum()))
            good.to_csv(REPORT/'gap_proxy_daily.csv')
        print(f'Finished {model}',flush=True)
    pd.DataFrame(summaries).to_csv(REPORT/'gap_cost_bridge.csv',index=False)
    pd.DataFrame(cost_rows).to_csv(REPORT/'gap_pair_costs.csv',index=False)
    pd.DataFrame(decision_rows).groupby(['model','reason'])['count'].sum().reset_index().to_csv(REPORT/'gap_entry_counts.csv',index=False)
    pd.DataFrame(proxy_rows).to_csv(REPORT/'gap_proxy_entries.csv',index=False)
    (REPORT/'gap_method.json').write_text(json.dumps(dict(pairs=len(defs),
        cost_method='Fixed inventories and hedges; cash financing recomputed exactly; no retraining or resizing. Cost add-backs are diagnostic, not executable policies.',
        proxy_method='Same common-complete pair union and fixed budgets. Original proxy rewards, common complete-outcome dates, 35-day runoff. Entry eligibility and then actual initial vega applied. Proxy cohort Sharpe is not daily marked Sharpe.'),indent=2)+'\n')
    print(pd.DataFrame(summaries).to_string(index=False))
    print(pd.DataFrame(proxy_rows).to_string(index=False))


if __name__=='__main__':main()
