"""Compare unchanged policies under quoted-side and halfway-to-mid fills."""
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    base=Path('artifacts/contracts/four_pairs')
    new=Path('artifacts/contracts/four_pairs_half_spread')
    report=Path('reports/selected_pair_backtest/half_spread')
    checks=[]
    for pair in ['AMD_NVDA','XOM_CVX','JPM_BAC','HD_LOW']:
        for model in ['rl_sized','nn_fixed','higher_iv']:
            a=base/'2025'/pair/model;b=new/'2025'/pair/model
            ta=pd.read_csv(a/'trades.csv');tb=pd.read_csv(b/'trades.csv')
            same=['date','cohort','secid','optionid','quantity','reason','commission']
            pd.testing.assert_frame_equal(ta[same],tb[same])
            na=pd.read_csv(a/'daily_nav.csv',index_col=0);nb=pd.read_csv(b/'daily_nav.csv',index_col=0)
            fees=ta.groupby('date').commission.sum().reindex(na.index,fill_value=0)
            expected_cost=fees+(na.option_execution_cost-fees)/2
            np.testing.assert_allclose(nb.option_execution_cost,expected_cost,rtol=0,atol=1e-8)
            for col in ['hedge_cost','stock_borrow_cost','stock_value','option_value']:
                np.testing.assert_allclose(na[col],nb[col],rtol=0,atol=1e-8)
            savings=(na.option_execution_cost-nb.option_execution_cost).cumsum()
            financing=(nb.interest-na.interest).cumsum()
            error=(nb.nav-na.nav-savings-financing).abs().max()
            assert error<1e-6
            checks.append(dict(pair=pair,model=model,spread_savings=float(savings.iloc[-1]),
                               interest_change=float(financing.iloc[-1]),nav_reconciliation_error=float(error)))
    pd.DataFrame(checks).to_csv(report/'comparison_checks.csv',index=False)
    old=pd.read_csv('reports/selected_pair_backtest/portfolio.csv')
    updated=pd.read_csv(report/'portfolio.csv')
    comparison=old.merge(updated,on=['window','model'],suffixes=('_bidask','_halfway'))
    comparison.to_csv(report/'portfolio_comparison.csv',index=False)
    print(comparison.to_string(index=False))
    print(pd.read_csv(report/'status.csv').query('model == "rl_sized"').to_string(index=False))
    print('All 12 2025 trade schedules and commissions match; savings reconcile to NAV.')


if __name__=='__main__':main()
