"""Supplement the common-roster table with full-roster results where known.

An unresolved book leaves that complete-universe NN row unavailable. This
does not change the matched comparison, policy or any simulated trade.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.run_atm_nn_cost_hurdles import OUTPUT, REPORT, HURDLES, PERIODS, fee_name
from scripts.run_selected_pair_backtest import scores
from scripts.run_contract_walkforward import digest


def main():
    assert json.loads((OUTPUT/'task_status.json').read_text())['status']=='complete'
    defs=json.loads((OUTPUT/'inputs.json').read_text())['definitions']
    coverage=json.loads((REPORT/'coverage.json').read_text())
    previous=pd.read_csv('reports/atm_higher_cost_hurdles/comparison.csv')
    previous['model']='rl_sized';previous['scope']='full_universe';previous['status']='complete'
    rows=previous.to_dict('records'); sources=[Path(__file__),REPORT/'coverage.json']
    for period,_ in PERIODS:
        source=REPORT/f'nn_fixed_fee_003_{period}_portfolio.csv'
        common=pd.read_csv(source,index_col=0,parse_dates=[0])
        cash=common.cash_benchmark/coverage['common_complete_pairs']
        assert abs(cash.iloc[0]-1e6)<1e-7
        sources.append(source)
        for fee in HURDLES:
            statuses=[]
            for d in defs:
                path=OUTPUT/d['pair']/fee_name(fee)/period/'status.json'
                sources.append(path);statuses.append(json.loads(path.read_text()))
            missing=[s['label'] for s in statuses if s['status']!='complete']
            row=dict(model='nn_fixed',training_hurdle=fee,period=period,scope='full_universe',
                     execution_commission=.03,pairs=len(defs),status='incomplete' if missing else 'complete',
                     unresolved_pairs=';'.join(missing))
            if not missing:
                book=pd.DataFrame({'nav':cash*len(defs),'cash_benchmark':cash*len(defs)})
                entries=active=0
                for d,s in zip(defs,statuses):
                    if not s.get('cash_only'):
                        path=OUTPUT/d['pair']/fee_name(fee)/period/'daily_nav.parquet'
                        nav=pd.read_parquet(path,columns=['nav','cash_benchmark']);sources.append(path)
                        assert nav.index.equals(book.index)
                        np.testing.assert_allclose(nav.cash_benchmark,cash,atol=1e-7,rtol=0)
                        book.nav+=nav.nav-cash
                    entries+=s['entries'];active+=int(s['entries']>0)
                book['return_']=book.nav.pct_change().fillna(0)
                book['excess_return']=book.return_-book.cash_benchmark.pct_change().fillna(0)
                row.update(entries=entries,active_pairs=active,
                           net_excess_pnl=float(book.nav.iloc[-1]-book.cash_benchmark.iloc[-1]),
                           **scores(book,1e6*len(defs)))
            rows.append(row)
    table=pd.DataFrame(rows).sort_values(['period','training_hurdle','model'])
    table.to_csv(REPORT/'full_roster_comparison.csv',index=False)
    (REPORT/'full_roster_sources.json').write_text(json.dumps(
        {str(p):digest(p) for p in sources},indent=2)+'\n')
    print(table[['model','training_hurdle','period','pairs','status','annualized_excess_sharpe','entries']].to_string(index=False),flush=True)


if __name__=='__main__':main()
