"""Ex-post concentration sensitivity; does not change the universe or select a policy."""
import pandas as pd
from scripts.run_tenor_comparison import OUTPUT,REPORT
from scripts.report_option_flex_ablation import sharpe

def main():
    pairs=pd.read_csv(REPORT/'pair_results.csv');rows=[]
    for tenor,g in pairs.groupby('tenor'):
        book=pd.read_csv(REPORT/f'{tenor}d_full_portfolio.csv',index_col=0,parse_dates=[0])
        for what,r in [('largest_winner',g.loc[g.net_excess_pnl.idxmax()]),('largest_loser',g.loc[g.net_excess_pnl.idxmin()])]:
            nav=pd.read_parquet(OUTPUT/r.pair/f'{tenor}d/daily_nav.parquet')
            # Replace the excluded sleeve with its cash benchmark; keep total capital fixed.
            reduced=book.nav-nav.nav+nav.cash_benchmark
            e=reduced.pct_change()-book.cash_benchmark.pct_change()
            rows.append(dict(tenor=tenor,diagnostic=what,pair=r.pair,label=r.label,entries=r.entries,
                             excluded_net_excess_pnl=r.net_excess_pnl,annualized_excess_sharpe_without_pair=sharpe(e.dropna())))
    result=pd.DataFrame(rows);result.to_csv(REPORT/'concentration_sensitivity.csv',index=False)
    print(result.to_string(index=False))

if __name__=='__main__':main()
