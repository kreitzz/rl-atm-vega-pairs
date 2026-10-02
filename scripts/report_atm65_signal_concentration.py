"""Descriptive concentration diagnostics; never feeds policy or roster choice."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.run_atm65_signal_mining import REPORT,OUT,source_folder
from scripts.run_contract_walkforward import digest

def main():
    assert json.loads((REPORT/'audit.json').read_text())['status']=='complete'
    pairs=pd.read_csv(REPORT/'pairs.csv');comparison=pd.read_csv(REPORT/'comparison.csv')
    rows=[]
    for (model,period),group in pairs.groupby(['model','period']):
        winners=group[group.net_excess_pnl>1e-7].sort_values('net_excess_pnl',ascending=False)
        losers=group[group.net_excess_pnl<-1e-7]
        book=pd.read_csv(REPORT/f'{model}_{period}_portfolio.csv',index_col=0,parse_dates=[0])
        row=dict(model=model,period=period,profitable_pairs=len(winners),losing_pairs=len(losers),
                 positive_pnl=float(winners.net_excess_pnl.sum()),negative_pnl=float(losers.net_excess_pnl.sum()))
        if len(winners):
            top=winners.iloc[0]
            row.update(largest_winner=top.label,largest_winner_pnl=float(top.net_excess_pnl),
                top_three_share_positive_pnl=float(winners.head(3).net_excess_pnl.sum()/winners.net_excess_pnl.sum()))
            for count in [1,3]:
                removed=pd.Series(0.,index=book.index)
                for r in winners.head(count).itertuples():
                    folder=source_folder(r.pair,model,period);status=json.loads((folder/'status.json').read_text())
                    nav=pd.read_parquet(status.get('nav_source',folder/'daily_nav.parquet'),columns=['nav','cash_benchmark'])
                    assert nav.index.equals(book.index)
                    removed+=nav.nav-nav.cash_benchmark
                substitute=book.nav-removed
                excess=(substitute.pct_change().fillna(0)-book.cash_benchmark.pct_change().fillna(0)).iloc[1:]
                row[f'sharpe_top_{count}_replaced_with_cash']=float(np.sqrt(252)*excess.mean()/excess.std()) if excess.std()>1e-12 else None
        rows.append(row)
    table=pd.DataFrame(rows);table.to_csv(REPORT/'concentration.csv',index=False)
    ranks=pd.read_csv(REPORT/'early_rank.csv')
    annual=pd.read_csv(REPORT/'annual.csv')
    lines=['','## Signal and concentration diagnostics','',
        'Ranks below use pre-2018 completed training cohorts only. Pair observations are non-overlapping within each pair but pairs can share underlyings. These averages are descriptive, not independent significance tests.','',
        '| Candidate | Mean within-pair rank correlation | Pairs | Cohorts |','|---|---:|---:|---:|']
    for r in ranks.head(6).itertuples():lines.append(f'| {r.feature} | {r.mean_spearman:+.3f} | {r.pairs} | {r.rows} |')
    lines+=['','| Model (full period) | Profitable / losing pairs | Top 3 share of positive P&L | Sharpe excluding largest winner |',
            '|---|---:|---:|---:|']
    for r in table[table.period.eq('full')].itertuples():
        share=getattr(r,'top_three_share_positive_pnl',np.nan);sharpe=getattr(r,'sharpe_top_1_replaced_with_cash',np.nan)
        lines.append(f'| {r.model} | {r.profitable_pairs} / {r.losing_pairs} | {share:.1%} | {sharpe:.3f} |')
    lines+=['','Excluding winners is an ex-post sensitivity diagnostic: those sleeves are replaced with cash, and no policy is retrained. It is not an executable selection rule.','',
        '| Year | Baseline | Dynamics | Dynamics + gamma | Reversal exit | Flat/reversal exit |','|---|---:|---:|---:|---:|---:|']
    order=['rl_sized','rl_dynamics','rl_dynamics_gamma','exit_reverse','exit_flat_or_reverse']
    yearly=annual[annual.period.eq('full')].pivot(index='year',columns='model',values='annualized_excess_sharpe')
    for year,r in yearly.iterrows():
        values=['cash' if pd.isna(r[m]) else f'{r[m]:.3f}' for m in order]
        lines.append('| '+str(year)+(' (partial)' if year==2025 else '')+' | '+' | '.join(values)+' |')
    lines+=['','[Concentration CSV](atm65_signal_mining/concentration.csv)','']
    report=REPORT.with_suffix('.md');content=report.read_text().split('\n## Signal and concentration diagnostics')[0]
    report.write_text(content+'\n'.join(lines))
    paths=[Path(__file__),REPORT/'pairs.csv',REPORT/'comparison.csv',REPORT/'annual.csv',REPORT/'early_rank.csv',REPORT/'audit.json']
    (REPORT/'concentration_inputs.json').write_text(json.dumps({str(p):digest(p) for p in paths},indent=2)+'\n')
    print(table.to_string(index=False),flush=True)

if __name__=='__main__':main()
