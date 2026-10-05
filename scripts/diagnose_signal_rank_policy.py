"""Descriptive diagnostics after the fixed signal-rank experiment; no model selection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.run_signal_rank_policy import OUTPUT, REPORT, VARIANTS, definitions
from scripts.report_signal_rank_policy import sharpe


def main():
    defs=definitions(); fits=[]; risks=[]; incomplete=[]
    for d in defs:
        for variant in VARIANTS:
            folder=OUTPUT/d['pair']/variant
            status=json.loads((folder/'status.json').read_text())
            if status['status']!='complete':
                incomplete.append(status)
                continue
            risks.append(dict(pair=d['pair'],variant=variant,peak_rss_bytes=status.get('worker_peak_rss_bytes',0)))
            if status.get('structural_cash'): continue
            try: f=pd.read_csv(folder/'fits.csv')
            except pd.errors.EmptyDataError: continue
            f['pair']=d['pair']; f['variant']=variant; fits.append(f)
    f=pd.concat(fits,ignore_index=True)
    f.groupby(['variant','status']).size().rename('pair_months').to_csv(REPORT/'fit_status_counts.csv')
    pd.DataFrame(risks).groupby('variant').peak_rss_bytes.max().to_csv(REPORT/'peak_memory.csv')
    pairs=pd.read_csv(REPORT/'matched_pair_results.csv'); rows=[]
    for name in VARIANTS:
        p=pairs[pairs.variant.eq(name)].sort_values('net_excess_pnl',ascending=False)
        positives=p.loc[p.net_excess_pnl.gt(0),'net_excess_pnl']
        book=pd.read_csv(REPORT/f'{name}_portfolio.csv',index_col=0,parse_dates=[0])
        for n in [0,1,3]:
            reduced=book[['nav','cash_benchmark']].copy()
            for pair in p.head(n).pair:
                folder=OUTPUT/pair/name; status=json.loads((folder/'status.json').read_text())
                nav=pd.read_parquet(status.get('nav_source',folder/'daily_nav.parquet'))
                reduced-=nav[['nav','cash_benchmark']]
            e=reduced.nav.pct_change()-reduced.cash_benchmark.pct_change()
            rows.append(dict(variant=name,excluded_top_pairs=n,annualized_excess_sharpe=sharpe(e.dropna()),
                             positive_pairs=int((p.net_excess_pnl>1e-5).sum()),negative_pairs=int((p.net_excess_pnl < -1e-5).sum()),
                             largest_winner=p.iloc[0].label,top3_share_of_positive_pnl=positives.head(3).sum()/positives.sum() if positives.sum() else np.nan))
    pd.DataFrame(rows).to_csv(REPORT/'concentration_sensitivity.csv',index=False)
    full_rows=[]
    for name in VARIANTS:
        statuses=[json.loads((OUTPUT/d['pair']/name/'status.json').read_text()) for d in defs]
        if any(r['status']!='complete' for r in statuses):
            full_rows.append(dict(variant=name,status='incomplete',pairs=len(defs)))
            continue
        book=None
        for d,r in zip(defs,statuses):
            b=pd.read_parquet(r.get('nav_source',OUTPUT/d['pair']/name/'daily_nav.parquet'))[['nav','cash_benchmark']]
            if book is None: book=b.copy()
            else:
                assert book.index.equals(b.index)
                book+=b
        e=book.nav.pct_change()-book.cash_benchmark.pct_change()
        full_rows.append(dict(variant=name,status='complete',pairs=len(defs),annualized_excess_sharpe=sharpe(e.dropna()),
                              net_excess_pnl=book.nav.iloc[-1]-book.cash_benchmark.iloc[-1],entries=sum(r['entries'] for r in statuses)))
    pd.DataFrame(full_rows).to_csv(REPORT/'full_universe_completeness.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(full_rows).to_string(index=False))
    (OUTPUT/'task_status.json').write_text(json.dumps(dict(status='complete',variants=list(VARIANTS),pairs=len(defs),
        report=str(REPORT.with_suffix('.md')),reference_promoted=False,incomplete_books=incomplete,
        tests='17 targeted tests passed',caveat='Adaptive research on inspected data, not untouched holdout'),indent=2)+'\n')

if __name__=='__main__': main()
