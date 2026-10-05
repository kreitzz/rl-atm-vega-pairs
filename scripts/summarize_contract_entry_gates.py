"""Breadth and concentration for already selected entry-gate experiments."""
import json
from pathlib import Path
import pandas as pd

from scripts.test_contract_entry_gates import SOURCE, OUTPUT, REPORT, MODELS


def main():
    status=pd.read_csv(REPORT/'status.csv')
    baseline=pd.read_csv('reports/aligned_contract_learning/status.csv')
    failed=set(status.loc[status.status.ne('complete'),'pair']) | set(baseline.loc[baseline.status.ne('complete'),'pair'])
    definitions=json.loads(Path('reports/aligned_contract_learning/universe.json').read_text())
    selected={r['model']:r['variant'] for r in json.loads((REPORT/'selected.json').read_text())}
    rows=[]
    for d in definitions:
        if d['pair'] in failed or d['group']!='broad':continue
        for model in MODELS:
            for variant in dict.fromkeys(['baseline',selected[model]]):
                folder=SOURCE/d['pair']/model if variant=='baseline' else OUTPUT/d['pair']/f'{model}_{variant}'
                nav=pd.read_parquet(folder/'daily_nav.parquet')
                excess_dollars=(nav.nav-nav.cash_benchmark).diff().fillna(0)
                for period,mask in [('development_2018_2021',nav.index.year<2022),('later_2022_2025',nav.index.year>=2022),('full',nav.index.year>0)]:
                    rows.append(dict(pair=d['pair'],label=d['label'],industry=d['industry'],model=model,variant=variant,
                                     period=period,excess_dollars=excess_dollars[mask].sum()))
    pairs=pd.DataFrame(rows);pairs.to_csv(REPORT/'selected_pair_contributions.csv',index=False)
    summary=[]
    for (model,variant,period),g in pairs.groupby(['model','variant','period']):
        positive=g[g.excess_dollars>1e-5].sort_values('excess_dollars',ascending=False)
        summary.append(dict(model=model,variant=variant,period=period,
            profitable_pairs=len(positive),losing_pairs=int(g.excess_dollars.lt(-1e-5).sum()),
            gross_positive_dollars=positive.excess_dollars.sum(),
            gross_negative_dollars=g.loc[g.excess_dollars.lt(0),'excess_dollars'].sum(),
            top_five_share_of_positive=positive.excess_dollars.head(5).sum()/positive.excess_dollars.sum() if len(positive) else None,
            top_five='; '.join(positive.label.head(5))))
    pd.DataFrame(summary).to_csv(REPORT/'selected_concentration.csv',index=False)
    pairs.groupby(['model','variant','period','industry']).excess_dollars.sum().reset_index().to_csv(REPORT/'selected_industry_contributions.csv',index=False)
    path=Path('reports/contract_entry_gates.md')
    common={d['pair'] for d in definitions if d['group']=='broad'}-failed
    counts=[]
    for model in MODELS:
        old=baseline[baseline.pair.isin(common)&baseline.model.eq(model)]
        variant=selected[model]
        new=old if variant=='baseline' else status[status.pair.isin(common)&status.model.eq(model)&status.variant.eq(variant)]
        counts.append(f"{model}: {int(old.entries.sum()):,} → {int(new.entries.sum()):,} entries ({variant})")
    summary_text='\nTrading activity on the common broad roster: '+ '; '.join(counts)+'.\n'
    summary_text+='\nThe development-selected RL filter improves later-period Sharpe but still loses versus cash. Much of the improvement is avoided trading; it is not evidence of positive net alpha. All tested NN variants also have negative development Sharpe, so the selected NN is a comparative benchmark, not a supported trading recommendation.\n'
    addition='\n[Selected-policy pair contributions](contract_entry_gates/selected_pair_contributions.csv) · [Profit concentration](contract_entry_gates/selected_concentration.csv) · [Industry contributions](contract_entry_gates/selected_industry_contributions.csv)\n'
    if summary_text not in path.read_text():path.write_text(path.read_text()+summary_text)
    if addition not in path.read_text():path.write_text(path.read_text()+addition)


if __name__=='__main__': main()
