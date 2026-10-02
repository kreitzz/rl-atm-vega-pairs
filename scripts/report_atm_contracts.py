"""Audit/refit the compact selected dataset without rereading full chains."""
import argparse
from collections import Counter
import json
from pathlib import Path
import pandas as pd
from scripts.black_scholes import fit_quotes
from scripts.select_atm_contracts import write_manifest, load_calendar


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('data/option_contracts'))
    p.add_argument('--selected',type=Path,default=Path('data/atm_30d'))
    p.add_argument('--refit',action='store_true')
    args=p.parse_args()
    summary=json.loads((args.selected/'summary.json').read_text())
    jobs=[r for r in json.loads((args.root/'requests.json').read_text()) if r['kind']=='quotes']
    sources={r['source'] for r in summary['partitions']}
    jobs=[r for r in jobs if r['path'] in sources]
    zero=pd.read_parquet(args.root/'zero_curve.parquet')
    divs=pd.read_parquet(args.root/'distributions.parquet')
    calendar=load_calendar(args.root)
    following=pd.Series(calendar[1:].to_numpy(), index=calendar[:-1])
    year=None; stocks=None; rows=[]; ids=set(); statuses=Counter(); seen=set()
    for part in summary['partitions']:
        if not part['selected']:continue
        path=Path(part['output']); q=pd.read_parquet(path)
        if args.refit:
            current=int(path.parent.name)
            if current != year:
                year=current
                stocks=pd.read_parquet(args.root/'underlying'/f'year={year}.parquet')
                stocks['date']=pd.to_datetime(stocks.date)
                stocks['entry_date']=stocks.date.map(following)
            q=fit_quotes(q.drop(columns='close'),stocks,zero,divs)
            q.to_parquet(path,index=False)
        assert q.entry_date.eq(q.date.map(following)).all(), 'Incorrect execution calendar'
        assert q.entry_dte.eq((q.expiry-q.entry_date).dt.days).all()
        keys=set(zip(q.secid,q.date))
        assert len(keys)==len(q) and not keys.intersection(seen),'Duplicate daily selection'
        seen.update(keys);ids.update(q.secid.astype(int));statuses.update(q.bs_status)
        assert q.entry_dte.between(21,45).all() and q.distance.le(.10).all()
        good=q[q.bs_status.eq('ok')]
        assert good.bs_iv.between(0,10).all() and good.bs_delta.between(0,1).all()
        assert good.bs_vega.ge(0).all() and good.bs_gamma.ge(0).all()
        assert good.empty or good.bs_error.abs().max()<1e-7
        part.update(selected=len(q),fitted=len(good),statuses=q.bs_status.value_counts().to_dict(),
                    max_abs_price_error=float(good.bs_error.abs().max()))
        for y,g in q.groupby(q.date.dt.year):
            rows.append(dict(year=int(y),selected=len(g),fitted=int(g.bs_status.eq('ok').sum()),
                             dte_sum=float(g.entry_dte.sum()),distance_sum=float(g.distance.sum()),
                             start=str(g.date.min().date()),end=str(g.date.max().date())))
    summary.update(selected=sum(x['selected'] for x in summary['partitions']),
                   fitted=sum(x['fitted'] for x in summary['partitions']),underlyings=len(ids),statuses=dict(statuses))
    (args.selected/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    write_manifest(args.root,args.selected,jobs)
    yearly=pd.DataFrame(rows).groupby('year').agg(dict(selected='sum',fitted='sum',dte_sum='sum',distance_sum='sum',start='min',end='max'))
    yearly['fit_rate']=yearly.fitted/yearly.selected
    yearly['mean_entry_dte']=yearly.pop('dte_sum')/yearly.selected
    yearly['mean_atm_distance']=yearly.pop('distance_sum')/yearly.selected
    dest=Path('reports/atm_contracts');dest.mkdir(exist_ok=True)
    yearly.to_csv(dest/'coverage.csv')
    (dest/'summary.json').write_text(json.dumps({k:v for k,v in summary.items() if k!='partitions'},indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='partitions'},indent=2))


if __name__=='__main__': main()
