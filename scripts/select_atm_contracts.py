"""Stream daily eligible ATM/~30D calls and fit their Black–Scholes IV/Greeks."""
import argparse
import json
from pathlib import Path
import resource
import numpy as np
import pandas as pd
from scripts.audit_contract_history import iter_contract_batches
from scripts.run_contract_pilot import normalize_quotes
from scripts.black_scholes import fit_quotes
from scripts.contract_engine import Settings
from dataclasses import asdict
from scripts.download_pair_contract_history import file_hash


SORT = ['secid','date','tenor_distance','distance','open_interest','optionid']
ASC = [True,True,True,True,False,True]


def load_calendar(root):
    dates = set()
    for path in sorted((root/'underlying').glob('year=*.parquet')):
        dates.update(pd.to_datetime(pd.read_parquet(path, columns=['date']).date.unique()))
    return pd.DatetimeIndex(sorted(dates))


def select_candidates(q, stocks, cfg):
    q = q.merge(stocks[['secid','date','close','entry_date']], on=['secid','date'], validate='many_to_one')
    q['entry_dte'] = (q.expiry-q.entry_date).dt.days
    q['distance'] = abs(q.strike/q.close-1)
    q['tenor_distance'] = abs(q.entry_dte-cfg.target_dte)
    mid = (q.bid+q.ask)/2
    eligible = (q.standard & q.multiplier.eq(100) & q.cp_flag.eq('C') &
                q.bid.gt(0) & q.ask.ge(q.bid) & q.strike.gt(0) & q.close.gt(0) &
                np.isfinite(q[['bid','ask','strike','multiplier','cfadj','delta','vega']]).all(axis=1) &
                q.delta.between(0,1) & q.vega.gt(0) & q.open_interest.ge(cfg.min_open_interest) &
                ((q.ask-q.bid)/mid).le(cfg.max_relative_spread) &
                q.entry_dte.between(cfg.min_dte,cfg.max_dte) & q.distance.le(cfg.max_moneyness_distance))
    return q[eligible].sort_values(SORT,ascending=ASC).drop_duplicates(['secid','date']).drop(columns='close')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('data/option_contracts'))
    p.add_argument('--output',type=Path,default=Path('data/atm_30d'))
    p.add_argument('--years',type=int,nargs='+')
    args = p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    jobs = [r for r in json.loads((args.root/'requests.json').read_text()) if r['kind']=='quotes']
    if args.years: jobs = [r for r in jobs if r['year'] in args.years]
    zero = pd.read_parquet(args.root/'zero_curve.parquet')
    dividends = pd.read_parquet(args.root/'distributions.parquet')
    # Stock sessions, not rate dates: curves can exist on exceptional market closures.
    calendar = load_calendar(args.root)
    following = pd.Series(calendar[1:].to_numpy(),index=calendar[:-1])
    summaries=[]; stocks=None; year=None
    for i,job in enumerate(jobs,1):
        if year != job['year']:
            year = job['year']
            stocks = pd.read_parquet(args.root/'underlying'/f'year={year}.parquet')
            stocks['date'] = pd.to_datetime(stocks.date)
            stocks['entry_date'] = stocks.date.map(following)
        subset = stocks[stocks.secid.isin(job['secids'])]
        best = None
        for raw in iter_contract_batches(args.root,job['secids'],job['start'],job['end']):
            candidates = select_candidates(normalize_quotes(raw),subset,Settings())
            best = candidates if best is None else pd.concat([best,candidates],ignore_index=True)
            best = best.sort_values(SORT,ascending=ASC).drop_duplicates(['secid','date'])
        if best is None or best.empty:
            summaries.append(dict(source=job['path'],selected=0,fitted=0));continue
        fitted = fit_quotes(best,subset,zero,dividends)
        target = args.output/str(year)/Path(job['path']).name
        target.parent.mkdir(exist_ok=True)
        fitted.to_parquet(target,index=False)
        summaries.append(dict(source=job['path'],output=str(target),selected=len(fitted),
                              fitted=int(fitted.bs_status.eq('ok').sum()),
                              max_abs_price_error=float(fitted.bs_error.abs().max()),
                              median_entry_dte=float(fitted.entry_dte.median()),
                              statuses=fitted.bs_status.value_counts().to_dict()))
        (args.output/'progress.json').write_text(json.dumps(dict(completed=i,total=len(jobs),partitions=summaries),indent=2))
        print(f'{i}/{len(jobs)} {year}: {len(fitted)} selected, {fitted.bs_status.eq("ok").sum()} fitted',flush=True)
    summary=dict(selected=sum(x['selected'] for x in summaries),fitted=sum(x['fitted'] for x in summaries),
                 peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,partitions=summaries)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    write_manifest(args.root, args.output, jobs)


def write_manifest(root, output, jobs):
    """Record exact inputs and conventions without loading large files."""
    files = [root/'requests.json', root/'zero_curve.parquet', root/'distributions.parquet']
    files += [root/'underlying'/f'year={year}.parquet' for year in sorted({r['year'] for r in jobs})]
    manifest = dict(selection=asdict(Settings()),
                    selection_timing='prior session quotes, DTE at next session entry',
                    model='European call, spot less PV of declared ordinary USD dividends',
                    year_basis=365, iv_units='decimal', vega_units='dollars/share/volatility point',
                    max_iv=10, source_files=[dict(path=str(f),sha256=file_hash(f)) for f in files],
                    code_files=[dict(path=str(f),sha256=file_hash(f)) for f in
                                [Path(__file__),Path('scripts/black_scholes.py'),Path('scripts/contract_engine.py')]],
                    quote_manifests=[json.loads((root/r['path']).with_suffix('.json').read_text()) for r in jobs])
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__': main()
