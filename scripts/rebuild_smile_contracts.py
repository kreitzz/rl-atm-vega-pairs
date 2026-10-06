"""Recover all three smile silos from raw chains under one selection protocol.

One worker, streamed partitions, resumable selection/history/asset phases.
Frozen studies are read-only. Only existing cached raw chains are required.
"""
from dataclasses import replace
from functools import lru_cache
import argparse
import gc
import json
from pathlib import Path
import resource
import time
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scripts.smile_contract_selection import select, reduce_choices
from scripts.run_contract_pilot import normalize_quotes
from scripts.option_flex.black_scholes import fit_quotes
from scripts.prepare_curve_contracts import HISTORY_COLUMNS
from scripts.curve_trading.config import config as old_config
from scripts.curve_trading.unit_paths import UnitPaths
from scripts.run_broad_contract_replay import ROOT, Market
from scripts.select_atm_contracts import load_calendar
from scripts.research_opposite_wing_signals import FX
from scripts.run_atm65_ablation import write_json
from scripts.run_contract_walkforward import digest

DATA = Path('data/recovered_smile')
OUT = Path('artifacts/contracts/recovered_smile_data')
REPORT = Path('reports/recovered_smile_data')
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def config():
    return replace(old_config(), min_open_interest=10, min_dte=21, max_dte=45)


def progress(stage, **values):
    write_json(OUT / 'progress.json', dict(stage=stage, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, **values))


def read(path, columns=None, filters=None):
    return pq.read_table(path, columns=columns, filters=filters, use_threads=False).to_pandas()


@lru_cache(maxsize=1)
def year_stocks(year):
    stocks = read(ROOT / 'underlying' / f'year={year}.parquet', ['date','secid','close'])
    stocks['date'] = pd.to_datetime(stocks.date)
    return stocks


def repaired_distributions():
    # Preserve the earlier studies' three declared-cash-event repairs.
    d = read(ROOT / 'distributions.parquet')
    for col in ['ex_date','declare_date']:
        d[col] = pd.to_datetime(d[col])
    exact = (d.cancel_flag.eq('0') & d.liquid_flag.eq('0') & d.approx_flag.eq('0') & d.adj_factor.eq(1) & d.amount.ge(0))
    supplemental = exact & d.secid.isin([108969,105003]) & d.distr_type.eq('5') & d.currency.eq('USD')
    d.loc[supplemental,'distr_type'] = '1'
    foreign = exact & d.secid.eq(127149) & d.distr_type.isin(['1','5']) & d.currency.eq('CAD') & d.declare_date.ge('2010-01-01')
    fx = pd.read_csv(FX)
    fx['observation_date'] = pd.to_datetime(fx.observation_date)
    fx['DEXCAUS'] = pd.to_numeric(fx.DEXCAUS, errors='coerce')
    fx = fx.dropna().sort_values('observation_date')
    events = d.loc[foreign, ['declare_date']].sort_values('declare_date')
    if len(events):
        rates = pd.merge_asof(events, fx, left_on='declare_date', right_on='observation_date', direction='backward', allow_exact_matches=False)
        assert rates.DEXCAUS.gt(0).all() and rates.observation_date.lt(rates.declare_date).all()
        d.loc[events.index,'amount'] = d.loc[events.index,'amount'].to_numpy() / rates.DEXCAUS.to_numpy()
    d.loc[foreign,'currency'], d.loc[foreign,'distr_type'] = 'USD','1'
    return d


def jobs(cp):
    root = ROOT if cp == 'C' else Path('data/curve_trading/puts_raw')
    requested = json.loads((root / 'requests.json').read_text())
    return root, sorted([x for x in requested if x['kind'] == 'quotes'], key=lambda x:(x['year'],x['path']))


def select_partitions(cp, cal):
    root, requested = jobs(cp)
    zero, divs = read(ROOT / 'zero_curve.parquet'), repaired_distributions()
    following = pd.Series(cal[1:].to_numpy(), index=cal[:-1])
    roster = json.loads(Path('artifacts/contracts/tail_put25_pooled/inputs.json').read_text())['definitions']
    sids = {s for d in roster for s in d['secids']}
    logs = []
    for n, job in enumerate(requested,1):
        source = root / job['path']
        stamp = DATA / 'metadata' / 'selection' / cp / str(job['year']) / (source.stem+'.json')
        if stamp.exists():
            record = json.loads(stamp.read_text())
            for path, value in record['files'].items():
                assert digest(path) == value, path
            logs.append(record)
            continue
        start = time.monotonic()
        best = None
        stats = dict(raw_rows=0, fitted_rows=0)
        stocks = year_stocks(job['year'])
        for batch in pq.ParquetFile(source).iter_batches(batch_size=32768,use_threads=False):
            q = normalize_quotes(batch.to_pandas())
            stats['raw_rows'] += len(q)
            entry_dte = (q.expiry-q.date.map(following)).dt.days
            width = (q.ask-q.bid)/((q.ask+q.bid)/2)
            good = (q.secid.isin(sids) & q.date.isin(cal[:-1]) & entry_dte.between(21,45) & q.standard & q.multiplier.eq(100) & q.bid.gt(0) & q.ask.ge(q.bid) & q.open_interest.ge(10) & width.le(.25))
            q = q.loc[good]
            if q.empty:
                continue
            if cp == 'C':
                spot = q[['date','secid','strike']].merge(stocks,on=['date','secid'],how='left',validate='many_to_one')
                q = q.loc[(spot.strike/spot.close-1).abs().le(.1).to_numpy()]
            if q.empty:
                continue
            q = fit_quotes(q, stocks, zero, divs)
            stats['fitted_rows'] += len(q)
            selected = select(q, following, cp)
            if len(selected):
                best = selected if best is None else reduce_choices(pd.concat([best,selected],ignore_index=True))
            if best is not None and len(best)>30000:
                raise MemoryError('Bounded selection partition exceeded 30,000 rows')
        files = {}
        if best is not None:
            for sid, frame in best.groupby('secid'):
                path = DATA / 'selection' / cp / str(job['year']) / f'{sid}.parquet'
                path.parent.mkdir(parents=True,exist_ok=True)
                frame.to_parquet(path,index=False)
                files[str(path)] = digest(path)
        record = dict(cp=cp,year=job['year'],source=str(source),source_metadata=json.loads(source.with_suffix('.json').read_text()), files=files, selected_rows=0 if best is None else len(best), elapsed_seconds=time.monotonic()-start, **stats)
        stamp.parent.mkdir(parents=True,exist_ok=True)
        write_json(stamp,record)
        logs.append(record)
        progress('selection',cp=cp,partition=n,total=len(requested),year=job['year'])
        print(f'Selection {cp} {n}/{len(requested)} {job["year"]}: {record["selected_rows"]:,} rows, {record["elapsed_seconds"]:.1f}s',flush=True)
        del best
        gc.collect()
    write_json(DATA / f'{cp}_selection_manifest.json',logs)


def extract_history(cp):
    root, requested = jobs(cp)
    zero, divs = read(ROOT / 'zero_curve.parquet'), repaired_distributions()
    logs = []
    for n,job in enumerate(requested,1):
        source = root / job['path']
        stamp = DATA / 'metadata' / 'history' / cp / str(job['year']) / (source.stem+'.json')
        if stamp.exists():
            logs.append(json.loads(stamp.read_text()))
            continue
        wanted = set()
        for year in [job['year']-1,job['year']]:
            for sid in job['secids']:
                path = DATA / 'selection' / cp / str(year) / f'{sid}.parquet'
                if path.exists():
                    wanted.update(read(path,['optionid']).optionid)
        writers, paths, count = {}, {}, 0
        start = time.monotonic()
        try:
            if wanted:
                for batch in pq.ParquetFile(source).iter_batches(batch_size=32768,use_threads=False):
                    raw = batch.to_pandas()
                    raw = raw.loc[raw.optionid.isin(wanted)]
                    if raw.empty:
                        continue
                    q = fit_quotes(normalize_quotes(raw),year_stocks(job['year']),zero,divs)
                    q['delta'], q['vega'] = q.bs_delta, q.bs_vega
                    q = q[HISTORY_COLUMNS]
                    for sid, frame in q.groupby('secid'):
                        sid = int(sid)
                        path = DATA / 'history' / cp / str(job['year']) / f'{sid}.parquet'
                        path.parent.mkdir(parents=True,exist_ok=True)
                        table = pa.Table.from_pandas(frame,preserve_index=False)
                        if sid not in writers:
                            writers[sid] = pq.ParquetWriter(path.with_suffix('.partial'),table.schema,compression='zstd')
                            paths[sid] = path
                        writers[sid].write_table(table)
                        count += len(frame)
        finally:
            for writer in writers.values():
                writer.close()
        files = {}
        for path in paths.values():
            path.with_suffix('.partial').replace(path)
            files[str(path)] = digest(path)
        record = dict(cp=cp,year=job['year'],source=str(source),files=files,rows=count,elapsed_seconds=time.monotonic()-start)
        stamp.parent.mkdir(parents=True,exist_ok=True)
        write_json(stamp,record)
        logs.append(record)
        progress('history',cp=cp,partition=n,total=len(requested),year=job['year'])
        print(f'History {cp} {n}/{len(requested)} {job["year"]}: {count:,} rows, {record["elapsed_seconds"]:.1f}s',flush=True)
        gc.collect()
    write_json(DATA / f'{cp}_history_manifest.json',logs)


def assets(cal):
    market = Market(cal)
    distributions = repaired_distributions()
    divs = distributions.loc[distributions.cancel_flag.eq('0') & distributions.distr_type.eq('1') & distributions.currency.eq('USD') & distributions.adj_factor.eq(1) & distributions.approx_flag.eq('0') & distributions.liquid_flag.eq('0')]
    defs = json.loads(Path('artifacts/contracts/tail_put25_pooled/inputs.json').read_text())['definitions']
    sids = sorted({s for d in defs for s in d['secids']})
    folder = OUT / 'market/assets'
    folder.mkdir(parents=True,exist_ok=True)
    records = []
    for n,sid in enumerate(sids,1):
        stamp = folder / f'{sid}.json'
        if stamp.exists():
            records.append(json.loads(stamp.read_text()))
            continue
        pieces, selections = [], []
        for cp in ['C','P']:
            for year in cal.year.unique():
                path = DATA / 'selection' / cp / str(year) / f'{sid}.parquet'
                if path.exists():
                    selections.append(read(path))
                path = DATA / 'history' / cp / str(year) / f'{sid}.parquet'
                if path.exists():
                    pieces.append(read(path))
        sel = pd.concat(selections,ignore_index=True) if selections else pd.DataFrame(columns=['date','expiry','secid','tenor','cp_flag','bucket','optionid','bs_iv','bs_vega'])
        q = pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame(columns=HISTORY_COLUMNS)
        assert not q.duplicated(['date','optionid']).any()
        stock = market.stocks.loc[market.stocks.secid.eq(sid)].set_index('date').reindex(cal)
        if sid == 106899:
            stock.loc[stock.index>='2019-05-31','cfadj'] /= 1.02
        stock['secid'], stock['rate'] = sid, market.rates
        stock.index.name = 'date'
        events = divs.loc[divs.secid.eq(sid)]
        if len(q) and len(sel):
            lookup = q.set_index(['date','optionid'])
            keys = pd.MultiIndex.from_frame(sel[['date','optionid']])
            for name, source in [('bs_iv','bs_iv'),('bs_vega','vega'),('bs_delta','delta')]:
                sel[name] = lookup[source].reindex(keys).to_numpy()
        for kind,frame in [('quotes',q),('stocks',stock.reset_index()),('selection',sel),('dividends',events)]:
            frame.to_parquet(folder / f'{sid}_{kind}.parquet',index=False)
        if len(q) and len(sel):
            unit = UnitPaths(q,stock.reset_index(),sel,cal,cal,events,replace(config(),option_commission=.65))
            joblib.dump(unit,folder / f'{sid}_unit.joblib',compress=0)
        record = dict(secid=sid,quotes=len(q),selections=len(sel))
        write_json(stamp,record)
        records.append(record)
        progress('assets',asset=n,total=len(sids),secid=sid)
        if n%10 == 0:
            print(f'Recovered assets {n}/{len(sids)}',flush=True)
        del q,sel,pieces,selections
        if 'unit' in locals():
            del unit
        gc.collect()
    pd.DataFrame(records).to_csv(REPORT / 'assets.csv',index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase',choices=['selection','history','assets'],required=True)
    p.add_argument('--cp',choices=['C','P'])
    args = p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    REPORT.mkdir(parents=True,exist_ok=True)
    cal = load_calendar(ROOT)
    cal = cal[(cal>='2010-01-01') & (cal<='2025-08-29')]
    signature = dict(code={str(path):digest(path) for path in [Path(__file__),Path('scripts/smile_contract_selection.py'),Path('scripts/option_flex/black_scholes.py')]}, min_open_interest=10, entry_dte=[21,45], target_dte=30, expiry_chosen_independently_by_bucket=True, delta_tolerance=.075, atm='closest strike within10% spot', evaluation_start='2018-01-01', signal_end='2025-06-30', runoff_end='2025-08-29')
    stamp = DATA / 'inputs.json'
    DATA.mkdir(parents=True,exist_ok=True)
    if stamp.exists():
        assert json.loads(stamp.read_text()) == signature, 'Preparation source changed after checkpointing'
    else:
        write_json(stamp,signature)
    if args.phase == 'selection':
        select_partitions(args.cp,cal)
    elif args.phase == 'history':
        extract_history(args.cp)
    else:
        assets(cal)
    progress(args.phase+'_complete',cp=args.cp)


if __name__ == '__main__':
    main()
