"""Frozen P25 signal and sizing transfer, without reconstructing its history.

This experiment transfers actual decisions, not just model architecture. It
does not train a wing-native model or promise equal profitability by wing.
"""
from dataclasses import replace
import argparse
import gc
import json
from pathlib import Path
import resource
import shutil
import time
import unittest

import numpy as np
import pandas as pd

from scripts.p25_signal_transfer import WINGS, choose_existing_first, transfer_rows
from scripts import uniform_smile_data as cached
from scripts import uniform_smile_engine as engine
from scripts.uniform_smile_risk import book_risk
from scripts.curve_trading.config import config
from scripts.run_atm65_ablation import write_json
from scripts.run_contract_walkforward import digest
from scripts.run_uniform_smile import verify, UNIFORM_CORPORATE_EVENTS
from scripts.run_aligned_contract_learning import mark_book

OUT = Path('artifacts/contracts/p25_signal_transfer')
REPORT = Path('reports/p25_signal_transfer')
FROZEN = Path('artifacts/contracts/same_structure_calendar/replays')
EXISTING = Path('artifacts/contracts/uniform_smile/market')
ADDITIONAL = Path('artifacts/contracts/recovered_smile_data/market')
FIELDS = ['date', 'pair', 'secid_a', 'secid_b', 'optionid_a', 'optionid_b', 'direction', 'base_level', 'wing']
PROTOCOL = (
    'Preserve all 608 frozen P25 qualified signal dates/pairs, directions and '
    'monthly learned floor-size decisions from same_structure_calendar. '
    'Transfer exactly the same decision stream to ATM calls and P75 puts; '
    'there is no second wing-native rank or profitability veto. Original '
    'selected contracts always take precedence. Missing wing choices can '
    'use already cached alternative selections only when signal OI>=100; '
    'fallback never replaces an existing contract. This fallback cache is '
    'not an exhaustive OI100 expiry search. Missing IDs reach execution '
    'explicitly and cannot disappear through future-path filtering. '
    'P25 selections are unchanged. Original 7-55D replay eligibility, '
    'costs (.03 commission, mid+.1 volpoint,1bp hedge), financing, dividends, '
    'vega matching, daily hedging and risk budgets are unchanged. '
    'Compare fixed50,fixed100,frozen RL50/75/100 at identical proposals, '
    'with separate wing and combined shared-book accounting. Additional '
    'existing-only floor replays isolate the fill-missing effect. '
    'Reproduce standalone frozen P25 fixed100 and floor daily NAV, trades '
    'and decisions before accepting other wing results. No retraining. '
    'This is a cross-wing transfer of the P25 model decisions, not '
    'independently trained wing RL. Previously inspected history and '
    'retrospective formation conditioning, no fresh holdout. One worker, '
    '50% of one core,one BLAS/Arrow thread,3GiB RSS stop.'
)


class Market(cached.ReplayMarket):
    def rate(self, date):
        if self.reported_year != date.year:
            self.reported_year = date.year
            print(f'{self.variant}: {date.date()} ({time.monotonic()-self.started:.1f}s)', flush=True)
        return float(self.rates.loc[date])


def track(sources, path):
    path = Path(path)
    if str(path) not in sources:
        sources[str(path)] = digest(path)
    return path


def progress(stage, **details):
    write_json(OUT/'progress.json', dict(stage=stage,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, **details))


def prepare(sources):
    anchor_path = track(sources, FROZEN/'floor_P25/scored_allocations.parquet')
    anchor = pd.read_parquet(anchor_path, columns=FIELDS).reset_index(drop=True)
    fixed = pd.read_parquet(track(sources, FROZEN/'fixed_100_P25/scored_allocations.parquet'), columns=FIELDS)
    pd.testing.assert_frame_equal(anchor.drop(columns='base_level'), fixed.drop(columns='base_level'))
    assert len(anchor) == 608 and anchor.base_level.isin([2, 3, 4]).all()
    maps, coverage = {}, []
    sids = sorted(set(anchor.secid_a) | set(anchor.secid_b))
    for n, sid in enumerate(sids, 1):
        old = pd.read_parquet(track(sources, EXISTING/'assets'/f'{sid}_selection.parquet'))
        new = pd.read_parquet(track(sources, ADDITIONAL/'assets'/f'{sid}_selection.parquet'))
        for wing in ['ATM', 'P75']:
            maps[(sid, wing)] = choose_existing_first(old, new, wing)
        if n % 20 == 0:
            print(f'Contract map {n}/{len(sids)}', flush=True)
        del old, new
    rows = []
    for wing in WINGS:
        mapped = transfer_rows(anchor, maps, wing)
        existing = transfer_rows(anchor, maps, wing, existing_only=True)
        mapped.to_parquet(OUT/f'{wing}_rows.parquet', index=False)
        existing.to_parquet(OUT/f'{wing}_existing_rows.parquet', index=False)
        missing = mapped[['optionid_a', 'optionid_b']].isna().any(axis=1)
        before_missing = existing[['optionid_a', 'optionid_b']].isna().any(axis=1)
        coverage.append(dict(wing=wing, frozen_proposals=len(mapped),
            original_contract_pair_available=int((~before_missing).sum()),
            filled_contract_pair_available=int((~missing).sum()),
            newly_covered_proposals=int((before_missing & ~missing).sum()),
            still_missing_proposals=int(missing.sum())))
        rows.append(mapped)
    pd.DataFrame(coverage).to_csv(REPORT/'coverage.csv', index=False)
    rows = pd.concat(rows, ignore_index=True)
    # Track selected IDs by source. Existing histories stay byte-equivalent;
    # alternative histories are only included for fallback IDs.
    needed = {'existing': {}, 'additional': {}}
    for side in ['a', 'b']:
        for sid, part in rows.groupby('secid_'+side):
            for source in needed:
                accepted = part['contract_source_'+side].isin([source, 'frozen_P25'] if source=='existing' else [source])
                needed[source].setdefault(int(sid), set()).update(part.loc[accepted, 'optionid_'+side].dropna().astype(int))
    folder = OUT/'market'
    folder.mkdir(exist_ok=True)
    for kind in ['stocks', 'dividends', 'rates']:
        source = track(sources, EXISTING/f'{kind}.parquet')
        shutil.copyfile(source, folder/f'{kind}.parquet')
    cal = pd.read_parquet(folder/'rates.parquet').index
    assert cal.max() == pd.Timestamp('2025-08-29')
    for year in sorted(set(cal[cal>='2018-01-01'].year)):
        target = folder/f'{year}_quotes.parquet'
        if target.exists():
            continue
        pieces = []
        for source, root in [('existing', EXISTING), ('additional', ADDITIONAL)]:
            for sid, ids in needed[source].items():
                if not ids:
                    continue
                path = track(sources, root/'assets'/f'{sid}_quotes.parquet')
                q = pd.read_parquet(path, filters=[('date','>=',pd.Timestamp(year,1,1)),
                    ('date','<',pd.Timestamp(year+1,1,1)), ('optionid','in',sorted(ids))])
                if len(q):
                    pieces.append(q)
        quotes = pd.concat(pieces, ignore_index=True).drop_duplicates(['date', 'optionid'], keep='first')
        quotes.to_parquet(target, index=False)
        print(f'Transfer quote cache {year}: {len(quotes):,}', flush=True)
        del quotes, pieces
        gc.collect()
        write_json(OUT/'sources.json', sources)
        progress('prepare', year=int(year))
    print(pd.DataFrame(coverage).to_string(index=False), flush=True)


def allocations(variant, wing=None, existing_only=False):
    pieces=[]
    for target in ([wing] if wing else WINGS):
        suffix='_existing_rows' if existing_only else '_rows'
        rows=pd.read_parquet(OUT/f'{target}{suffix}.parquet')
        if variant in ['fixed_50', 'fixed_100']:
            rows['base_level']=2 if variant=='fixed_50' else 4
        elif variant!='floor':
            raise ValueError(variant)
        pieces.append(rows)
    return pd.concat(pieces,ignore_index=True).sort_values(['date','pair','wing'],kind='stable').reset_index(drop=True)


def assert_reference(name, nav, trades, decisions, sources):
    reference=FROZEN/name
    old_nav=pd.read_parquet(track(sources,reference/'daily_nav.parquet'))
    pd.testing.assert_frame_equal(nav,old_nav,check_exact=False,rtol=1e-12,atol=1e-6)
    for current, filename, dates in [(trades,'trades.csv',['date']),
                                     (decisions,'decisions.csv',['signal_date','entry_date'])]:
        old=pd.read_csv(track(sources,reference/filename),parse_dates=dates)
        pd.testing.assert_frame_equal(current.reset_index(drop=True),old.reset_index(drop=True),
            check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-7)
    print(f'Frozen reference reproduced: {name}',flush=True)


def replay(sources):
    cached.OUT=OUT
    market=Market()
    cal=market.rates.index[market.rates.index>='2018-01-01']
    # Standalone P25 controls must pass before any transfer is accepted.
    arms=[('fixed_100','P25',False),('floor','P25',False)]
    arms += [(variant,wing,False) for variant in ['fixed_50','fixed_100','floor']
             for wing in [None,'ATM','P25','P75'] if (variant,wing,False) not in arms]
    arms += [('floor',wing,True) for wing in [None,'ATM','P75']]
    for variant,wing,existing_only in arms:
        name=variant+('_'+wing if wing else '')+('_existing' if existing_only else '')
        folder=OUT/'replays'/name
        folder.mkdir(parents=True,exist_ok=True)
        if (folder/'status.json').exists():
            continue
        rows=allocations(variant,wing,existing_only)
        rows.to_parquet(folder/'scored_allocations.parquet',index=False)
        multiplier=3 if wing is None else 1
        cfg=replace(config(),option_commission=.03,capital=1_000_000*multiplier*345,
                    max_gross_vega=20_000*multiplier*345)
        assert cfg.min_open_interest==100 and cfg.min_dte==7 and cfg.max_dte==55
        market.variant,market.started,market.reported_year=name,time.monotonic(),None
        print(f'Transfer replay {name}: {len(rows):,} frozen decisions',flush=True)
        saved=engine.book_risk
        try:
            engine.book_risk=book_risk
            nav,trades,decisions,_=engine.simulate(market,rows,cal,cfg,1_000_000*multiplier,
                wing_capital=1_000_000,corporate_events=UNIFORM_CORPORATE_EVENTS)
        finally:
            engine.book_risk=saved
        mark_book(nav,market.rates,cfg.capital)
        audit=verify(nav,trades,decisions,rows,cfg)
        if name in ['fixed_100_P25','floor_P25']:
            assert_reference(name,nav,trades,decisions,sources)
        nav.to_parquet(folder/'daily_nav.parquet')
        trades.to_csv(folder/'trades.csv',index=False)
        decisions.to_csv(folder/'decisions.csv',index=False)
        write_json(folder/'status.json',dict(status='complete',variant=variant,wing=wing,
            existing_only=existing_only,proposals=len(rows),net_excess_pnl=float(nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1]),**audit))
        progress('replay',arm=name,**audit)
        write_json(OUT/'sources.json',sources)
        print(f'{name}: Sharpe {audit["annualized_excess_sharpe"]:.6f}, {audit["entries"]} entries',flush=True)


def summarize(sources):
    results,annual=[],[]
    base=allocations('floor')
    for variant in ['fixed_50','fixed_100']:
        pd.testing.assert_frame_equal(base.drop(columns='base_level'),allocations(variant).drop(columns='base_level'))
    for folder in sorted((OUT/'replays').iterdir()):
        record=json.loads((folder/'status.json').read_text())
        assert record['status']=='complete'
        results.append(dict(model=folder.name,**record))
        nav=pd.read_parquet(folder/'daily_nav.parquet')
        for year,g in nav.groupby(nav.index.year):
            r=g.excess_return.iloc[1:] if year==2018 else g.excess_return
            annual.append(dict(model=folder.name,year=int(year),annualized_excess_sharpe=float(np.sqrt(252)*r.mean()/r.std()) if r.std()>1e-12 else None))
    result=pd.DataFrame(results)
    assert len(result)==15
    result.to_csv(REPORT/'comparison.csv',index=False)
    columns=['model','annualized_excess_sharpe','entries','net_excess_pnl']
    result[columns].to_csv(REPORT/'results.csv',index=False)
    pd.DataFrame(annual).pivot(index='year',columns='model',values='annualized_excess_sharpe').to_csv(REPORT/'annual_comparison.csv')
    write_json(REPORT/'audit.json',dict(status='complete',frozen_p25_standalone_nav_trades_decisions_reproduced=True,
        same_signal_dates_directions_sizes_across_wings=True,no_wing_specific_signal_veto=True,
        existing_contracts_never_replaced=True,unchanged_oi100_and_7_55d=True,
        identical_sizing_control_proposals=True,all_ledgers_verified=True,replays=15,
        no_retraining=True,no_fresh_holdout=True,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    for path,value in sources.items():
        assert digest(path)==value,path
    (REPORT/'findings.txt').write_text(PROTOCOL+'\n\n'+result[columns].to_string(index=False)+'\n')
    write_json(REPORT/'output_hashes.json',{str(p):digest(p) for root in [OUT,REPORT] for p in root.rglob('*') if p.is_file() and p.name!='output_hashes.json'})
    print(result[columns].to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=['test','prepare','replay','all'],default='all')
    args=p.parse_args()
    if args.phase=='test':
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests',pattern='test_p25_signal_transfer.py'))
        if not result.wasSuccessful():
            raise SystemExit(1)
        return
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    path=OUT/'sources.json'
    sources=json.loads(path.read_text()) if path.exists() else {}
    for code in [Path(__file__),Path('scripts/p25_signal_transfer.py'),Path('scripts/uniform_smile_engine.py'),Path('scripts/uniform_smile_risk.py')]:
        if str(code) in sources:
            assert digest(code)==sources[str(code)],code
        track(sources,code)
    write_json(OUT/'inputs.json',dict(protocol=PROTOCOL,source=str(FROZEN/'floor_P25'),no_retraining=True))
    if args.phase in ['prepare','all']:
        prepare(sources)
        write_json(OUT/'sources.json',sources)
    if args.phase in ['replay','all']:
        replay(sources)
        summarize(sources)


if __name__=='__main__':
    main()
