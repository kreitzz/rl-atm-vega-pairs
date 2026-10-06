"""Same-structure RL comparison after recovering smile contract coverage."""
from dataclasses import replace
import argparse
import gc
import json
from pathlib import Path
import resource
import time
import joblib
import numpy as np
import pandas as pd
from scripts import rebuild_smile_contracts as rebuilt
from scripts import uniform_smile_data as data
from scripts import run_same_structure_smile as common
from scripts import uniform_smile_engine as engine
from scripts.uniform_smile_risk import book_risk
from scripts.run_same_structure_calendar import formation_dates, filter_calendar
from scripts.run_uniform_smile import verify, UNIFORM_CORPORATE_EVENTS
from scripts.run_atm65_ablation import write_json, read_actions
from scripts.run_contract_walkforward import digest
from scripts.run_aligned_contract_learning import mark_book
from scripts.select_atm_contracts import load_calendar
from scripts.run_broad_contract_replay import ROOT, Market
from scripts.contract_learning_labels import beta_screen
from scripts.run_aligned_contract_learning import asset_paths
from scripts.retrain_atm_commission import fast_memory
from scripts.run_tail_put25_pooled import prior_vix

OUT = Path('artifacts/contracts/recovered_smile')
REPORT = Path('reports/recovered_smile')
PROTOCOL = (
    'Controlled coverage recovery from the same-structure original-calendar study. '
    'All three wings select their own closest usable expiry to30D, with21-45D '
    'at next-session entry, OI>=10, relative width<=25%, positive standard '
    '100-share quotes and successful BS IV fit. ATM calls use closest strike '
    'within10% spot; P25/P75 puts use fitted absolute delta+/-0.075. '
    'Expiry distance precedes strike/delta distance. No next-session quote '
    'or future holding-path availability enters contract selection. '
    'All remaining signal definitions,27 oriented features,monthly pooled '
    '540model fits,hard91/365day directional-profit and21/63 count gates, '
    'beta0.85-1.15/correlation>=0.5,0.65 training hurdle,0.03 actual commissions, '
    'mid+.1volpoint,hedge/funding,risk limits and ladders are unchanged. '
    'The original formation DATE INDEX is applied equally across all wings '
    'in evaluation, matching the existing calendar control. All345 roster '
    'pairs are retained;122 have that historical calendar. This is inherited '
    'retrospective coverage conditioning, not a new pair discovery test. '
    'Completed finite labels only for training; never filter evaluated '
    'signals on forward label validity. Fixed50/75/100 and two RL ladders '
    'share proposals; floor50/75/100,flex0/25/50/75/100. All wing results '
    'and combined results reported, with a same-skip full-size control. '
    'One worker,50%of one core,3GiB stop. Frozen checkpoints are untouched. '
    'Already inspected history; no fresh holdout or sequential inventory RL.'
)


class Sources(dict):
    def track(self, path):
        path = Path(path)
        if str(path) not in self:
            self[str(path)] = digest(path)
        return path


class Loader(data.Loader):
    def asset(self, sid):
        if sid in self.cache:
            return self.cache[sid]
        folder = data.OUT / 'market/assets'
        stock = pd.read_parquet(folder / f'{sid}_stocks.parquet')
        sel = pd.read_parquet(folder / f'{sid}_selection.parquet')
        path = folder / f'{sid}_unit.joblib'
        unit = joblib.load(path,mmap_mode='r') if path.exists() else None
        assert unit is None or unit.cfg.min_open_interest == 10
        returns = (stock.set_index('date').close * stock.set_index('date').cfadj).pct_change(fill_method=None)
        result = dict(unit=unit,sel=sel,returns=returns)
        self.cache[sid] = result
        return result


def setup():
    common.OUT, common.REPORT = OUT, REPORT
    data.OUT, data.REPORT = rebuilt.OUT, rebuilt.REPORT
    common.config, data.config = rebuilt.config, rebuilt.config
    data.Loader = Loader
    calendars, calendar_sources = formation_dates()
    original = common.allocations
    def allocations(variant, cal):
        rows = original(variant, cal)
        return rows.loc[filter_calendar(rows,calendars)].reset_index(drop=True)
    common.allocations = allocations
    return calendars, calendar_sources


def compact(cal, calendars):
    folder = data.OUT / 'market/assets'
    sids = sorted({int(x) for pair in calendars for x in pair.split('__')})
    needed = {sid:set() for sid in sids}
    for wing in common.WINGS:
        rows = pd.read_parquet(OUT / f'{wing}_rows.parquet')
        rows = rows.loc[rows.signal_eligible & rows.date.between('2018-01-01','2025-06-30')].reset_index(drop=True)
        rows = rows.loc[filter_calendar(rows,calendars)]
        for suffix in ['a','b']:
            for sid,g in rows.groupby('secid_'+suffix):
                needed[int(sid)].update(g['optionid_'+suffix].dropna().astype(int))
    for kind in ['stocks','dividends']:
        pd.concat([pd.read_parquet(folder / f'{sid}_{kind}.parquet') for sid in sids],ignore_index=True).to_parquet(data.OUT / 'market' / f'{kind}.parquet',index=False)
    market = Market(cal)
    market.rates.rename('rate').to_frame().to_parquet(data.OUT / 'market/rates.parquet')
    for year in cal[cal>='2018-01-01'].year.unique():
        path = data.OUT / 'market' / f'{year}_quotes.parquet'
        if path.exists():
            continue
        parts = []
        for sid in sids:
            if needed[sid]:
                parts.append(pd.read_parquet(folder / f'{sid}_quotes.parquet',filters=[('date','>=',pd.Timestamp(year,1,1)),('date','<',pd.Timestamp(year+1,1,1)),('optionid','in',sorted(needed[sid]))]))
        q = pd.concat(parts,ignore_index=True)
        assert not q.duplicated(['date','optionid']).any()
        q.to_parquet(path,index=False)
        print(f'Recovered replay quotes {year}: {len(q):,}',flush=True)
        del q,parts
        gc.collect()


def prepare_pairs(defs, market, sources):
    """Compute paths only where at least one observation enters the study.

The scope check uses IV, historical beta and inherited dates, never future
path validity. Skipped optional memory is explicitly unavailable, not a veto
on any otherwise admitted observation.
"""
    cal, loader, coverage = market.calendar, Loader(market), []
    vix = prior_vix(cal)
    sources.track(Path('data/raw/VIX_History.csv'))
    for n,d in enumerate(defs,1):
        folder = data.OUT / 'pairs' / d['pair']
        folder.mkdir(parents=True,exist_ok=True)
        stamp = folder / 'prepared.json'
        if stamp.exists():
            coverage.extend(json.loads(stamp.read_text()))
            continue
        loader.retain(d['secids'])
        a,b = [loader.asset(sid) for sid in d['secids']]
        returns = []
        for path, asset in zip(asset_paths(d),[a,b]):
            returns.append(pd.read_parquet(sources.track(path),columns=['ret']).ret.reindex(cal) if path.exists() else asset['returns'])
        returns = pd.concat(returns,axis=1)
        returns.columns = d['secids']
        screen = beta_screen(returns,cal,beta_tolerance=.15)
        native = Path('artifacts/contracts/tail_put25_strategy') / d['pair'] / 'states.parquet'
        legacy = Path('artifacts/contracts/tail_calendar_ablation') / d['pair'] / 'legacy_calendar_memory/actions.csv'
        pre = pd.read_parquet(sources.track(native),columns=[]).index if native.exists() else pd.DatetimeIndex([])
        post = read_actions(sources.track(legacy)).index if legacy.exists() else pd.DatetimeIndex([])
        context = np.where(cal<'2018-01-01',cal.isin(pre),cal.isin(post))
        records = []
        for wing,(cp,bucket) in common.WINGS.items():
            chosen = [x['sel'].loc[x['sel'].cp_flag.eq(cp) & x['sel'].bucket.eq(bucket)] for x in [a,b]]
            memory_cols = [f'{prefix}_{side}_{days}d' for days in [91,365] for side in ['short','long'] for prefix in ['profit','count']]
            memory = pd.DataFrame(np.nan,index=cal,columns=memory_cols)
            panel = data.policy.states(*chosen,returns,screen,cal,memory,vix,.65)
            panel['context'] = context
            panel['candidate'] = panel.observed & panel.eligible & panel.context
            needed = bool(panel.candidate.any())
            width = rebuilt.config().max_dte + 2
            daily = np.zeros((len(cal),8,width),np.float32)
            ends = np.arange(len(cal))
            vegas = np.zeros((len(cal),8))
            status = np.full(len(cal),'no_entry',object)
            if needed and all(x['unit'] is not None for x in [a,b]):
                daily,ends,vegas,status = data.pair_labels(a['unit'],b['unit'],(30,cp,int(bucket) if bucket!='atm' else bucket,int(bucket) if bucket!='atm' else bucket),replace(rebuilt.config(),option_commission=.65))
                reward = np.divide(daily[:,[0,-1]].sum(axis=2)*450,vegas[:,[0,-1]],out=np.full((len(cal),2),np.nan),where=vegas[:,[0,-1]]>0)
                memory = fast_memory(cal,cal[ends],reward,vegas[:,[0,-1]]>0)
                panel = data.policy.states(*chosen,returns,screen,cal,memory,vix,.65)
                panel['context'] = context
                panel['candidate'] = panel.observed & panel.eligible & panel.context
                del reward
            test = panel.observed & (cal>='2018-01-01') & (cal<=cal[-1]-pd.Timedelta(days=60))
            panel.loc[test].reset_index().rename(columns={'index':'date'}).to_parquet(folder / f'{wing}_observed.parquet',index=False)
            idx = np.flatnonzero(panel.candidate)
            meta = panel.iloc[idx].copy()
            meta['date'], meta['position'] = meta.index, idx
            meta['end'], meta['label_row'] = cal[ends[idx]], np.arange(len(idx))
            labels = np.zeros((len(idx),9,width),np.float32)
            labels[:,1:] = daily[idx]
            meta['finite_label'] = np.isfinite(labels).all(axis=(1,2)) & (status[idx]=='complete') & (vegas[idx,0]>0) & (vegas[idx,-1]>0)
            meta['target_long'], meta['target_short'] = labels[:,8].sum(axis=1), labels[:,1].sum(axis=1)
            meta['pair'],meta['wing'],meta['secid_a'],meta['secid_b'] = d['pair'],wing,*d['secids']
            for name,selected in zip(['optionid_a','optionid_b'],chosen):
                meta[name] = selected.set_index('date').optionid.reindex(meta.index).to_numpy()
            meta.reset_index(drop=True).to_parquet(folder / f'{wing}_rows.parquet',index=False)
            np.save(folder / f'{wing}_labels.npy',labels)
            records.append(dict(pair=d['pair'],label=d['label'],wing=wing,observed_test_rows=int(test.sum()),candidates=len(meta),finite_training_candidates=int(meta.finite_label.sum()),evaluated_candidates=int(meta.date.ge('2018-01-01').sum()),paths_computed=needed))
            del panel,meta,daily,labels,memory
        write_json(stamp,records)
        coverage.extend(records)
        common.progress('pairs',pair=n,total=len(defs))
        if n%20 == 0:
            print(f'Recovered pair labels {n}/{len(defs)}',flush=True)
        gc.collect()
    loader.cache.clear()
    pd.DataFrame(coverage).to_csv(data.REPORT / 'coverage.csv',index=False)
    for wing in common.WINGS:
        rows = pd.concat([pd.read_parquet(data.OUT / 'pairs' / d['pair'] / f'{wing}_rows.parquet') for d in defs],ignore_index=True)
        rows.to_parquet(data.OUT / f'{wing}_rows.parquet',index=False)


def fixed_wing_replays(cal):
    market = common.Market()
    cfg = replace(rebuilt.config(), option_commission=.03, capital=345_000_000,max_gross_vega=20_000*345)
    for wing in common.WINGS:
        folder = OUT / 'replays' / f'fixed_100_{wing}'
        folder.mkdir(parents=True,exist_ok=True)
        if (folder / 'status.json').exists():
            continue
        rows = common.allocations('fixed_100',cal)
        rows = rows.loc[rows.wing.eq(wing)].copy()
        rows.to_parquet(folder / 'scored_allocations.parquet',index=False)
        market.variant, market.started, market.reported_year = folder.name,time.monotonic(),None
        saved = engine.book_risk
        try:
            engine.book_risk = book_risk
            nav,trades,decisions,_ = engine.simulate(market,rows,cal[cal>='2018-01-01'],cfg,1_000_000,wing_capital=1_000_000,corporate_events=UNIFORM_CORPORATE_EVENTS)
        finally:
            engine.book_risk = saved
        mark_book(nav,market.rates,cfg.capital)
        audit = verify(nav,trades,decisions,rows,cfg)
        nav.to_parquet(folder / 'daily_nav.parquet')
        trades.to_csv(folder / 'trades.csv',index=False)
        decisions.to_csv(folder / 'decisions.csv',index=False)
        write_json(folder / 'status.json',dict(status='complete',variant='fixed_100',wing=wing,proposals=len(rows),qualified_candidates=len(rows),learned_skips=0,net_excess_pnl=float(nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1]),**audit))
        print(f'{folder.name}: Sharpe {audit["annualized_excess_sharpe"]:.6f}, {audit["entries"]} entries',flush=True)


def summarize(sources):
    result = common.summarize(sources)
    old = pd.read_csv('reports/same_structure_calendar/comparison.csv')
    keys = ['model','annualized_excess_sharpe','entries','net_excess_pnl']
    compare = result[keys].merge(old[keys],on='model',how='left',suffixes=('_recovered','_previous'))
    compare.to_csv(REPORT / 'coverage_effect.csv',index=False)
    result[['model','annualized_excess_sharpe','entries','net_excess_pnl']].to_csv(REPORT / 'results.csv',index=False)
    for path,value in sources.items():
        assert digest(path)==value, f'Pinned source changed: {path}'
    write_json(REPORT / 'audit.json',dict(status='complete',same_structure_across_all_wings=True,replays=len(result),models=540,selection_changes_only=True,no_fresh_holdout=True,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    write_json(REPORT / 'output_hashes.json',{str(p):digest(p) for root in [OUT,REPORT] for p in root.rglob('*') if p.is_file() and p.name!='output_hashes.json'})
    (REPORT / 'findings.txt').write_text(PROTOCOL+'\n\n'+compare.to_string(index=False)+'\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase',choices=['pairs','prepare','train','replay','summarize'],required=True)
    args = p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    REPORT.mkdir(parents=True,exist_ok=True)
    calendars, calendar_sources = setup()
    cal = load_calendar(ROOT)
    cal = cal[(cal>='2010-01-01') & (cal<='2025-08-29')]
    sources = Sources(calendar_sources)
    if (OUT / 'sources.json').exists():
        for path,value in json.loads((OUT / 'sources.json').read_text()).items():
            assert digest(path)==value, f'Pinned source changed: {path}'
            sources[path]=value
    for path in [Path(__file__),Path('scripts/rebuild_smile_contracts.py'),Path('scripts/smile_contract_selection.py'),Path('scripts/same_structure_smile.py'),Path('scripts/run_same_structure_smile.py'),Path('scripts/uniform_smile_data.py'),Path('scripts/uniform_smile_engine.py'),Path('scripts/uniform_smile_risk.py'),rebuilt.DATA / 'inputs.json']:
        sources.track(path)
    write_json(OUT / 'inputs.json',dict(protocol=PROTOCOL,definitions_path='artifacts/contracts/tail_put25_pooled/inputs.json',only_changes=['selection_min_open_interest_10','closest_usable_30D_expiry_independently_per_bucket'],features=common.policy.FEATURES,training_commission=.65,actual_commission=.03,same_structure_across_wings=True))
    defs = json.loads(Path('artifacts/contracts/tail_put25_pooled/inputs.json').read_text())['definitions']
    if args.phase == 'pairs':
        market = Market(cal)
        prepare_pairs(defs,market,sources)
    elif args.phase == 'prepare':
        common.prepare(sources)
        compact(cal,calendars)
    elif args.phase == 'train':
        common.train(cal)
    elif args.phase == 'replay':
        common.replay(cal)
        fixed_wing_replays(cal)
        summarize(sources)
    else:
        summarize(sources)
    write_json(OUT / 'sources.json',sources)
    common.progress(args.phase+'_complete')


if __name__ == '__main__':
    main()
