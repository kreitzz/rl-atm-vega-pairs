"""Three fixed training hurdles for the frozen ungated P75 flexible sizing setup."""
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import resource
import time
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from scripts.put75_cost_hurdle import with_training_commission
from scripts.run_smile_sizing import scoped_inputs, SizingMarket
from scripts.research_opposite_wing_signals import AccountingReplay, OUT as SIGNAL
from scripts.curve_trading.config import DATA, config, RUNOFF
from scripts.curve_trading.unit_paths import UnitPaths, pair_labels
from scripts.smile_sizing_policy import FEATURES, directional_labels, training_mask, fit_sizes, predict, translate
from scripts.tail_put25_pooled import aggregate_chunk
from scripts.run_broad_contract_replay import Market, ROOT
from scripts.select_atm_contracts import load_calendar
from scripts.run_atm65_ablation import read_actions, write_json
from scripts.run_contract_walkforward import digest
from scripts.replay_frozen_atm_commission import cash_book
from scripts.run_aligned_contract_learning import mark_book
from scripts.run_selected_pair_backtest import audit, scores
from scripts.repair_tail_put25_pooled import save

FROZEN = Path('artifacts/contracts/smile_sizing')
FROZEN_REPORT = Path('reports/smile_sizing')
OUT = Path('artifacts/contracts/put75_cost_hurdles')
REPORT = Path('reports/put75_cost_hurdles')
HURDLES = {'100':1., '150':1.5}
LEVELS = [0,1,2,3,4]


class Sources(dict):
    def __init__(self):
        super().__init__()
        self.expected = json.loads((FROZEN/'market_sources.json').read_text())
        self.expected.update(json.loads((FROZEN_REPORT/'output_hashes.json').read_text()))
        self.verified = set()

    def track(self,path):
        path = Path(path); key = str(path)
        if key not in self:
            value = digest(path)
            if key in self.expected:
                assert value == self.expected[key], f'Frozen input changed: {key}'
                self.verified.add(key)
            self[key] = value
        return path


class Put75Market(SizingMarket):
    def __init__(self,market,cfg,sources,units):
        super().__init__(market,cfg,sources)
        self.units = units

    def asset(self,sid):
        if sid in self.cache:
            return self.cache[sid]
        cal = self.market.calendar; pieces = []
        for year in cal.year.unique():
            path = DATA/'selection/P'/str(year)/f'{sid}.parquet'
            if path.exists():
                frame = pd.read_parquet(self.sources.track(path))
                pieces.append(frame.loc[frame.tenor.eq(30)&frame.bucket.eq('75')])
        sel = pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame(columns=['date','expiry','tenor','cp_flag','bucket','optionid'])
        for col in ['date','expiry']:
            sel[col] = pd.to_datetime(sel[col])
        ids = sel.optionid.unique().tolist(); pieces = []
        for year in cal.year.unique():
            path = DATA/'history/P'/str(year)/f'{sid}.parquet'
            if ids and path.exists():
                pieces.append(pd.read_parquet(self.sources.track(path),filters=[('optionid','in',ids)]))
        q = pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame()
        stock = self.market.stocks.loc[self.market.stocks.secid.eq(sid)].set_index('date').reindex(cal)
        stock['secid'] = sid; stock['rate'] = self.market.rates; stock.index.name = 'date'
        divs = self.market.divs.loc[self.market.divs.secid.eq(sid)]
        q,stock,sel,divs = scoped_inputs([dict(q=q,s=stock.reset_index(),sel=sel,divs=divs)],[sid])
        unit = UnitPaths(q,stock,sel,cal,cal,divs,self.cfg) if len(q) and self.units else None
        result = dict(q=q,s=stock,sel=sel,divs=divs,unit=unit)
        self.cache[sid] = result
        return result


def progress(stage,**details):
    write_json(OUT/'progress.json',{**details,'stage':stage,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


def prepare(defs,market,cfg,sources):
    loader = Put75Market(market,replace(cfg,option_commission=.65),sources,units=True)
    logs = []
    for n,d in enumerate(defs,1):
        dest = OUT/'pairs'/d['pair']; dest.mkdir(parents=True,exist_ok=True)
        stamp = dest/'prepared.json'
        if stamp.exists():
            logs.append(json.loads(stamp.read_text())); continue
        base = FROZEN/'pairs'/d['pair']/'P75'
        meta = pd.read_parquet(sources.track(base/'training_rows.parquet'))
        frozen = np.load(sources.track(base/'labels.npy'),mmap_mode='r')
        indices = meta.position.to_numpy(int)
        if len(meta):
            loader.retain(d['secids']); assets = [loader.asset(s) for s in d['secids']]
            assert all(a['unit'] is not None for a in assets)
            panel = pd.read_parquet(sources.track(SIGNAL/'pairs'/d['pair']/'P75/states.parquet'),columns=['wing_iv_rank'])
            ranks = panel.wing_iv_rank.iloc[indices]
            units = [a['unit'] for a in assets]
            daily,ends,vegas,status = pair_labels(*units,(30,'P',75,75),loader.cfg)
            reproduced = directional_labels(daily[indices],ranks).astype(np.float32)
            np.testing.assert_array_equal(reproduced,frozen)
            finite = np.isfinite(reproduced).all(axis=(1,2))&(status[indices]=='complete')&(vegas[indices,0]>0)&(vegas[indices,-1]>0)
            np.testing.assert_array_equal(finite,meta.finite_label)
            np.testing.assert_array_equal(market.calendar[ends[indices]].values,pd.DatetimeIndex(meta.end).values)
            del daily,reproduced
            for key,hurdle in HURDLES.items():
                adjusted = [with_training_commission(u,hurdle) for u in units]
                new,other_ends,other_vegas,other_status = pair_labels(*adjusted,(30,'P',75,75),replace(cfg,option_commission=hurdle))
                np.testing.assert_array_equal(other_ends,ends)
                np.testing.assert_array_equal(other_vegas,vegas)
                np.testing.assert_array_equal(other_status,status)
                labels = directional_labels(new[indices],ranks).astype(np.float32)
                np.testing.assert_array_equal(np.isfinite(labels),np.isfinite(frozen))
                np.save(dest/f'labels_{key}.npy',labels)
                del adjusted,new,labels
            del assets,units,ends,vegas,status,other_ends,other_vegas,other_status
        else:
            for key in HURDLES:
                np.save(dest/f'labels_{key}.npy',np.asarray(frozen))
        record = dict(pair=d['pair'],label=d['label'],training_rows=len(meta),finite_training_rows=int(meta.finite_label.sum()),baseline_labels_exactly_reproduced=True)
        write_json(stamp,record); logs.append(record)
        write_json(OUT/'market_sources.json',sources); progress('prepare',pair=n,roster=len(defs),label=d['label'])
        if len(meta) or n%40 == 0:
            print(f'P75 hurdle labels {n}/{len(defs)} {d["label"]}: {len(meta)} rows',flush=True)
        del frozen,meta; gc.collect()
    loader.cache.clear(); gc.collect()
    pd.DataFrame(logs).to_csv(REPORT/'training_coverage.csv',index=False)


def train(defs,cal,sources):
    meta = pd.concat([pd.read_parquet(sources.track(FROZEN/'pairs'/d['pair']/'P75/training_rows.parquet')) for d in defs],ignore_index=True)
    logs = []
    for month in pd.period_range('2018-01-01',cal[-1]-pd.Timedelta(days=RUNOFF),freq='M'):
        cutoff = month.start_time; fitdate = str(cutoff.date())
        data = meta.loc[training_mask(meta,cutoff)].reset_index(drop=True)
        assert len(data)>=16 and pd.Timestamp(data.end.max())<cutoff
        for key,hurdle in HURDLES.items():
            dest = OUT/'models'/key; dest.mkdir(parents=True,exist_ok=True)
            stamp = dest/f'{fitdate}.json'
            if stamp.exists():
                logs.append(json.loads(stamp.read_text())); continue
            target = np.zeros(len(data))
            for pair,rows in data.groupby('pair',sort=False).groups.items():
                rows = np.asarray(rows); disk = np.load(OUT/'pairs'/pair/f'labels_{key}.npy',mmap_mode='r')
                target[rows] = disk[data.loc[rows,'label_row'].to_numpy(int),4].sum(axis=1)
                del disk
            minleaf = max(8,min(63,len(data)//8))
            tree = DecisionTreeRegressor(max_leaf_nodes=8,min_samples_leaf=minleaf,random_state=2026).fit(data[FEATURES],target)
            leaves = np.unique(tree.apply(data[FEATURES])); states = np.searchsorted(leaves,tree.apply(data[FEATURES]))
            first = int(data.position.min()+1); last = int(cal.get_indexer(pd.DatetimeIndex(data.end)).max())
            book = np.zeros((last-first+1,len(leaves)*5))
            for pair,rows in data.groupby('pair',sort=False).groups.items():
                rows = np.asarray(rows); disk = np.load(OUT/'pairs'/pair/f'labels_{key}.npy',mmap_mode='r')
                labels = disk[data.loc[rows,'label_row'].to_numpy(int)]
                aggregate_chunk(book,states[rows],labels,data.loc[rows,'position'].to_numpy(int),first)
                del disk,labels
            policy,details = fit_sizes(book,tree,leaves,LEVELS)
            joblib.dump(policy,dest/f'{fitdate}.joblib')
            record = dict(hurdle=hurdle,fit=fitdate,training_rows=len(data),training_pairs=int(data.pair.nunique()),
                          last_training_outcome=str(pd.Timestamp(data.end.max()).date()),leaves=len(leaves),**details)
            write_json(stamp,record); logs.append(record)
            del tree,book,policy,target; gc.collect()
        progress('train',fit=fitdate,models=len(logs))
        if month.month in [1,7]:
            print(f'P75 hurdle fits {fitdate}: {len(data)} completed rows',flush=True)
        del data
    pd.DataFrame(logs).to_csv(REPORT/'fits.csv',index=False)


def resolved_full(root,sources):
    if (root/'reference.json').exists():
        return Path(json.loads(sources.track(root/'reference.json').read_text())['source'])
    return root/'full'


def replay(defs,market,cfg,sources):
    cal = market.calendar; evaluation = cal[cal>='2018-01-01']
    cash = cash_book(evaluation,market.rates,cfg.capital)
    loader = Put75Market(market,cfg,sources,units=False)
    models = {key:{p.stem:joblib.load(p) for p in (OUT/'models'/key).glob('*.joblib')} for key in HURDLES}
    coverage = []
    for n,d in enumerate(defs,1):
        frozen = FROZEN/'pairs'/d['pair']/'P75'
        base = read_actions(sources.track(SIGNAL/'pairs'/d['pair']/'P75/spread/actions.csv'))
        state = pd.read_parquet(sources.track(frozen/'features.parquet'))
        dates = state.index[state.observed&(state.index>='2018-01-01')&(state.index<=cal[-1]-pd.Timedelta(days=RUNOFF))]
        candidates = [frozen/'flex']; engine = stock = None
        for key,hurdle in HURDLES.items():
            dest = OUT/'pairs'/d['pair']/key; full = dest/'full'; full.mkdir(parents=True,exist_ok=True)
            allocation = pd.Series(index=dates,dtype=int,name='level')
            for month in dates.to_period('M').unique():
                idx = dates[dates.to_period('M')==month]
                allocation.loc[idx] = predict(models[key][str(month.start_time.date())],state.loc[idx,FEATURES])
            actions = base.copy().rename('code')
            actions.loc[dates] = translate(allocation.to_numpy(int),base.reindex(dates,fill_value=0),'P',75)
            assert actions.loc[base.eq(0)].eq(0).all()
            actions.to_csv(dest/'actions.csv'); (allocation/4).rename('allocation').to_csv(dest/'scored_allocations.csv')
            coverage.append(dict(pair=d['pair'],label=d['label'],hurdle=hurdle,scored_rows=len(dates),
                                 rule_signals=int(base.ne(0).sum()),sized_signals=int(actions.ne(0).sum()),learned_skips=int((base.ne(0)&actions.eq(0)).sum())))
            if (full/'status.json').exists() or (dest/'reference.json').exists():
                candidates.append(dest); continue
            matched = None
            for other in candidates:
                old = read_actions(sources.track(other/'actions.csv'))
                if actions.index.equals(old.index) and np.array_equal(actions.to_numpy(),old.to_numpy()):
                    matched = resolved_full(other,sources); break
            if matched is not None:
                sources.track(matched/'status.json')
                write_json(dest/'reference.json',dict(source=str(matched),identical_actual_actions=True,actual_commission=.03))
            elif not actions.ne(0).any():
                write_json(full/'status.json',dict(status='complete',cash_only=True,entries=0))
            else:
                if engine is None:
                    loader.retain(d['secids']); assets = [loader.asset(sid) for sid in d['secids']]
                    q,stock,sel,divs = [pd.concat([a[c] for a in assets],ignore_index=True) for c in ['q','s','sel','divs']]
                    engine = AccountingReplay(q,stock,sel,d['secids'],divs,cfg)
                nav,trades,decisions = engine.run(actions,evaluation); mark_book(nav,market.rates,cfg.capital)
                if trades.empty:
                    checks = audit(nav,trades,stock,cfg,d['secids'])
                    np.testing.assert_allclose(nav.nav,cash.nav,atol=1e-7,rtol=0)
                    decisions.to_csv(full/'decisions.csv',index=False)
                    write_json(full/'status.json',dict(status='complete',cash_only=True,entries=0,**checks))
                else:
                    save(full,nav,trades,decisions,cfg,stock,d)
                del nav,trades,decisions
            candidates.append(dest)
        progress('replay',pair=n,roster=len(defs),label=d['label']); write_json(OUT/'market_sources.json',sources)
        if n%20==0:
            print(f'P75 hurdle replay {n}/{len(defs)} {d["label"]}',flush=True)
        del engine,stock; gc.collect()
    loader.cache.clear(); gc.collect()
    pd.DataFrame(coverage).to_csv(REPORT/'scoring_coverage.csv',index=False)


def summarize(defs,market,cfg,sources):
    cal = market.calendar[market.calendar>='2018-01-01']; cash = cash_book(cal,market.rates,cfg.capital)
    books = {}; details = {}; pair_rows = []; allocation_rows = []
    frozen_comparison = pd.read_csv(sources.track(FROZEN_REPORT/'comparison.csv')).set_index('model')
    for name in ['atm_frozen','put25_frozen','P75_flex','P75_floor']:
        book = pd.read_csv(sources.track(FROZEN_REPORT/f'{name}_full_portfolio.csv'),index_col=0,parse_dates=[0],float_precision='round_trip')
        assert book.index.equals(cal)
        books[name] = book[['nav','cash_benchmark']]
        np.testing.assert_allclose(scores(book,cfg.capital*len(defs))['annualized_excess_sharpe'],frozen_comparison.loc[name,'annualized_excess_sharpe'],atol=1e-10,rtol=0)
    for key,hurdle in HURDLES.items():
        name = f'P75_flex_{key}'; book = cash[['nav','cash_benchmark']]*len(defs)
        detail = dict(training_hurdle=hurdle,entries=0,active_pairs=0,commission=0.,option_cost=0.,hedge_cost=0.,risk_breach_pair_sessions=0)
        for d in defs:
            root = OUT/'pairs'/d['pair']/key; full = resolved_full(root,sources)
            status = json.loads(sources.track(full/'status.json').read_text()); assert status['status']=='complete'
            nav = cash if status.get('cash_only') else pd.read_parquet(sources.track(full/'daily_nav.parquet'))
            assert nav.index.equals(cal)
            book += nav[['nav','cash_benchmark']]-cash[['nav','cash_benchmark']]
            for field in ['entries','commission','option_cost','hedge_cost']:
                detail[field] += status.get(field,0)
            detail['active_pairs'] += int(status['entries']>0)
            detail['risk_breach_pair_sessions'] += status.get('risk_breach_sessions',0)
            pair_rows.append(dict(model=name,pair=d['pair'],label=d['label'],entries=status['entries'],net_excess_pnl=float(nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1])))
            alloc = pd.read_csv(root/'scored_allocations.csv',index_col=0,parse_dates=[0]).allocation
            base = read_actions(SIGNAL/'pairs'/d['pair']/'P75/spread/actions.csv')
            for level,count in alloc.loc[base.reindex(alloc.index,fill_value=0).ne(0)].value_counts().items():
                allocation_rows.append(dict(model=name,allocation=float(level),signals=int(count)))
        books[name] = book; details[name] = detail
    details['P75_flex'] = dict(training_hurdle=.65,entries=261,active_pairs=40)
    combinations = {'ATM_plus_put25':['atm_frozen','put25_frozen']}
    for name in ['P75_floor','P75_flex','P75_flex_100','P75_flex_150']:
        combinations[f'put25_plus_{name}'] = ['put25_frozen',name]
        combinations[f'ATM_put25_plus_{name}'] = ['atm_frozen','put25_frozen',name]
    for name,members in combinations.items():
        books[name] = sum(books[m][['nav','cash_benchmark']] for m in members)
    rows = []; annual = []
    for name,book in books.items():
        book = book.copy(); book['return_'] = book.nav.pct_change().fillna(0)
        book['excess_return'] = book.return_-book.cash_benchmark.pct_change().fillna(0)
        sleeves = len(combinations[name]) if name in combinations else 1
        rows.append(dict(model=name,funded_sleeve_sets=sleeves,net_excess_pnl=float(book.nav.iloc[-1]-book.cash_benchmark.iloc[-1]),
                         **details.get(name,{}),**scores(book,cfg.capital*len(defs)*sleeves)))
        book.to_csv(REPORT/f'{name}_full_portfolio.csv')
        for year,g in book.groupby(book.index.year):
            annual.append(dict(model=name,year=int(year),annualized_excess_sharpe=float(np.sqrt(252)*g.excess_return.mean()/g.excess_return.std()) if g.excess_return.std()>1e-12 else None))
    result = pd.DataFrame(rows); result.to_csv(REPORT/'comparison.csv',index=False)
    pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False)
    pd.DataFrame(pair_rows).to_csv(REPORT/'pairs.csv',index=False)
    pd.DataFrame(allocation_rows).groupby(['model','allocation']).signals.sum().reset_index().to_csv(REPORT/'allocations.csv',index=False)
    print(result[['model','annualized_excess_sharpe','net_excess_pnl','entries']].to_string(index=False),flush=True)
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument('--phase',choices=['all','test','prepare','train','replay','summarize'],default='all'); args=p.parse_args()
    if args.phase=='test':
        suite = unittest.defaultTestLoader.discover('tests',pattern='test_put75_cost_hurdle.py')
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            raise SystemExit(1)
        return
    started = time.monotonic(); OUT.mkdir(parents=True,exist_ok=True); REPORT.mkdir(parents=True,exist_ok=True)
    sources = Sources()
    prior_run_audit = json.loads((REPORT/'audit.json').read_text()) if args.phase=='summarize' and (REPORT/'audit.json').exists() else None
    if (OUT/'market_sources.json').exists():
        for path,value in json.loads((OUT/'market_sources.json').read_text()).items():
            if args.phase=='summarize' and path==str(Path(__file__)) and digest(path)!=value:
                # A duplicate reporting keyword stopped the original job after
                # all fits/replays and ledger checks completed. Preserve that
                # driver and allow its reporting-only repair without refitting.
                archive = OUT/'run_before_reporting_fix.py'
                assert digest(archive)==value, 'Reporting repair needs the exact original driver'
                sources.track(archive); sources[path]=digest(path)
                continue
            assert digest(path)==value, f'Resumed source changed: {path}'
            sources[path] = value
            if path in sources.expected:
                assert value==sources.expected[path]; sources.verified.add(path)
    for path in [Path(__file__),Path('scripts/put75_cost_hurdle.py'),Path('tests/test_put75_cost_hurdle.py'),
                 Path('scripts/run_smile_sizing.py'),Path('scripts/smile_sizing_policy.py'),
                 Path('scripts/research_opposite_wing_signals.py'),Path('scripts/analyze_tail_universe_change.py'),
                 Path('scripts/curve_trading/unit_paths.py'),Path('scripts/curve_trading/policy.py'),
                 Path('scripts/option_flex/pooled_tail_engine.py'),Path('scripts/option_flex/contract_risk.py'),
                 ROOT/'zero_curve.parquet',ROOT/'distributions.parquet',Path('data/raw/opposite_wing_DEXCAUS.csv')]:
        sources.track(path)
    cal = load_calendar(ROOT); cal=cal[(cal>='2010-01-01')&(cal<='2025-08-29')]
    for year in cal.year.unique():
        sources.track(ROOT/'underlying'/f'year={year}.parquet')
    defs = json.loads(sources.track(Path('artifacts/contracts/tail_put25_pooled/inputs.json')).read_text())['definitions']
    market = Market(cal); cfg = replace(config(),option_commission=.03)
    protocol = ('P75 only: fixed 0.65/1.00/1.50 training commission hurdles, actual commission 0.03. '
                'Frozen 0.65 flexible policy and ledgers reused; two higher hurdles retrained monthly on five years of completed standalone integer-contract outcomes. '
                'All 345 roster pairs and every observed state retained, same ungated rule directions/features/calendars; native profitability memory remains a frozen input. '
                'Flexible 0/25/50/75/100 sizing, pooled direct-Sharpe allocation search, not a sequential inventory-aware RL agent. '
                'Unit cash flows change entry commissions only; pair financing and cash-sign borrowing recomputed exactly. Baseline labels must reproduce bit-for-bit. '
                'Identical actual actions may reuse audited actual ledgers. Actual risk limits, daily delta hedging, initial vega matching and own-expiry hold remain unchanged. '
                'Frozen ATM/downside policies, equal-capital separately funded sleeve combinations, no shared inventory or fitted combination weights. '
                'Known wide held quotes left unchanged equally across trials. Previously inspected history; exploratory sensitivity, not an untouched holdout or proof of an optimal hurdle. '
                'One worker, 50% of one core, 3 GiB RSS stop, single BLAS/Arrow thread.')
    write_json(OUT/'inputs.json',dict(hurdles={'.65':'frozen baseline',**HURDLES},actual_commission=.03,features=FEATURES,levels=LEVELS,protocol=protocol))
    if args.phase in ['all','prepare']:
        prepare(defs,market,cfg,sources)
    if args.phase in ['all','train']:
        train(defs,cal,sources)
    if args.phase in ['all','replay']:
        replay(defs,market,cfg,sources)
    if args.phase in ['all','summarize']:
        result = summarize(defs,market,cfg,sources)
        fits = pd.read_csv(REPORT/'fits.csv'); assert len(fits)==180
        assert (pd.to_datetime(fits.last_training_outcome)<pd.to_datetime(fits.fit)).all()
        coverage = pd.read_csv(REPORT/'scoring_coverage.csv')
        assert len(coverage)==690 and coverage.groupby('hurdle').scored_rows.sum().eq(58518).all()
        for d in defs:
            for key in HURDLES:
                full = resolved_full(OUT/'pairs'/d['pair']/key,sources)
                assert json.loads((full/'status.json').read_text())['status']=='complete'
        record = dict(status='complete',models_retrained=180,models_frozen=90,roster_pairs=len(defs),all_states_scored=True,
                      actual_commission=.03,frozen_source_files_verified=len(sources.verified),combined_shared_inventory=False,
                      elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      finalization_only=args.phase=='summarize')
        if prior_run_audit is not None:
            record['research_run_elapsed_seconds'] = prior_run_audit.get('research_run_elapsed_seconds',prior_run_audit['elapsed_seconds'])
            record['research_run_peak_rss_bytes'] = prior_run_audit.get('research_run_peak_rss_bytes',prior_run_audit['peak_rss_bytes'])
            record['reporting_keyword_repaired'] = True
        write_json(REPORT/'audit.json',record)
        (REPORT/'findings.txt').write_text(result.to_string(index=False)+'\n\n'+json.dumps(record,indent=2)+'\n\n'+protocol+'\n')
        progress('complete',**record)
    for path,value in sources.items():
        assert digest(path)==value, f'Source changed during run: {path}'
    write_json(OUT/'market_sources.json',sources)
    if args.phase in ['all','summarize']:
        hashes = {str(path):digest(path) for root in [OUT,REPORT] for path in root.rglob('*') if path.is_file() and path.name!='output_hashes.json'}
        for path in [Path(__file__),Path('scripts/put75_cost_hurdle.py'),Path('tests/test_put75_cost_hurdle.py')]:
            hashes[str(path)] = digest(path)
        write_json(REPORT/'output_hashes.json',hashes)


if __name__ == '__main__':
    main()
