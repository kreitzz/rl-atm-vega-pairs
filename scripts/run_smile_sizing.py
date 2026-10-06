"""Bounded wing-specific sizing, then fixed-weight stacks of funded sleeves."""
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
from sklearn.tree import DecisionTreeRegressor

from scripts.smile_sizing_policy import FEATURES,LADDERS,features,translate,directional_labels,training_mask,fit_sizes,predict
from scripts.research_opposite_wing_signals import OUT as SIGNAL,REPORT as SIGNAL_REPORT,SILOS,working_inputs as inherited_inputs,AccountingReplay
from scripts.analyze_tail_universe_change import BASE
from scripts.curve_trading.config import DATA,config,RUNOFF
from scripts.curve_trading.unit_paths import UnitPaths,pair_labels
from scripts.tail_put25_pooled import aggregate_chunk
from scripts.run_broad_contract_replay import ROOT,Market
from scripts.select_atm_contracts import load_calendar
from scripts.run_atm65_ablation import read_actions,write_json
from scripts.run_contract_walkforward import digest
from scripts.replay_frozen_atm_commission import cash_book
from scripts.run_aligned_contract_learning import mark_book
from scripts.run_selected_pair_backtest import scores,audit
from scripts.repair_tail_put25_pooled import save
from scripts.retrain_atm_commission import fast_memory

OUT=Path('artifacts/contracts/smile_sizing')
REPORT=Path('reports/smile_sizing')
ATM=Path('reports/atm65_employee_sentiment/all_rl_sentiment_full_portfolio.csv')
PUT=Path('reports/tail_put25_residual_rl/context_residual_rl_full_portfolio.csv')
DIVIDEND_REPAIR_IDS={105003,127149}


def scoped_inputs(assets,secids):
    q,s,sel,divs=inherited_inputs(assets,secids)
    # The frozen helper adds both repaired issuers whenever either is present.
    # UnitPaths expects dividends for its one asset, and the replay expects
    # only its held pair. Do not duplicate the other issuer's cash events.
    divs=divs.loc[divs.secid.isin(secids)].copy()
    return q,s,sel,divs


class SizingMarket:
    def __init__(self,market,cfg,sources):
        self.market,self.cfg,self.sources=market,cfg,sources;self.cache={}

    def retain(self,secids):
        for sid in list(self.cache):
            if sid not in secids:del self.cache[sid]
        gc.collect()

    def asset(self,sid):
        if sid in self.cache:return self.cache[sid]
        cal=self.market.calendar;selected=[];quotes=[]
        for cp,bucket in SILOS:
            parts=[]
            for year in cal.year.unique():
                path=DATA/'selection'/cp/str(year)/f'{sid}.parquet'
                if path.exists():
                    self.sources.setdefault(str(path),digest(path))
                    s=pd.read_parquet(path);parts.append(s[s.tenor.eq(30)&s.bucket.eq(str(bucket))])
            if not parts:continue
            sel=pd.concat(parts,ignore_index=True);selected.append(sel);ids=sel.optionid.unique().tolist()
            for year in cal.year.unique():
                path=DATA/'history'/cp/str(year)/f'{sid}.parquet'
                if ids and path.exists():
                    self.sources.setdefault(str(path),digest(path))
                    quotes.append(pd.read_parquet(path,filters=[('optionid','in',ids)]))
        sel=pd.concat(selected,ignore_index=True) if selected else pd.DataFrame(columns=['date','expiry','tenor','cp_flag','bucket','optionid'])
        for col in ['date','expiry']:sel[col]=pd.to_datetime(sel[col])
        q=pd.concat(quotes,ignore_index=True) if quotes else pd.DataFrame()
        stock=self.market.stocks[self.market.stocks.secid.eq(sid)].set_index('date').reindex(cal)
        stock['secid']=sid;stock['rate']=self.market.rates;stock.index.name='date'
        divs=self.market.divs[self.market.divs.secid.eq(sid)]
        q,s,sel,divs=scoped_inputs([dict(q=q,s=stock.reset_index(),sel=sel,divs=divs)],[sid])
        unit=UnitPaths(q,s,sel,cal,cal,divs,self.cfg) if len(q) else None
        result=dict(q=q,s=s,sel=sel,divs=divs,unit=unit);self.cache[sid]=result
        return result


def prepare(defs,market,cfg,sources):
    cal=market.calendar;loader=SizingMarket(market,replace(cfg,option_commission=.65),sources)
    logs=[]
    for n,d in enumerate(defs,1):
        folder=OUT/'pairs'/d['pair'];folder.mkdir(parents=True,exist_ok=True)
        completed=folder/'prepared.json'
        if completed.exists():logs.extend(json.loads(completed.read_text()));continue
        pre=Path('artifacts/contracts/tail_put25_strategy')/d['pair']/'states.parquet'
        context_dates=pd.read_parquet(pre,columns=[]).index if pre.exists() else pd.DatetimeIndex([])
        loader.retain(d['secids']);assets=[loader.asset(sid) for sid in d['secids']];rows=[]
        for cp,bucket in SILOS:
            silo=f'{cp}{bucket}';dest=folder/silo;dest.mkdir(exist_ok=True)
            panel=pd.read_parquet(SIGNAL/'pairs'/d['pair']/silo/'states.parquet')
            x=features(panel);x['observed']=panel.observed;x.to_parquet(dest/'features.parquet')
            context=np.where(cal<'2018-01-01',cal.isin(context_dates),panel.legacy)
            extreme=panel.wing_iv_rank.le(.2)|panel.wing_iv_rank.gt(.8)
            candidate=panel.observed&panel.eligible&extreme&context
            indices=np.flatnonzero(candidate);meta=x.loc[candidate,FEATURES].copy()
            meta['date']=meta.index;meta['position']=indices;meta['pair']=d['pair']
            meta['label_row']=np.arange(len(meta));meta['end']=meta.index
            meta['finite_label']=False;meta['target']=0.
            labels=np.zeros((len(meta),5,cfg.max_dte+2),dtype=np.float32)
            if all(a['unit'] is not None for a in assets) and (len(meta) or set(d['secids']).intersection(DIVIDEND_REPAIR_IDS)):
                daily,ends,vegas,status=pair_labels(*[a['unit'] for a in assets],(30,cp,bucket,bucket),loader.cfg)
                labels=directional_labels(daily[indices],panel.wing_iv_rank.iloc[indices]).astype(np.float32)
                meta['end']=cal[ends[indices]]
                meta['finite_label']=np.isfinite(labels).all(axis=(1,2))&(status[indices]=='complete')&(vegas[indices,0]>0)&(vegas[indices,-1]>0)
                meta['target']=labels[:,4].sum(axis=1)
                if set(d['secids']).intersection(DIVIDEND_REPAIR_IDS):
                    reward=np.divide(daily[:,[0,-1]].sum(axis=2)*cfg.entry_vega_per_leg,vegas[:,[0,-1]],
                        out=np.full((len(cal),2),np.nan),where=vegas[:,[0,-1]]>0)
                    memory=fast_memory(cal,cal[ends],reward,vegas[:,[0,-1]]>0)
                    long=panel.wing_iv_rank.le(.2)
                    for days in [91,365]:
                        for prefix in ['profit','count']:
                            panel[f'{prefix}_{days}d']=np.where(long,memory[f'{prefix}_long_{days}d'],memory[f'{prefix}_short_{days}d'])
                    x=features(panel);x['observed']=panel.observed;x.to_parquet(dest/'features.parquet')
                    meta.loc[:,FEATURES]=x.loc[candidate,FEATURES].to_numpy()
                    del memory,reward
                del daily
            np.save(dest/'labels.npy',labels);meta.reset_index(drop=True).to_parquet(dest/'training_rows.parquet')
            rows.append(dict(pair=d['pair'],label=d['label'],silo=silo,candidate_training_rows=len(meta),
                finite_training_rows=int(meta.finite_label.sum()),observed_evaluation_rows=int((panel.observed&(cal>='2018-01-01')&(cal<=cal[-1]-pd.Timedelta(days=RUNOFF))).sum())))
        write_json(completed,rows);logs.extend(rows);write_json(OUT/'market_sources.json',sources)
        write_json(OUT/'progress.json',dict(stage='prepare',pair=n,roster=len(defs),label=d['label']))
        if n%20==0:print(f'Smile labels {n}/{len(defs)} {d["label"]}',flush=True)
        del assets,labels,meta;gc.collect()
    loader.cache.clear();gc.collect();pd.DataFrame(logs).to_csv(REPORT/'training_coverage.csv',index=False)


def train(defs,cal):
    logs=[]
    for cp,bucket in SILOS:
        silo=f'{cp}{bucket}'
        meta=pd.concat([pd.read_parquet(OUT/'pairs'/d['pair']/silo/'training_rows.parquet') for d in defs],ignore_index=True)
        for month in pd.period_range('2018-01-01',cal[-1]-pd.Timedelta(days=RUNOFF),freq='M'):
            cutoff=month.start_time;fitdate=str(cutoff.date())
            data=meta.loc[training_mask(meta,cutoff)].reset_index(drop=True)
            if len(data):assert pd.Timestamp(data.end.max())<cutoff
            paths={v:OUT/'models'/silo/v for v in LADDERS}
            if all((p/f'{fitdate}.json').exists() for p in paths.values()):
                logs.extend(json.loads((p/f'{fitdate}.json').read_text()) for p in paths.values());continue
            tree=None;leaves=np.array([0]);book=None;minleaf=None
            if len(data)>=16:
                minleaf=max(8,min(63,len(data)//8))
                tree=DecisionTreeRegressor(max_leaf_nodes=8,min_samples_leaf=minleaf,random_state=2026).fit(data[FEATURES],data.target)
                leaves=np.unique(tree.apply(data[FEATURES]));states=np.searchsorted(leaves,tree.apply(data[FEATURES]))
                first=int(data.position.min()+1);last=int(cal.get_indexer(pd.DatetimeIndex(data.end)).max())
                book=np.zeros((last-first+1,len(leaves)*5))
                for pair,rows in data.groupby('pair',sort=False).groups.items():
                    rows=np.asarray(rows);disk=np.load(OUT/'pairs'/pair/silo/'labels.npy',mmap_mode='r')
                    labels=disk[data.loc[rows,'label_row'].to_numpy(int)]
                    aggregate_chunk(book,states[rows],labels,data.loc[rows,'position'].to_numpy(int),first)
                    del disk,labels
            for variant,levels in LADDERS.items():
                dest=paths[variant];dest.mkdir(parents=True,exist_ok=True)
                if book is None:
                    policy=(None,np.array([0]),np.array([4]));details=dict(allocations=[1.])
                else:policy,details=fit_sizes(book,tree,leaves,levels)
                row=dict(silo=silo,variant=variant,fit=fitdate,status='enabled' if book is not None else 'global_full_size_cold_start',
                    training_rows=len(data),training_pairs=int(data.pair.nunique()),
                    last_training_outcome=str(pd.Timestamp(data.end.max()).date()) if len(data) else None,
                    leaves=len(leaves),min_leaf=minleaf,split_features=[FEATURES[int(f)] for f in tree.tree_.feature if f>=0] if tree is not None else [],**details)
                joblib.dump(policy,dest/f'{fitdate}.joblib');write_json(dest/f'{fitdate}.json',row);logs.append(row)
            if month.month in [1,7]:print(f'Smile fits {silo} {fitdate}: {len(data)} rows / {data.pair.nunique()} pairs',flush=True)
            del tree,book,data;gc.collect()
        del meta;gc.collect()
    pd.DataFrame(logs).to_csv(REPORT/'fits.csv',index=False)


def replay(defs,market,cfg,sources):
    cal=market.calendar;evalcal=cal[cal>='2018-01-01'];cash=cash_book(evalcal,market.rates,cfg.capital)
    loader=SizingMarket(market,cfg,sources);models={};coverage=[]
    for cp,bucket in SILOS:
        silo=f'{cp}{bucket}'
        for variant in LADDERS:
            models[silo,variant]={p.stem:joblib.load(p) for p in (OUT/'models'/silo/variant).glob('*.joblib')}
    for n,d in enumerate(defs,1):
        engine=stock=None
        for cp,bucket in SILOS:
            silo=f'{cp}{bucket}';source=SIGNAL/'pairs'/d['pair']/silo/'spread/actions.csv';base=read_actions(source)
            state=pd.read_parquet(OUT/'pairs'/d['pair']/silo/'features.parquet')
            dates=state.index[state.observed&(state.index>='2018-01-01')&(state.index<=cal[-1]-pd.Timedelta(days=RUNOFF))]
            for variant in ['rule',*LADDERS]:
                dest=OUT/'pairs'/d['pair']/silo/variant;full=dest/'full';full.mkdir(parents=True,exist_ok=True)
                if variant=='rule' and not set(d['secids']).intersection(DIVIDEND_REPAIR_IDS):
                    write_json(dest/'reference.json',dict(source=str(source.parent/'full'),accounting_unchanged=True));continue
                allocation=pd.Series(index=dates,dtype=int,name='level');global_cold=0
                if variant=='rule':allocation[:]=4
                else:
                    for month in dates.to_period('M').unique():
                        idx=dates[dates.to_period('M')==month];model=models[silo,variant][str(month.start_time.date())]
                        allocation.loc[idx]=predict(model,state.loc[idx,FEATURES]);global_cold+=len(idx) if model[0] is None else 0
                actions=base.copy().rename('code')
                actions.loc[dates]=translate(allocation.to_numpy(int),base.reindex(dates,fill_value=0),cp,bucket)
                assert actions[base.eq(0)].eq(0).all()
                if variant=='floor':assert actions.ne(0).sum()==base.ne(0).sum()
                actions.to_csv(dest/'actions.csv');(allocation/4).rename('allocation').to_csv(dest/'scored_allocations.csv')
                if variant!='rule':
                    coverage.append(dict(pair=d['pair'],label=d['label'],silo=silo,variant=variant,scored_rows=len(dates),
                        global_cold_start_rows=global_cold,rule_signals=int(base.ne(0).sum()),
                        sized_signals=int(actions.ne(0).sum()),learned_skips=int((base.ne(0)&actions.eq(0)).sum())))
                if (full/'status.json').exists():continue
                if not actions.ne(0).any():write_json(full/'status.json',dict(status='complete',cash_only=True,entries=0));continue
                if engine is None:
                    loader.retain(d['secids']);assets=[loader.asset(sid) for sid in d['secids']]
                    q,stock,sel,divs=[pd.concat([a[c] for a in assets],ignore_index=True) for c in ['q','s','sel','divs']]
                    engine=AccountingReplay(q,stock,sel,d['secids'],divs,cfg)
                nav,trades,decisions=engine.run(actions,evalcal);mark_book(nav,market.rates,cfg.capital)
                if trades.empty:
                    checks=audit(nav,trades,stock,cfg,d['secids']);np.testing.assert_allclose(nav.nav,cash.nav,atol=1e-7,rtol=0)
                    decisions.to_csv(full/'decisions.csv',index=False)
                    write_json(full/'status.json',dict(status='complete',cash_only=True,entries=0,**checks))
                else:save(full,nav,trades,decisions,cfg,stock,d)
        write_json(OUT/'market_sources.json',sources);write_json(OUT/'progress.json',dict(stage='replay',pair=n,roster=len(defs),label=d['label']))
        if n%20==0:print(f'Smile replay {n}/{len(defs)} {d["label"]}',flush=True)
        del engine,stock;gc.collect()
    loader.cache.clear();gc.collect();pd.DataFrame(coverage).to_csv(REPORT/'scoring_coverage.csv',index=False)


def summarize(defs,market,cfg):
    cal=market.calendar[market.calendar>='2018-01-01'];cash=cash_book(cal,market.rates,cfg.capital)
    cash_portfolio=cash[['nav','cash_benchmark']]*len(defs);books={};rows=[];pair_rows=[];allocations=[];decisions=[];annual=[]
    for cp,bucket in SILOS:
        silo=f'{cp}{bucket}'
        for variant in ['rule',*LADDERS]:
            name=f'{silo}_{variant}';book=cash_portfolio.copy();entries=active=breaches=0
            costs=dict(commission=0.,option_cost=0.,hedge_cost=0.)
            for d in defs:
                root=OUT/'pairs'/d['pair']/silo/variant;full=root/'full'
                if (root/'reference.json').exists():full=Path(json.loads((root/'reference.json').read_text())['source'])
                status=json.loads((full/'status.json').read_text());assert status['status']=='complete'
                nav=cash if status.get('cash_only') else pd.read_parquet(full/'daily_nav.parquet');assert nav.index.equals(cal)
                book+=nav[['nav','cash_benchmark']]-cash[['nav','cash_benchmark']]
                entries+=status['entries'];active+=int(status['entries']>0);breaches+=status.get('risk_breach_sessions',0)
                for key in costs:costs[key]+=status.get(key,0.)
                pair_rows.append(dict(model=name,pair=d['pair'],label=d['label'],entries=status['entries'],net_excess_pnl=float(nav.nav.iloc[-1]-nav.cash_benchmark.iloc[-1])))
                if variant!='rule':
                    alloc=pd.read_csv(root/'scored_allocations.csv',index_col=0,parse_dates=[0]).allocation
                    base=read_actions(SIGNAL/'pairs'/d['pair']/silo/'spread/actions.csv')
                    for level,count in alloc[base.reindex(alloc.index,fill_value=0).ne(0)].value_counts().items():allocations.append(dict(model=name,pair=d['pair'],allocation=float(level),signals=int(count)))
                for key,count in status.get('decision_counts',{}).items():decisions.append(dict(model=name,pair=d['pair'],decision=key,count=count))
            books[name]=book;rows.append(dict(model=name,entries=entries,active_pairs=active,risk_breach_pair_sessions=breaches,**costs))
    for name,path in [('put25_frozen',PUT),('atm_frozen',ATM)]:
        book=pd.read_csv(path,index_col=0,parse_dates=[0],float_precision='round_trip');assert book.index.equals(cal)
        np.testing.assert_allclose(book.cash_benchmark,cash_portfolio.cash_benchmark,atol=1e-5,rtol=0)
        books[name]=book[['nav','cash_benchmark']]
    combinations={'ATM_plus_put25':['atm_frozen','put25_frozen']}
    for silo in ['C25','P75']:
        for variant in ['rule',*LADDERS]:
            combinations[f'put25_plus_{silo}_{variant}']=['put25_frozen',f'{silo}_{variant}']
            combinations[f'whole_smile_{silo}_{variant}']=['atm_frozen','put25_frozen',f'{silo}_{variant}']
    for name,members in combinations.items():books[name]=sum(books[m][['nav','cash_benchmark']] for m in members)
    detail={r['model']:r for r in rows};comparison=[];concentration=[]
    for name,book in books.items():
        book=book.copy();book['return_']=book.nav.pct_change().fillna(0);book['excess_return']=book.return_-book.cash_benchmark.pct_change().fillna(0)
        book.to_csv(REPORT/f'{name}_full_portfolio.csv')
        sleeve_count=len(combinations[name]) if name in combinations else 1
        excess=book.nav-book.cash_benchmark
        comparison.append(dict(model=name,funded_sleeve_sets=sleeve_count,
            net_excess_pnl=float(excess.iloc[-1]),max_excess_drawdown_dollars=float((excess-excess.cummax()).min()),
            **{k:v for k,v in detail.get(name,{}).items() if k!='model'},**scores(book,cfg.capital*len(defs)*sleeve_count)))
        for year,g in book.groupby(book.index.year):
            annual.append(dict(model=name,year=int(year),annualized_excess_sharpe=float(np.sqrt(252)*g.excess_return.mean()/g.excess_return.std()) if g.excess_return.std()>1e-12 else None,
                total_return_percent=100*((1+g.return_).prod()-1),excess_return_percent=100*((1+g.return_).prod()-(1+cash.return_.loc[g.index]).prod())))
        if name in detail:
            leaders=pd.DataFrame(pair_rows).query('model==@name').nlargest(3,'net_excess_pnl');stripped=book[['nav','cash_benchmark']].copy()
            silo,variant=name.split('_')
            for i,r in enumerate(leaders.itertuples(),1):
                if r.entries==0:continue
                root=OUT/'pairs'/r.pair/silo/variant;full=root/'full'
                if (root/'reference.json').exists():full=Path(json.loads((root/'reference.json').read_text())['source'])
                nav=pd.read_parquet(full/'daily_nav.parquet')
                stripped-=nav[['nav','cash_benchmark']]-cash[['nav','cash_benchmark']]
                if i in [1,3]:
                    stripped['return_']=stripped.nav.pct_change().fillna(0);stripped['excess_return']=stripped.return_-stripped.cash_benchmark.pct_change().fillna(0)
                    concentration.append(dict(model=name,winners_removed=i,**scores(stripped,cfg.capital*len(defs))))
    result=pd.DataFrame(comparison);result.to_csv(REPORT/'comparison.csv',index=False)
    pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False);pd.DataFrame(pair_rows).to_csv(REPORT/'pairs.csv',index=False)
    pd.DataFrame(allocations).to_csv(REPORT/'allocations_by_pair.csv',index=False)
    if allocations:pd.DataFrame(allocations).groupby(['model','allocation']).signals.sum().reset_index().to_csv(REPORT/'allocations.csv',index=False)
    pd.DataFrame(decisions).to_csv(REPORT/'decisions.csv',index=False);pd.DataFrame(concentration).to_csv(REPORT/'concentration.csv',index=False)
    pd.DataFrame({n:(b.nav.pct_change().fillna(0)-b.cash_benchmark.pct_change().fillna(0)) for n,b in books.items() if n not in combinations}).corr().to_csv(REPORT/'daily_correlations.csv')
    print(result[['model','annualized_excess_sharpe','net_excess_pnl','entries']].to_string(index=False),flush=True)
    # Source CSV serialization / re-reading can move daily-return Sharpe in
    # the twelfth decimal; frozen source files are separately hash-verified.
    np.testing.assert_allclose(result.loc[result.model.eq('put25_frozen'),'annualized_excess_sharpe'],.7920411985037874,atol=1e-10,rtol=0)
    np.testing.assert_allclose(result.loc[result.model.eq('atm_frozen'),'annualized_excess_sharpe'],.3680386150089804,atol=1e-10,rtol=0)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--finalize',action='store_true');args=parser.parse_args()
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    defs=json.loads((BASE/'inputs.json').read_text())['definitions'];sources={}
    for path in [SIGNAL/'inputs.json',SIGNAL/'market_sources.json',Path('artifacts/contracts/tail_put25_residual_rl/inputs.json'),Path('artifacts/contracts/tail_put25_residual_rl/market_sources.json')]:
        data=json.loads(path.read_text());sources.update(data.get('sources',data))
    for study in ['opposite_wing_signals','opposite_wing_signals_checks','tail_put25_residual_rl']:sources.update(json.loads(Path(f'reports/{study}/output_hashes.json').read_text()))
    sources.update(json.loads(Path('config/atm30_fee065_employee_2026-10-01_checkpoint.json').read_text())['files'])
    if (OUT/'market_sources.json').exists():sources.update(json.loads((OUT/'market_sources.json').read_text()))
    for path in [Path(__file__),Path('scripts/smile_sizing_policy.py'),Path('tests/test_smile_sizing.py'),ATM,PUT]:sources[str(path)]=digest(path)
    for path,sha in sources.items():assert digest(path)==sha,f'Frozen input changed: {path}'
    protocol='Two wing-specific monthly five-year pooled direct-Sharpe sizing fits; all345 roster pairs retained, all observed states scored. Deploy exactly the saved ungated 30D C25 and P75 spread-rule signals and directions. Ladders50/75/100 and0/25/50/75/100; no optional-feature or per-pair model activation veto; any global cold start keeps full size. Compact causal relative-IV/factor-IV/risk-reversal/right-slope features plus market context and completed native profit memory; memory is an input, not an entry veto. Training uses only completed standalone integer-contract paths with0.65 commission hurdle; actual replay pays0.03, mid+.1volpoint perleg, daily delta hedge, previous financing and risk limits, initial vega matching10pct, hold own expiry with documented corporate exits. Training context: original put-state dates pre2018, legacy pair/date calendar thereafter. All observed states remain scored outside that calendar. Frozen .792 put25 sizing and .368 ATM center are not retrained. Predeclared equal-capital combinations of funded sleeve ledgers, no optimized hindsight wing weights. P75 and C25 tested as alternative upside implementations, never both in a primary stack. Stacks use separately funded/risk-limited pair-wing accounts; cash and financing are additive, no shared-name hedge netting or global risk consolidation, not a common-inventory strategy. Training paths are additive standalone cohorts, not sequential inventory-aware RL. Inspected history, multiple comparisons, no new holdout; frozen formation/calendar conditioning and no American early assignment; TECK cash uses fixed declaration-time FX approximation.'
    protocol+=' Dividend cash tables are scoped per issuer; rebuild FCX/TECK label histories and native memory, and compare against full-size controls with the same repaired cash accounting. Previous opposite-wing study remains archived; its affected dividend flows are not reused for this matched comparison.'
    write_json(OUT/'inputs.json',dict(features=FEATURES,ladders=LADDERS,sources=sources,protocol=protocol))
    cal=load_calendar(ROOT);cal=cal[(cal>='2010-01-01')&(cal<='2025-08-29')]
    market=Market(cal);cfg=replace(config(),option_commission=.03)
    if not args.finalize:
        prepare(defs,market,cfg,sources);train(defs,cal);replay(defs,market,cfg,sources)
    result=summarize(defs,market,cfg)
    coverage=pd.read_csv(REPORT/'scoring_coverage.csv');expected=pd.read_csv(SIGNAL_REPORT/'coverage.csv').groupby('silo').observed_rows.sum()
    for (silo,variant),g in coverage.groupby(['silo','variant']):assert int(g.scored_rows.sum())==int(expected[silo])
    fits=pd.read_csv(REPORT/'fits.csv');assert len(fits)==360
    known=fits.last_training_outcome.notna();assert (pd.to_datetime(fits.loc[known,'last_training_outcome'])<pd.to_datetime(fits.loc[known,'fit'])).all()
    statuses=[json.loads(p.read_text()) for p in (OUT/'pairs').glob('*/*/*/full/status.json') if p.parent.parent.name in LADDERS];assert len(statuses)==1380 and all(s['status']=='complete' for s in statuses)
    for path,sha in sources.items():assert digest(path)==sha,f'Frozen source changed: {path}'
    write_json(OUT/'market_sources.json',sources)
    record=dict(status='complete',models=len(fits),roster_pairs=len(defs),strategy_books=len(statuses),
        scored_rows_by_wing_and_variant=[dict(silo=s,variant=v,rows=int(n)) for (s,v),n in coverage.groupby(['silo','variant']).scored_rows.sum().items()],
        global_cold_start_months=int(fits.status.ne('enabled').sum()),source_files_verified=len(sources),
        frozen_atm_and_put25_reproduced=True,combined_shared_inventory=False,
        finalization_only=args.finalize,
        elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_json(REPORT/'audit.json',record);(REPORT/'findings.txt').write_text(result.to_string(index=False)+'\n\n'+json.dumps(record,indent=2)+'\n\n'+protocol)
    hashes={str(p):digest(p) for root in [OUT,REPORT] for p in root.rglob('*') if p.is_file() and p.name!='output_hashes.json'}
    for path in [Path(__file__),Path('scripts/smile_sizing_policy.py'),Path('tests/test_smile_sizing.py')]:hashes[str(path)]=digest(path)
    write_json(REPORT/'output_hashes.json',hashes)


if __name__=='__main__':main()
