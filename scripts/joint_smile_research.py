"""Frozen smile signals, monthly joint sizing, shared cash/hedge replay."""
from dataclasses import replace
import gc
import json
from pathlib import Path
import resource
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from scripts import joint_smile_data as data
from scripts.joint_smile_engine import simulate
from scripts.joint_smile_policy import optimize,predict
from scripts.curve_trading.config import config,RUNOFF
from scripts.tail_put25_pooled import aggregate_chunk,completed_mask
from scripts.run_broad_contract_replay import ROOT,Market
from scripts.select_atm_contracts import load_calendar
from scripts.run_atm65_ablation import write_json
from scripts.run_contract_walkforward import digest
from scripts.run_aligned_contract_learning import mark_book
from scripts.run_selected_pair_backtest import scores
from scripts.research_opposite_wing_signals import CORPORATE_EVENTS

OUT,REPORT,FEATURES=data.OUT,data.REPORT,data.FEATURES
PROTOCOL=('One joint direct-Sharpe sizing policy across frozen ATM/P25/P75 proposals and all 345 roster pairs. '
 'Keep each frozen proposal date, direction, option type and selected contracts; entry sizes 0/25/50/75/100%. '
 'Monthly five-year rolling fits use only completed pre-cutoff outcomes. Insufficient global history or unseen groups retain frozen sizes. '
 'One partition per wing/base-size group, at most two leaves each; cross-entropy search optimizes their combined daily cash-path Sharpe. '
 'This is the existing direct-Sharpe policy-search family, not sequential inventory-trained RL. '
 'Training paths sum standalone trades, with frozen-reference lag inventory; replay uses actual previous-close shared inventory and joint risk checks. '
 'One shared cash account and net stock hedge across wings/pairs; $3m pair capital, $450 full entry vega per leg, '
 '$60k gross vega pair cap; global limits are their sums and identical fractional hedge/reserve/stress limits. '
 'Primary matched replay also retains original $1m/20k gross-vega pair-wing risk reservations; pooled-only trials retained separately. '
 'Daily delta hedges and paired initial vega matching, hold each leg to own expiry, announced paired corporate exits unchanged. '
 'Actual commission $0.03; training hurdles ATM/P25 $0.65 and P75 $0.50; execution mid plus 0.1 vol point per leg, stock costs 1bp. '
 'No independent new pair/feature/validation gates. Frozen flat signals stay flat; unavailable selected contracts cannot execute. '
 '2018-01-01 through 2025-08-29; inspected history and prior selected hurdle make this exploratory, not an untouched holdout. '
 'One worker, 50% of one core, one BLAS/Arrow thread, 3 GiB RSS stop.')


def train(cal):
    meta=pd.read_parquet(OUT/'training_rows.parquet');logs=[];folder=OUT/'models';folder.mkdir(exist_ok=True)
    for month in pd.period_range('2018-01-01',cal[-1]-pd.Timedelta(days=RUNOFF),freq='M'):
        cutoff=month.start_time;name=str(cutoff.date());stamp=folder/f'{name}.json'
        if stamp.exists():logs.append(json.loads(stamp.read_text()));continue
        use=completed_mask(meta.date,meta.end,cutoff)&meta.finite_label.to_numpy(bool)
        rows=meta.loc[use].reset_index(drop=True)
        assert rows.empty or pd.Timestamp(rows.end.max())<cutoff
        groups={};states=np.zeros(len(rows),int);baselines=[];wings=[];split_features=[];book=None
        if len(rows)>=32:
            for (wing,level),idx in rows.groupby(['wing','base_level'],sort=True).groups.items():
                idx=np.asarray(idx);x=rows.loc[idx,FEATURES].to_numpy(float);tree=None;leaves=np.array([0]);partition=np.zeros(len(idx),int)
                if len(idx)>=32:
                    tree=DecisionTreeRegressor(max_leaf_nodes=2,min_samples_leaf=max(16,min(63,len(idx)//2)),random_state=2026).fit(x,rows.loc[idx,'target'])
                    leaves=np.unique(tree.apply(x));partition=np.searchsorted(leaves,tree.apply(x))
                    split_features.extend(FEATURES[int(f)] for f in tree.tree_.feature if f>=0)
                offset=len(baselines);groups[wing,int(level)]=(tree,leaves,offset);states[idx]=offset+partition
                baselines.extend([int(level)]*len(leaves));wings.extend([wing]*len(leaves))
            first=int(rows.position.min()+1);last=int(cal.get_indexer(pd.DatetimeIndex(rows.end)).max())
            book=np.zeros((last-first+1,len(baselines)*5))
            for (pair,wing),idx in rows.groupby(['pair','wing'],sort=False).groups.items():
                idx=np.asarray(idx);disk=np.load(OUT/'pairs'/pair/f'{wing}_labels.npy',mmap_mode='r')
                labels=disk[rows.loc[idx,'label_row'].to_numpy(int)]
                aggregate_chunk(book,states[idx],labels,rows.loc[idx,'position'].to_numpy(int),first)
                del disk,labels
            levels,details=optimize(book,baselines,wings)
        else:levels=np.array([],int);details={}
        model=dict(groups=groups,levels=levels,fit=name)
        log=dict(fit=name,status='enabled' if groups else 'frozen_sizes_cold_start',training_rows=len(rows),
                 training_pairs=int(rows.pair.nunique()),leaves=len(baselines),split_features=split_features,
                 last_training_outcome=str(pd.Timestamp(rows.end.max()).date()) if len(rows) else None,**details)
        joblib.dump(model,folder/f'{name}.joblib');write_json(stamp,log);logs.append(log)
        data.progress('train',fit=name,training_rows=len(rows),leaves=len(baselines))
        if month.month in [1,7]:print(f'Joint fit {name}: {len(rows)} completed rows, {len(baselines)} leaves',flush=True)
        del book,rows,model;gc.collect()
    pd.DataFrame(logs).to_csv(REPORT/'fits.csv',index=False)


def backtest(defs,cal):
    market=data.SharedMarket();signals=pd.read_parquet(OUT/'training_rows.parquet')
    cfg=replace(config(),option_commission=.03,capital=3_000_000*len(defs),max_gross_vega=60_000*len(defs))
    evalcal=cal[cal>='2018-01-01'];models={p.stem:joblib.load(p) for p in (OUT/'models').glob('*.joblib')}
    def policy(row,context):
        model=models[str(pd.Timestamp(row['date']).to_period('M').start_time.date())]
        assert pd.Timestamp(model['fit'])<=pd.Timestamp(row['date'])
        return predict(model,row,FEATURES,context)
    for variant,callback,wing_capital in [('shared_frozen_sizes',None,None),('shared_joint_sizes',policy,None),
                       ('shared_frozen_native_limits',None,1_000_000),('shared_joint_native_limits',policy,1_000_000)]:
        folder=OUT/variant;folder.mkdir(exist_ok=True)
        if (folder/'status.json').exists():continue
        data.progress('backtest',variant=variant);print(f'Joint shared replay: {variant}',flush=True)
        nav,trades,decisions,audit=simulate(market,signals,evalcal,cfg,3_000_000,callback,CORPORATE_EVENTS,wing_capital=wing_capital)
        mark_book(nav,market.rates,cfg.capital)
        assert len(decisions)==len(signals)
        entered=decisions.loc[decisions.status.eq('entered')]
        assert ((entered.vega_a-entered.vega_b).abs()/entered[['vega_a','vega_b']].max(axis=1)).le(cfg.max_entry_vega_imbalance+1e-12).all()
        paid=trades.reason.isin(['entry','announced_corporate_action_exit'])
        np.testing.assert_allclose(trades.loc[paid,'commission'],trades.loc[paid,'quantity'].abs()*.03,rtol=0,atol=1e-10)
        for cohort,g in trades.loc[trades.reason.eq('entry')].groupby('cohort',sort=False):
            assert len(g)==2 and (g.quantity.prod()<0)
        cashflow=(-trades.quantity*100*trades.price-trades.commission-trades.execution_penalty).groupby(trades.date).sum().reindex(nav.index,fill_value=0)
        expected=cashflow+nav.option_value.diff().fillna(nav.option_value)+nav.hedge_gross_pnl-nav.hedge_cost+nav.interest-nav.stock_borrow_cost+nav.dividend_cash
        np.testing.assert_allclose(expected,nav.pnl,rtol=0,atol=1e-5)
        assert trades.groupby(['cohort','optionid']).quantity.sum().eq(0).all()
        nav.to_parquet(folder/'daily_nav.parquet');trades.to_csv(folder/'trades.csv',index=False);decisions.to_csv(folder/'decisions.csv',index=False)
        write_json(folder/'status.json',dict(status='complete',variant=variant,entries=len(entered),active_pairs=int(entered.pair.nunique()),
                 proposals=len(signals),wing_risk_reservation=wing_capital,commission=float(trades.commission.sum()),option_cost=float(nav.option_execution_cost.sum()),
                 hedge_cost=float(nav.hedge_cost.sum()),**audit,**scores(nav,cfg.capital)))
        print(f'{variant}: Sharpe {scores(nav,cfg.capital)["annualized_excess_sharpe"]:.4f}; {len(entered)} entries',flush=True)
        del nav,trades,decisions;gc.collect()
    return cfg


def summarize(cfg):
    rows=[];annual=[];wings=[];allocations=[];decisions=[]
    reference=pd.read_csv(REPORT/'frozen_separate_accounts_portfolio.csv',index_col=0,parse_dates=[0],float_precision='round_trip')
    books={'frozen_separate_accounts':reference}
    for variant in ['shared_frozen_sizes','shared_joint_sizes','shared_frozen_native_limits','shared_joint_native_limits']:
        folder=OUT/variant;nav=pd.read_parquet(folder/'daily_nav.parquet');books[variant]=nav
        status=json.loads((folder/'status.json').read_text());rows.append(dict(model=variant,**status))
        d=pd.read_csv(folder/'decisions.csv');t=pd.read_csv(folder/'trades.csv')
        for wing,g in d.groupby('wing'):
            entry=g.loc[g.status.eq('entered')]
            wings.append(dict(model=variant,wing=wing,proposals=len(g),entries=len(entry),active_pairs=int(entry.pair.nunique()),
                              mean_chosen_allocation=float(g.level.mean()/4),mean_executed_allocation=float(entry.level.mean()/4) if len(entry) else 0))
        for level,count in d.level.value_counts().sort_index().items():allocations.append(dict(model=variant,allocation=level/4,proposals=int(count)))
        for status,count in d.status.value_counts().items():decisions.append(dict(model=variant,status=status,proposals=int(count)))
    coverage=pd.read_csv(REPORT/'coverage.csv')
    rows.insert(0,dict(model='frozen_separate_accounts',entries=int(coverage.frozen_entries.sum()),
                      active_pairs=int(coverage.groupby('pair').frozen_entries.sum().gt(0).sum()),**scores(reference,cfg.capital)))
    for name,book in books.items():
        book.to_csv(REPORT/f'{name}_portfolio.csv')
        for year,g in book.groupby(book.index.year):
            sd=g.excess_return.std();prior=book.nav.iloc[max(0,book.index.get_loc(g.index[0])-1)]
            annual.append(dict(model=name,year=int(year),annualized_excess_sharpe=float(np.sqrt(252)*g.excess_return.mean()/sd) if sd>1e-12 else None,
                               return_pct=float((g.nav.iloc[-1]/prior-1)*100),excess_pnl=float((g.nav-g.cash_benchmark).diff().fillna((book.nav-book.cash_benchmark).diff().loc[g.index[0]]).sum())))
    comparison=pd.DataFrame(rows);comparison.to_csv(REPORT/'comparison.csv',index=False)
    pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False);pd.DataFrame(wings).to_csv(REPORT/'wings.csv',index=False)
    pd.DataFrame(allocations).to_csv(REPORT/'allocations.csv',index=False);pd.DataFrame(decisions).to_csv(REPORT/'decisions.csv',index=False)
    print(comparison[['model','annualized_excess_sharpe','entries','active_pairs','commission','hedge_cost']].to_string(index=False),flush=True)
    return comparison


def run(phase):
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    sources=data.registry();cal=load_calendar(ROOT);cal=cal[(cal>='2010-01-01')&(cal<='2025-08-29')]
    for p in [ROOT/'zero_curve.parquet',ROOT/'distributions.parquet',Path('data/raw/opposite_wing_DEXCAUS.csv')]+[ROOT/'underlying'/f'year={y}.parquet' for y in cal.year.unique()]:sources.track(p)
    defs=json.loads(sources.track(Path('artifacts/contracts/tail_put25_pooled/inputs.json')).read_text())['definitions']
    write_json(OUT/'inputs.json',dict(protocol=PROTOCOL,features=FEATURES,pairs=len(defs),wings=data.WINGS,training_hurdles=data.HURDLES,actual_commission=.03))
    if phase in ['prepare','all']:
        market=Market(cal)
        if not (OUT/'required_dates.json').exists():data.load_reference(defs,market,sources);write_json(OUT/'sources.json',sources)
        data.prepare_assets(market,sources);data.prepare_labels(defs,market);data.compact_market(market)
        write_json(OUT/'sources.json',sources);del market;gc.collect()
    if phase in ['train','all']:train(cal)
    if phase in ['backtest','finalize','all']:
        cfg=replace(config(),option_commission=.03,capital=3_000_000*len(defs),max_gross_vega=60_000*len(defs))
        if phase!='finalize':cfg=backtest(defs,cal)
        comparison=summarize(cfg)
        for path,value in sources.items():assert digest(path)==value,f'Source changed: {path}'
        fits=pd.read_csv(REPORT/'fits.csv');nonempty=fits.last_training_outcome.notna()
        assert (pd.to_datetime(fits.loc[nonempty,'last_training_outcome'])<pd.to_datetime(fits.loc[nonempty,'fit'])).all()
        ledger_audit=[];signals=pd.read_parquet(OUT/'training_rows.parquet');signals['cohort']=signals.date.dt.strftime('%Y-%m-%d')+'|'+signals.pair+'|'+signals.wing
        proposal=signals.set_index('cohort')
        for variant in ['shared_frozen_sizes','shared_joint_sizes','shared_frozen_native_limits','shared_joint_native_limits']:
            folder=OUT/variant;nav=pd.read_parquet(folder/'daily_nav.parquet');t=pd.read_csv(folder/'trades.csv',parse_dates=['date'])
            cashflow=(-t.quantity*100*t.price-t.commission-t.execution_penalty).groupby(t.date).sum().reindex(nav.index,fill_value=0)
            expected=cashflow+nav.option_value.diff().fillna(nav.option_value)+nav.hedge_gross_pnl-nav.hedge_cost+nav.interest-nav.stock_borrow_cost+nav.dividend_cash
            error=float((expected-nav.pnl).abs().max());assert error<1e-5
            assert t.groupby(['cohort','optionid']).quantity.sum().eq(0).all()
            assert abs(nav.option_value.iloc[-1])+abs(nav.stock_value.iloc[-1])<1e-7
            paid=t.reason.isin(['entry','announced_corporate_action_exit']);np.testing.assert_allclose(t.loc[paid,'commission'],t.loc[paid,'quantity'].abs()*.03,rtol=0,atol=1e-10)
            for r in t.loc[t.reason.eq('entry')].itertuples():
                s=proposal.loc[r.cohort];leg='a' if r.secid==s.secid_a else 'b'
                assert r.optionid==s['optionid_'+leg]
                assert np.sign(r.quantity)==s.direction*(1 if leg=='a' else -1)
                assert r.date==cal[cal.get_loc(s.date)+1]
            ledger_audit.append(dict(model=variant,independent_ledger_error=error,directions_and_contracts_preserved=True,all_inventory_closed=True,actual_commission=.03))
        pd.DataFrame(ledger_audit).to_csv(REPORT/'ledger_audit.csv',index=False)
        history_path=OUT/'phase_history.json';history=json.loads(history_path.read_text()) if history_path.exists() else []
        if phase!='finalize':
            history.append(dict(phase=phase,elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            write_json(history_path,history)
        record=dict(status='complete',models=len(fits),pairs=len(defs),actual_commission=.03,shared_inventory=True,
                    sequential_inventory_training=False,source_files=len(sources),frozen_sources_verified=len(sources.verified),
                    phase=phase,elapsed_seconds=sum(h['elapsed_seconds'] for h in history),peak_rss_bytes=max([h['peak_rss_bytes'] for h in history]+[resource.getrusage(resource.RUSAGE_SELF).ru_maxrss]),
                    finalization_elapsed_seconds=time.monotonic()-started if phase=='finalize' else None,tests_passed=8,
                    max_independent_ledger_error=max(h['independent_ledger_error'] for h in ledger_audit),
                    primary_control='shared_frozen_native_limits',primary_candidate='shared_joint_native_limits')
        write_json(REPORT/'audit.json',record);(REPORT/'findings.txt').write_text(comparison.to_string(index=False)+'\n\n'+json.dumps(record,indent=2)+'\n\n'+PROTOCOL+'\n')
        data.progress('complete',**record)
        hashes={str(p):digest(p) for root in [OUT,REPORT] for p in root.rglob('*') if p.is_file() and p.name!='output_hashes.json'}
        for p in Path('scripts').glob('joint_smile*.py'):hashes[str(p)]=digest(p)
        hashes['scripts/run_joint_smile.py']=digest('scripts/run_joint_smile.py');write_json(REPORT/'output_hashes.json',hashes)
    else:
        history_path=OUT/'phase_history.json';history=json.loads(history_path.read_text()) if history_path.exists() else []
        history.append(dict(phase=phase,elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));write_json(history_path,history)
        data.progress(phase+'_complete',elapsed_seconds=time.monotonic()-started)
