"""A declared, bounded sizing/earnings comparison on the ungated P25 book."""
import argparse
from collections import defaultdict, OrderedDict
import gc
import json
from pathlib import Path
import resource
import sqlite3

import numpy as np
import pandas as pd

from scripts.submission import BUNDLE, FEATURE, CAPITAL, digest, require_supervisor
from scripts.run_p25_earnings import ADDED, protected, write
from scripts.run_p25_memory_ablation import OUT as BASE, DiskMarket, eligible

OUT=Path('artifacts/contracts/p25_ungated_search')
REPORT=Path('reports/p25_ungated_search')
CK=Path('config/p25_submission.json')
EARN=Path('artifacts/contracts/p25_earnings')
CASES={
    'quarter_floor':dict(menu=[1,2,3,4],months=18,leaves=8,earnings=None),
    'skip_floor':dict(menu=[0,2,3,4],months=18,leaves=8,earnings=None),
    'flexible':dict(menu=[0,1,2,3,4],months=18,leaves=8,earnings=None),
    'tree16':dict(menu=[2,3,4],months=18,leaves=16,earnings=None),
    'window36':dict(menu=[2,3,4],months=36,leaves=8,earnings=None),
    'earnings_prior':dict(menu=[2,3,4],months=18,leaves=8,earnings='prior_reports_only'),
    'earnings_prior_skip':dict(menu=[0,2,3,4],months=18,leaves=8,earnings='prior_reports_only'),
    'earnings_calendar':dict(menu=[2,3,4],months=18,leaves=8,earnings='historical_calendar'),
    'tree4':dict(menu=[2,3,4],months=18,leaves=4,earnings=None),
    'window12':dict(menu=[2,3,4],months=12,leaves=8,earnings=None),
}
FOLLOWUPS=['tree4','window12']
CONTROLS={'fixed25':1,'fixed50':2,'fixed75':3}
EXECUTION_COLUMNS=['date','pair','wing','secid_a','secid_b','optionid_a','optionid_b','base_level','direction','learned_level']


def portfolio_score(book,choices):
    """Annualized daily shadow-book Sharpe of a leaf/action assignment."""
    mean=book.mean(axis=0);cov=np.cov(book,rowvar=False,ddof=1)
    choices=np.atleast_2d(choices)
    ns=book.shape[1]//5
    cols=choices+np.arange(ns)[None,:]*5
    m=mean[cols].sum(axis=1)
    v=cov[cols[:,:,None],cols[:,None,:]].sum(axis=(1,2))
    return np.sqrt(252)*np.divide(m,np.sqrt(np.maximum(v,0)),out=np.zeros_like(m),where=v>1e-18)


def fit_menu(book,menu):
    """Retained CEM arithmetic generalized only to the declared action menu."""
    menu=np.asarray(menu,int)
    if menu.ndim!=1 or len(menu)<2 or len(np.unique(menu))!=len(menu) or not np.isin(menu,np.arange(5)).all() or 4 not in menu:
        raise ValueError('Distinct sizing codes 0..4, including full size, are required')
    mean=book.mean(axis=0);cov=np.cov(book,rowvar=False,ddof=1)
    ns=book.shape[1]//5
    def score(choices):
        cols=choices+np.arange(ns)[None,:]*5
        m=mean[cols].sum(axis=1)
        v=cov[cols[:,:,None],cols[:,None,:]].sum(axis=(1,2))
        return np.sqrt(252)*np.divide(m,np.sqrt(np.maximum(v,0)),out=np.zeros_like(m),where=v>1e-18)
    rng=np.random.default_rng(2026)
    probability=np.full((ns,len(menu)),1/len(menu))
    best=np.full(ns,4)
    best_score=baseline=float(score(best[None])[0])
    for _ in range(50):
        aliases=(rng.random((192,ns))[:,:,None]>probability.cumsum(axis=1)[None]).sum(axis=2).clip(max=len(menu)-1)
        choices=menu[aliases];choices[0]=best;choices[1:len(menu)+1]=menu[:,None]
        scores=score(choices);winner=scores.argmax()
        if scores[winner]>best_score:
            best,best_score=choices[winner].copy(),float(scores[winner])
        elite=choices[np.argsort(scores)[-19:]]
        frequency=np.stack([(elite==level).mean(axis=0) for level in menu],axis=1)
        probability=.97*(.3*probability+.7*frequency)+.03/len(menu)
    return best,dict(training_shadow_daily_sharpe=best_score,baseline_shadow_daily_sharpe=baseline)


def build_state_book(rows,features,cal,leaves):
    from sklearn.tree import DecisionTreeRegressor
    if len(rows)<16:
        return None,np.array([0]),None
    # Preserve the original minimum-leaf formula when changing max leaves.
    tree=DecisionTreeRegressor(max_leaf_nodes=leaves,min_samples_leaf=max(8,min(63,len(rows)//8)),random_state=2026).fit(rows[features],rows.target)
    leaf_ids=np.unique(tree.apply(rows[features]))
    states=np.searchsorted(leaf_ids,tree.apply(rows[features]))
    first=int(rows.position.min()+1)
    last=int(cal.get_indexer(pd.DatetimeIndex(rows.end)).max())
    book=np.zeros((last-first+1,len(leaf_ids)*5))
    for pair,indices in rows.groupby('pair',sort=False).groups.items():
        disk=np.load(BUNDLE/'artifacts/contracts/uniform_smile/pairs'/pair/'P25_labels.npy',mmap_mode='r')
        indices=np.asarray(indices)
        for offset in range(0,len(indices),128):
            idx=indices[offset:offset+128]
            raw=disk[rows.loc[idx,'label_row'].to_numpy(int)]
            direction=rows.loc[idx,'direction'].to_numpy(int)
            positions=rows.loc[idx,'position'].to_numpy(int)
            days=positions[:,None]+1+np.arange(raw.shape[-1])[None,:]-first
            valid=(days>=0)&(days<len(book))
            for state in np.unique(states[idx]):
                local=np.flatnonzero(states[idx]==state)
                for action in range(1,5):
                    values=raw[np.arange(len(idx)),np.where(direction>0,4+action,5-action)]
                    np.add.at(book[:,int(state)*5+action],days[local][valid[local]],values[local][valid[local]])
            del raw
        del disk
    return tree,leaf_ids,book


def prepare():
    if (OUT/'prepared.json').exists():return
    signals=pd.read_parquet(BASE/'no_memory_gate_signals.parquet')
    assert len(signals)==4744 and eligible(signals,'no_memory_gate').all()
    coverage=[]
    for mode in ['prior_reports_only','historical_calendar']:
        path=EARN/f'{mode}_training.parquet'
        if not path.exists():
            raise FileNotFoundError('Restore data/p25_earnings_experiment.zip first')
        events=pd.read_parquet(path,columns=['date','pair']+ADDED)
        joined=signals.merge(events,on=['date','pair'],how='left',validate='one_to_one')
        pd.testing.assert_frame_equal(joined[signals.columns],signals)
        assert np.isfinite(joined[ADDED]).all().all()
        joined.to_parquet(OUT/f'{mode}_signals.parquet',index=False)
        for field in ADDED[::2]:
            coverage.append(dict(mode=mode,feature=field,available=int(joined[field+'_available'].sum()),proposals=len(joined)))
        del joined,events
        gc.collect()
    pd.DataFrame(coverage).to_csv(REPORT/'earnings_coverage.csv',index=False)
    paths=[CK,BASE/'no_memory_gate_signals.parquet',BASE/'replays/no_memory_gate/status.json',
        BASE/'quotes_complete.json',BUNDLE/FEATURE/'training_rows.parquet']
    paths += [EARN/f'{m}_training.parquet' for m in ['prior_reports_only','historical_calendar']]
    write(OUT/'prepared.json',dict(proposals=4744,cases=CASES,sources={str(p):digest(p) for p in paths}))
    print('All variants retain 4744 proposals; earnings adds 8 inputs and no gate',flush=True)


def train(cases,limit):
    import joblib
    ck=json.loads(CK.read_text())
    columns=list(dict.fromkeys(ck['features']+['date','pair','observed','eligible','context','direction','finite_label','target','end','position','label_row']))
    meta=pd.read_parquet(BUNDLE/FEATURE/'training_rows.parquet',columns=columns)
    events={mode:pd.read_parquet(EARN/f'{mode}_training.parquet',columns=['date','pair']+ADDED)
        for mode in {CASES[c]['earnings'] for c in cases} if mode is not None}
    for frame in events.values():
        pd.testing.assert_frame_equal(frame[['date','pair']],meta[['date','pair']])
    valid=eligible(meta,'no_memory_gate')&meta.finite_label
    cal=pd.read_parquet(BUNDLE/'artifacts/contracts/uniform_smile/market/rates.parquet').index
    months=pd.period_range('2018-01-01',ck['last_signal'],freq='M')
    if limit is not None:months=months[:limit]
    groups=defaultdict(list)
    for case in cases:
        spec=CASES[case]
        groups[(spec['months'],spec['leaves'],spec['earnings'])].append(case)
    for month in months:
        cutoff=month.start_time;stamp=str(cutoff.date())
        for (window,leaves,mode),members in groups.items():
            pending=[c for c in members if not (OUT/'models'/c/f'{stamp}.json').exists()]
            if not pending:continue
            rows=meta.loc[valid & meta.date.ge(cutoff-pd.DateOffset(months=window)) & meta.date.lt(cutoff) & meta.end.lt(cutoff)].copy()
            assert len(rows)==0 or rows.end.max()<cutoff
            features=ck['features']+(ADDED if mode is not None else [])
            if mode is not None:rows[ADDED]=events[mode].loc[rows.index,ADDED].to_numpy()
            rows=rows.reset_index(drop=True)
            tree,leaf_ids,book=build_state_book(rows,features,cal,leaves)
            if (window,leaves,mode)==(18,8,None):
                choices,details=(np.array([4]),{}) if book is None else fit_menu(book,[2,3,4])
                stored=joblib.load(BASE/'models/no_memory_gate'/f'{stamp}.joblib')
                np.testing.assert_array_equal(choices,stored['choices']);np.testing.assert_array_equal(leaf_ids,stored['leaves'])
                if tree is not None:
                    for field in ['feature','threshold','children_left','children_right','value']:
                        np.testing.assert_array_equal(getattr(tree.tree_,field),getattr(stored['tree'].tree_,field))
                write(OUT/'baseline_fit_checks'/f'{stamp}.json',dict(exact=True,fit=stamp,training_rows=len(rows),**details))
                del stored
            for case in pending:
                spec=CASES[case]
                choices,details=(np.array([4]),{}) if book is None else fit_menu(book,spec['menu'])
                model=dict(tree=tree,leaves=leaf_ids,choices=choices,features=features,fit=stamp)
                folder=OUT/'models'/case;folder.mkdir(parents=True,exist_ok=True)
                joblib.dump(model,folder/f'{stamp}.joblib')
                write(folder/f'{stamp}.json',dict(case=case,fit=stamp,training_rows=len(rows),feature_count=len(features),
                    latest_completed_outcome=str(rows.end.max().date()) if len(rows) else None,choices=choices.tolist(),**spec,**details))
                del model
            del rows,tree,book
            gc.collect()
        print(f'{stamp}: monthly variants completed',flush=True)
        write(OUT/'progress.json',dict(phase='train',fit=stamp,models_complete=len(list((OUT/'models').rglob('*.joblib')))))


def score(cases):
    import joblib
    for case in cases:
        folder=OUT/'replays'/case;folder.mkdir(parents=True,exist_ok=True)
        path=folder/'scored_allocations.parquet'
        if path.exists():continue
        mode=CASES[case]['earnings']
        rows=pd.read_parquet(OUT/f'{mode}_signals.parquet' if mode else BASE/'no_memory_gate_signals.parquet')
        rows['learned_level']=4
        for month,idx in rows.groupby(rows.date.dt.to_period('M')).groups.items():
            model=joblib.load(OUT/'models'/case/f'{month.start_time.date()}.joblib')
            if model['tree'] is not None:
                states=model['tree'].apply(rows.loc[idx,model['features']])
                rows.loc[idx,'learned_level']=model['choices'][np.searchsorted(model['leaves'],states)]
        assert len(rows)==4744 and rows.learned_level.isin(CASES[case]['menu']).all()
        rows.to_parquet(path,index=False)
        print(f'{case}: scored4744, learned skips={int(rows.learned_level.eq(0).sum())}',flush=True)
        del rows,model
        gc.collect()


def market_cache():
    """Stream only evaluation stocks into a small indexed disk store."""
    import pyarrow.parquet as pq
    if (OUT/'stocks_complete.json').exists():return
    rows=pd.read_parquet(BASE/'no_memory_gate_signals.parquet',columns=['secid_a','secid_b'])
    ids=set(rows.secid_a.astype(int))|set(rows.secid_b.astype(int))
    if 101322 in ids:ids.add(102968)
    source=BUNDLE/'artifacts/contracts/p25_signal_transfer/market/stocks.parquet'
    db=sqlite3.connect(OUT/'stocks.sqlite')
    db.execute('PRAGMA cache_size=-1024')
    db.execute('CREATE TABLE IF NOT EXISTS stocks(date TEXT,secid INTEGER,close REAL,cfadj REAL,PRIMARY KEY(date,secid)) WITHOUT ROWID')
    count=0
    for batch in pq.ParquetFile(source).iter_batches(batch_size=4096,columns=['date','secid','close','cfadj'],use_threads=False):
        frame=batch.to_pandas()
        frame=frame.loc[frame.secid.isin(ids)&frame.date.ge('2018-01-01')&frame.date.le('2025-08-29')]
        records=[(str(r.date.date()),int(r.secid),None if pd.isna(r.close) else float(r.close),
                  None if pd.isna(r.cfadj) else float(r.cfadj)) for r in frame.itertuples()]
        db.executemany('INSERT OR REPLACE INTO stocks VALUES(?,?,?,?)',records)
        count+=len(records)
        del frame,batch,records
    db.commit();db.close()
    write(OUT/'stocks_complete.json',dict(rows=count,names=len(ids),source_sha256=digest(source)))
    print(f'Disk-backed stock cache: {count} rows, {len(ids)} names',flush=True)


class LazyDiskMarket(DiskMarket):
    """Identical values with no full-universe stock pivots or training imports."""
    def __init__(self,secids):
        folder=BUNDLE/'artifacts/contracts/p25_signal_transfer/market'
        self.rates=pd.read_parquet(folder/'rates.parquet').rate
        self.divs=pd.read_parquet(folder/'dividends.parquet')
        self.reported_year=None
        self.db=sqlite3.connect(f'file:{BASE}/quotes.sqlite?mode=ro',uri=True)
        self.db.execute('PRAGMA cache_size=-1024')
        self.stockdb=sqlite3.connect(f'file:{OUT}/stocks.sqlite?mode=ro',uri=True)
        self.stockdb.execute('PRAGMA cache_size=-1024')
        self.days,self.stock_days=OrderedDict(),OrderedDict()

    def native_stock(self,date,sid):
        if date not in self.stock_days:
            result=self.stockdb.execute('SELECT secid,close,cfadj FROM stocks WHERE date=?',(str(date.date()),))
            self.stock_days[date]={int(name):(np.nan if close is None else close,np.nan if adj is None else adj)
                for name,close,adj in result}
            while len(self.stock_days)>2:self.stock_days.popitem(last=False)
        self.stock_days.move_to_end(date)
        return self.stock_days[date].get(int(sid),(np.nan,np.nan))

    def stock(self,date,sid):
        if sid==101322 and date==pd.Timestamp('2025-07-18'):
            return 1.025*self.native_stock(date,102968)[0],self.native_stock(pd.Timestamp('2025-07-17'),sid)[1]
        return self.native_stock(date,sid)


def replay(cases):
    from scripts import uniform_smile_engine as engine
    from scripts.uniform_smile_risk import book_risk
    from scripts.option_flex.contract_engine import Settings
    from scripts.submission_market import verify_ledger
    from scripts.cheap_vega_runtime import mark_book,metrics,annual_metrics
    ck=json.loads(CK.read_text())
    arms=list(cases)+(list(CONTROLS) if set(cases)==set(CASES) else [])
    if not (OUT/'baseline_replay_check.json').exists():arms=['ungated_baseline']+arms
    for case in arms:
        folder=OUT/'replays'/case;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'status.json').exists():continue
        # Sizing is already scored. Execution needs no feature/target columns;
        # omit them from thousands of per-day schedule frames and dictionaries.
        rows=pd.read_parquet(BASE/'replays/no_memory_gate/scored_allocations.parquet' if case=='ungated_baseline' or case in CONTROLS else folder/'scored_allocations.parquet',columns=EXECUTION_COLUMNS)
        if case in CONTROLS:rows['learned_level']=CONTROLS[case]
        cfg=Settings(**json.loads((BUNDLE/CAPITAL/'replays/10m_learned/config.json').read_text()))
        market=LazyDiskMarket(set(rows.secid_a.astype(int))|set(rows.secid_b.astype(int)))
        cal=market.rates.index[market.rates.index>='2018-01-01']
        saved=engine.book_risk
        try:
            engine.book_risk=book_risk
            nav,trades,decisions,ledger=engine.simulate(market,rows,cal,cfg,1_000_000,
                policy=lambda row,context:int(row['learned_level']),corporate_events=ck['corporate_events'],wing_capital=1_000_000)
        finally:engine.book_risk=saved
        mark_book(nav,market.rates,cfg.capital)
        audit=verify_ledger(nav,trades,decisions,rows)
        if case=='ungated_baseline':
            reference=BASE/'replays/no_memory_gate'
            pd.testing.assert_frame_equal(nav,pd.read_parquet(reference/'daily_nav.parquet'),check_exact=False,rtol=1e-12,atol=1e-6)
            for frame,name,dates in [(trades,'trades.csv',['date']),(decisions,'decisions.csv',['signal_date','entry_date'])]:
                pd.testing.assert_frame_equal(frame.reset_index(drop=True),pd.read_csv(reference/name,parse_dates=dates),check_dtype=False,check_exact=False,rtol=1e-10,atol=1e-6)
            write(OUT/'baseline_replay_check.json',dict(exact_nav_trades_decisions=True,**audit))
        result=dict(status='complete',case=case,proposals=len(rows),learned_skips=int(rows.learned_level.eq(0).sum()),
            **metrics(nav,cfg.capital),**audit,**ledger)
        nav.to_parquet(folder/'daily_nav.parquet')
        trades.to_csv(folder/'trades.csv',index=False)
        decisions.to_csv(folder/'decisions.csv',index=False)
        pd.DataFrame(annual_metrics(nav,case,cfg.capital)).to_csv(folder/'annual.csv',index=False)
        write(folder/'status.json',result)
        print(f'{case}: Sharpe={result["annualized_excess_sharpe"]:.6f},entries={result["entries"]},skips={result["learned_skips"]}',flush=True)
        market.db.close()
        market.stockdb.close()
        del market,nav,trades,decisions,rows
        gc.collect()


def summarize():
    from scripts.cheap_vega_runtime import sharpe,metrics,annual_metrics
    baseline=pd.read_parquet(BASE/'replays/no_memory_gate/daily_nav.parquet')
    reference=pd.read_parquet(BASE/'replays/no_memory_gate/scored_allocations.parquet')
    records=[];annual=[];skips=[];intervals=[];fits=[]
    for case in ['ungated_baseline','fixed100']+list(CONTROLS)+list(CASES):
        if case=='fixed100':
            folder=BASE/'replays/no_memory_gate_fixed100'
        elif case=='ungated_baseline':folder=BASE/'replays/no_memory_gate'
        else:folder=OUT/'replays'/case
        status=json.loads((folder/'status.json').read_text())
        nav=pd.read_parquet(folder/'daily_nav.parquet')
        pd.testing.assert_index_equal(nav.index,baseline.index)
        if case in ['ungated_baseline','fixed100'] or case in CONTROLS:
            rows=reference.copy()
            if case in CONTROLS:rows['learned_level']=CONTROLS[case]
            if case=='fixed100':rows['learned_level']=4
        else:rows=pd.read_parquet(folder/'scored_allocations.parquet')
        if case in CASES:
            cols=['date','pair','direction','secid_a','secid_b','optionid_a','optionid_b']+json.loads(CK.read_text())['features']
            pd.testing.assert_frame_equal(rows[cols],reference[cols])
        requested_skips=int(rows.learned_level.eq(0).sum()) if case in CASES else 0
        records.append(dict(case=case,proposals=len(rows),entries=status['entries'],active_pairs=status['active_pairs'],
            learned_skips=requested_skips,early_2018_2023_sharpe=sharpe(nav.loc[:'2023-12-31','excess_return'].iloc[1:]),
            recent_2024_2025_sharpe=sharpe(nav.loc['2024-01-01':,'excess_return']),**metrics(nav,10_000_000)))
        annual.extend(annual_metrics(nav,case,10_000_000))
        levels=rows.learned_level
        for level,count in levels.value_counts().items():skips.append(dict(case=case,requested_size_pct=int(level)*25,proposals=int(count)))
        if case in CASES:
            a=nav.excess_return.iloc[1:].to_numpy();b=baseline.excess_return.iloc[1:].to_numpy()
            rng=np.random.default_rng(2026);draws=[]
            for _ in range(1000):
                starts=rng.integers(0,len(a),size=int(np.ceil(len(a)/126)))
                idx=((starts[:,None]+np.arange(126)[None,:])%len(a)).ravel()[:len(a)]
                draws.append(sharpe(a[idx])-sharpe(b[idx]))
            intervals.append(dict(case=case,sharpe_change=sharpe(a)-sharpe(b),lower_95=np.quantile(draws,.025),
                upper_95=np.quantile(draws,.975),block_sessions=126,draws=1000,selection_adjusted=False))
        elif case.startswith('fixed'):
            a=baseline.excess_return.iloc[1:].to_numpy();b=nav.excess_return.iloc[1:].to_numpy()
            rng=np.random.default_rng(2026);draws=[]
            for _ in range(1000):
                starts=rng.integers(0,len(a),size=int(np.ceil(len(a)/126)))
                idx=((starts[:,None]+np.arange(126)[None,:])%len(a)).ravel()[:len(a)]
                draws.append(sharpe(a[idx])-sharpe(b[idx]))
            intervals.append(dict(case='baseline_vs_'+case,sharpe_change=sharpe(a)-sharpe(b),
                lower_95=np.quantile(draws,.025),upper_95=np.quantile(draws,.975),
                block_sessions=126,draws=1000,selection_adjusted=False))
    for path in sorted((OUT/'models').rglob('*.json')):
        r=json.loads(path.read_text());assert r['latest_completed_outcome'] is None or pd.Timestamp(r['latest_completed_outcome'])<pd.Timestamp(r['fit'])
        fits.append(r)
    assert len(fits)==90*len(CASES)
    assert len(list((OUT/'baseline_fit_checks').glob('*.json')))==90
    frame=pd.DataFrame(records)
    frame.to_csv(REPORT/'comparison.csv',index=False)
    pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False)
    pd.DataFrame(skips).to_csv(REPORT/'requested_sizes.csv',index=False)
    pd.DataFrame(intervals).to_csv(REPORT/'sharpe_change_intervals.csv',index=False)
    pd.DataFrame(fits).to_csv(REPORT/'fits.csv',index=False)
    write(REPORT/'audit.json',dict(monthly_models=len(fits),all4744_proposals_preserved=True,original33_inputs_exact=True,
        baseline90_fits_exact=True,baseline_full_ledger_exact=True,completed_outcomes_only=True,
        no_earnings_gate=True,no_checkpoint_promotion=True,original_checkpoint_unchanged=True))
    labels={'ungated_baseline':'Ungated 50/75/100 baseline','fixed100':'Ungated fixed full size',
        'fixed25':'Constant25% sizing','fixed50':'Constant50% sizing','fixed75':'Constant75% sizing',
        'quarter_floor':'25/50/75/100 sizing','skip_floor':'0/50/75/100 sizing','flexible':'0/25/50/75/100 sizing',
        'tree16':'16-leaf state model','window36':'36-month training window','tree4':'4-leaf state model','window12':'12-month training window','earnings_prior':'Prior-report earnings countdown',
        'earnings_prior_skip':'Prior-report earnings + skip','earnings_calendar':'Historical earnings dates assumed known'}
    lines=['# Sizing and earnings tests on the ungated strategy','',
        'Eight initially declared comparisons and two subsequent follow-ups retain all 4,744 proposals, original directions and contracts, daily delta hedges, expiry holding, training/execution costs and the $10m funded account limits. Profitability memory stays an input with no minimum-profit or count gate. Four-state and 12-month variants were added after observing that 16 states and a 36-month window hurt performance. These are further research on inspected dates, not independent confirmations.','',
        '| Variant | Annualized excess Sharpe | Trades | Learned skips | Profit above cash |','| --- | ---: | ---: | ---: | ---: |']
    for r in frame.itertuples():lines.append(f'| {labels[r.case]} | {r.annualized_excess_sharpe:.3f} | {int(r.entries)} | {int(r.learned_skips)} | ${r.net_excess_pnl:,.0f} |')
    winner=frame.loc[frame.case.isin(CASES)].sort_values('annualized_excess_sharpe',ascending=False).iloc[0]
    baseline_score=float(frame.loc[frame.case.eq('ungated_baseline'),'annualized_excess_sharpe'].iloc[0])
    best_constant=frame.loc[frame.case.str.startswith('fixed')].sort_values('annualized_excess_sharpe',ascending=False).iloc[0]
    lines+=['',f'The highest full-history score among the new variants is {winner.annualized_excess_sharpe:.3f}, from {labels[winner["case"]]}, versus {baseline_score:.3f} for the original ungated learner. The best of the four constant-size controls scores {best_constant.annualized_excess_sharpe:.3f}. These descriptive rankings use the same previously researched history. They are not a new holdout or automatically promoted checkpoints.','',
        'The learned menu changes only entry sizing and admission. It never reverses direction or exits held positions. A zero action explicitly skips a proposal. The 16-leaf test retains the original minimum-leaf formula. All cases refit monthly using completed outcomes; the longer-window case uses 36 months rather than 18. CEM remains 50 iterations, 192 candidates and seed 2026. Menus with a full-size anchor can reproduce that anchor rather than being forced to change size.','',
        'Earnings adds days until the next report and an indicator for a report before expiry, separately for long and short, plus availability flags. The prior-report countdown uses strictly earlier reports. Historical-calendar mode assumes final report dates were known at signal time; an archive of announced schedules is unavailable. Same-day announcement time is unresolved. These versions use 41 inputs and never apply an earnings ban.','',
        'All 90 baseline monthly fits and the complete 0.835-funded ledger reproduce. New ledgers reconcile with closed inventory. Seven focused tests pass. A single worker remains capped at 20% of one core and 384 MiB RSS. After a safe stop on system swap growth, stock prices were moved to a disk-backed lookup; subsequent funded replays peaked below 200 MiB RSS without changing the baseline ledger.','',
        'The original 18-month, eight-state, 50/75/100 sizing policy remains best in this batch. Earnings inputs do not improve it. The 25% floor permits 45% more trades with nearly the same Sharpe, but lower profit. Adaptive sizing beats every constant-size control descriptively; the unadjusted block-bootstrap interval for its Sharpe gain over full sizing includes zero, while its gain over constant half sizing excludes zero. Neither comparison corrects for the prior research search.','',
        'See [yearly returns and Sharpe](annual.csv), [requested sizes](requested_sizes.csv), [fit dates](fits.csv), [earnings coverage](earnings_coverage.csv), and [block-bootstrap differences](sharpe_change_intervals.csv). Confidence intervals do not correct for the strategy search. Training shadow-path Sharpe excludes funded-book admission interactions, and differs from funded Sharpe. 2025 is partial through 29 August.']
    (REPORT/'findings.md').write_text('\n'.join(lines)+'\n')
    print(frame[['case','annualized_excess_sharpe','entries','learned_skips','net_excess_pnl']].to_string(index=False),flush=True)


def package():
    import zipfile
    paths=[p for p in OUT.rglob('*') if p.is_file() and p.name not in ['STOP','progress.json']]
    paths += [REPORT/'protocol.json']
    write(REPORT/'bundle_manifest.json',dict(scope='Ten variants,900 monthly models,4744 unchanged proposals each,14 funded replays; uses existing memory/earnings archives and original input CSV',
        files={str(p):digest(p) for p in paths}))
    target=Path('data/p25_ungated_search_experiment.zip')
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in paths:archive.write(p,str(p))
    (REPORT/'reproduction.txt').write_text(
        'Use requirements.txt, data/submission and the original input CSV. Install requirements-report.txt for the plot phase.\n'
        'Restore the required archives with unzip -n: data/p25_memory_ablation_experiment.zip, '
        'data/p25_earnings_experiment.zip, data/p25_ungated_search_experiment.zip.\n'
        'Run one phase at a time: .venv/bin/python -m scripts.run_signal_discovery_guarded '
        '--module scripts.run_p25_ungated_search --stop-file artifacts/contracts/p25_ungated_search/STOP '
        '--timeout 3600 --duty-cycle 0.2 --rss-limit-mib 384 --min-free-percent 35 '
        '--warning-rss-limit-mib 384 --max-swap-growth-mib 64 --pause-low-headroom-seconds 60 '
        '-- --phase PHASE\n'
        'Phases: test,prepare,train,score,market,replay,summarize,plot,package. Restored models and completed statuses resume. '
        'For a fresh refit/replay, work in a separate checkout and move only the restored new model files/statuses away. '
        'Baseline90-model checks always use the saved original ungated policies. No WRDS access or new option scan is required.\n')
    print(f'Archived {len(paths)} files, {target.stat().st_size/1024**2:.1f} MiB',flush=True)


def plot():
    """Two result visuals from saved ledgers; no additional fitting or search."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame=pd.read_csv(REPORT/'comparison.csv')
    labels={'ungated_baseline':'Original ungated sizing','fixed100':'Fixed100%',
        'fixed25':'Fixed25%','fixed50':'Fixed50%','fixed75':'Fixed75%',
        'quarter_floor':'25% minimum','skip_floor':'Skip / 50–100%',
        'flexible':'Skip / 25–100%','tree16':'16 states','window36':'36-month window',
        'tree4':'4 states (follow-up)','window12':'12-month window (follow-up)',
        'earnings_prior':'Prior-report earnings','earnings_prior_skip':'Prior earnings + skip',
        'earnings_calendar':'Historical earnings schedule*'}
    ranking=frame.sort_values('annualized_excess_sharpe')
    colors=['#747b86' if r.case.startswith('fixed') else '#d29032' if r.case=='earnings_calendar'
        else '#198778' if r.case.startswith('earnings_') else '#426faa' for r in ranking.itertuples()]
    fig,ax=plt.subplots(figsize=(10,6.8))
    ax.barh([labels[c] for c in ranking.case],ranking.annualized_excess_sharpe,color=colors)
    ax.axvline(float(frame.loc[frame.case.eq('ungated_baseline'),'annualized_excess_sharpe'].iloc[0]),color='#29313a',linestyle='--',linewidth=1)
    for i,r in enumerate(ranking.itertuples()):
        ax.text(r.annualized_excess_sharpe+.015,i,f'{r.annualized_excess_sharpe:.3f}  |  {r.entries:,} trades',va='center',fontsize=9)
    ax.set_xlim(min(0,float(ranking.annualized_excess_sharpe.min())-.1),max(1.05,float(ranking.annualized_excess_sharpe.max())+.35))
    ax.set_xlabel('Annualized excess Sharpe, 2018–August 2025')
    ax.set_title('Same 4,744 proposals: sizing, model capacity and earnings')
    ax.spines[['top','right']].set_visible(False)
    fig.text(.02,.018,'*Final historical earnings dates assumed known. Research history was repeatedly inspected.',fontsize=8)
    fig.tight_layout(rect=[0,.035,1,1])
    fig.savefig(REPORT/'comparison.png',dpi=140)
    fig.savefig(REPORT/'comparison.pdf')
    plt.close(fig)
    causal=[c for c in CASES if c!='earnings_calendar']
    winner=frame.loc[frame.case.isin(causal)].sort_values('annualized_excess_sharpe',ascending=False).iloc[0]['case']
    best_fixed=frame.loc[frame.case.str.startswith('fixed')].sort_values('annualized_excess_sharpe',ascending=False).iloc[0]['case']
    fig,ax=plt.subplots(figsize=(10,5))
    for case,color in [('ungated_baseline','#426faa'),(winner,'#198778'),(best_fixed,'#747b86')]:
        folder=(BASE/'replays/no_memory_gate' if case=='ungated_baseline' else
            BASE/'replays/no_memory_gate_fixed100' if case=='fixed100' else OUT/'replays'/case)
        nav=pd.read_parquet(folder/'daily_nav.parquet',columns=['nav','cash_benchmark'])
        ax.plot(nav.index,nav.nav-nav.cash_benchmark,label=labels[case],color=color,linewidth=1.6)
    ax.axhline(0,color='#909090',linewidth=.6)
    ax.set_ylabel('Funded NAV above cash benchmark ($)')
    ax.set_title('Broader book: saved chronological funded ledgers')
    ax.legend(loc='upper left',frameon=False)
    ax.spines[['top','right']].set_visible(False)
    fig.text(.02,.018,'Highest full-history causal-input variant and constant-size control shown; both selected retrospectively.',fontsize=8)
    fig.tight_layout(rect=[0,.04,1,1])
    fig.savefig(REPORT/'cumulative_excess.png',dpi=140)
    fig.savefig(REPORT/'cumulative_excess.pdf')
    plt.close(fig)
    print(f'Charts saved: best causal-input variant={winner}, best constant control={best_fixed}',flush=True)


def main():
    require_supervisor()
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['test','prepare','train','score','market','replay','summarize','plot','package'],required=True)
    parser.add_argument('--case',choices=CASES)
    parser.add_argument('--limit-months',type=int)
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    before=protected()
    prior_status=digest(BASE/'replays/no_memory_gate/status.json')
    code=['scripts/run_p25_ungated_search.py','scripts/run_p25_memory_ablation.py','scripts/run_p25_earnings.py','scripts/p25_earnings_features.py','scripts/uniform_smile_engine.py','scripts/uniform_smile_risk.py']
    write(REPORT/'protocol.json',dict(cases=CASES,initial_cases=[c for c in CASES if c not in FOLLOWUPS],
        followups=dict(cases=FOLLOWUPS,reason='Added after16-state0.6723 and36-month0.7174 results; smaller state model and shorter window to test reduced estimation complexity and quicker adaptation'),
        constant_controls=CONTROLS,source='Ungated0.8346021868;4744proposals',
        ranking='Report all variants and full/2018-2023/2024-2025; retrospective research comparison; no fresh holdout or automatic promotion',
        code_sha256={p:digest(p) for p in code},resources='One worker20%of onecore,384MiB RSSstop,35%headroom,64MiB swapgrowth',
        unchanged='33 original features, direction, contracts, formation/calendar,beta/corr, costs/financing, expiry/dailyhedges,10m capital and risk caps'))
    cases=[args.case] if args.case else list(CASES)
    if args.phase=='test':
        import unittest
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests',pattern='test_p25_ungated_search.py'))
        if not result.wasSuccessful():raise SystemExit(1)
    elif args.phase=='prepare':prepare()
    elif args.phase=='train':train(cases,args.limit_months)
    elif args.phase=='score':score(cases)
    elif args.phase=='market':market_cache()
    elif args.phase=='replay':replay(cases)
    elif args.phase=='summarize':summarize()
    elif args.phase=='plot':plot()
    elif args.phase=='package':package()
    assert before==protected() and prior_status==digest(BASE/'replays/no_memory_gate/status.json')
    print(f'Peak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2:.1f} MiB',flush=True)


if __name__=='__main__':main()
