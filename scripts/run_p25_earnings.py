"""Bounded earnings ablation around the retained $10m P25 policy."""
import argparse
import gc
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd

from scripts import p25_earnings_features as earnings
from scripts.submission import BUNDLE, FEATURE, CAPITAL, digest, require_supervisor

OUT = Path('artifacts/contracts/p25_earnings')
REPORT = Path('reports/p25_earnings')
RAW = Path('/Users/adamkreitzman/Desktop/RL_Converts/convert-rv-rl/data/raw/wrds/compustat/underlying_quarterly_earnings.parquet')
CHECKPOINT = Path('config/p25_submission.json')
ADDED = [c for f in earnings.FIELDS for c in [f, f+'_available']]
PROTOCOL = {
    'comparison': 'Retained $10m 1.003 Sharpe policy versus two declared earnings-state additions',
    'cases': earnings.CASES,
    'new_inputs': ADDED,
    'unchanged': 'All 608 proposals, direction, contracts, 33 original inputs, monthly 18-month completed-outcome fits, 8-leaf tree, CEM50x192 seed2026, 50/75/100 sizing, costs, financing, expiry holding and risk limits',
    'earnings': 'Historical_calendar assumes final historical report dates known at signal close. Prior_reports_only uses strictly earlier reports and median of last eight observed inter-report gaps, clipped70-120days, fallback91days. Countdowns capped90days; missing inputs receive zero and explicit availability, never a proposal veto. ETF has no issuer earnings event.',
    'timing': 'Before-open versus after-close release time is unknown. Same-day reports are excluded from prior-report information; known-calendar mode counts same-day events as possibly inside the contract.',
    'limitations': 'Previously explored evaluation history, retrospective pair formation, revised vendor data and no fresh holdout. Earnings features change partition/sizing, not the directional entry rule. No automatic checkpoint promotion.',
    'resources': 'One worker,20%of one core,384MiB RSS stop,35%headroom,64MiB maximum swap growth',
}


def protected():
    return {str(p): digest(p) for p in [CHECKPOINT, Path('data/p25_variance_premium_inputs.csv'),
            Path('reports/p25_capital_sensitivity/comparison.csv'),
            BUNDLE/FEATURE/'training_rows.parquet', BUNDLE/FEATURE/'signal_rows.parquet']}


def calendar_for_assets(secids):
    metadata = json.loads((BUNDLE/'artifacts/broad_pairs/assets.json').read_text())
    selected = [item for item in metadata if item['secid'] in secids]
    saved_calendar=REPORT/'earnings_calendar.csv'
    if not RAW.exists() and saved_calendar.exists():
        saved=pd.read_csv(saved_calendar,parse_dates=['earnings_date'])
        output={sid:pd.DatetimeIndex(saved.loc[saved.secid.eq(sid),'earnings_date']).unique().sort_values() for sid in secids}
        flags={sid:any(item['secid']==sid and item['etf'] for item in selected) for sid in secids}
        return output,flags
    raw = pd.read_parquet(RAW, columns=['permno', 'rdq', 'datadate'])
    for c in ['rdq', 'datadate']:
        raw[c] = pd.to_datetime(raw[c])
    corrections = pd.read_csv('config/universe_earnings_overrides.csv', parse_dates=['datadate', 'earnings_date'])
    for row in corrections.itertuples():
        mask = raw.permno.eq(row.permno) & raw.datadate.eq(row.datadate)
        if not mask.any():
            raise ValueError(f'Missing earnings override {row.ticker} {row.datadate}')
        raw.loc[mask, 'rdq'] = row.earnings_date
    supplement = pd.read_csv('config/earnings_supplement.csv', parse_dates=['earnings_date'])
    output, flags, records = {}, {}, []
    for sid in sorted(secids):
        items = [item for item in selected if item['secid']==sid]
        if not items:
            output[sid], flags[sid] = pd.DatetimeIndex([]), False
            continue
        if len({item['etf'] for item in items}) != 1:
            raise ValueError(f'Conflicting ETF classification for {sid}')
        flags[sid] = bool(items[0]['etf'])
        dates = []
        for item in items:
            events = pd.concat([raw.loc[raw.permno.eq(item['permno']), 'rdq'],
                                supplement.loc[supplement.permno.eq(item['permno']), 'earnings_date']]).dropna()
            valid = pd.Series(False, index=events.index)
            for start, end in item['intervals']:
                valid |= events.between(pd.Timestamp(start), pd.Timestamp(end))
            values = pd.DatetimeIndex(events[valid]).unique().sort_values()
            dates.extend(values.tolist())
            records.extend(dict(secid=sid, permno=item['permno'], ticker=item['ticker'],
                                earnings_date=str(date.date())) for date in values)
        output[sid] = pd.DatetimeIndex(dates).unique().sort_values()
    pd.DataFrame(records).drop_duplicates().to_csv(REPORT/'earnings_calendar.csv', index=False)
    return output, flags


def prepare():
    if (OUT/'prepared.json').exists():
        return
    ck = json.loads(CHECKPOINT.read_text())
    columns = list(dict.fromkeys(ck['features'] + ['date', 'pair', 'wing', 'secid_a', 'secid_b',
        'optionid_a', 'optionid_b', 'direction', 'dte_gap', 'observed', 'eligible', 'context',
        'signal_eligible', 'finite_label', 'target', 'end', 'position', 'label_row']))
    meta = pd.read_parquet(BUNDLE/FEATURE/'training_rows.parquet', columns=columns)
    proposals = pd.read_parquet(BUNDLE/FEATURE/'signal_rows.parquet')
    assert len(meta)==36327 and len(proposals)==608
    secids = set(meta.secid_a.astype(int)) | set(meta.secid_b.astype(int))
    events, etfs = calendar_for_assets(secids)
    coverage = []
    for case in earnings.CASES:
        result = meta.copy()
        for col in ADDED:
            result[col] = 0.
        for _, idx in meta.groupby('pair', sort=False).groups.items():
            group = meta.loc[idx]
            a, b = int(group.secid_a.iloc[0]), int(group.secid_b.iloc[0])
            valid_dte = group.dte_mean_available.eq(1) & group.oriented_dte_gap_available.eq(1)
            da = (group.dte_mean + group.dte_gap/2).where(valid_dte)
            db = (group.dte_mean - group.dte_gap/2).where(valid_dte)
            la = earnings.leg_features(group.date, da, events[a], etfs[a], case)
            lb = earnings.leg_features(group.date, db, events[b], etfs[b], case)
            added = earnings.orient_legs(la, lb, group.direction)
            result.loc[idx, ADDED] = added[ADDED].to_numpy()
        pd.testing.assert_frame_equal(result[meta.columns], meta)
        assert np.isfinite(result[ck['features']+ADDED]).all().all()
        result.to_parquet(OUT/f'{case}_training.parquet', index=False)
        joined = proposals.merge(result[['date','pair']+ADDED], on=['date','pair'], how='left', validate='one_to_one')
        pd.testing.assert_frame_equal(joined[proposals.columns], proposals)
        assert np.isfinite(joined[ADDED]).all().all()
        joined.to_parquet(OUT/f'{case}_signals.parquet', index=False)
        for f in earnings.FIELDS:
            coverage.append(dict(case=case, feature=f, rows_available=int(result[f+'_available'].sum()),
                                 proposal_available=int(joined[f+'_available'].sum())))
        print(f'{case}: prepared {len(result)} rows, all {len(joined)} proposals retained', flush=True)
        del result, joined
        gc.collect()
    pd.DataFrame(coverage).to_csv(REPORT/'coverage.csv', index=False)
    source_paths = ([RAW] if RAW.exists() else []) + [REPORT/'earnings_calendar.csv', CHECKPOINT, BUNDLE/FEATURE/'training_rows.parquet', BUNDLE/FEATURE/'signal_rows.parquet',
                    BUNDLE/'artifacts/broad_pairs/assets.json', Path('config/universe_earnings_overrides.csv'),
                    Path('config/earnings_supplement.csv')]
    write(OUT/'prepared.json', dict(rows=len(meta), proposals=608, names=len(secids),
          calendar_missing_names=[sid for sid in secids if len(events[sid])==0 and not etfs[sid]],
          features=ck['features']+ADDED, sources={str(p):digest(p) for p in source_paths}))
    progress('prepared')


def fit_sizes(book):
    """Exact retained CEM arithmetic, without importing broad research pipelines."""
    mean = book.mean(axis=0)
    cov = np.cov(book, rowvar=False, ddof=1)
    ns = book.shape[1]//5
    def score(choices):
        cols = choices + np.arange(ns)[None, :]*5
        m = mean[cols].sum(axis=1)
        v = cov[cols[:,:,None],cols[:,None,:]].sum(axis=(1,2))
        return np.sqrt(252)*np.divide(m,np.sqrt(np.maximum(v,0)),out=np.zeros_like(m),where=v>1e-18)
    menu = np.array([2,3,4])
    rng = np.random.default_rng(2026)
    probability = np.full((ns,3),1/3)
    best = np.full(ns,4)
    best_score = baseline = float(score(best[None])[0])
    for _ in range(50):
        aliases = (rng.random((192,ns))[:,:,None]>probability.cumsum(axis=1)[None]).sum(axis=2).clip(max=2)
        choices=menu[aliases]; choices[0]=best; choices[1:4]=menu[:,None]
        scores=score(choices); winner=scores.argmax()
        if scores[winner]>best_score:
            best,best_score=choices[winner].copy(),float(scores[winner])
        elite=choices[np.argsort(scores)[-19:]]
        frequency=np.stack([(elite==level).mean(axis=0) for level in menu],axis=1)
        probability=.97*(.3*probability+.7*frequency)+.03/3
    return best, dict(training_shadow_daily_sharpe=best_score, baseline_shadow_daily_sharpe=baseline)


def train_one(rows, features, cal):
    import joblib
    from sklearn.tree import DecisionTreeRegressor
    tree, leaves, choices, details = None, np.array([0]), np.array([4]), {}
    if len(rows)>=16:
        tree=DecisionTreeRegressor(max_leaf_nodes=8,min_samples_leaf=max(8,min(63,len(rows)//8)),random_state=2026).fit(rows[features],rows.target)
        leaves=np.unique(tree.apply(rows[features]))
        states=np.searchsorted(leaves,tree.apply(rows[features]))
        first=int(rows.position.min()+1)
        last=int(cal.get_indexer(pd.DatetimeIndex(rows.end)).max())
        book=np.zeros((last-first+1,len(leaves)*5))
        for pair, indices in rows.groupby('pair',sort=False).groups.items():
            idx=np.asarray(indices)
            disk=np.load(BUNDLE/'artifacts/contracts/uniform_smile/pairs'/pair/'P25_labels.npy',mmap_mode='r')
            raw=disk[rows.loc[idx,'label_row'].to_numpy(int)]
            direction=rows.loc[idx,'direction'].to_numpy(int)
            labels=np.zeros((len(idx),5,raw.shape[-1]),raw.dtype)
            for size in range(1,5):
                labels[:,size]=raw[np.arange(len(idx)),np.where(direction>0,4+size,5-size)]
            positions=rows.loc[idx,'position'].to_numpy(int)
            days=positions[:,None]+1+np.arange(labels.shape[-1])[None,:]-first
            valid=(days>=0)&(days<len(book))
            for leaf in np.unique(states[idx]):
                local=np.flatnonzero(states[idx]==leaf)
                for action in range(5):
                    np.add.at(book[:,int(leaf)*5+action],days[local][valid[local]],labels[local,action][valid[local]])
            del disk,raw,labels
        choices,details=fit_sizes(book)
        del book
    return dict(tree=tree,leaves=leaves,choices=choices,features=features), details


def train(cases, limit):
    import joblib
    ck=json.loads(CHECKPOINT.read_text())
    cal=pd.read_parquet(BUNDLE/'artifacts/contracts/uniform_smile/market/rates.parquet').index
    months=pd.period_range('2018-01-01',ck['last_signal'],freq='M')
    if limit is not None:
        months=months[:limit]
    # An exact initial-fold check validates the lightweight optimizer against
    # the archived model before introducing any new state inputs.
    if not (OUT/'baseline_fit_check.json').exists():
        meta=pd.read_parquet(BUNDLE/FEATURE/'training_rows.parquet')
        cutoff=pd.Timestamp('2018-01-01')
        rows=meta.loc[meta.signal_eligible & meta.finite_label & meta.date.ge(cutoff-pd.DateOffset(months=18)) & meta.date.lt(cutoff) & meta.end.lt(cutoff)].reset_index(drop=True)
        model,details=train_one(rows,ck['features'],cal)
        stored=joblib.load(BUNDLE/FEATURE/'models/variance_premium/2018-01-01.joblib')
        for field in ['choices','leaves']:
            np.testing.assert_array_equal(model[field],stored[field])
        for field in ['feature','threshold','children_left','children_right','value']:
            np.testing.assert_array_equal(getattr(model['tree'].tree_,field),getattr(stored['tree'].tree_,field))
        write(OUT/'baseline_fit_check.json',dict(exact=True,training_rows=len(rows),**details))
        print('Original initial fit exactly reproduced',flush=True)
        del meta,rows,model,stored
        gc.collect()
    for case in cases:
        meta=pd.read_parquet(OUT/f'{case}_training.parquet')
        features=ck['features']+ADDED
        for month in months:
            cutoff=month.start_time; stamp=str(cutoff.date())
            folder=OUT/'models'/case;folder.mkdir(parents=True,exist_ok=True)
            if (folder/f'{stamp}.json').exists():
                continue
            rows=meta.loc[meta.signal_eligible & meta.finite_label & meta.date.ge(cutoff-pd.DateOffset(months=18)) & meta.date.lt(cutoff) & meta.end.lt(cutoff)].reset_index(drop=True)
            assert len(rows)==0 or rows.end.max()<cutoff
            started=time.monotonic()
            model,details=train_one(rows,features,cal)
            model['fit']=stamp
            joblib.dump(model,folder/f'{stamp}.joblib')
            write(folder/f'{stamp}.json',dict(case=case,fit=stamp,training_rows=len(rows),features=features,
                  latest_completed_outcome=str(rows.end.max().date()) if len(rows) else None,choices=model['choices'].tolist(),**details))
            progress('training',case=case,fit=stamp,fit_wall_seconds=time.monotonic()-started)
            print(f'{case} {stamp}: {len(rows)} completed rows, sizes {model["choices"].tolist()}',flush=True)
            del rows,model
            gc.collect()
        del meta
        gc.collect()


def score(cases):
    import joblib
    for case in cases:
        folder=OUT/'replays'/case;folder.mkdir(parents=True,exist_ok=True)
        path=folder/'scored_allocations.parquet'
        if path.exists():
            continue
        rows=pd.read_parquet(OUT/f'{case}_signals.parquet')
        rows['learned_level']=4
        for month,idx in rows.groupby(rows.date.dt.to_period('M')).groups.items():
            model=joblib.load(OUT/'models'/case/f'{month.start_time.date()}.joblib')
            if model['tree'] is not None:
                states=model['tree'].apply(rows.loc[idx,model['features']])
                rows.loc[idx,'learned_level']=model['choices'][np.searchsorted(model['leaves'],states)]
        assert len(rows)==608 and rows.learned_level.isin([2,3,4]).all()
        rows.to_parquet(path,index=False)
        print(f'{case}: scored all 608 proposals',flush=True)
        del rows,model
        gc.collect()


def replay(cases):
    from scripts import uniform_smile_engine as engine
    from scripts.uniform_smile_risk import book_risk
    from scripts.option_flex.contract_engine import Settings
    from scripts.submission_market import FrozenMarket, verify_ledger
    from scripts.cheap_vega_runtime import mark_book,metrics,annual_metrics
    ck=json.loads(CHECKPOINT.read_text())
    class LeanMarket(FrozenMarket):
        def __init__(self,folder,secids):
            # Preserve the exact source values and quote lookup conventions,
            # while avoiding full-universe pivot buffers. CVX supports the
            # existing Hess corporate-event mark; it is not a new trade.
            from collections import OrderedDict
            self.folder=folder
            ids=set(secids)
            if 101322 in ids:
                ids.add(102968)
            stock=pd.read_parquet(folder/'stocks.parquet',columns=['date','secid','close','cfadj'],
                                  filters=[('secid','in',sorted(ids))])
            self.spots=stock.pivot(index='date',columns='secid',values='close')
            self.adjustments=stock.pivot(index='date',columns='secid',values='cfadj')
            self.rates=pd.read_parquet(folder/'rates.parquet').rate
            self.divs=pd.read_parquet(folder/'dividends.parquet')
            self.years,self.days=OrderedDict(),OrderedDict()
            self.reported_year=None
            del stock
            gc.collect()
            import pyarrow as pa
            pa.default_memory_pool().release_unused()
    # Check that the smaller market view preserves the original complete ledger.
    if not (OUT/'baseline_replay_check.json').exists():
        cases=['retained_baseline']+list(cases)
    for case in cases:
        folder=OUT/'replays'/case;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'status.json').exists():
            continue
        rows=pd.read_parquet(BUNDLE/FEATURE/'replays/variance_premium/scored_allocations.parquet'
                             if case=='retained_baseline' else folder/'scored_allocations.parquet')
        assert len(rows)==608 and rows.learned_level.isin([2,3,4]).all()
        cfg=Settings(**json.loads((BUNDLE/CAPITAL/'replays/10m_learned/config.json').read_text()))
        secids=set(rows.secid_a.astype(int))|set(rows.secid_b.astype(int))
        market=LeanMarket(BUNDLE/'artifacts/contracts/p25_signal_transfer/market',secids)
        cal=market.rates.index[market.rates.index>='2018-01-01']
        saved=engine.book_risk
        try:
            engine.book_risk=book_risk
            nav,trades,decisions,ledger=engine.simulate(market,rows,cal,cfg,1_000_000,
                policy=lambda row,context:int(row['learned_level']),corporate_events=ck['corporate_events'],wing_capital=1_000_000)
        finally:
            engine.book_risk=saved
        mark_book(nav,market.rates,cfg.capital)
        audit=verify_ledger(nav,trades,decisions,rows)
        if case=='retained_baseline':
            reference=BUNDLE/CAPITAL/'replays/10m_learned'
            pd.testing.assert_frame_equal(nav,pd.read_parquet(reference/'daily_nav.parquet'),check_exact=False,rtol=1e-12,atol=1e-6)
            for frame,name,dates in [(trades,'trades.csv',['date']),
                                     (decisions,'decisions.csv',['signal_date','entry_date'])]:
                pd.testing.assert_frame_equal(frame.reset_index(drop=True),pd.read_csv(reference/name,parse_dates=dates),
                                               check_dtype=False,check_exact=False,rtol=1e-10,atol=1e-6)
            write(OUT/'baseline_replay_check.json',dict(exact_nav_trades_decisions=True,**audit))
        result=dict(status='complete',case=case,**metrics(nav,cfg.capital),**audit,**ledger)
        nav.to_parquet(folder/'daily_nav.parquet')
        trades.to_csv(folder/'trades.csv',index=False)
        decisions.to_csv(folder/'decisions.csv',index=False)
        rows.to_parquet(folder/'scored_allocations.parquet',index=False)
        pd.DataFrame(annual_metrics(nav,case,cfg.capital)).to_csv(folder/'annual.csv',index=False)
        write(folder/'status.json',result)
        print(f'{case}: Sharpe {result["annualized_excess_sharpe"]:.6f}, {result["entries"]} entries',flush=True)
        progress('replayed',case=case,sharpe=result['annualized_excess_sharpe'])
        del market,nav,trades,decisions,rows
        gc.collect()


def summarize():
    from scripts.cheap_vega_runtime import metrics,annual_metrics,sharpe
    import joblib
    ck=json.loads(CHECKPOINT.read_text())
    comparisons,annual,allocation,usage,event_sizes,uplift=[],[],[],[],[],[]
    reference=BUNDLE/CAPITAL/'replays/10m_learned'
    baseline=pd.read_parquet(reference/'daily_nav.parquet')
    base_decisions=pd.read_csv(reference/'decisions.csv')
    comparisons.append(dict(case='retained_baseline',entries=ck['results']['10m_learned']['entries'],
        early_2018_2023_sharpe=sharpe(baseline.loc[:'2023-12-31','excess_return'].iloc[1:]),
        recent_2024_2025_sharpe=sharpe(baseline.loc['2024-01-01':,'excess_return']),**metrics(baseline,10_000_000)))
    annual.extend(annual_metrics(baseline,'retained_baseline',10_000_000))
    original=pd.read_parquet(BUNDLE/FEATURE/'replays/variance_premium/scored_allocations.parquet')
    calendar_signals=pd.read_parquet(OUT/'historical_calendar_signals.parquet')
    long_event=calendar_signals.long_earnings_before_expiry.eq(1)
    short_event=calendar_signals.short_earnings_before_expiry.eq(1)
    event_groups=np.where(long_event & short_event,'both',
                 np.where(long_event,'long_only',np.where(short_event,'short_only','neither')))
    for label in ['both','long_only','short_only','neither']:
        take=event_groups==label
        event_sizes.append(dict(case='retained_baseline',event_group=label,proposals=int(take.sum()),
                                mean_requested_size=float(original.loc[take,'learned_level'].mean()/4)))
    for case in earnings.CASES:
        folder=OUT/'replays'/case
        if not (folder/'status.json').exists():
            raise RuntimeError(f'Missing completed replay: {case}')
        result=json.loads((folder/'status.json').read_text())
        nav=pd.read_parquet(folder/'daily_nav.parquet')
        pd.testing.assert_index_equal(nav.index,baseline.index)
        comparisons.append(result)
        annual.extend(pd.read_csv(folder/'annual.csv').to_dict('records'))
        scored=pd.read_parquet(folder/'scored_allocations.parquet')
        pd.testing.assert_frame_equal(scored[original.columns.drop('learned_level')],original.drop(columns='learned_level'))
        changed=int((scored.learned_level.to_numpy()!=original.learned_level.to_numpy()).sum())
        decisions=pd.read_csv(folder/'decisions.csv')
        left=set(zip(base_decisions.loc[base_decisions.status.eq('entered'),'signal_date'],
                     base_decisions.loc[base_decisions.status.eq('entered'),'pair']))
        right=set(zip(decisions.loc[decisions.status.eq('entered'),'signal_date'],
                      decisions.loc[decisions.status.eq('entered'),'pair']))
        allocation.append(dict(case=case,proposals=608,changed_sizes=changed,
            accepted_in_common=len(left & right),newly_accepted=len(right-left),no_longer_accepted=len(left-right),
            size_50=int(scored.learned_level.eq(2).sum()),size_75=int(scored.learned_level.eq(3).sum()),size_100=int(scored.learned_level.eq(4).sum())))
        for label in ['both','long_only','short_only','neither']:
            take=event_groups==label
            event_sizes.append(dict(case=case,event_group=label,proposals=int(take.sum()),
                                    mean_requested_size=float(scored.loc[take,'learned_level'].mean()/4)))
        models=[joblib.load(path) for path in sorted((OUT/'models'/case).glob('*.joblib'))]
        for f in ck['features']+ADDED:
            imports=[m['tree'].feature_importances_[m['features'].index(f)] if m['tree'] is not None else 0. for m in models]
            usage.append(dict(case=case,feature=f,months_used=int(np.count_nonzero(imports)),
                              mean_tree_importance=float(np.mean(imports))))
        del models
        excess=nav.excess_return.iloc[1:].to_numpy()
        control=baseline.excess_return.iloc[1:].to_numpy()
        result['sharpe_change_vs_baseline']=sharpe(excess)-sharpe(control)
        result['early_2018_2023_sharpe']=sharpe(nav.loc[:'2023-12-31','excess_return'].iloc[1:])
        result['recent_2024_2025_sharpe']=sharpe(nav.loc['2024-01-01':,'excess_return'])
        result['modeled_held_marks']=int(nav.causal_held_mark_fallbacks.sum())
        rng=np.random.default_rng(2026);n=len(excess);block=126;draws=[]
        for _ in range(1000):
            starts=rng.integers(0,n,size=int(np.ceil(n/block)))
            idx=((starts[:,None]+np.arange(block))%n).ravel()[:n]
            draws.append(sharpe(excess[idx])-sharpe(control[idx]))
        uplift.append(dict(case=case,sharpe_change=result['sharpe_change_vs_baseline'],
                           lower=float(np.quantile(draws,.025)),upper=float(np.quantile(draws,.975)),
                           block_sessions=block,draws=1000,selection_adjusted=False))
        comparisons[-1]=result
        del nav,scored
        gc.collect()
    frame=pd.DataFrame(comparisons)
    frame.to_csv(REPORT/'comparison.csv',index=False)
    pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False)
    pd.DataFrame(allocation).to_csv(REPORT/'allocations.csv',index=False)
    pd.DataFrame(event_sizes).to_csv(REPORT/'event_sizing.csv',index=False)
    pd.DataFrame(usage).to_csv(REPORT/'feature_usage.csv',index=False)
    pd.DataFrame(uplift).to_csv(REPORT/'sharpe_change_intervals.csv',index=False)
    logs=[]
    for case in earnings.CASES:
        records=[json.loads(path.read_text()) for path in sorted((OUT/'models'/case).glob('*.json'))]
        assert len(records)==90, (case,len(records))
        assert all(not r['latest_completed_outcome'] or pd.Timestamp(r['latest_completed_outcome'])<pd.Timestamp(r['fit']) for r in records)
        logs.extend(records)
    pd.DataFrame(logs).to_csv(REPORT/'fits.csv',index=False)
    write(REPORT/'audit.json',dict(status='complete',monthly_models=len(logs),all_608_proposals_preserved=True,
          no_earnings_gate=True,baseline_initial_fit_exact=True,completed_outcomes_only=True,
          baseline_full_nav_trades_decisions_exact=json.loads((OUT/'baseline_replay_check.json').read_text())['exact_nav_trades_decisions'],
          original_checkpoint_unchanged=True,known_calendar_is_assumption=True,no_checkpoint_promotion=True,
          peak_worker_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2))
    labels={'retained_baseline':'Retained policy','historical_calendar':'Historical dates assumed known',
            'prior_reports_only':'Countdown from prior reports'}
    report=['# Earnings-aware sizing on the retained strategy','',
        'Adding earnings inputs did not improve the retained policy in this first comparison. Both variants changed sizing while keeping all 608 proposals, option contracts, direction rules, costs and account limits. The saved 1.003 Sharpe checkpoint remains unchanged.','',
        '| Policy | Annualized excess Sharpe | Trades | Profit above cash | Annualized growth relative to cash |',
        '| --- | ---: | ---: | ---: | ---: |']
    for r in frame.itertuples():
        report.append(f'| {labels[r.case]} | {r.annualized_excess_sharpe:.3f} | {int(r.entries)} | ${r.net_excess_pnl:,.0f} | {r.relative_to_cash_cagr_pct:.3f}% |')
    report += ['', 'Each leg gets days until earnings and an indicator for earnings before its selected expiry. Four availability flags bring the input count from 33 to 41. Missing information never removes a proposal. The action menu remains 50%, 75% and 100%, with monthly fits on 18 months of completed outcomes. There is no earnings exit or entry ban.','',
        'The historical-date version assumes the eventual reported earnings date was known at the signal close, with a countdown capped at 90 days. We lack an archive of schedules as announced at the time. The other version estimates the countdown from strictly earlier reports; changing future report dates does not change that input. Before-open versus after-close timing is unresolved. Neither is a new untouched test period.','',
        '## Yearly annualized excess Sharpe','',
        '| Year | Retained | Historical dates | Prior reports |','| --- | ---: | ---: | ---: |']
    yearly=pd.DataFrame(annual).pivot(index='year',columns='case',values='annualized_excess_sharpe')
    for year,r in yearly.iterrows():
        report.append(f'| {int(year)}'+(' (partial)' if year==2025 else '')+f' | {r.retained_baseline:.3f} | {r.historical_calendar:.3f} | {r.prior_reports_only:.3f} |')
    report += ['', '2025 ends on 29 August. Full percentage returns are in [annual.csv](annual.csv).', '',
        'The largest reductions in return above cash occur in 2018 and 2022. In 2022 the earnings-aware policies have a higher yearly Sharpe but less profit. The weak 2024 result barely changes. Earnings input therefore does not resolve the strategy’s recent losing regime in this comparison.', '',
        '## What changed','']
    for r in allocation:
        report.append(f"{labels[r['case']]} changes the requested size on {r['changed_sizes']} of 608 proposals. It shares {r['accepted_in_common']} accepted entries with the original book, admits {r['newly_accepted']} different entries and no longer admits {r['no_longer_accepted']} original entries.")
    report += ['', 'The earnings features are evaluated through the existing allocation model; this result does not establish that earnings are economically irrelevant. Added information can change tree partitions and estimated sizing rules without improving performance on subsequent trades. The sample supplies roughly 60–300 completed examples per monthly fit.','',
        'The original first fit and the complete funded baseline reproduce exactly. Six tests check event boundaries, direction, missing data, ETF treatment and independence from future reports. Both new ledgers reconcile and finish with all inventory closed. The first replay attempt hit the memory stop; model scoring was separated from replay and stock reads were restricted to proposed names. The optimized replay preserved the original ledger exactly.','',
        'Additional files: [feature usage](feature_usage.csv), [event-conditioned sizes](event_sizing.csv), [allocation changes](allocations.csv), [block-bootstrap intervals](sharpe_change_intervals.csv), [protocol](protocol.json). Bootstrap intervals do not correct for strategy selection. No model, window or threshold was selected using this experiment’s outcomes. The wider strategy was previously selected on these dates.']
    (REPORT/'findings.md').write_text('\n'.join(report)+'\n')
    print(frame[['case','annualized_excess_sharpe','entries','net_excess_pnl']].to_string(index=False),flush=True)
    progress('complete')


def package():
    """Share input-only CSVs and resumable experiment caches without raw WRDS scans."""
    import zipfile
    base=pd.read_csv('data/p25_variance_premium_inputs.csv',parse_dates=['date'])
    datasets=[]
    for case in earnings.CASES:
        meta=pd.read_parquet(OUT/f'{case}_training.parquet',columns=['date','pair']+ADDED)
        joined=base.merge(meta,on=['date','pair'],how='left',validate='one_to_one')
        pd.testing.assert_frame_equal(joined[base.columns],base)
        assert len(joined)==36327 and not set(joined)&{'reward','target','end','label_row','finite_label'}
        path=Path('data')/f'p25_earnings_{case}_inputs.csv'
        joined.to_csv(path,index=False)
        datasets.append(path)
        del meta,joined
        gc.collect()
    paths=[p for p in OUT.rglob('*') if p.is_file() and p.name not in ['STOP','progress.json']]
    paths+=datasets
    target=Path('data/p25_earnings_experiment.zip')
    manifest=dict(scope='Both earnings-state additions, 180 fitted monthly policies, funded replays and input-only CSVs',
                  baseline_checkpoint=str(CHECKPOINT),baseline_sha256=digest(CHECKPOINT),
                  files={str(p):digest(p) for p in paths})
    write(REPORT/'bundle_manifest.json',manifest)
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for path in paths:
            z.write(path,str(path))
    (REPORT/'reproduction.txt').write_text(
        'Use requirements.txt and the existing submission cache in data/submission.\n'
        'Restore saved earnings models and replay states: unzip -n data/p25_earnings_experiment.zip\n'
        'Run one phase at a time with .venv/bin/python -m scripts.run_signal_discovery_guarded '
        '--module scripts.run_p25_earnings --stop-file artifacts/contracts/p25_earnings/STOP '
        '--timeout 1800 --duty-cycle 0.2 --rss-limit-mib 384 --min-free-percent 35 '
        '--warning-rss-limit-mib 384 --max-swap-growth-mib 64 --pause-low-headroom-seconds 60 '
        '-- --phase PHASE\n'
        'Phases: test, prepare, train, score, replay, summarize, package. '
        'The saved models and completed statuses make these resumable. To refit from scratch, '
        'use a separate checkout and remove only its restored earnings models before train. '
        'To rerun a portfolio, preserve its status.json under a different name first.\n'
        'Calendar source: corrected Compustat quarterly report dates; the exported earnings_calendar.csv '
        'can rebuild the feature panel when the original WRDS path is unavailable. '
        'Historical_calendar assumes future event dates known; prior_reports_only uses earlier reports. '
        'Original checkpoints and paper are unchanged.\n')
    print(f'Shared two input-only CSVs and {len(paths)} cache files; bundle {target.stat().st_size/1024**2:.1f} MiB',flush=True)


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2)+'\n')


def progress(stage, **values):
    write(OUT/'progress.json', dict(stage=stage,
          peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2, **values))


def inspect_inputs():
    meta = pd.read_parquet(BUNDLE/FEATURE/'training_rows.parquet')
    proposals = pd.read_parquet(BUNDLE/FEATURE/'signal_rows.parquet')
    print(json.dumps(dict(training_rows=len(meta), proposals=len(proposals),
                         columns=list(meta.columns), proposal_columns=list(proposals.columns)), indent=2), flush=True)


def main():
    require_supervisor()
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['inspect', 'test', 'prepare', 'train', 'score', 'replay', 'summarize', 'package'], required=True)
    parser.add_argument('--case', choices=earnings.CASES)
    parser.add_argument('--limit-months', type=int)
    args = parser.parse_args()
    if args.phase == 'inspect':
        inspect_inputs()
        return
    if args.phase == 'test':
        import unittest
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests', pattern='test_p25_earnings.py'))
        if not result.wasSuccessful():
            raise SystemExit(1)
        return
    OUT.mkdir(parents=True,exist_ok=True)
    REPORT.mkdir(parents=True,exist_ok=True)
    before=protected()
    write(REPORT/'protocol.json',PROTOCOL)
    cases=[args.case] if args.case else earnings.CASES
    if args.phase in ['prepare','all']:
        prepare()
    if args.phase in ['train','all']:
        train(cases,args.limit_months)
    if args.phase == 'score':
        score(cases)
    if args.phase in ['replay','all']:
        replay(cases)
    if args.phase in ['summarize','all']:
        summarize()
    if args.phase == 'package':
        package()
    assert before==protected(), 'Protected source changed during the experiment'


if __name__ == '__main__':
    main()
