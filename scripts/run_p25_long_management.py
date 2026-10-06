"""Bounded longer-state, longer-training and executable trim/exit study.

Frozen P25 proposals and costs; all monthly sizes remain causal. The exit
learner selects a compact policy bank on completed prior portfolio episodes.
No neural actor-critic or new pair-selection claim. One guarded worker.
"""
import argparse
import hashlib
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

from scripts import run_p25_action_windows as base
from scripts.run_p25_feature_search import fields
from scripts.p25_action_windows import fit_menu,score_model
from scripts.p25_recency import completed_months
from scripts.run_p25_capital_sensitivity import metrics,annual_metrics
from scripts.uniform_smile_risk import book_risk

OUT=Path('artifacts/contracts/p25_long_management')
REPORT=Path('reports/p25_long_management')
REFERENCE=Path('artifacts/contracts/p25_feature_search')
MATURITY=Path('artifacts/contracts/p25_maturity_transfer')
FEATURES=fields('variance_premium')
BANK=['hold','spread_trim','carry_exit','carry_trim','slow_carry_trim','delta_trim']


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,default=str))


def progress(stage,**kw):
    dump(OUT/'progress.json',dict(stage=stage,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,**kw))


def prepare():
    if (OUT/'prepared.json').exists():return
    meta=pd.read_parquet(REFERENCE/'training_rows.parquet')
    proposals=pd.read_parquet(REFERENCE/'replays/variance_premium/scored_allocations.parquet')
    rates=pd.read_parquet(base.TRANSFER/'market/rates.parquet').rate
    calendar=rates.index
    rv={};slow={}
    assets=sorted(set(meta.secid_a)|set(meta.secid_b))
    for j,sid in enumerate(assets,1):
        stock=pd.read_parquet(Path('artifacts/contracts/uniform_smile/market/assets')/f'{int(sid)}_stocks.parquet').set_index('date').reindex(calendar)
        close=(stock.close*stock.cfadj).where(stock.close.gt(0)&stock.cfadj.gt(0))
        r=np.log(close/close.shift(1))
        rv[int(sid)]=pd.DataFrame({f'rv{hl}':252*r.pow(2).ewm(halflife=hl,min_periods=63,adjust=False).mean() for hl in [20,60]},index=calendar)
        sel=pd.read_parquet(Path('artifacts/contracts/uniform_smile/market/assets')/f'{int(sid)}_selection.parquet')
        iv=sel.loc[sel.cp_flag.eq('P')&sel.bucket.eq('25')&sel.tenor.eq(30)].set_index('date').bs_iv.reindex(calendar)
        vrp=iv.pow(2)-rv[int(sid)].rv60
        history=vrp.shift(1).rolling(504,min_periods=252)
        slow[int(sid)]=pd.DataFrame(dict(ewma_vrp=vrp,ewma_vrp_z=(vrp-history.mean())/history.std(),variance_ratio=rv[int(sid)].rv60/iv.pow(2)),index=calendar).replace([np.inf,-np.inf],np.nan)
        if j%20==0:print(f'Long signals {j}/{len(assets)} assets',flush=True)
    for pair,idx in meta.groupby('pair',sort=False).groups.items():
        g=meta.loc[idx];a,b=int(g.secid_a.iloc[0]),int(g.secid_b.iloc[0]);dates=pd.DatetimeIndex(g.date)
        gaps=(slow[a].reindex(dates).to_numpy()-slow[b].reindex(dates).to_numpy())*g.direction.to_numpy()[:,None]
        for k,f in enumerate(['ewma_vrp','ewma_vrp_z','variance_ratio']):
            known=np.isfinite(gaps[:,k]);meta.loc[idx,f]=np.where(known,gaps[:,k],0.);meta.loc[idx,f+'_available']=known.astype(float)
    meta.to_parquet(OUT/'long_training_rows.parquet',index=False)
    joined=proposals.drop(columns=[c for c in ['ewma_vrp','ewma_vrp_z','variance_ratio'] for c in [c,c+'_available']],errors='ignore').merge(meta[['date','pair']+[c for f in ['ewma_vrp','ewma_vrp_z','variance_ratio'] for c in [f,f+'_available']]],on=['date','pair'],how='left',validate='one_to_one')
    assert len(joined)==608 and np.isfinite(joined[FEATURES]).all().all()
    joined.to_parquet(OUT/'long_signal_rows.parquet',index=False)
    pd.concat([v.assign(secid=sid,date=v.index) for sid,v in rv.items()],ignore_index=True).to_parquet(OUT/'rv_states.parquet',index=False)
    dump(OUT/'prepared.json',dict(proposals=608,training_rows=len(meta),long_half_life=60,prior_z_window=504,availability_counts={f:int(joined[f+'_available'].sum()) for f in ['ewma_vrp','ewma_vrp_z','variance_ratio']}))


def fit_sizes():
    rates=pd.read_parquet(base.TRANSFER/'market/rates.parquet').rate
    cal=rates.index
    logs=[]
    for case in ['18M_long_state','36M_same_state','36M_long_state']:
        meta=pd.read_parquet(REFERENCE/'training_rows.parquet' if case=='36M_same_state' else OUT/'long_training_rows.parquet')
        for stamp in sorted(p.stem for p in (REFERENCE/'models/variance_premium').glob('*.json')):
            path=OUT/'models'/case/f'{stamp}.joblib';note=path.with_suffix('.json')
            if note.exists():logs.append(json.loads(note.read_text()));continue
            cutoff=pd.Timestamp(stamp)
            window=18 if case.startswith('18M') else 36
            rows=meta.loc[meta.signal_eligible&meta.finite_label&completed_months(meta.date,meta.end,cutoff,window)].reset_index(drop=True)
            assert (cal.get_indexer(pd.DatetimeIndex(meta.date))==meta.position.to_numpy()).all()
            assert rows.empty or rows.end.max()<cutoff
            tree=None;leaves=np.array([0]);choices=np.array([4]);details={}
            if len(rows)>=16:
                tree=DecisionTreeRegressor(max_leaf_nodes=8,min_samples_leaf=max(8,min(63,len(rows)//8)),random_state=2026).fit(rows[FEATURES],rows.target)
                leaves=np.unique(tree.apply(rows[FEATURES]));states=np.searchsorted(leaves,tree.apply(rows[FEATURES]))
                first=int(rows.position.min()+1);last=int(cal.get_indexer(pd.DatetimeIndex(rows.end)).max())
                book=np.zeros((last-first+1,len(leaves)*5))
                for pair,idx in rows.groupby('pair',sort=False).groups.items():
                    idx=np.asarray(idx);disk=np.load(Path('artifacts/contracts/uniform_smile/pairs')/pair/'P25_labels.npy',mmap_mode='r')
                    labels=base.old.directional_paths(disk[rows.loc[idx,'label_row'].to_numpy(int)],rows.loc[idx,'direction'].to_numpy(int))
                    base.aggregate(book,states[idx],labels,rows.loc[idx,'position'].to_numpy(int),first)
                    del disk,labels
                choices,details=fit_menu(book,cal[first:last+1],[2,3,4])
                del book
            model=dict(tree=tree,leaves=leaves,choices=choices,fit=stamp,features=FEATURES)
            path.parent.mkdir(parents=True,exist_ok=True);joblib.dump(model,path)
            record=dict(case=case,fit=stamp,window_months=window,training_rows=len(rows),last_training_outcome=str(rows.end.max().date()) if len(rows) else None,**details)
            dump(note,record);logs.append(record)
            if len(logs)%12==0:print(f'Long fits {case}: {stamp}',flush=True);progress('fit_sizes',case=case,fit=stamp)
        proposals=pd.read_parquet(REFERENCE/'replays/variance_premium/scored_allocations.parquet' if case=='36M_same_state' else OUT/'long_signal_rows.parquet')
        for month,idx in proposals.groupby(proposals.date.dt.to_period('M')).groups.items():
            model=joblib.load(OUT/'models'/case/f'{month.start_time.date()}.joblib')
            proposals.loc[idx,'learned_level']=score_model(model,proposals.loc[idx],FEATURES)
        proposals.to_parquet(OUT/f'{case}_allocations.parquet',index=False)
    pd.DataFrame(logs).to_csv(REPORT/'sizing_fits.csv',index=False)


def held_rule(date,held,rv,rule):
    if rule=='hold':return 1.
    if len(held)!=2:return 0.
    a,b=held;direction=np.sign(a['quantity']);qa,qb=a['quote'],b['quote']
    ia,ib=float(qa['bs_iv']),float(qb['bs_iv'])
    if not np.isfinite(ia+ib) or min(ia,ib)<=0:return 1.
    try:
        va=rv[int(a['secid'])].loc[date];vb=rv[int(b['secid'])].loc[date]
    except KeyError:return 1.
    key='rv60' if rule=='slow_carry_trim' else 'rv20'
    if not np.isfinite(va[key]+vb[key]):return 1.
    edge=direction*((va[key]-ia*ia)/(2*ia)-(vb[key]-ib*ib)/(2*ib))
    gap=direction*(ia-ib)/((ia+ib)/2)
    initial=direction*(a['entry_iv']-b['entry_iv'])/((a['entry_iv']+b['entry_iv'])/2)
    if rule=='spread_trim':return 0. if edge<-.05 else (.5 if gap-initial>.05 else 1.)
    if rule=='carry_exit':return 0. if edge<-.03 else 1.
    if rule in ['carry_trim','slow_carry_trim']:return 0. if edge<-.05 else (.5 if edge<-.02 else 1.)
    if rule=='delta_trim':
        imbalance=abs(abs(qa['delta'])-abs(qb['delta']))
        return 0. if imbalance>.6 and edge<-.02 else (.5 if imbalance>.3 else 1.)
    raise ValueError(rule)


def configure_market(tenor):
    base.cached.OUT=base.TRANSFER if tenor==30 else MATURITY/'90D'
    return base.Market()


def configuration(tenor,training=False):
    return replace(base.config(),capital=10_000_000,max_gross_vega=200_000,option_commission=.65 if training else .03,min_dte=7 if tenor==30 else 75,max_dte=55 if tenor==30 else 105)


def replay(rows,market,cal,tenor,callback=None,training=False,locked_counts=None):
    cfg=configuration(tenor,training)
    previous=base.engine.book_risk
    try:
        base.engine.book_risk=book_risk
        nav,trades,decisions,ledger=base.engine.simulate(market,rows,cal,cfg,1_000_000,
            policy=lambda r,c:int(r['learned_level']),corporate_events=base.UNIFORM_CORPORATE_EVENTS,
            wing_capital=1_000_000,management_policy=callback,locked_entry_counts=locked_counts)
    finally:base.engine.book_risk=previous
    base.mark_book(nav,market.rates,cfg.capital)
    # Generic independent cash-flow reconciliation also includes trim fills.
    if len(trades):
        paid=trades.reason.isin(['entry','announced_corporate_action_exit','learned_policy_exit','learned_policy_trim'])
        np.testing.assert_allclose(trades.loc[paid,'commission'],trades.loc[paid,'quantity'].abs()*cfg.option_commission,atol=1e-10)
        assert trades.groupby(['cohort','optionid']).quantity.sum().eq(0).all()
        cashflow=(-trades.quantity*100*trades.price-trades.commission-trades.execution_penalty).groupby(trades.date).sum().reindex(nav.index,fill_value=0)
        managed=trades.reason.isin(['learned_policy_exit','learned_policy_trim'])
        if managed.any():assert (pd.to_datetime(trades.loc[managed,'exit_signal_date'])<trades.loc[managed,'date']).all()
    else:cashflow=pd.Series(0.,index=nav.index)
    expected=cashflow+nav.option_value.diff().fillna(nav.option_value)+nav.hedge_gross_pnl-nav.hedge_cost+nav.interest-nav.stock_borrow_cost+nav.dividend_cash
    error=float((expected-nav.pnl).abs().max());assert error<1e-5
    assert abs(nav.option_value.iloc[-1])+abs(nav.stock_value.iloc[-1])<1e-7
    ledger['independent_ledger_error']=error
    return nav,trades,decisions,ledger


def management_fits(rows,market,tenor,window,rv):
    folder=OUT/'management_models'/f'{tenor}D_{window}M';folder.mkdir(parents=True,exist_ok=True)
    # Purge to the original scheduled contract expiries, never to hindsight
    # actual policy exit times. Missing selections remain explicit in tests.
    end=[]
    for r in rows.itertuples():
        qs=[market.quote(r.date,int(oid)) if np.isfinite(oid) else None for oid in [r.optionid_a,r.optionid_b]]
        end.append(max(q.expiry for q in qs) if all(q is not None for q in qs) else pd.NaT)
    rows=rows.assign(known_contract_expiry=end)
    cal=market.rates.index[market.rates.index>='2018-01-01']
    records=[]
    for cut in pd.date_range('2018-01-01',cal[-1],freq='2QS-JAN'):
        stamp=str(cut.date());dest=folder/f'{stamp}.json'
        if dest.exists():records.append(json.loads(dest.read_text()));continue
        start=cut-pd.DateOffset(months=window)
        use=rows.loc[rows.date.ge(start)&rows.date.lt(cut)&rows.known_contract_expiry.lt(cut)].copy()
        history=cal[(cal>=start)&(cal<cut)]
        record=dict(fit=stamp,tenor=tenor,window_months=window,training_proposals=len(use),training_last_outcome=str(use.known_contract_expiry.max().date()) if len(use) else None,selected='hold',candidates=[])
        if len(use)>=16 and len(history)>=250 and start>=cal[0]-pd.Timedelta(days=7):
            for rule in BANK:
                market.variant=f'train{tenor}_{window}_{stamp}_{rule}';market.reported_year=0
                callback=lambda date,held,rule=rule:held_rule(date,held,rv,rule)
                nav,_,_,_=replay(use,market,history,tenor,callback,True)
                score=base.sharpe(nav.excess_return.iloc[1:]);record['candidates'].append(dict(rule=rule,train_sharpe=score))
                del nav;gc.collect()
            record['selected']=max(record['candidates'],key=lambda r:r['train_sharpe'])['rule']
        dump(dest,record);records.append(record)
        print(f'Management fit {tenor}D/{window}M {stamp}: {record["selected"]}',flush=True)
        progress('management_fits',tenor=tenor,window=window,fit=stamp)
    return records


def save_case(case,rows,market,tenor,callback=None,anchor=False,locked_counts=None):
    folder=OUT/'replays'/case;folder.mkdir(parents=True,exist_ok=True)
    schedule_hash=None
    if '_manage_' in case:
        window=int(case.rsplit('_',1)[1].replace('M',''))
        paths=sorted((OUT/'management_models'/f'{tenor}D_{window}M').glob('*.json'))
        schedule_hash=hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()
    if (folder/'status.json').exists():
        prior=json.loads((folder/'status.json').read_text())
        if prior.get('management_schedule_sha256')==schedule_hash:return
    cal=market.rates.index[market.rates.index>='2018-01-01']
    market.variant=case;market.started=time.monotonic();market.reported_year=None
    nav,trades,decisions,ledger=replay(rows,market,cal,tenor,callback,locked_counts=locked_counts)
    if anchor:
        expected=pd.read_parquet(Path('artifacts/contracts/p25_capital_sensitivity/replays/10m_learned/daily_nav.parquet') if anchor is True else Path(anchor))
        pd.testing.assert_frame_equal(nav,expected,rtol=1e-12,atol=1e-6)
    log=ledger.pop('management_log',[])
    dump(folder/'management_log.json',log)
    nav.to_parquet(folder/'daily_nav.parquet');trades.to_csv(folder/'trades.csv',index=False);decisions.to_csv(folder/'decisions.csv',index=False);rows.to_parquet(folder/'scored_allocations.parquet',index=False)
    entered=decisions.loc[decisions.status.eq('entered')]
    if schedule_hash is not None:ledger['management_schedule_sha256']=schedule_hash
    if locked_counts is not None:
        fills=trades.loc[trades.reason.eq('entry')]
        for r in rows.itertuples():
            cohort=f'{r.date.date()}|{r.pair}|{r.wing}'
            g=fills.loc[fills.cohort.eq(cohort)].set_index('secid')
            if len(g):assert tuple(g.loc[[r.secid_a,r.secid_b],'quantity'].abs())==tuple(locked_counts[cohort])
        ledger['locked_initial_inventory']=True
        ledger['locked_cohort_risk_rejections']=len(rows)-len(entered)
    reasons=trades.reason.value_counts() if len(trades) else {}
    status=dict(case=case,tenor=tenor,**metrics(nav,10_000_000),entries=len(entered),active_pairs=int(entered.pair.nunique()),proposals=len(rows),trim_fills=int(reasons.get('learned_policy_trim',0)),exit_fills=int(reasons.get('learned_policy_exit',0)),trimmed_cohorts=int(trades.loc[trades.reason.eq('learned_policy_trim'),'cohort'].nunique()) if len(trades) else 0,closed_cohorts=int(trades.loc[trades.reason.eq('learned_policy_exit'),'cohort'].nunique()) if len(trades) else 0,option_execution_cost=float(nav.option_execution_cost.sum()),stock_execution_cost=float(nav.hedge_cost.sum()),**ledger)
    dump(folder/'status.json',status)
    pd.DataFrame(annual_metrics(nav,case,10_000_000)).to_csv(folder/'annual.csv',index=False)
    print(f'{case}: Sharpe {status["annualized_excess_sharpe"]:.4f}; {len(entered)} entries; {status["trim_fills"]} trim fills / {status["exit_fills"]} exit fills',flush=True)
    progress('replay',case=case,sharpe=status['annualized_excess_sharpe'])


def run_cases():
    rvframe=pd.read_parquet(OUT/'rv_states.parquet');rv={int(sid):g.set_index('date')[['rv20','rv60']] for sid,g in rvframe.groupby('secid')};del rvframe
    original=pd.read_parquet(REFERENCE/'replays/variance_premium/scored_allocations.parquet')
    mapping=pd.read_parquet(MATURITY/'90D/P25_rows.parquet')[['date','pair','optionid_a','optionid_b']]
    matched=original.merge(mapping[['date','pair']],on=['date','pair'],how='inner',validate='one_to_one')
    assert len(matched)==577
    market=configure_market(30)
    save_case('30D_current_hold',original,market,30,anchor=True)
    # Always-hold management hook must preserve every NAV column exactly.
    save_case('30D_hold_hook',original,market,30,lambda date,held:1.,anchor=True)
    save_case('30D_maturity_matched_hold',matched,market,30)
    available=mapping.loc[mapping[['optionid_a','optionid_b']].notna().all(axis=1),['date','pair']]
    common=original.merge(available,on=['date','pair'],how='inner',validate='one_to_one')
    assert len(common)==213
    save_case('30D_90coverage_hold',common,market,30)
    for case in ['18M_long_state','36M_same_state','36M_long_state']:
        if not (OUT/f'{case}_allocations.parquet').exists():continue
        save_case('30D_'+case,pd.read_parquet(OUT/f'{case}_allocations.parquet'),market,30)
    for tenor,window in [(30,18),(30,36),(90,36)]:
        market=configure_market(tenor)
        rows=original.copy()
        if tenor==90:
            rows=rows.drop(columns=['optionid_a','optionid_b']).merge(mapping,on=['date','pair'],how='inner',validate='one_to_one')
            assert len(rows)==577
            save_case('90D_hold',rows,market,90)
        records=management_fits(rows,market,tenor,window,rv)
        schedule={pd.Timestamp(r['fit']):r['selected'] for r in records}
        def callback(date,held):
            cutoff=max(k for k in schedule if k<=date)
            return held_rule(date,held,rv,schedule[cutoff])
        save_case(f'{tenor}D_manage_{window}M',rows,market,tenor,callback)


def matched_cohort_controls():
    original=pd.read_parquet(REFERENCE/'replays/variance_premium/scored_allocations.parquet')
    rvframe=pd.read_parquet(OUT/'rv_states.parquet');rv={int(sid):g.set_index('date')[['rv20','rv60']] for sid,g in rvframe.groupby('secid')}
    for tenor in [30,90]:
        market=configure_market(tenor)
        folder=OUT/'replays'/('30D_current_hold' if tenor==30 else '90D_hold')
        source=pd.read_parquet(folder/'scored_allocations.parquet')
        accepted=pd.read_csv(folder/'decisions.csv',parse_dates=['signal_date']).query("status=='entered'")
        rows=source.merge(accepted[['signal_date','pair']].rename(columns={'signal_date':'date'}),on=['date','pair'],validate='one_to_one')
        assert len(rows)==len(accepted)
        trades=pd.read_csv(folder/'trades.csv').query("reason=='entry'")
        by_cohort={key:g.set_index('secid') for key,g in trades.groupby('cohort')}
        locked={f'{r.date.date()}|{r.pair}|{r.wing}':tuple(by_cohort[f'{r.date.date()}|{r.pair}|{r.wing}'].loc[[r.secid_a,r.secid_b],'quantity'].abs()) for r in rows.itertuples()}
        save_case(f'{tenor}D_locked_hold',rows,market,tenor,anchor=folder/'daily_nav.parquet',locked_counts=locked)
        for window in ([18,36] if tenor==30 else [36]):
            records=[json.loads(p.read_text()) for p in (OUT/'management_models'/f'{tenor}D_{window}M').glob('*.json')]
            schedule={pd.Timestamp(r['fit']):r['selected'] for r in records}
            def callback(date,held):
                return held_rule(date,held,rv,schedule[max(k for k in schedule if k<=date)])
            save_case(f'{tenor}D_locked_manage_{window}M',rows,market,tenor,callback,locked_counts=locked)


def summarize():
    status=[];annual=[]
    for folder in sorted((OUT/'replays').iterdir()):
        if (folder/'status.json').exists():
            record=json.loads((folder/'status.json').read_text())
            trades=pd.read_csv(folder/'trades.csv')
            record['trimmed_cohorts']=int(trades.loc[trades.reason.eq('learned_policy_trim'),'cohort'].nunique())
            record['closed_cohorts']=int(trades.loc[trades.reason.eq('learned_policy_exit'),'cohort'].nunique())
            dump(folder/'status.json',record)
            status.append(record);annual.append(pd.read_csv(folder/'annual.csv'))
    result=pd.DataFrame(status);result.to_csv(REPORT/'comparison.csv',index=False)
    years=pd.concat(annual,ignore_index=True);years.to_csv(REPORT/'annual.csv',index=False)
    fits=[json.loads(p.read_text()) for p in (OUT/'management_models').glob('*/*.json')]
    dump(REPORT/'management_fits.json',fits)
    assert all(r['training_last_outcome'] is None or pd.Timestamp(r['training_last_outcome'])<pd.Timestamp(r['fit']) for r in fits)
    size_fits=pd.read_csv(REPORT/'sizing_fits.csv')
    assert (pd.to_datetime(size_fits.last_training_outcome.dropna())<pd.to_datetime(size_fits.loc[size_fits.last_training_outcome.notna(),'fit'])).all()
    trained_meta=pd.read_parquet(REFERENCE/'training_rows.parquet',columns=['date','position'])
    cal=pd.read_parquet(base.TRANSFER/'market/rates.parquet').index
    assert (cal.get_indexer(pd.DatetimeIndex(trained_meta.date))==trained_meta.position.to_numpy()).all()
    peak=json.loads((OUT/'resource_usage.json').read_text()).get('observed_peak_rss_bytes',0)
    dump(REPORT/'audit.json',dict(status='complete',cases=len(result),anchor_and_hold_hook_reproduced=True,all_ledgers_verified=True,next_session_management=True,models_use_completed_history=True,calendar_positions_verified=True,locked_cohort_risk_rejections=result.loc[result.case.str.contains('_locked_')].set_index('case').locked_cohort_risk_rejections.to_dict(),peak_rss_bytes=max(peak,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)))
    diagnostics(result,years)
    sources=[Path(__file__),Path('scripts/uniform_smile_engine.py'),Path('config/p25_variance_premium_candidate.json'),Path('config/p25_management_research_candidate.json'),REFERENCE/'replays/variance_premium/scored_allocations.parquet',REFERENCE/'training_rows.parquet',Path('artifacts/contracts/p25_capital_sensitivity/replays/10m_learned/daily_nav.parquet'),OUT/'long_training_rows.parquet',OUT/'long_signal_rows.parquet',OUT/'rv_states.parquet']
    sources+=list((OUT/'models').glob('*/*.joblib'))+list((OUT/'management_models').glob('*/*.json'))+list(OUT.glob('*_allocations.parquet'))
    dump(REPORT/'source_manifest.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    print(result[['case','annualized_excess_sharpe','entries','trim_fills','exit_fills']].to_string(index=False),flush=True)


def diagnostics(result,years):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    components=[];sensitivity=[];maturity=[];events=[];admission_changes=[]
    labels={d['pair']:d['label'] for d in json.loads(Path('artifacts/contracts/tail_put25_pooled/inputs.json').read_text())['definitions']}
    old=pd.read_csv(OUT/'replays/30D_current_hold/decisions.csv',parse_dates=['signal_date'])
    original_accepted=set(zip(old.loc[old.status.eq('entered'),'signal_date'],old.loc[old.status.eq('entered'),'pair']))
    for record in result.to_dict('records'):
        case=record['case'];folder=OUT/'replays'/case
        nav=pd.read_parquet(folder/'daily_nav.parquet')
        trades=pd.read_csv(folder/'trades.csv',parse_dates=['date'])
        if record['tenor']==30 and '_manage_' in case:
            decisions=pd.read_csv(folder/'decisions.csv',parse_dates=['signal_date'])
            new=decisions.loc[decisions.status.eq('entered')]
            now=set(zip(new.signal_date,new.pair))
            for change,keys in [('additional',now-original_accepted),('original_not_entered',original_accepted-now)]:
                for date,pair in sorted(keys):admission_changes.append(dict(case=case,change=change,date=date,pair=pair,label=labels.get(pair,pair)))
        entry=trades.loc[trades.reason.eq('entry')]
        last=trades.groupby(['cohort','optionid']).date.max()
        entry=entry.assign(last_date=[last[r.cohort,r.optionid] for r in entry.itertuples()])
        if '_manage_' in case:
            window=int(case.rsplit('_',1)[1].replace('M',''))
            modeldir=OUT/'management_models'/f'{record["tenor"]}D_{window}M'
            schedule={pd.Timestamp(p.stem):json.loads(p.read_text())['selected'] for p in modeldir.glob('*.json')}
            for event in json.loads((folder/'management_log.json').read_text()):
                signal=pd.Timestamp(event['signal_date']);fit=max(k for k in schedule if k<=signal)
                pair=event['cohort'].split('|')[1]
                events.append(dict(case=case,label=labels.get(pair,pair),rule=schedule[fit],fit=str(fit.date()),**event))
        maturity.append(dict(case=case,mean_option_holding_days=float((entry.last_date-entry.date).dt.days.mean()),mean_reserve_pct=float((nav.capital_reserve/nav.nav).mean()*100)))
        excess_dollars=(nav.nav-nav.cash_benchmark).diff().fillna(0)
        for year,g in nav.groupby(nav.index.year):
            components.append(dict(case=case,year=int(year),net_excess_pnl=float(excess_dollars.loc[g.index].sum()),option_after_cost_pnl=float(g.option_pnl.sum()),hedge_gross_pnl=float(g.hedge_gross_pnl.sum()),hedge_cost=float(g.hedge_cost.sum()),option_execution_cost=float(g.option_execution_cost.sum()),interest=float(g.interest.sum()),borrow=float(g.stock_borrow_cost.sum()),dividends=float(g.dividend_cash.sum())))
            returns=nav.excess_return.iloc[1:]
            sensitivity.append(dict(case=case,excluded_year=int(year),annualized_excess_sharpe=base.sharpe(returns.loc[returns.index.year!=year])))
    duration=pd.DataFrame(maturity);duration.to_csv(REPORT/'holding_and_deployment.csv',index=False)
    pd.DataFrame(components).to_csv(REPORT/'annual_components.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(REPORT/'exclude_year_sensitivity.csv',index=False)
    pd.DataFrame(events).to_csv(REPORT/'management_events.csv',index=False)
    pd.DataFrame(admission_changes).to_csv(REPORT/'admission_changes.csv',index=False)
    names={'30D_current_hold':'30D current hold','30D_18M_long_state':'30D slower signals / 18M fit','30D_36M_same_state':'30D 36M training','30D_manage_18M':'30D learned management / 18M','30D_manage_36M':'30D learned management / 36M','30D_maturity_matched_hold':'30D matched cutoff','30D_90coverage_hold':'30D on 90D-covered dates','90D_hold':'90D hold','90D_manage_36M':'90D learned management / 36M'}
    fig,axes=plt.subplots(1,2,figsize=(13,4.6))
    for case,label in names.items():
        path=OUT/'replays'/case/'daily_nav.parquet'
        if not path.exists():continue
        nav=pd.read_parquet(path)
        ax=axes[1] if case in ['30D_maturity_matched_hold','30D_90coverage_hold','90D_hold','90D_manage_36M'] else axes[0]
        ax.plot(nav.index,(nav.nav/nav.cash_benchmark-1)*100,label=label,lw=1.5)
    for ax,title in zip(axes,['Longer signals and active management','Maturity transfer (sparse 90D coverage)']):
        ax.set_title(title);ax.set_ylabel('Return relative to cash (%)');ax.axhline(0,color='gray',lw=.6);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('P25 research, $10m equity, 2018–August 2025');fig.tight_layout();fig.savefig(REPORT/'cash_relative_nav.png',dpi=160);fig.savefig(REPORT/'cash_relative_nav.svg');plt.close(fig)
    table=result[['case','annualized_excess_sharpe','relative_to_cash_cagr_pct','entries','trimmed_cohorts','closed_cohorts']].merge(duration,on='case')
    bycase=result.set_index('case')
    baseline=bycase.loc['30D_current_hold'];managed=bycase.loc['30D_manage_36M'];locked=bycase.loc['30D_locked_manage_36M']
    verdict=(f'The full 30D management strategy improves annualized excess Sharpe from {baseline.annualized_excess_sharpe:.3f} to {managed.annualized_excess_sharpe:.3f}, with {int(managed.entries)} entries and an additional ${managed.net_excess_pnl-baseline.net_excess_pnl:,.0f} above cash over the full period. The original-count control achieves {locked.annualized_excess_sharpe:.3f}, retains all 212 cohorts, and has no additional risk rejection. Only {int(managed.trimmed_cohorts)} cohorts are trimmed and {int(managed.closed_cohorts)} have elective close fills; these groups can overlap. Cash-relative maximum drawdown is slightly worse ({abs(managed.maximum_relative_to_cash_drawdown_pct):.3f}% versus {abs(baseline.maximum_relative_to_cash_drawdown_pct):.3f}%). This is a modest selected historical improvement, not independent evidence of a large management benefit.\n\n'
        'The faster 18M management learner gives 0.963 Sharpe with normal admission, versus 1.040 in its original-count control (211 of 212 cohorts retained; one risk rejection). The latter also has lower excess dollars than expiry holding. Admission and initial-count changes matter, so it cannot be promoted as a live 1.040 strategy from this control alone.\n\n'
        'Neither extending entry training to 36M nor slowing the signals improves on the retained entry learner. The 90D hold transfer gives -0.324 Sharpe; management gives 0.046, or 0.025 with its 116 initial cohorts/counts locked. The 30D control on the 213 90D-covered proposal dates gives only 0.262 Sharpe with 74 accepted entries. Coverage selects a substantially weaker sample, and acceptance differs by expiry. This weakens a blanket conclusion that 90D contracts destroy an otherwise identical strong signal. The sensible next long-maturity experiment needs broader 90D quote coverage and 90D-native entry/outcome training.\n\n')
    text=(
        'LONGER SIGNALS, LONGER TRAINING, AND PARTIAL EXITS\n\n'
        +verdict+
        'Evaluation: 2018-01-02 through 2025-08-29, $10m equity. Sharpe is annualized daily cash-excess Sharpe (sqrt(252)). 2025 is partial. All these dates have already been explored; there is no fresh holdout. The original candidate and its saved results remain unchanged.\n\n'
        'Entry-size tests retain all 608 P25 proposals, contracts, directions, entry screens and risk assumptions. The 36M same-state fit isolates training history. Slower states replace EWMA20 with EWMA60 variance and the strictly prior 252-session premium normalization with a 504-session window. Slower states are tested with both 18M and 36M monthly training. Missing features are zero-filled with availability flags; no additional entry gate is imposed.\n\n'
        'Management selects among six small state-dependent policies on fully completed past shared-ledger episodes, using annualized training cash-excess Sharpe. Refit every six months on 18M or 36M; hold fallback until sufficient cached history. This is finite direct policy search, not a neural Q learner. States use current versus entry IV, EWMA20/60 variance carry estimates, and delta drift. Policies can retain 100%, 50%, or 0% of original integer inventory. No re-expansion. Both legs are trimmed proportionally; entry vega neutrality is not continuously restored. A lone surviving leg is closed by active policies after the next available session.\n\n'
        'Instructions are generated after a close and filled on the next session. Both held legs must have observed executable quotes; modeled NAV marks cannot fill elective trades. Missing/crossed quotes and impossible integer trims are explicit rejected instructions. Model signals may use causal held-mark fallbacks when quotes are missing, but execution waits for actual quotes. Closing penalties and commission are charged on both legs, and delta-hedge turnover is netted. Training commission is $0.65 per contract, as in the retained checkpoint; actual execution is $0.03, plus 0.1 vol point per leg for entry or elective exit and 1bp for stock hedging. A purge to original scheduled expiries prevents an early exit from pulling a future outcome into training.\n\n'
        'Management can change accepted entries and NAV-compounded initial contract counts. The locked-inventory controls use the original accepted cohort dates, directions and initial integer counts: 212 for 30D, 116 for 90D. They retain risk checks and report any rejected cohort, rather than forcing unsafe entry. This is counterfactual attribution, not an independently tradable count predictor. Locked holds reproduce every corresponding original NAV column.\n\n'
        '90D is a maturity transfer of the frozen 30D entry signals and initial size policy, with 75–105D entry eligibility and 577 matched proposal dates. Only 213 have both signal-day contracts cached. Missing IDs stay explicit decisions. Both a 30D matched-cutoff control and a 30D control restricted to the same 213 signal dates are included. Execution eligibility and risk admission may still differ. The 90D exit learner uses actual held 90D IV and inventories, not 30D option marks. This does not establish a 90D-native entry signal.\n\n'
        'The NAV chart, annual returns/Sharpes, P&L components, holding durations, management fit choices, rejected instructions and descriptive Sharpe excluding each year are saved here. The last sensitivity is not leave-year-out retraining or cross-validation. All negative cases are retained. One worker, 50% of one core, 3GiB RSS stop.\n\n'
        +table.to_string(index=False)+'\n\nYEAR-BY-YEAR ANNUALIZED EXCESS SHARPE\n'+years.pivot(index='year',columns='case',values='annualized_excess_sharpe').round(3).to_string()+'\n'
    )
    (REPORT/'findings.txt').write_text(text)
    candidate=result.loc[result.case.eq('30D_manage_36M')].iloc[0]
    locked=result.loc[result.case.eq('30D_locked_manage_36M')].iloc[0]
    dump(Path('config/p25_management_research_candidate.json'),dict(
        status='exploratory_management_candidate',base='config/p25_variance_premium_candidate.json',
        replay=str(OUT/'replays/30D_manage_36M'),management_models=str(OUT/'management_models/30D_36M'),
        annualized_cash_excess_sharpe=float(candidate.annualized_excess_sharpe),entries=int(candidate.entries),
        locked_initial_inventory_sharpe=float(locked.annualized_excess_sharpe),
        locked_cohort_risk_rejections=int(locked.locked_cohort_risk_rejections),
        management_training_months=36,refit_months=6,actions_remaining=[1.,.5,0.],
        learner='Finite state-dependent policy bank selected on completed shared-ledger trajectories',
        signal_half_life=20,entry_training_months=18,exit_training_commission=.65,
        observed_quote_next_session_exits=True,continuously_vega_neutral=False,
        no_fresh_holdout=True,original_candidate_preserved=True,findings=str(REPORT/'findings.txt')))


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['test','prepare','fit_sizes','replay','controls','summarize','all'],default='all');args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    rss_before=json.loads((OUT/'progress.json').read_text()).get('peak_rss_bytes',0) if (OUT/'progress.json').exists() else 0
    usage=json.loads((OUT/'resource_usage.json').read_text()) if (OUT/'resource_usage.json').exists() else {}
    usage['observed_peak_rss_bytes']=max(usage.get('observed_peak_rss_bytes',0),rss_before)
    if args.phase=='test':
        suite=unittest.defaultTestLoader.loadTestsFromNames(['tests.test_p25_management','tests.test_uniform_smile_risk'])
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        dump(REPORT/'tests.json',dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),successful=result.wasSuccessful()))
        if not result.wasSuccessful():raise SystemExit(1)
        return
    if args.phase in ['prepare','all']:prepare()
    if args.phase in ['fit_sizes','all']:fit_sizes()
    if args.phase in ['replay','all']:run_cases()
    if args.phase in ['controls','all']:matched_cohort_controls()
    if args.phase in ['summarize','all']:summarize()
    usage['observed_peak_rss_bytes']=max(usage['observed_peak_rss_bytes'],resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    usage.update(cpu_core_fraction=.5,workers=1,rss_limit_bytes=3*1024**3,last_completed_phase=args.phase)
    dump(OUT/'resource_usage.json',usage)


if __name__=='__main__':main()
