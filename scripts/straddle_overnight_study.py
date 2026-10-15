"""Declared feature/learner study of actual-contract short-straddle entry sizing.

All numerical phases require the single-worker research guard. Prior experiments
and immutable reward archives are reused without overwriting their outputs.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from scripts import straddle_feature_transfer as parent
from scripts import straddle_transfer_fit as streaming
from paper_code.metrics import sharpe,period_metrics,paired_interval

CONFIG=Path('config/straddle_overnight_study.json');CFG=json.loads(CONFIG.read_text())
OUT,REPORT=Path(CFG['output_root']),Path(CFG['report_root']);DATA=Path(CFG['source_root'])
FORECAST=Path(CFG['forecast_root'])
CORE=['residual_memory','iv','vix']
TRANSFER=CORE+['profit_91d','profit_365d','count_91d','count_365d',
    'profit_91d_available','profit_365d_available','variance_ratio','ewma_vrp_z',
    'iv_rank','dte','commission_vol_points','quoted_vol_width','normalized_memory','fast_rv_ratio']
MARKET=['spot_market_beta126','spot_market_corr126','iv_vix_beta126','iv_vix_corr126',
        'iv_spot_beta126','jump_share63']
PREDICTED=['forecast_gap','forecast_gap_yield','forecast_variance_carry','forecast_disagreement','forecast_available']
FEATURES=dict(core=CORE,transfer=TRANSFER,market=TRANSFER+parent.expanded(MARKET),
              market_forecast=TRANSFER+parent.expanded(MARKET)+PREDICTED)
ORIGINAL_FIT=parent.core.fit_model


def specifications():
    result=[dict(name=f'{f}_tree4_w{w}',features=f,kind='tree',leaves=4,months=w)
            for f in CFG['feature_groups'] for w in CFG['training_windows_months']]
    result += [dict(name=f'market_{kind}_w{w}',features='market',kind='tree' if kind=='tree8' else 'ridge',
                    leaves=8 if kind=='tree8' else 3,ridge=10.,months=w)
               for kind in ['tree8','ridge'] for w in CFG['training_windows_months']]
    return result


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def sources():
    paths=[Path(__file__),CONFIG,Path(parent.__file__),Path(streaming.__file__),
           Path(parent.core.__file__),Path(parent.unit.__file__),Path(parent.unit.engine.__file__),
           DATA/'training_manifest.json',FORECAST/'training_manifest.json',
           Path('scripts/run_capped_straddle_book.py'),Path('config/atm_rv_persistence.json')]
    value={str(p):digest(p) for p in paths};p=OUT/'sources.json'
    if p.exists():assert json.loads(p.read_text())==value,'Changed sources; create a new study root'
    else:parent.core.dump(p,value)


def returns_for(sid,cal):
    stock=pd.read_parquet(parent.unit.base.SOURCE/f'{sid}_stocks.parquet',
        columns=['date','close','cfadj'],use_threads=False).set_index('date').reindex(cal)
    for r in parent.unit.base.CFG.get('observed_stock_repairs',[]):
        if r['secid']==sid:stock.loc[pd.Timestamp(r['date']),['close','cfadj']]=[r['close'],r['cfadj']]
    p=(stock.close*stock.cfadj).where(stock.close.gt(0)&stock.cfadj.gt(0))
    return np.log(p/p.shift(1))


def market_proxy():
    p=OUT/'market_proxy.parquet'
    if p.exists():return pd.read_parquet(p,use_threads=False)
    cal=parent.unit.base.calendar();total=np.zeros(len(cal));count=np.zeros(len(cal))
    # Existing retrospective native universe; each return is observed at that
    # day's close. Remove known ETF rows rather than double-count index holdings.
    assets=json.loads(Path('data/submission/artifacts/broad_pairs/assets.json').read_text())
    etfs={int(a['secid']) for a in assets if a.get('etf')}
    ids=[i for i in parent.unit.base.assets() if i not in etfs]
    for sid in ids:
        r=returns_for(sid,cal).to_numpy(float);valid=np.isfinite(r)
        total+=np.where(valid,r,0.);count+=valid
    result=pd.DataFrame(dict(return_sum=total,return_count=count),index=cal)
    result.to_parquet(p)
    parent.core.dump(REPORT/'market_proxy.json',dict(secids=ids,excluded_cached_etfs=sorted(etfs),
        definition='Equal-weight log-return proxy from available cached stock closes; each name is removed from its own market proxy. No future returns or option outcomes.',
        minimum_available_names=50,formation='Uses the already retrospective cached universe; not a point-in-time index.'))
    return result


def beta_corr(y,x,window=126,minimum=63):
    joined=pd.concat([y.rename('y'),x.rename('x')],axis=1)
    y=joined.y.where(joined.x.notna());x=joined.x.where(joined.y.notna())
    covariance=y.rolling(window,min_periods=minimum).cov(x)
    variance=x.rolling(window,min_periods=minimum).var()
    return covariance/variance.where(variance>1e-12),y.rolling(window,min_periods=minimum).corr(x)


def market_features(r,iv,vix,proxy):
    finite=r.notna().astype(float)
    factor=(proxy.return_sum-r.fillna(0))/(proxy.return_count-finite).where(proxy.return_count-finite>=50)
    b,c=beta_corr(r,factor);iv_change=iv.diff();vix_change=(vix/100).diff()
    vb,vc=beta_corr(iv_change,vix_change)
    sb,_=beta_corr(iv_change/iv.shift(1).where(iv.shift(1)>0),r)
    sq=r.pow(2)
    jump=sq.rolling(63,min_periods=42).max()/sq.rolling(63,min_periods=42).sum().where(lambda s:s>0)
    return pd.DataFrame(dict(spot_market_beta126=b,spot_market_corr126=c,
        iv_vix_beta126=vb,iv_vix_corr126=vc,iv_spot_beta126=sb,jump_share63=jump))


def prepare_features(limit):
    folder=OUT/'features';folder.mkdir(parents=True,exist_ok=True)
    proxy=market_proxy();cal=proxy.index
    v=pd.read_csv('data/raw/VIX_History.csv',usecols=['DATE','CLOSE'],parse_dates=['DATE']).set_index('DATE').CLOSE
    vix=v.sort_index().reindex(cal,method='ffill')
    roster=pd.read_parquet(DATA/'base_proposals.parquet',columns=['date','secid'],use_threads=False)
    done=0
    for sid,g in roster.groupby('secid',sort=True):
        p=folder/f'{sid}.parquet'
        if p.exists():continue
        if done>=limit:break
        r=returns_for(int(sid),cal)
        iv=pd.read_parquet(parent.unit.base.OUT/'panel'/f'{sid}.parquet',
                           columns=['date','iv'],use_threads=False).set_index('date').iv.reindex(cal)
        f=market_features(r,iv,vix,proxy).reindex(g.date).reset_index(drop=True)
        f['date']=g.date.to_numpy();f['secid']=int(sid)
        for c in MARKET:
            f[c]=f[c].replace([np.inf,-np.inf],np.nan)
            f[c+'_available']=f[c].notna().astype(float);f[c]=f[c].fillna(0.)
        f.to_parquet(p,index=False);done+=1
        del f;gc.collect();pa.default_memory_pool().release_unused()
        if done%20==0:print(json.dumps(dict(phase='market_features',new_names=done)),flush=True)
    print(json.dumps(dict(phase='market_features_done',total=len(list(folder.glob('*.parquet'))))),flush=True)


def assemble():
    base_cols=list(dict.fromkeys(parent.unit.bandit.EXECUTION+['secid']+
        [c for c in TRANSFER if c not in ['normalized_memory','fast_rv_ratio']]+['rv5']))
    rows=pd.read_parquet(DATA/'proposals.parquet',columns=base_cols,use_threads=False)
    rows['normalized_memory']=-rows.residual_memory/rows.iv.clip(lower=.01)
    rows['fast_rv_ratio']=rows.rv5/rows.iv.clip(lower=.01)
    for c in ['count_91d','count_365d']:rows[c]=np.log1p(rows[c].clip(lower=0))
    f=pd.concat([pd.read_parquet(p,use_threads=False) for p in sorted((OUT/'features').glob('*.parquet'))],ignore_index=True)
    rows=rows.merge(f,on=['date','secid'],how='left',validate='one_to_one');del f
    predictions=pd.read_parquet(FORECAST/'proposals.parquet',columns=['date','secid']+PREDICTED,use_threads=False)
    rows=rows.merge(predictions,on=['date','secid'],how='left',validate='one_to_one');del predictions
    names=list(dict.fromkeys(sum(FEATURES.values(),[])))
    assert len(rows)==35774 and np.isfinite(rows[names]).all().all()
    rows.to_parquet(OUT/'proposals.parquet',index=False)
    p=OUT/'base_proposals.parquet'
    if not p.exists():p.symlink_to((DATA/'base_proposals.parquet').resolve())
    meta=pd.read_parquet(DATA/'training.parquet',use_threads=False)
    meta=meta.drop(columns=[c for c in names if c not in ['date','secid'] and c in meta])
    meta=meta.merge(rows[['date','secid']+names],on=['date','secid'],how='left',validate='many_to_one')
    assert np.isfinite(meta.loc[meta.finite_label,names]).all().all()
    meta.to_parquet(OUT/'training.parquet',index=False)
    parent.core.dump(OUT/'training_manifest.json',{str(p):digest(p) for p in
        [OUT/'proposals.parquet',OUT/'training.parquet',DATA/'training_manifest.json']})
    parent.core.dump(REPORT/'protocol.json',dict(configuration=CFG,specifications=specifications(),
        features=FEATURES,proposals=len(rows),valid_rewards=int(meta.finite_label.sum()),
        actions=[.5,.75,1.],unchanged_costs_and_risk_limits=True,
        feature_availability={c:float(rows[c+'_available'].mean()) for c in MARKET},
        reward_scope='Exact independently simulated integer-contract reward paths; training omits funded admission interactions, which are applied in actual replay.',
        interpretation='Compare matched feature groups within a training window; state-count and ridge variants are separate learner ablations.'))
    parent.core.dump(REPORT/'evaluation_plan.json',dict(control='core_tree4_w36',selection='Frozen validation selectors plus one validation winner per feature group.'))
    print(json.dumps(dict(phase='assembled',proposals=len(rows),procedures=len(specifications()))),flush=True)


def verify_training():
    for p,h in json.loads((OUT/'training_manifest.json').read_text()).items():assert digest(p)==h,p


def fit_model(meta,paths,cutoff,spec,cal):
    old=parent.CFG['minimum_leaf_samples']
    if spec.get('leaves')==8:
        n=len(parent.unit.bandit.completed_rows(meta,cutoff,spec['months']))
        parent.CFG['minimum_leaf_samples']=max(8,min(63,n//8))
    try:
        model=ORIGINAL_FIT(meta,paths,cutoff,spec,cal)
        model['effective_minimum_leaf']=parent.CFG['minimum_leaf_samples'];return model
    finally:parent.CFG['minimum_leaf_samples']=old


def configure():
    parent.configure();parent.CFG=dict(CFG,specifications=specifications());parent.OUT=OUT;parent.REPORT=REPORT
    parent.core.CFG=parent.CFG;parent.core.OUT=OUT;parent.core.REPORT=REPORT;parent.core.DATA=DATA
    parent.core.FEATURES=FEATURES;parent.core.fit_model=fit_model
    parent.unit.CFG=parent.CFG;parent.unit.OUT=OUT;parent.verify_training=verify_training


def selected_cases():return json.loads((REPORT/'selection.json').read_text())['later_cases']


def allocations(name,later):
    verify_training()
    if later:assert name in selected_cases()
    marker=OUT/'cases'/name/('evaluation_allocations.json' if later else 'validation_allocations.json')
    if marker.exists():return
    spec=next(s for s in specifications() if s['name']==name)
    roster=pd.read_parquet(OUT/'base_proposals.parquet',
        columns=list(dict.fromkeys(parent.unit.bandit.EXECUTION+['secid'])),use_threads=False)
    keys=pd.MultiIndex.from_frame(roster[['date','secid']]);assert keys.is_unique
    levels=np.full(len(roster),CFG['warmup_action'],int)
    meta,paths=parent.core.load_labels();cal=parent.unit.base.calendar()
    end=pd.Timestamp(CFG['evaluation_end'] if later else CFG['validation_end'])
    needed=list(dict.fromkeys(['date','secid']+FEATURES[spec['features']]))
    for month in pd.period_range(roster.date.min(),min(end,roster.date.max()),freq='M'):
        p=OUT/'models'/name/f'{month}.json'
        if p.exists():model=json.loads(p.read_text())
        else:model=fit_model(meta,paths,month.start_time,spec,cal);parent.core.dump(p,model)
        f=pd.read_parquet(OUT/'proposals.parquet',columns=needed,
            filters=[('date','>=',month.start_time),('date','<',(month+1).start_time)],use_threads=False)
        if len(f):streaming.scatter_levels(keys,f,parent.core.predict(model,f),levels)
        assert model['last_completed'] is None or pd.Timestamp(model['last_completed'])<month.start_time
        del f;gc.collect();pa.default_memory_pool().release_unused()
    roster['learned_level']=levels
    dest=OUT/'cases'/name/'allocations/tree18_test.parquet';dest.parent.mkdir(parents=True,exist_ok=True)
    roster.to_parquet(dest,index=False);parent.core.dump(marker,dict(case=name,later=later,proposals=len(roster)))
    print(json.dumps(dict(phase='allocations',case=name,later=later,proposals=len(roster))),flush=True)


def book(name,months,later):
    if later:
        assert name in selected_cases()
        # Funded helper has one-primary authorization; run its unchanged
        # execution body after enforcing this study's frozen multi-case list.
        parent.unit.OUT=OUT/'cases'/name
        original=parent.unit.engine.simulate
        parent.unit.engine.simulate=lambda *a,**k:parent.native_ladder(original,*a,**k)
        try:parent.unit.replay(months)
        finally:parent.unit.engine.simulate=original
    else:parent.book(name,months,False)


def validation_scores():
    baseline=pd.read_parquet('artifacts/contracts/atm_rv_straddle_fixed50/replays/tree18/daily_nav.parquet',use_threads=False).excess_return.loc['2018':'2021']
    records=[]
    for spec in specifications():
        nav,d=parent.core.funded_nav(spec['name'],CFG['validation_end']);r=nav.excess_return
        folds=[]
        for y in range(2018,2022):
            keep=r.index.year!=y;folds.append(sharpe(r.loc[keep])-sharpe(baseline.reindex(r.loc[keep].index)))
        records.append(dict(case=spec['name'],feature_group=spec['features'],specification=spec,
            robust_delta=float(min(folds)),leave_year_out_deltas=folds,
            **period_metrics(nav,d,'2018-01-01',CFG['validation_end'])))
    robust=sorted(records,key=lambda r:(-r['robust_delta'],-r['annualized_excess_sharpe'],r['case']))[0]
    point=sorted(records,key=lambda r:(-r['annualized_excess_sharpe'],r['case']))[0]
    winners={f:sorted([r for r in records if r['feature_group']==f],key=lambda r:(-r['robust_delta'],-r['annualized_excess_sharpe'],r['case']))[0]['case'] for f in CFG['feature_groups']}
    later=sorted(set(list(winners.values())+[robust['case'],point['case']]))
    record=dict(study_id='straddle_overnight_study',selected=robust['case'],point_selected=point['case'],
        feature_winners=winners,later_cases=later,candidates=records,
        preferred_policy=robust['case'] if robust['annualized_excess_sharpe']>sharpe(baseline) else 'fixed50',
        evaluation_status=CFG['evaluation_status'])
    p=REPORT/'selection.json'
    if p.exists():assert json.loads(p.read_text())==record
    else:parent.core.dump(p,record)
    pd.DataFrame(records).drop(columns=['specification']).to_csv(REPORT/'validation_candidates.csv',index=False)
    print(json.dumps(dict(primary=robust['case'],point=point['case'],later_cases=later,
        preferred=record['preferred_policy'],validation_sharpe=robust['annualized_excess_sharpe'])),flush=True)


def analyze(name):
    folder=OUT/'cases'/name/'replays/tree18';nav=pd.read_parquet(folder/'daily_nav.parquet',use_threads=False)
    d=pd.read_csv(folder/'decisions.csv',parse_dates=['signal_date','entry_date'])
    assert len(d)==35774
    error=float((nav.option_pnl+nav.hedge_gross_pnl+nav.dividend_cash+nav.interest-nav.hedge_cost-nav.stock_borrow_cost-nav.pnl).abs().max())
    assert error<1e-5
    rows=[];annual=[]
    for period,start,end in parent.core.PERIODS:rows.append(dict(case=name,period=period,**period_metrics(nav,d,start,end)))
    for y in range(2018,2026):
        m=period_metrics(nav,d,f'{y}-01-01',min(f'{y}-12-31',CFG['evaluation_end']))
        part=nav.loc[str(y)]
        m.update({f'dollar_{c}':float(part[c].sum()) for c in ['option_pnl','hedge_gross_pnl','dividend_cash','interest','hedge_cost','stock_borrow_cost']})
        annual.append(dict(case=name,year=y,**m))
    detail=REPORT/'cases'/name;detail.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(detail/'results.csv',index=False);pd.DataFrame(annual).to_csv(detail/'annual.csv',index=False)
    statuses=d.groupby('status',dropna=False).size().rename('count');statuses.to_csv(detail/'admission.csv')
    decisions=pd.read_parquet(OUT/'cases'/name/'allocations/tree18_test.parquet',columns=['date','secid','learned_level'],use_threads=False)
    decisions['period']=np.where(decisions.date.lt('2022-01-01'),'validation','evaluation')
    decisions.groupby(['period','learned_level']).size().rename('proposals').to_csv(detail/'action_counts.csv')
    usage=[];fits=[]
    for p in sorted((OUT/'models'/name).glob('*.json')):
        m=json.loads(p.read_text());assert m['last_completed'] is None or pd.Timestamp(m['last_completed'])<pd.Timestamp(m['cutoff'])
        seen=set()
        for leaf in m.get('tree',[]):
            for depth,node in enumerate(leaf):
                feature,threshold,_=node;key=(tuple(tuple(x) for x in leaf[:depth]),feature,threshold)
                if key not in seen:usage.append(dict(month=p.stem,feature=m['features'][feature]));seen.add(key)
        fits.append(dict(month=p.stem,samples=m['training_samples'],states=len(m['choices']),
            min_occupancy=min(m.get('occupancy',[0])),training_shadow_sharpe=m.get('training_shadow_sharpe')))
    pd.DataFrame(usage,columns=['month','feature']).to_csv(detail/'partition_features.csv',index=False)
    pd.DataFrame(fits).to_csv(detail/'fit_diagnostics.csv',index=False)
    baseline=pd.read_parquet('artifacts/contracts/atm_rv_straddle_fixed50/replays/tree18/daily_nav.parquet',columns=['excess_return'],use_threads=False).excess_return
    interval=paired_interval(nav.excess_return.loc['2022':],baseline.loc['2022':],block=63,draws=1000)
    parent.core.dump(detail/'audit.json',dict(ledger_error=error,proposals=len(d),monthly_fits=len(fits),strict_completed_outcomes=True,comparison_to_fixed50=interval))
    print(json.dumps(dict(phase='analyzed',case=name,metrics=rows)),flush=True)


def replication():
    """The unchanged three-input control must reproduce the earlier experiment."""
    name='core_tree4_w36'
    old=Path('artifacts/contracts/straddle_feature_transfer/cases/dense_control36')
    a=pd.read_parquet(OUT/'cases'/name/'allocations/tree18_test.parquet',
        columns=['date','secid','learned_level'],use_threads=False)
    b=pd.read_parquet(old/'allocations/tree18_test.parquet',columns=['date','secid','learned_level'],use_threads=False)
    pd.testing.assert_frame_equal(a.loc[a.date.le(CFG['validation_end'])].reset_index(drop=True),
                                  b.loc[b.date.le(CFG['validation_end'])].reset_index(drop=True))
    nav,_=parent.core.funded_nav(name,CFG['validation_end'])
    previous=pd.read_parquet(old/'replays/tree18/daily_nav.parquet',columns=['excess_return'],use_threads=False)
    np.testing.assert_allclose(nav.excess_return,previous.excess_return.reindex(nav.index),atol=1e-12,rtol=0)
    parent.core.dump(REPORT/'replication_audit.json',dict(original='dense_control36',replicated=name,
        validation_sharpe=sharpe(nav.excess_return),unchanged_allocations=True,unchanged_daily_returns=True))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',required=True,
        choices=['test','features','assemble','fit','book','select','analyze','replication'])
    p.add_argument('--case');p.add_argument('--later',action='store_true');p.add_argument('--months',type=int,default=8)
    p.add_argument('--limit',type=int,default=250);a=p.parse_args();configure()
    if a.phase=='test':
        import unittest
        s=unittest.defaultTestLoader.loadTestsFromNames(['tests.test_straddle_overnight_study','tests.test_research_memory_guard'])
        if not unittest.TextTestRunner(verbosity=2).run(s).wasSuccessful():raise SystemExit(1)
        return
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True);sources()
    if a.phase=='features':prepare_features(a.limit)
    elif a.phase=='assemble':assemble()
    elif a.phase=='fit':allocations(a.case,a.later)
    elif a.phase=='book':book(a.case,a.months,a.later)
    elif a.phase=='select':validation_scores()
    elif a.phase=='analyze':analyze(a.case)
    elif a.phase=='replication':replication()


if __name__=='__main__':main()
