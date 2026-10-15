"""Allocate using the underlying volatility opportunities rather than P&L alone."""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import atm_p25_allocation_q as old
from scripts import atm_p25_sharpe_policy as direct
from paper_code.metrics import paired_interval,sharpe

CFG_PATH=Path('config/atm_p25_opportunity_policy.json');CFG=json.loads(CFG_PATH.read_text())
OUT=Path('artifacts/contracts/atm_p25_opportunity_policy');REPORT=Path('reports/atm_p25_opportunity_policy')
ATM_SIGNAL=Path('artifacts/contracts/atm_rv_bandit_bottom10/proposals.parquet')
PAIR_SIGNAL=Path('artifacts/contracts/p25_feature_search/signal_rows.parquet')
FEATURES=['atm_iv_rv_gap','vix','atm_completed_premium','pair_rank_strength','pair_vrp_gap','pair_proposals21','pair_reserve','atm_reserve']


def opportunity_features(dates,atm,pair):
    """Every rolling value includes data through its own close, never later rows."""
    a=atm.copy();a['atm_iv_rv_gap']=(a.iv-a.rv21)/a.iv.clip(lower=.01)
    a['atm_completed_premium']=-a.residual_memory/a.iv.clip(lower=.01)
    z=a.groupby('date')[['atm_iv_rv_gap','atm_completed_premium']].mean().reindex(dates).ffill().rolling(21,min_periods=1).mean()
    pair=pair.copy();pair['pair_proposals21']=1.
    p=pair.groupby('date').agg(pair_rank_strength=('rank_strength','mean'),pair_vrp_gap=('oriented_vrp_gap','mean'),pair_proposals21=('pair_proposals21','sum'))
    z['pair_rank_strength']=p.pair_rank_strength.reindex(dates).ffill().fillna(0)
    z['pair_vrp_gap']=p.pair_vrp_gap.reindex(dates).ffill().fillna(0)
    z['pair_proposals21']=p.pair_proposals21.reindex(dates,fill_value=0).rolling(21,min_periods=1).sum()
    for label,source in [('atm',a),('pair',pair)]:
        observed=source.groupby('date').size().index
        known=pd.Series(observed,index=observed).reindex(dates).ffill()
        assert known.dropna().le(known.dropna().index).all()
        z[label+'_source_date']=known
    return z


def prepare(variant):
    data=direct.prepare(variant,'balanced')
    a=pd.read_parquet(ATM_SIGNAL,columns=['date','iv','rv21','residual_memory'],use_threads=False)
    p=pd.read_parquet(PAIR_SIGNAL,columns=['date','rank_strength','oriented_vrp_gap'],use_threads=False)
    features=opportunity_features(data['pair'].index,a,p)
    for m in data['months']:
        f=dict(zip(old.FEATURES,m['features']))
        row=features.loc[m['feature_date']] if m['feature_date'] is not None else {c:(pd.NaT if c.endswith('_date') else np.nan) for c in features}
        for column in ['atm_iv_rv_gap','atm_completed_premium','pair_rank_strength','pair_vrp_gap','pair_proposals21']:f[column]=row[column]
        m['state']=np.array([f[k] for k in FEATURES],float)
        m['ready']=m['ready'] and bool(np.isfinite(m['state']).all())
        m['signal_source_dates']={k:str(row[k]) for k in ['atm_source_date','pair_source_date']}
        if m['feature_date'] is not None:
            for k in ['atm_source_date','pair_source_date']:
                if pd.notna(row[k]):assert row[k]<=m['feature_date']
    return data


def fit(data,index,window,kind):
    cutoff=data['months'][index]['cutoff']
    train=[m for m in data['months'][:index] if m['ready'] and m['cutoff']>=cutoff-pd.DateOffset(months=window)]
    if len(train)<CFG['minimum_training_months']:return None
    assert max(m['end'] for m in train)<cutoff
    x=np.stack([m['state'] for m in train])
    y=np.array([(m['paths'][1,2]-m['paths'][1,0]).sum()/max(np.sqrt((m['paths'][1,1]**2).sum()),1e-8) for m in train])
    leaves=direct.partition(x,y,'momentum' if kind=='edge' else kind)
    states=direct.contexts(x,leaves)
    choices,score,reference,objective,n=direct.search(train,states,len(leaves))
    return dict(leaves=leaves,choices=choices.tolist(),training_months=len(train),
        last_training_end=str(max(m['end'] for m in train).date()),cutoff=str(cutoff.date()),
        training_sharpe=score,baseline_training_sharpe=reference,objective=objective,policies_enumerated=n,
        occupancy=np.bincount(states,minlength=len(leaves)).tolist(),window=window,partition=kind)


def name(window,kind):return f'opportunity_w{window}_{kind}'


def model_for(data,variant,index,window,kind):
    path=OUT/'models'/variant/name(window,kind)/(data['months'][index]['month']+'.json')
    if path.exists():return json.loads(path.read_text())
    model=fit(data,index,window,kind)
    if model is not None:old.dump(path,model)
    return model


def replay(data,variant,case,stop,window=None,kind=None,ensemble=False,cost_multiplier=1.):
    previous=np.array([1.,0.,0.]);daily=[];decisions=[]
    for i,m in enumerate(data['months']):
        if m['cutoff']>pd.Timestamp(stop):break
        models=[];actions=[]
        if m['ready']:
            specs=[(w,k) for w in CFG['windows'] for k in CFG['partitions']] if ensemble else [(window,kind)]
            for w,k in specs:
                model=model_for(data,variant,i,w,k);action=1
                if model is not None:
                    assert pd.Timestamp(model['last_training_end'])<m['cutoff']
                    action=model['choices'][int(direct.contexts(m['state'],model['leaves'])[0])];models.append(model)
                actions.append(action)
            allocation=m['targets'][actions].mean(axis=0)
        else:allocation=np.array([1.,0.,0.])
        r,after,cost=old.month_path(m['gross'],m['cash'],m['cost_rate'],previous,allocation,cost_multiplier)
        daily.append(pd.Series(r,index=data['pair'].index[m['indices']]))
        decisions.append(dict(date=m['cutoff'],feature_date=m['feature_date'],
            pair_weight=allocation[0],atm_weight=allocation[1],cash_weight=allocation[2],
            resizing_cost_fraction=cost,fitted=bool(models),model_count=len(models),actions=actions,
            last_training_end=max([m['last_training_end'] for m in models],default=None),
            minimum_training_months=min([m['training_months'] for m in models],default=0),
            **m['signal_source_dates']))
        previous=after
    r=pd.concat(daily).rename('excess_return');d=pd.DataFrame(decisions)
    folder=OUT/'replays'/variant/case;folder.mkdir(parents=True,exist_ok=True)
    suffix='validation' if stop==CFG['validation_end'] else 'full'
    pd.DataFrame(dict(excess_return=r,cash_return=data['cash'].reindex(r.index))).to_parquet(folder/(suffix+'.parquet'))
    d.to_csv(folder/(suffix+'_decisions.csv'),index=False)
    return r,d


def check_sources(data,variant):
    paths=data['sources']+[CFG_PATH,Path(__file__),Path(direct.__file__),direct.CFG_PATH,ATM_SIGNAL,PAIR_SIGNAL,Path('scripts/report_atm_p25_combination.py')]
    record={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    marker=OUT/'source_hashes'/(variant+'.json')
    if marker.exists():assert json.loads(marker.read_text())==record,'Source changed; use new experiment directory'
    else:old.dump(marker,record)


def validation():
    rows=[]
    for variant in ['tree18','fixed200']:
        data=prepare(variant);check_sources(data,variant)
        for window in CFG['windows']:
            for kind in CFG['partitions']:
                case=name(window,kind)
                r,_=replay(data,variant,case,CFG['validation_end'],window,kind)
                row=dict(atm_variant=variant,window=window,partition=kind,case=case,validation_sharpe=sharpe(r.iloc[1:]))
                rows.append(row);pd.DataFrame(rows).to_csv(REPORT/'validation_candidates.csv',index=False);old.progress(phase='validation',**row)
        r,_=replay(data,variant,'opportunity_ensemble',CFG['validation_end'],ensemble=True)
        rows.append(dict(atm_variant=variant,window=0,partition='ensemble',case='opportunity_ensemble',validation_sharpe=sharpe(r.iloc[1:])))
        pd.DataFrame(rows).to_csv(REPORT/'validation_candidates.csv',index=False)
        del data;gc.collect()


def select():
    rows=pd.read_csv(REPORT/'validation_candidates.csv');assert len(rows)==14
    selected={v:g.sort_values(['validation_sharpe','case'],ascending=[False,True]).iloc[0].to_dict() for v,g in rows.groupby('atm_variant')}
    old.dump(REPORT/'selection.json',dict(selected=selected,validation_only=True,scope=CFG['scope']))
    old.progress(phase='selection',selected=selected)


def full_replays():
    selected=json.loads((REPORT/'selection.json').read_text())['selected']
    for variant,record in selected.items():
        data=prepare(variant);check_sources(data,variant)
        kwargs=dict(window=int(record['window']),kind=record['partition'],ensemble=record['partition']=='ensemble')
        r,d=replay(data,variant,record['case'],CFG['test_end'],**kwargs)
        saved=pd.read_parquet(OUT/'replays'/variant/record['case']/'validation.parquet',use_threads=False)
        np.testing.assert_allclose(r.loc[saved.index],saved.excess_return,atol=1e-14)
        r,sd=replay(data,variant,record['case']+'_cost5',CFG['test_end'],cost_multiplier=5.,**kwargs)
        np.testing.assert_allclose(d.atm_weight,sd.atm_weight)
        old.progress(phase='replay_complete',variant=variant)
        del data;gc.collect()


def report():
    selected=json.loads((REPORT/'selection.json').read_text())['selected'];rows=[];annual=[];intervals=[]
    for variant,record in selected.items():
        books={}
        for case in [record['case'],record['case']+'_cost5']:
            folder=OUT/'replays'/variant/case;book=pd.read_parquet(folder/'full.parquet',use_threads=False);books[case]=book
            d=pd.read_csv(folder/'full_decisions.csv',parse_dates=['date','feature_date','last_training_end','atm_source_date','pair_source_date'])
            assert d.feature_date.iloc[1:].lt(d.date.iloc[1:]).all()
            assert d.loc[d.fitted,'last_training_end'].lt(d.loc[d.fitted,'date']).all()
            for key in ['atm_source_date','pair_source_date']:assert d.loc[d[key].notna(),key].le(d.loc[d[key].notna(),'feature_date']).all()
            np.testing.assert_allclose(d[['pair_weight','atm_weight','cash_weight']].sum(axis=1),1)
            for period,start,stop in old.blend.PERIODS:
                part=d.loc[d.date.between(start,stop)]
                rows.append(dict(atm_variant=variant,case=case,period=period,**old.blend.stats(book.excess_return,book.cash_return,start,stop),mean_atm_capital_pct=100*part.atm_weight.mean(),resizing_cost_bps=part.resizing_cost_fraction.sum()*10000))
            for year in range(2018,2026):annual.append(dict(atm_variant=variant,case=case,year=year,**old.blend.stats(book.excess_return,book.cash_return,f'{year}-01-01',min(f'{year}-12-31',CFG['test_end']))))
        baseline=pd.read_parquet(old.OUT/'replays'/variant/'rolling_risk_balance'/'full.parquet',use_threads=False)
        intervals.append(dict(atm_variant=variant,control='rolling_risk_balance',**paired_interval(books[record['case']].excess_return.loc[CFG['test_start']:],baseline.excess_return.loc[CFG['test_start']:],block=63,draws=1000)))
    result=pd.DataFrame(rows);result.to_csv(REPORT/'results.csv',index=False)
    pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False);pd.DataFrame(intervals).to_csv(REPORT/'sharpe_intervals.csv',index=False)
    old.dump(REPORT/'protocol.json',dict(config=CFG,selection=selected,features=FEATURES,
        optimization='Sequential daily training Sharpe net of resizing costs, exhaustive 3^K action maps, at most three states; same 0.05 tilt preference as direct-search control.',
        feature_timing='Daily ATM proposal IV/RV21 and completed residual memory, averaged then trailing21-smoothed. Last known put-pair proposal rank/VRP and trailing21 proposal count. VIX and funded reserve from prior close. No future outcomes or future proposals enter state.',scope=CFG['scope']))
    print(result.loc[result.period.isin(['validation','test']),['atm_variant','case','period','annualized_excess_sharpe','annualized_growth_relative_to_cash_pct']].to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',required=True,choices=['inspect','test','validation','select','replay','report'])
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    if args.phase=='inspect':
        import pyarrow.parquet as pq
        for path in [ATM_SIGNAL,PAIR_SIGNAL]:print(str(path),pq.read_schema(path).names,flush=True)
    elif args.phase=='test':
        import unittest
        suite=unittest.defaultTestLoader.loadTestsFromNames(['tests.test_atm_p25_opportunity_policy','tests.test_atm_p25_sharpe_policy','tests.test_research_memory_guard'])
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():raise SystemExit(1)
    else:dict(validation=validation,select=select,replay=full_replays,report=report)[args.phase]()


if __name__=='__main__':main()
