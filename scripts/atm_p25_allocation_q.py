"""Monthly fitted Q allocation over two frozen strategy return indices.

Prior allocation is part of state. Actions change subsequent holdings, their
drift and future resizing costs. This is Bellman fitted Q-iteration, with a
matched gamma=0 allocator as a control. Fractional strategy units approximate
capital allocation; this is not a shared option-contract ledger.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource

import numpy as np
import pandas as pd

from scripts import report_atm_p25_combination as blend
from paper_code.metrics import paired_interval, sharpe

CFG_PATH=Path('config/atm_p25_q_allocation.json')
CFG=json.loads(CFG_PATH.read_text())
OUT=Path('artifacts/contracts/atm_p25_q_allocation')
REPORT=Path('reports/atm_p25_q_allocation')
ACTIONS=CFG['actions'];N=len(ACTIONS)
FEATURES=['pair_sharpe63','atm_sharpe63','log_vol_ratio','correlation63',
          'pair_drawdown63','atm_drawdown63','vix','log_atm_vol','pair_reserve','atm_reserve']


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,default=str)+'\n')


def progress(**values):
    print(json.dumps(dict(peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,**values),default=str),flush=True)


def target_weights(pair_vol,atm_vol):
    """Actions allocate risk contributions; convert them to capital without leverage."""
    p=max(float(pair_vol),1e-8);a=max(float(atm_vol),1e-8)
    result=np.zeros((N,3));result[0,2]=1
    for index,risk_share in enumerate([0.,.25,.5,.75,1.],1):
        w=(risk_share/a)/((1-risk_share)/p+risk_share/a)
        result[index]=[1-w,w,0]
    return result


def drift(weights,gross):
    value=np.asarray(weights,float)*np.prod(gross,axis=0)
    return value/value.sum()


def month_path(gross,cash,cost_rate,previous,target,cost_multiplier=1.):
    """First session belongs to old weights; resize at that close, then hold."""
    weights=np.asarray(previous,float).copy();returns=np.empty(len(gross));charge=0.
    for t,change in enumerate(gross):
        growth=float(weights@change)
        weights=weights*change/growth
        if t==0:
            charge=float(np.abs(target[:2]-weights[:2])@cost_rate)*cost_multiplier
            if not 0<=charge<1:raise ValueError('Invalid resizing fee')
            growth*=1-charge;weights=np.asarray(target,float).copy()
        returns[t]=growth-1-cash[t]
    np.testing.assert_allclose(weights.sum(),1,atol=1e-12)
    return returns,weights,charge


def contract_counts(trades,dates):
    """Known-close outstanding quantities, including natural exits and settlements."""
    held={};values=[];groups={d:g for d,g in trades.groupby('date',sort=False)}
    for date in dates:
        if date in groups:
            for row in groups[date].itertuples():
                key=(row.cohort,int(row.optionid));held[key]=held.get(key,0)+int(row.quantity)
                if not held[key]:del held[key]
        values.append(sum(abs(x) for x in held.values()))
    assert not held,'Source ledger must finish flat'
    return np.asarray(values)


def market_features(pair,atm):
    """A row describes only returns and holdings observed through that close."""
    a,b=pair.excess_return,atm.excess_return
    av=a.rolling(63,min_periods=21).std();bv=b.rolling(63,min_periods=21).std()
    z=pd.DataFrame(index=a.index)
    z['pair_sharpe63']=np.sqrt(252)*a.rolling(63,min_periods=21).mean()/av.replace(0,np.nan)
    z['atm_sharpe63']=np.sqrt(252)*b.rolling(63,min_periods=21).mean()/bv.replace(0,np.nan)
    z['log_vol_ratio']=np.log(bv.clip(lower=1e-8)/av.clip(lower=1e-8))
    z['correlation63']=a.rolling(63,min_periods=21).corr(b)
    for name,book in [('pair',pair),('atm',atm)]:
        relative=book.nav/book.cash_benchmark
        z[name+'_drawdown63']=relative/relative.rolling(63,min_periods=21).max()-1
        z[name+'_reserve']=book.capital_reserve/book.nav
    vix=pd.read_csv('data/raw/VIX_History.csv',usecols=['DATE','CLOSE'],parse_dates=['DATE']).set_index('DATE').CLOSE
    z['vix']=vix.reindex(z.index,method='ffill')/20
    z['log_atm_vol']=np.log(bv.clip(lower=1e-8))
    z[['pair_sharpe63','atm_sharpe63','correlation63']]=z[['pair_sharpe63','atm_sharpe63','correlation63']].fillna(0)
    z['pair_vol']=av;z['atm_vol']=bv
    z['ready']=np.arange(len(z))>=62
    return z


def prepare(variant):
    columns=['nav','return_','excess_return','cash_benchmark','capital_reserve','gross_vega','gross_hedge_notional']
    pair_path=blend.PAIR/'daily_nav.parquet';atm_path=blend.ATM/variant/'test_nav.parquet'
    pair=pd.read_parquet(pair_path,columns=columns,use_threads=False)
    atm=pd.read_parquet(atm_path,columns=columns,use_threads=False)
    cash=blend.aligned_cash(pair,atm)
    costs=[];sources=[pair_path,atm_path,CFG_PATH,Path(__file__),Path('data/raw/VIX_History.csv')]
    for folder,book in [(blend.PAIR,pair),(blend.ATM/variant,atm)]:
        path=folder/'trades.csv';trades=pd.read_csv(path,usecols=['date','cohort','optionid','quantity'],parse_dates=['date'])
        counts=contract_counts(trades,book.index)
        costs.append((.1*book.gross_vega+.0001*book.gross_hedge_notional+.03*counts)/book.nav)
        sources.append(path)
    rates=np.column_stack(costs);features=market_features(pair,atm)
    gross=np.column_stack([1+pair.return_,1+atm.return_,1+cash])
    months=[];previous=None
    for period,indices in pd.Series(np.arange(len(pair)),index=pair.index).groupby(pair.index.to_period('M')):
        ii=indices.to_numpy();prior=int(ii[0])-1
        f=features.iloc[prior] if prior>=0 else features.iloc[0]*np.nan
        ready=prior>=0 and bool(features.ready.iloc[prior])
        targets=target_weights(f.pair_vol,f.atm_vol) if ready else target_weights(.01,.01)
        before=np.tile([1.,0.,0.],(N,1)) if previous is None else np.stack([drift(w,previous['gross'][1:]) for w in previous['targets']])
        month=dict(month=str(period),cutoff=pair.index[ii[0]],end=pair.index[ii[-1]],
            feature_date=pair.index[prior] if prior>=0 else None,
            ready=ready,features=f[FEATURES].to_numpy(float),indices=ii,
            targets=targets,before=before,gross=gross[ii],cash=cash.iloc[ii].to_numpy(),cost_rate=rates[ii[0]],
            after=np.stack([drift(w,gross[ii][1:]) for w in targets]))
        paths=np.empty((N,N,len(ii)))
        for p in range(N):
            for a in range(N):paths[p,a]=month_path(month['gross'],month['cash'],month['cost_rate'],before[p],targets[a])[0]
        month['paths']=paths;months.append(month);previous=month
    return dict(pair=pair,atm=atm,cash=cash,months=months,sources=sources)


def states_features(month,previous):
    return np.r_[month['features'],np.asarray(previous)[:2]]


def design(states,actions,center,scale):
    z=np.clip((np.asarray(states)-center)/scale,-5,5)
    z=np.column_stack([np.ones(len(z)),z])
    return (np.eye(N)[np.asarray(actions,int),:,None]*z[:,None,:]).reshape(len(z),-1)


def solve_fqi(x,rewards,next_x,discount,alpha,iterations):
    """Repeated least-squares Bellman backups, with one common regularized basis."""
    projector=np.linalg.solve(x.T@x+alpha*np.eye(x.shape[1]),x.T)
    beta=projector@rewards
    for _ in range(iterations-1 if discount else 0):
        continuation=(next_x@beta).max(axis=1)
        beta=projector@(rewards+discount*continuation)
    if not np.isfinite(beta).all():raise ValueError('Nonfinite Q critic')
    return beta


def fit(data,index,window,risk,discount):
    current=data['months'][index];cutoff=current['cutoff']
    train=[i for i,m in enumerate(data['months'][:index]) if m['ready'] and m['cutoff']>=cutoff-pd.DateOffset(months=window)]
    if len(train)<CFG['minimum_training_months']:return None
    assert max(data['months'][i]['end'] for i in train)<cutoff
    # Risk-adjusted reward calibration uses this training window only.
    reference=np.concatenate([data['months'][i]['paths'][3,3] for i in train])
    volatility=max(float(reference.std(ddof=1)),1e-8)
    penalty=risk*max(float(reference.mean()/volatility),.01)
    states=[];actions=[];rewards=[];next_states=[]
    for i in train:
        m=data['months'][i];nxt=data['months'][i+1]
        assert nxt['feature_date']<=current['feature_date']
        for previous in range(N):
            state=states_features(m,m['before'][previous])
            for action in range(N):
                z=m['paths'][previous,action]/volatility
                states.append(state);actions.append(action);rewards.append(float((z-.5*penalty*z*z).sum()))
                next_states.append(states_features(nxt,m['after'][action]))
    states=np.asarray(states);next_states=np.asarray(next_states)
    center=states.mean(axis=0);scale=np.maximum(states.std(axis=0),.1)
    x=design(states,actions,center,scale)
    expanded=np.repeat(next_states,N,axis=0)
    next_x=design(expanded,np.tile(np.arange(N),len(next_states)),center,scale).reshape(len(next_states),N,-1)
    beta=solve_fqi(x,np.asarray(rewards),next_x,discount,CFG['ridge_alpha'],CFG['bellman_iterations'])
    return dict(center=center.tolist(),scale=scale.tolist(),beta=beta.tolist(),training_months=len(train),
        last_training_end=str(max(data['months'][i]['end'] for i in train).date()),cutoff=str(cutoff.date()),
        training_reward_volatility=volatility,variance_penalty=penalty,window=window,risk=risk,discount=discount)


def values(model,month,previous):
    state=states_features(month,previous)
    x=design(np.tile(state,(N,1)),np.arange(N),np.asarray(model['center']),np.asarray(model['scale']))
    return x@np.asarray(model['beta'])


def case_name(window,risk,prefix='q'):
    return f'{prefix}_w{window}_r{risk:g}'.replace('.','p')


def replay(data,variant,case,stop,window=None,risk=None,discount=None,mode=None,cost_multiplier=1.):
    previous=np.array([1.,0.,0.]);daily=[];decisions=[]
    fixed_atm=1-blend.validation_weight(data['pair'].excess_return,data['atm'].excess_return) if mode=='fixed' else None
    for i,m in enumerate(data['months']):
        if m['cutoff']>pd.Timestamp(stop):break
        model=None;q=None
        if mode=='pair':action=1
        elif mode=='atm':action=5
        elif mode in ['rolling','fixed']:action=3 if m['ready'] else 1
        elif not m['ready']:action=1
        else:
            path=OUT/'models'/variant/case.removesuffix('_cost5')/(m['month']+'.json')
            if path.exists():model=json.loads(path.read_text())
            else:
                model=fit(data,i,window,risk,discount)
                if model is not None:dump(path,model)
            action=3
            if model is not None:
                assert pd.Timestamp(model['last_training_end'])<m['cutoff']
                q=values(model,m,previous);action=int(np.argmax(q))
        target=m['targets'][action]
        if mode=='fixed' and m['cutoff']>=pd.Timestamp(CFG['test_start']):
            target=np.array([1-fixed_atm,fixed_atm,0.])
        path,after,cost=month_path(m['gross'],m['cash'],m['cost_rate'],previous,target,cost_multiplier)
        daily.append(pd.Series(path,index=data['pair'].index[m['indices']]))
        decisions.append(dict(date=m['cutoff'],feature_date=m['feature_date'],action=ACTIONS[action],action_code=action,
            pair_weight=target[0],atm_weight=target[1],cash_weight=target[2],
            previous_pair_weight=previous[0],previous_atm_weight=previous[1],reallocation_cost_fraction=cost,
            fitted=model is not None,training_months=model['training_months'] if model else 0,
            last_training_end=model['last_training_end'] if model else None,q_values=q.tolist() if q is not None else None))
        previous=after
    returns=pd.concat(daily).rename('excess_return')
    folder=OUT/'replays'/variant/case;folder.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(dict(excess_return=returns,cash_return=data['cash'].reindex(returns.index))).to_parquet(folder/('validation.parquet' if stop==CFG['validation_end'] else 'full.parquet'))
    pd.DataFrame(decisions).to_csv(folder/('validation_decisions.csv' if stop==CFG['validation_end'] else 'decisions.csv'),index=False)
    return returns,pd.DataFrame(decisions)


def check_sources(data):
    sources=data['sources']+[Path('scripts/report_atm_p25_combination.py')]
    record={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    marker=OUT/'source_hashes'/(data['sources'][1].parent.name+'.json')
    if marker.exists():assert json.loads(marker.read_text())==record,'Sources changed; preserve this run and use a new output root'
    else:dump(marker,record)


def validation():
    records=[]
    for variant in ['tree18','fixed200']:
        data=prepare(variant);check_sources(data)
        for window in CFG['training_windows_months']:
            for risk in CFG['risk_penalty_multipliers']:
                case=case_name(window,risk)
                r,d=replay(data,variant,case,CFG['validation_end'],window,risk,CFG['discount'])
                record=dict(atm_variant=variant,case=case,window=window,risk=risk,validation_sharpe=sharpe(r.iloc[1:]),fitted_months=int(d.fitted.sum()))
                records.append(record);progress(phase='validation',**record)
                pd.DataFrame(records).to_csv(REPORT/'validation_candidates.csv',index=False)
                del r,d;gc.collect()
        del data;gc.collect()


def select():
    rows=pd.read_csv(REPORT/'validation_candidates.csv')
    assert len(rows)==12
    selected={}
    for variant,g in rows.groupby('atm_variant'):
        selected[variant]=g.sort_values(['validation_sharpe','window','risk'],ascending=[False,False,False]).iloc[0].to_dict()
    dump(REPORT/'selection.json',dict(selected=selected,validation_only=True,source_hashes={v:json.loads((OUT/'source_hashes'/(v+'.json')).read_text()) for v in ['tree18','fixed200']}))
    progress(phase='selection',selected=selected)


def test_replays():
    selection=json.loads((REPORT/'selection.json').read_text())
    for variant in ['tree18','fixed200']:
        data=prepare(variant);check_sources(data)
        winner=selection['selected'][variant]
        window=int(winner['window']);risk=float(winner['risk']);case=winner['case']
        r,decisions=replay(data,variant,case,CFG['test_end'],window,risk,CFG['discount'])
        saved=pd.read_parquet(OUT/'replays'/variant/case/'validation.parquet').excess_return
        np.testing.assert_allclose(r.reindex(saved.index),saved,atol=1e-14)
        names=[case]
        control=case_name(window,risk,'gamma0')
        replay(data,variant,control,CFG['test_end'],window,risk,0.);names.append(control)
        for label,mode in [('rolling_risk_balance','rolling'),('fixed_validation_balance','fixed'),('pair_only','pair'),('atm_only','atm')]:
            path,_=replay(data,variant,label,CFG['test_end'],mode=mode);names.append(label)
            if mode in ['pair','atm']:
                source=data[mode if mode=='pair' else 'atm'].excess_return
                np.testing.assert_allclose(path,source,atol=1e-12)
        # Same frozen decisions under five times the additional resizing penalty.
        stress,stressed=replay(data,variant,case+'_cost5',CFG['test_end'],window,risk,CFG['discount'],cost_multiplier=5.)
        pd.testing.assert_series_equal(decisions.action_code,stressed.action_code)
        names.append(case+'_cost5')
        dump(OUT/variant/'cases.json',dict(cases=names,selected=case,gamma0=control))
        progress(phase='test_complete',atm_variant=variant,cases=names)
        del data;gc.collect()


def report():
    records=[];annual=[];intervals=[];weights=[]
    for variant in ['tree18','fixed200']:
        manifest=json.loads((OUT/variant/'cases.json').read_text());books={}
        for case in manifest['cases']:
            folder=OUT/'replays'/variant/case
            nav=pd.read_parquet(folder/'full.parquet',use_threads=False);books[case]=nav
            choices=pd.read_csv(folder/'decisions.csv',parse_dates=['date','feature_date','last_training_end'])
            assert choices.feature_date.iloc[1:].lt(choices.date.iloc[1:]).all()
            assert choices.loc[choices.fitted,'last_training_end'].lt(choices.loc[choices.fitted,'date']).all()
            np.testing.assert_allclose(choices[['pair_weight','atm_weight','cash_weight']].sum(axis=1),1,atol=1e-12)
            assert choices[['pair_weight','atm_weight','cash_weight']].ge(0).all().all()
            for period,start,stop in blend.PERIODS:
                part=choices.loc[choices.date.between(start,stop)]
                records.append(dict(atm_variant=variant,case=case,period=period,
                    **blend.stats(nav.excess_return,nav.cash_return,start,stop),
                    mean_atm_capital_pct=100*part.atm_weight.mean(),mean_cash_capital_pct=100*part.cash_weight.mean(),
                    fitted_months=int(part.fitted.sum()),resizing_cost_bps=float(part.reallocation_cost_fraction.sum()*10000)))
                for action,n in part.action.value_counts().items():weights.append(dict(atm_variant=variant,case=case,period=period,action=action,months=int(n)))
            for year in range(2018,2026):annual.append(dict(atm_variant=variant,case=case,year=year,**blend.stats(nav.excess_return,nav.cash_return,f'{year}-01-01',min(f'{year}-12-31',CFG['test_end']))))
        selected=manifest['selected']
        for control in ['fixed_validation_balance','rolling_risk_balance',manifest['gamma0']]:
            intervals.append(dict(atm_variant=variant,control=control,**paired_interval(books[selected].excess_return.loc[CFG['test_start']:],books[control].excess_return.loc[CFG['test_start']:],block=63,draws=1000)))
    results=pd.DataFrame(records);results.to_csv(REPORT/'results.csv',index=False)
    pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False)
    pd.DataFrame(intervals).to_csv(REPORT/'sharpe_intervals.csv',index=False)
    pd.DataFrame(weights).to_csv(REPORT/'action_counts.csv',index=False)
    dump(REPORT/'protocol.json',dict(specification=CFG,selection=json.loads((REPORT/'selection.json').read_text()),
        warmup='No pre-2018 funded histories. Before 12 completed feature-ready months, use causal trailing-volatility risk balance; before 63 observed sessions, use the put-pair sleeve.',
        benchmark='Rolling risk balance until the end of validation; then fixed validation-volatility weights for the later period, with the same monthly execution delay and resizing penalty.',
        checks='Identical cash/calendar, causal features/completed outcomes, unchanged validation Q replay, standalone-index replication, simplex weights, unchanged Q decisions in cost stress.',
        limitation=CFG['scope'],historical_test_previously_examined=True))
    (REPORT/'FINDINGS.txt').write_text(__doc__+'\n\n'+results.to_string(index=False)+'\n\n'+json.dumps(CFG,indent=2)+'\n')
    print(results.loc[results.period.isin(['validation','test']),['atm_variant','case','period','annualized_excess_sharpe','annualized_growth_relative_to_cash_pct','mean_atm_capital_pct','mean_cash_capital_pct']].to_string(index=False),flush=True)


def charts():
    import os
    os.environ.setdefault('MPLCONFIGDIR','artifacts/contracts/atm_rv_persistence/matplotlib_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results=pd.read_csv(REPORT/'results.csv');selection=json.loads((REPORT/'selection.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,variant in zip(axes,['tree18','fixed200']):
        meta=json.loads((OUT/variant/'cases.json').read_text())
        cases=['fixed_validation_balance','rolling_risk_balance',meta['gamma0'],meta['selected']]
        for i,(case,label) in enumerate(zip(cases,['Frozen balance','Rolling balance','Gamma 0 allocator','Q allocator'])):
            rows=results.loc[results.atm_variant.eq(variant)&results.case.eq(case)].set_index('period')
            ax.bar(np.arange(2)+(i-1.5)*.18,rows.loc[['validation','test'],'annualized_excess_sharpe'],width=.18,label=label)
        ax.set_xticks([0,1],['2018–2021 validation','2022–Aug 2025 test'],fontsize=9)
        ax.set_title('ATM bandit' if variant=='tree18' else 'ATM fixed 200%');ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Annualized excess Sharpe');axes[1].legend(fontsize=8)
    fig.suptitle('Monthly Q capital allocation versus matched allocation rules')
    fig.tight_layout();fig.savefig(REPORT/'comparison.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
    for ax,variant in zip(axes,['tree18','fixed200']):
        case=selection['selected'][variant]['case']
        rows=pd.read_csv(OUT/'replays'/variant/case/'decisions.csv',parse_dates=['date'])
        ax.stackplot(rows.date,100*rows.pair_weight,100*rows.atm_weight,100*rows.cash_weight,labels=['Put-pair sleeve','ATM sleeve','Cash'],step='post')
        ax.axvline(pd.Timestamp(CFG['test_start']),color='black',ls='--',lw=.7)
        ax.set(ylim=(0,100),ylabel='Capital weight (%)',title=variant)
    axes[0].legend(loc='upper left',fontsize=8);fig.tight_layout();fig.savefig(REPORT/'allocations.png',dpi=140);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['test','validation','select','replay','report','charts'],required=True)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    if args.phase=='test':
        import unittest
        suite=unittest.defaultTestLoader.loadTestsFromNames(['tests.test_atm_p25_allocation_q','tests.test_research_memory_guard'])
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():raise SystemExit(1)
    else:dict(validation=validation,select=select,replay=test_replays,report=report,charts=charts)[args.phase]()


if __name__=='__main__':main()
