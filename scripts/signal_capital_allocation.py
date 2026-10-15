"""Allocate between two unchanged best bandits; fixed endpoints replicate sources.

No inherited overlay, portfolio stop, or forced fixed-policy warmup. Training
counterfactuals are completed market weeks, with total weight one per week.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import atm_p25_allocation_q as engine
from scripts import put_straddle_q_opportunity as opportunity
from scripts.research_atomic_io import atomic_dump
from scripts.report_atm_p25_combination import stats,aligned_cash

CONFIG=Path('config/signal_capital_allocation.json');CFG=json.loads(CONFIG.read_text())
ROOT,REPORT=Path(CFG['output_root']),Path(CFG['report_root'])
SHARES=np.array(CFG['straddle_weights']);N=len(SHARES)
TARGETS=np.column_stack([1-SHARES,SHARES,np.zeros(N)])
PERIODS=[('validation','2018-01-01','2021-12-31'),('evaluation','2022-01-01',CFG['evaluation_end']),('full','2018-01-01',CFG['evaluation_end'])]
FEATURES=opportunity.GROUPS['economic']+['relative_sharpe63']


def dump(path,value):opportunity.dump(path,value)


def sources():
    paths=[CONFIG,Path(__file__),Path(engine.__file__),Path('scripts/report_atm_p25_combination.py'),Path(CFG['prepared_source'])]
    for name in ['put_book','straddle_book']:
        paths.extend([Path(CFG[name])/'daily_nav.parquet',Path(CFG[name])/'trades.csv'])
    record={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    path=ROOT/'sources.json'
    if path.exists():assert json.loads(path.read_text())==record,'Sources changed; use a new study root'
    else:dump(path,record)


def load():
    import joblib
    return joblib.load(ROOT/'prepared.joblib')


def prepare():
    import joblib
    if (ROOT/'prepared.joblib').exists():return
    data=joblib.load(CFG['prepared_source']);columns=['nav','return_','excess_return','cash_benchmark']
    for key,source in [('pair','put_book'),('straddle','straddle_book')]:
        book=pd.read_parquet(Path(CFG[source])/'daily_nav.parquet',columns=columns,use_threads=False)
        pd.testing.assert_frame_equal(data[key][columns],book)
    aligned_cash(data['pair'],data['straddle'])
    known=engine.market_features(data['pair'],data['straddle'])
    relative=(known.atm_sharpe63-known.pair_sharpe63).replace([np.inf,-np.inf],np.nan).fillna(0).clip(-5,5)
    periods=[];before=TARGETS.copy()
    for m in data['weeks']:
        prior=int(m['indices'][0])-1
        if prior>=0:assert known.index[prior]==m['feature_date'] and m['feature_date']<m['cutoff']
        market=np.r_[m['markets']['economic'],relative.iloc[prior] if prior>=0 else 0.]
        paths=np.stack([[engine.month_path(m['gross'],m['cash'],m['costs'][0],p,a)[0] for a in TARGETS] for p in before])
        after=np.stack([engine.drift(a,m['gross'][1:]) for a in TARGETS])
        periods.append(dict(cutoff=m['cutoff'],end=m['end'],feature_date=m['feature_date'],market=market,
            indices=m['indices'],gross=m['gross'],cash=m['cash'],costs=m['costs'][0],before=before.copy(),after=after,paths=paths))
        before=after
    prepared=dict(pair=data['pair'],straddle=data['straddle'],cash=data['cash'],periods=periods)
    atomic_dump(prepared,ROOT/'prepared.joblib',joblib.dump)
    dump(REPORT/'protocol.json',dict(configuration=CFG,features=FEATURES,targets=TARGETS.tolist()))


def design(states,actions,center,scale):
    z=np.column_stack([np.ones(len(states)),np.clip((np.asarray(states)-center)/scale,-5,5)])
    a=SHARES[np.asarray(actions,int)]
    return (np.column_stack([np.ones(len(a)),a,a*a])[:,:,None]*z[:,None,:]).reshape(len(a),-1)


def case(risk,gamma):return f'q_lambda{risk:g}_g{gamma:g}'


def fit(data,index):
    periods=data['periods'];current=periods[index];cutoff=current['cutoff']
    train=[i for i,m in enumerate(periods[:index]) if m['end']<cutoff and m['cutoff']>=cutoff-pd.Timedelta(weeks=CFG['training_weeks'])]
    if len(train)<CFG['minimum_training_weeks']:return None
    last=max(periods[i]['end'] for i in train);assert last<cutoff
    states=np.concatenate([np.repeat(np.column_stack([np.tile(periods[i]['market'],(N,1)),periods[i]['before'][:,1]]),N,axis=0) for i in train])
    following=np.concatenate([np.column_stack([np.tile(periods[i+1]['market'],(N,1)),periods[i]['after'][:,1]]) for i in train])
    assert all(periods[i+1]['feature_date']<=current['feature_date'] for i in train)
    center=states.mean(axis=0);scale=np.maximum(states.std(axis=0),.1)
    actions=np.tile(np.arange(N),len(train)*N)
    x=design(states,actions,center,scale);coef=x.shape[1]
    nx=design(np.repeat(following,N,axis=0),np.tile(np.arange(N),len(following)),center,scale).reshape(len(following),N,coef)
    projection=x.reshape(len(train),N,N,coef).sum(axis=1).reshape(-1,coef).T/(N*N)
    ref=np.concatenate([periods[i]['paths'][3,3] for i in train]);vol=max(float(ref.std(ddof=1)),1e-8)
    profit=np.concatenate([periods[i]['paths'].sum(axis=-1).ravel()/vol for i in train])
    squares=np.concatenate([(periods[i]['paths']**2).sum(axis=-1).ravel()/vol**2 for i in train])
    penalty=CFG['ridge_alpha']*np.eye(coef);penalty[0,0]=.01
    inverse=np.linalg.inv(x.T@x/(N*N)+penalty)
    moments=x.T@np.column_stack([profit,squares])/(N*N)
    month=str(cutoff.to_period('M'))
    for risk in CFG['risk_preferences']:
        reward=moments[:,0]-risk*moments[:,1];immediate=inverse@reward
        for gamma in [CFG['discount'],0.]:
            beta=immediate.copy()
            if gamma:
                for _ in range(CFG['iterations']-1):beta=inverse@(reward+gamma*(projection@(nx@beta).max(axis=1)))
            assert np.isfinite(beta).all()
            model=dict(beta=beta.tolist(),center=center.tolist(),scale=scale.tolist(),
                cutoff=str(cutoff.date()),last_training_end=str(last.date()),training_weeks=len(train),
                reference_volatility=vol,risk=risk,gamma=gamma,features=FEATURES+['previous_straddle_weight'])
            dump(ROOT/'models'/case(risk,gamma)/(month+'.json'),model)
    print(json.dumps(dict(month=month,training_weeks=len(train),coefficients=coef)),flush=True)
    return True


def model_for(data,index,risk,gamma):
    m=data['periods'][index];path=ROOT/'models'/case(risk,gamma)/(str(m['cutoff'].to_period('M'))+'.json')
    if not path.exists() and fit(data,index) is None:return None
    model=json.loads(path.read_text())
    if pd.Timestamp(model['cutoff'])>m['cutoff']:return None
    assert pd.Timestamp(model['last_training_end'])<m['cutoff']
    return model


def replay(data,name,stop,risk=None,gamma=None,constant=None):
    folder=ROOT/'replays'/name;stem='validation' if stop==CFG['validation_end'] else 'full'
    if (folder/(stem+'_complete.json')).exists():return pd.read_parquet(folder/(stem+'.parquet'),use_threads=False)
    # Fixed policies start at their OWN mix. Only Q has a learning fallback.
    previous=TARGETS[constant if constant is not None else 3].copy();nav=CFG['capital'];rows=[];decisions=[]
    for i,m in enumerate(data['periods']):
        if m['cutoff']>pd.Timestamp(stop):break
        model=None
        if constant is not None:action=constant
        else:
            model=model_for(data,i,risk,gamma);action=3
            if model is not None:
                state=np.r_[m['market'],previous[1]]
                scores=design(np.tile(state,(N,1)),np.arange(N),np.array(model['center']),np.array(model['scale']))@np.array(model['beta'])
                action=int(np.argmax(scores))
        excess,after,charge=engine.month_path(m['gross'],m['cash'],m['costs'],previous,TARGETS[action])
        gross=1+excess+m['cash'];wealth=nav*np.cumprod(gross)
        for j,pos in enumerate(m['indices']):
            rows.append(dict(date=data['pair'].index[pos],excess_return=excess[j],cash_return=m['cash'][j],nav=wealth[j]))
        decisions.append(dict(date=m['cutoff'],feature_date=m['feature_date'],put_weight=TARGETS[action,0],straddle_weight=TARGETS[action,1],
            action=action,fitted=model is not None,last_training_end=model['last_training_end'] if model else None,
            resizing_cost_fraction=charge))
        nav=float(wealth[-1]);previous=after
    book=pd.DataFrame(rows).set_index('date');folder.mkdir(parents=True,exist_ok=True)
    book.to_parquet(folder/(stem+'.parquet'));pd.DataFrame(decisions).to_csv(folder/(stem+'_decisions.csv'),index=False)
    dump(folder/(stem+'_complete.json'),dict(rows=len(book)))
    return book


def metrics(book,start,stop):
    result=stats(book.excess_return,book.cash_return,start,stop)
    part=book.loc[start:stop];prior=max(0,book.index.get_loc(part.index[0])-1)
    benchmark=CFG['capital']*(1+book.cash_return).cumprod()
    result['profit_above_cash']=float((part.nav.iloc[-1]-benchmark.loc[part.index[-1]])-(book.nav.iloc[prior]-benchmark.iloc[prior]))
    return result


def fixed():
    data=load();records=[];checks=[]
    for action,weight in enumerate(SHARES):
        book=replay(data,f'fixed_{weight:g}',CFG['evaluation_end'],constant=action)
        if weight in [0,1]:
            source=data['pair' if weight==0 else 'straddle']
            np.testing.assert_allclose(book.excess_return,source.excess_return,atol=1e-14,rtol=0)
            np.testing.assert_allclose(book.nav,source.nav,atol=1e-6,rtol=0)
        for period,start,end in PERIODS:
            record=dict(kind='fixed',case=f'fixed_{weight:g}',period=period,straddle_target=weight,**metrics(book,start,end));records.append(record)
            if weight in [0,1]:
                expected=stats(source.excess_return,data['cash'],start,end)['annualized_excess_sharpe']
                np.testing.assert_allclose(record['annualized_excess_sharpe'],expected,atol=1e-12,rtol=0)
                checks.append(dict(period=period,straddle_target=weight,source_sharpe=expected,replayed_sharpe=record['annualized_excess_sharpe']))
    pd.DataFrame(records).to_csv(REPORT/'fixed_results.csv',index=False)
    dump(REPORT/'endpoint_verification.json',dict(daily_returns_and_nav_reproduced=True,checks=checks))
    print(pd.DataFrame(checks).to_string(index=False),flush=True)


def run(stop):
    assert (REPORT/'endpoint_verification.json').exists(),'Fixed endpoints must pass first'
    data=load();records=[]
    for risk in CFG['risk_preferences']:
        for gamma in [CFG['discount'],0.]:
            name=case(risk,gamma);book=replay(data,name,stop,risk,gamma)
            for period,start,end in PERIODS:
                if pd.Timestamp(start)>pd.Timestamp(stop):continue
                records.append(dict(kind='q' if gamma else 'myopic',case=name,period=period,risk=risk,gamma=gamma,**metrics(book,start,min(end,stop))))
    pd.DataFrame(records).to_csv(REPORT/('validation.csv' if stop==CFG['validation_end'] else 'q_results.csv'),index=False)


def select():
    frame=pd.read_csv(REPORT/'validation.csv')
    r=frame.loc[frame.kind.eq('q')&frame.period.eq('validation')].sort_values(['annualized_excess_sharpe','case'],ascending=[False,True]).iloc[0]
    value=dict(case=r['case'],risk=float(r.risk),rule=CFG['selection'])
    path=REPORT/'selection.json'
    if path.exists():assert json.loads(path.read_text())==value
    else:dump(path,value)


def analyze():
    frame=pd.concat([pd.read_csv(REPORT/'fixed_results.csv'),pd.read_csv(REPORT/'q_results.csv')],ignore_index=True)
    frame.to_csv(REPORT/'results.csv',index=False);timing=[];annual=[]
    for path in ROOT.glob('models/*/*.json'):
        m=json.loads(path.read_text());assert pd.Timestamp(m['last_training_end'])<pd.Timestamp(m['cutoff'])
        timing.append(dict(path=str(path),cutoff=m['cutoff'],last_training_end=m['last_training_end']))
    for name in frame.case.unique():
        folder=ROOT/'replays'/name;book=pd.read_parquet(folder/'full.parquet',use_threads=False)
        d=pd.read_csv(folder/'full_decisions.csv',parse_dates=['date','feature_date','last_training_end'])
        fitted=d.loc[d.fitted];assert fitted.last_training_end.lt(fitted.date).all()
        assert d.loc[d.feature_date.notna(),'feature_date'].lt(d.loc[d.feature_date.notna(),'date']).all()
        np.testing.assert_allclose(d.put_weight+d.straddle_weight,1,atol=1e-12)
        early=folder/'validation.parquet'
        if early.exists():
            old=pd.read_parquet(early,use_threads=False);np.testing.assert_allclose(book.nav.loc[old.index],old.nav,atol=1e-6,rtol=0)
        for year in range(2018,2026):annual.append(dict(case=name,year=year,**metrics(book,f'{year}-01-01',min(f'{year}-12-31',CFG['evaluation_end']))))
    pd.DataFrame(timing).to_csv(REPORT/'model_timing.csv',index=False);pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False)
    choice=json.loads((REPORT/'selection.json').read_text())
    frame.loc[frame.case.eq(choice['case'])].to_csv(REPORT/'validation_selected_results.csv',index=False)
    dump(REPORT/'verification.json',dict(endpoint_paths_verified=True,no_added_stop=True,no_fixed_warmup=True,
        one_capital_budget=True,completed_model_timing_checks=len(timing),selected_case=choice['case']))
    (REPORT/'FINDINGS.txt').write_text(CFG['scope']+'\n\n'+frame.loc[frame.period.ne('full')&frame.kind.ne('myopic')].to_string(index=False)+'\n')
    print(frame.loc[frame.kind.ne('myopic')&frame.period.eq('evaluation')].to_string(index=False),flush=True)


def charts():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame=pd.read_csv(REPORT/'results.csv');selected=json.loads((REPORT/'selection.json').read_text())['case']
    for period,title in [('validation','2018–2021 validation'),('evaluation','2022–August 2025 evaluation')]:
        fig,ax=plt.subplots(figsize=(11,6.5));fig.subplots_adjust(left=.10,right=.95,bottom=.20,top=.77)
        for kind,color,label in [('fixed','#85919b','Fixed put / straddle capital splits'),('q','#126a91','Dynamic Q (reward preferences)')]:
            group=frame.loc[frame.period.eq(period)&frame.kind.eq(kind)].sort_values('profit_above_cash')
            ax.plot(group.profit_above_cash/1e6,group.annualized_excess_sharpe,'o-',color=color,lw=2.5,ms=6,label=label)
            for r in group.itertuples():
                text=f'{1-r.straddle_target:.0%}/{r.straddle_target:.0%}' if kind=='fixed' else f'λ={r.risk:g}'
                ax.annotate(text,(r.profit_above_cash/1e6,r.annualized_excess_sharpe),xytext=(6,10),textcoords='offset points',fontsize=9,color=color)
        chosen=frame.loc[frame.period.eq(period)&frame.case.eq(selected)].iloc[0]
        ax.scatter(chosen.profit_above_cash/1e6,chosen.annualized_excess_sharpe,s=125,facecolors='none',edgecolors='#126a91',lw=1.5,zorder=5)
        fig.text(.10,.94,'Fixed versus dynamic allocation across two signals',fontsize=18,weight='bold')
        fig.text(.10,.88,title+' · Best put-pair bandit + best straddle bandit · One capital budget',fontsize=11,color='#536572')
        ax.set(xlabel='Cumulative profit above cash ($ millions)',ylabel='Annualized excess Sharpe')
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.14);ax.margins(.15,.22)
        ax.legend(frameon=False,fontsize=10,loc='lower left',bbox_to_anchor=(0,1.01),ncol=2,borderaxespad=0)
        fig.text(.10,.12,'Both pure endpoints reproduce the original strategy returns. No added portfolio stop or fixed-policy warmup.',fontsize=9)
        fig.text(.10,.075,'Q changes the capital split weekly; fixed portfolios keep their target split. Both pay the same resizing-cost model.',fontsize=8.5,color='#657681')
        fig.text(.10,.035,'All preferences shown; ring marks validation-selected Q. Q uses 50/50 until 52 completed weeks. Previously researched dates.',fontsize=8,color='#657681')
        fig.savefig(REPORT/('fixed_vs_dynamic_'+period+'.png'),dpi=170);fig.savefig(REPORT/('fixed_vs_dynamic_'+period+'.pdf'));plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',required=True,choices=['test','prepare','fixed','validation','select','evaluate','analyze','charts']);a=p.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True);sources()
    if a.phase=='test':
        import unittest
        if not unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('tests.test_signal_capital_allocation')).wasSuccessful():raise SystemExit(1)
    elif a.phase=='validation':run(CFG['validation_end'])
    elif a.phase=='evaluate':run(CFG['evaluation_end'])
    else:globals()[a.phase]()


if __name__=='__main__':main()
