"""Direct profit and Sharpe-change rewards on the verified two-bandit allocator."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scripts import signal_capital_allocation as base

CONFIG=Path('config/signal_allocation_objectives.json');DECLARED=json.loads(CONFIG.read_text())
CFG=dict(base.CFG,**DECLARED);ROOT,REPORT,SOURCE=map(Path,[CFG['output_root'],CFG['report_root'],CFG['source_root']])
N=base.N


def objectives():
    rows=[dict(key='profit',family='shared',weight=1.,label='Profit')]
    for family in CFG['sharpe_rewards']:
        for w in CFG['profit_weights'][:-1]:
            rows.append(dict(key=f'{family}_profit{w:g}',family=family,weight=w,
                label='Sharpe' if w==0 else ('Balanced' if w==.5 else f'{w:.0%} profit')))
    return rows


def name(key,gamma):return f'{key}_g{gamma:g}'


def sources():
    paths=[CONFIG,Path(__file__),Path(base.__file__),Path(base.engine.__file__),SOURCE/'prepared.joblib',
        SOURCE/'sources.json',Path('reports/signal_capital_allocation/fixed_results.csv'),
        Path('reports/signal_capital_allocation/endpoint_verification.json')]
    record={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};path=ROOT/'sources.json'
    if path.exists():assert json.loads(path.read_text())==record
    else:base.dump(path,record)


def load():
    import joblib
    return joblib.load(SOURCE/'prepared.joblib')


def sharpe_from_moments(total,squares,count):
    mean=np.asarray(total)/count
    variance=np.maximum((np.asarray(squares)-np.asarray(total)**2/count)/(count-1),1e-16)
    return np.sqrt(252)*mean/np.sqrt(variance)


def reward_parts(paths,cash,reference,total,squares,count):
    """Exact finite replacement and its derivative use only completed labels."""
    sums=paths.sum(axis=-1);second=(paths*paths).sum(axis=-1)
    delta=sums-reference.sum();delta2=second-(reference*reference).sum()
    profit=CFG['capital']/CFG['profit_scale_dollars']*(np.prod(1+paths+cash,axis=-1)-np.prod(1+cash))
    replacement=(sharpe_from_moments(total+delta,squares+delta2,count)
        -sharpe_from_moments(total,squares,count))*count/252
    mean=total/count;sd=np.sqrt(max((squares-total**2/count)/(count-1),1e-16));correction=count/(count-1)
    local=((1/sd+correction*mean**2/sd**3)*delta-correction*mean/(2*sd**3)*delta2)/np.sqrt(252)
    return np.stack([profit,local,replacement],axis=-1)


def fit(data,index):
    periods=data['periods'];current=periods[index];cutoff=current['cutoff']
    train=[i for i,m in enumerate(periods[:index]) if m['end']<cutoff and m['cutoff']>=cutoff-pd.Timedelta(weeks=CFG['training_weeks'])]
    if len(train)<CFG['minimum_training_weeks']:return None
    last=max(periods[i]['end'] for i in train);assert last<cutoff
    states=np.concatenate([np.repeat(np.column_stack([np.tile(periods[i]['market'],(N,1)),periods[i]['before'][:,1]]),N,axis=0) for i in train])
    following=np.concatenate([np.column_stack([np.tile(periods[i+1]['market'],(N,1)),periods[i]['after'][:,1]]) for i in train])
    assert all(periods[i+1]['feature_date']<=current['feature_date'] for i in train)
    center=states.mean(axis=0);scale=np.maximum(states.std(axis=0),.1)
    x=base.design(states,np.tile(np.arange(N),len(train)*N),center,scale);coef=x.shape[1]
    nx=base.design(np.repeat(following,N,axis=0),np.tile(np.arange(N),len(following)),center,scale).reshape(len(following),N,coef)
    projection=x.reshape(len(train),N,N,coef).sum(axis=1).reshape(-1,coef).T/(N*N)
    ref=np.concatenate([periods[i]['paths'][3,3] for i in train]);total=float(ref.sum());second=float(ref@ref);count=len(ref)
    rewards=np.concatenate([reward_parts(periods[i]['paths'],periods[i]['cash'],periods[i]['paths'][3,3],total,second,count).reshape(N*N,3) for i in train])
    penalty=CFG['ridge_alpha']*np.eye(coef);penalty[0,0]=.01
    inverse=np.linalg.inv(x.T@x/(N*N)+penalty);moments=x.T@rewards/(N*N)
    month=str(cutoff.to_period('M'))
    for obj in objectives():
        column=1 if obj['family']=='local' else 2;w=obj['weight']
        reward=w*moments[:,0]+(1-w)*moments[:,column];immediate=inverse@reward
        for gamma in [CFG['discount'],0.]:
            beta=immediate.copy();diagnostics=[];previous_actions=None;previous_gaps=None
            if gamma:
                for iteration in range(1,CFG['iterations']):
                    values=nx@beta;beta=inverse@(reward+gamma*(projection@values.max(axis=1)))
                    if iteration in [19,49,99,199]:
                        values=nx@beta;greedy=values.argmax(axis=1);gaps=values-values[:,[0]]
                        diagnostics.append(dict(updates=iteration+1,
                            action_agreement=None if previous_actions is None else float((greedy==previous_actions).mean()),
                            max_action_gap_change=None if previous_gaps is None else float(np.max(np.abs(gaps-previous_gaps)))))
                        previous_actions,previous_gaps=greedy,gaps
            assert np.isfinite(beta).all()
            model=dict(beta=beta.tolist(),center=center.tolist(),scale=scale.tolist(),cutoff=str(cutoff.date()),
                last_training_end=str(last.date()),training_weeks=len(train),objective=obj,gamma=gamma,
                reference_sharpe=float(sharpe_from_moments(total,second,count)),reference_mean=total/count,
                reference_sessions=count,features=base.FEATURES+['previous_straddle_weight'],iteration_checks=diagnostics)
            base.dump(ROOT/'models'/name(obj['key'],gamma)/(month+'.json'),model)
    print(json.dumps(dict(month=month,training_weeks=len(train),objectives=len(objectives()))),flush=True)
    return True


def model_for(data,index,key,gamma):
    m=data['periods'][index];path=ROOT/'models'/name(key,gamma)/(str(m['cutoff'].to_period('M'))+'.json')
    if not path.exists() and fit(data,index) is None:return None
    model=json.loads(path.read_text())
    if pd.Timestamp(model['cutoff'])>m['cutoff']:return None
    assert pd.Timestamp(model['last_training_end'])<m['cutoff']
    return model


def configure():
    base.CFG=CFG;base.ROOT,base.REPORT=ROOT,REPORT;base.load=load;base.model_for=model_for


def run(stop):
    assert (REPORT/'endpoint_verification.json').exists()
    data=load();records=[]
    for obj in objectives():
        for gamma in [CFG['discount'],0.]:
            key=name(obj['key'],gamma);book=base.replay(data,key,stop,obj['key'],gamma)
            for period,start,end in base.PERIODS:
                if pd.Timestamp(start)>pd.Timestamp(stop):continue
                records.append(dict(kind='q' if gamma else 'myopic',case=key,objective=obj['key'],family=obj['family'],
                    profit_weight=obj['weight'],objective_label=obj['label'],gamma=gamma,period=period,
                    **base.metrics(book,start,min(end,stop))))
    pd.DataFrame(records).to_csv(REPORT/('validation.csv' if stop==CFG['validation_end'] else 'q_results.csv'),index=False)


def select():
    frame=pd.read_csv(REPORT/'validation.csv')
    candidates=frame.loc[frame.kind.eq('q')&frame.period.eq('validation')&frame.profit_weight.eq(0)]
    chosen=candidates.sort_values(['annualized_excess_sharpe','family'],ascending=[False,True]).iloc[0]
    value=dict(family=chosen.family,validation_pure_sharpe=float(chosen.annualized_excess_sharpe),rule=CFG['selection'])
    path=REPORT/'selection.json'
    if path.exists():assert json.loads(path.read_text())==value
    else:base.dump(path,value)
    candidates.to_csv(REPORT/'sharpe_reward_validation.csv',index=False)


def analyze():
    frame=pd.concat([pd.read_csv(REPORT/'fixed_results.csv'),pd.read_csv(REPORT/'q_results.csv')],ignore_index=True)
    frame.to_csv(REPORT/'results.csv',index=False);timing=[];checks=[]
    for path in ROOT.glob('models/*/*.json'):
        m=json.loads(path.read_text());assert pd.Timestamp(m['last_training_end'])<pd.Timestamp(m['cutoff'])
        timing.append(dict(path=str(path),cutoff=m['cutoff'],last_training_end=m['last_training_end']))
        for check in m['iteration_checks']:checks.append(dict(case=path.parent.name,month=path.stem,**check))
    pd.DataFrame(timing).to_csv(REPORT/'model_timing.csv',index=False);pd.DataFrame(checks).to_csv(REPORT/'iteration_checks.csv',index=False)
    choice=json.loads((REPORT/'selection.json').read_text())
    curve=frame.loc[frame.kind.eq('q')&(frame.family.eq(choice['family'])|frame.family.eq('shared'))]
    curve.to_csv(REPORT/'selected_reward_curve.csv',index=False)
    old=pd.read_csv('reports/signal_capital_allocation/fixed_results.csv')
    pd.testing.assert_frame_equal(pd.read_csv(REPORT/'fixed_results.csv'),old)
    base.dump(REPORT/'verification.json',dict(fixed_curves_unchanged=True,endpoint_paths_verified=True,
        model_timing_checks=len(timing),selected_family=choice['family'],discount=CFG['discount'],bellman_updates=CFG['iterations']))
    (REPORT/'FINDINGS.txt').write_text(CFG['scope']+'\n\n'+curve.loc[curve.period.ne('full')].to_string(index=False)+'\n')
    print(curve.loc[curve.period.eq('evaluation')].to_string(index=False),flush=True)


def charts():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fixed=pd.read_csv(REPORT/'fixed_results.csv');curve=pd.read_csv(REPORT/'selected_reward_curve.csv')
    choice=json.loads((REPORT/'selection.json').read_text())
    for period,title in [('validation','2018–2021 validation'),('evaluation','2022–August 2025 evaluation')]:
        fig,ax=plt.subplots(figsize=(11,6.5));fig.subplots_adjust(left=.10,right=.95,bottom=.20,top=.77)
        f=fixed.loc[fixed.period.eq(period)].sort_values('profit_above_cash');q=curve.loc[curve.period.eq(period)].sort_values('profit_above_cash')
        ax.plot(f.profit_above_cash/1e6,f.annualized_excess_sharpe,'o-',color='#85919b',lw=2.5,label='Fixed capital splits')
        ax.plot(q.profit_above_cash/1e6,q.annualized_excess_sharpe,'o-',color='#126a91',lw=2.5,label='Dynamic Q: profit–Sharpe reward')
        for r in f.itertuples():ax.annotate(f'{1-r.straddle_target:.0%}/{r.straddle_target:.0%}',(r.profit_above_cash/1e6,r.annualized_excess_sharpe),xytext=(6,10),textcoords='offset points',fontsize=9,color='#85919b')
        for r in q.itertuples():ax.annotate(r.objective_label,(r.profit_above_cash/1e6,r.annualized_excess_sharpe),xytext=(6,12),textcoords='offset points',fontsize=9,color='#126a91')
        fig.text(.10,.94,'Rewarding profit versus rewarding Sharpe',fontsize=18,weight='bold')
        fig.text(.10,.88,title+' · Dynamic allocation across the same best two bandits · Discount 0.99',fontsize=11,color='#536572')
        ax.set(xlabel='Cumulative profit above cash ($ millions)',ylabel='Annualized excess Sharpe')
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.14);ax.margins(.18,.25)
        ax.legend(frameon=False,fontsize=10,loc='lower left',bbox_to_anchor=(0,1.01),ncol=2,borderaxespad=0)
        fig.text(.10,.12,'Q endpoints target Sharpe and net dollar profit; intermediate points blend those rewards. All five objectives shown.',fontsize=9)
        fig.text(.10,.075,'Sharpe reward formulation selected on validation and frozen for the whole curve. Same fixed mixes, budget and costs.',fontsize=8.5,color='#657681')
        fig.text(.10,.035,'Sharpe-change reward is a historical replacement surrogate, not a guarantee of globally optimal Sharpe. Previously researched dates.',fontsize=8,color='#657681')
        fig.savefig(REPORT/('profit_sharpe_objectives_'+period+'.png'),dpi=170);fig.savefig(REPORT/('profit_sharpe_objectives_'+period+'.pdf'));plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True,choices=['test','fixed','validation','select','evaluate','analyze','charts']);a=p.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True);configure();sources()
    if a.phase=='test':
        import unittest
        if not unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('tests.test_signal_allocation_objectives')).wasSuccessful():raise SystemExit(1)
    elif a.phase=='fixed':base.fixed()
    elif a.phase=='validation':run(CFG['validation_end'])
    elif a.phase=='evaluate':run(CFG['evaluation_end'])
    else:globals()[a.phase]()


if __name__=='__main__':main()
