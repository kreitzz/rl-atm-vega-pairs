"""Profit versus return per unit of each candidate capital allocation's risk."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts import signal_allocation_objectives as parent

CONFIG=Path('config/signal_allocation_forecast_reward.json');DECLARED=json.loads(CONFIG.read_text())
CFG=dict(parent.CFG,**DECLARED);ROOT,REPORT=Path(CFG['output_root']),Path(CFG['report_root'])
BASE_FIT=parent.fit;CURRENT=[];POSITION=0;COVARIANCES=[]


def covariance_forecast(pair,straddle,start,cutoff,half_life):
    sample=pd.concat([pair,straddle],axis=1).loc[lambda z:(z.index>=start)&(z.index<cutoff)].to_numpy()
    assert len(sample)>1 and np.isfinite(sample).all()
    weights=np.exp2(-np.arange(len(sample)-1,-1,-1)/half_life);weights/=weights.sum()
    centered=sample-(weights[:,None]*sample).sum(axis=0)
    return (centered.T@(weights[:,None]*centered))/(1-weights@weights)


def objectives():
    rows=[dict(key='profit',family='shared',weight=1.,label='Profit')]
    for family,half in [('local',63),('replacement',126)]:
        for w in CFG['profit_weights'][:-1]:
            rows.append(dict(key=f'forecast{half}_profit{w:g}',family=family,weight=w,
                label='Sharpe-focused' if w==0 else ('Balanced' if w==.5 else f'{w:.0%} profit')))
    return rows


def reward_parts(paths,cash,reference,total,squares,count):
    global POSITION
    m=CURRENT[POSITION];POSITION+=1
    profit=CFG['capital']/CFG['profit_scale_dollars']*(np.prod(1+paths+cash,axis=-1)-np.prod(1+cash))
    columns=[profit]
    for covariance in COVARIANCES:
        prior=np.einsum('ij,jk,ik->i',m['before'][:,:2],covariance,m['before'][:,:2])
        target=np.einsum('ij,jk,ik->i',parent.base.TARGETS[:,:2],covariance,parent.base.TARGETS[:,:2])
        variance=(prior[:,None]+(paths.shape[-1]-1)*target[None,:])/paths.shape[-1]
        sd=np.sqrt(np.maximum(variance,1e-16))
        columns.append(paths.sum(axis=-1)/sd/np.sqrt(252))
    return np.stack(columns,axis=-1)


def fit(data,index):
    global CURRENT,POSITION,COVARIANCES
    cutoff=data['periods'][index]['cutoff']
    CURRENT=[m for m in data['periods'][:index] if m['end']<cutoff and m['cutoff']>=cutoff-pd.Timedelta(weeks=CFG['training_weeks'])]
    POSITION=0
    if len(CURRENT)<CFG['minimum_training_weeks']:return None
    COVARIANCES=[covariance_forecast(data['pair'].excess_return,data['straddle'].excess_return,
        CURRENT[0]['cutoff'],cutoff,half) for half in [63,126]]
    result=BASE_FIT(data,index)
    assert POSITION==len(CURRENT)
    return result


def sources():
    paths=[CONFIG,Path(__file__),Path(parent.__file__),Path(parent.base.__file__),parent.SOURCE/'prepared.joblib',parent.SOURCE/'sources.json']
    record={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};path=ROOT/'sources.json'
    if path.exists():assert json.loads(path.read_text())==record
    else:parent.base.dump(path,record)


def configure():
    parent.CONFIG=CONFIG;parent.CFG=CFG;parent.ROOT,parent.REPORT=ROOT,REPORT
    parent.objectives=objectives;parent.reward_parts=reward_parts;parent.fit=fit;parent.sources=sources
    parent.configure()


def charts():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fixed=pd.read_csv(REPORT/'fixed_results.csv');curve=pd.read_csv(REPORT/'selected_reward_curve.csv')
    choice=json.loads((REPORT/'selection.json').read_text());description=CFG['family_meanings'][choice['family']]
    for period,title in [('validation','2018–2021 validation'),('evaluation','2022–August 2025 evaluation')]:
        fig,ax=plt.subplots(figsize=(11,6.5));fig.subplots_adjust(left=.10,right=.95,bottom=.20,top=.77)
        f=fixed.loc[fixed.period.eq(period)].sort_values('profit_above_cash');q=curve.loc[curve.period.eq(period)].sort_values('profit_above_cash')
        ax.plot(f.profit_above_cash/1e6,f.annualized_excess_sharpe,'o-',color='#85919b',lw=2.5,label='Fixed capital splits')
        ax.plot(q.profit_above_cash/1e6,q.annualized_excess_sharpe,'o-',color='#126a91',lw=2.5,label='Dynamic Q: profit versus risk-adjusted reward')
        for r in f.itertuples():ax.annotate(f'{1-r.straddle_target:.0%}/{r.straddle_target:.0%}',(r.profit_above_cash/1e6,r.annualized_excess_sharpe),xytext=(6,10),textcoords='offset points',fontsize=9,color='#85919b')
        for r in q.itertuples():ax.annotate(r.objective_label,(r.profit_above_cash/1e6,r.annualized_excess_sharpe),xytext=(6,12),textcoords='offset points',fontsize=9,color='#126a91')
        fig.text(.10,.94,'Rewarding profit versus risk-adjusted performance',fontsize=18,weight='bold')
        fig.text(.10,.88,title+' · Same two best bandits · Discount 0.99',fontsize=11,color='#536572')
        ax.set(xlabel='Cumulative profit above cash ($ millions)',ylabel='Annualized excess Sharpe')
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.14);ax.margins(.18,.25)
        ax.legend(frameon=False,fontsize=10,loc='lower left',bbox_to_anchor=(0,1.01),ncol=2,borderaxespad=0)
        fig.text(.10,.12,'The reward endpoints target net dollar profit and net return per unit of forecast allocation risk. All five goals shown.',fontsize=9)
        fig.text(.10,.075,'Risk forecast selected on validation: '+description+'. Same fixed mixes, budget, costs and next-close execution.',fontsize=8.5,color='#657681')
        fig.text(.10,.035,'Forecast-risk normalization is a Sharpe surrogate; realized Sharpe and P&L are measured separately. Previously researched dates.',fontsize=8,color='#657681')
        fig.savefig(REPORT/('reward_frontier_'+period+'.png'),dpi=170);fig.savefig(REPORT/('reward_frontier_'+period+'.pdf'));plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True,choices=['test','fixed','validation','select','evaluate','analyze','charts']);a=p.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True);configure();sources()
    if a.phase=='test':
        import unittest
        if not unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('tests.test_signal_allocation_forecast_reward')).wasSuccessful():raise SystemExit(1)
    elif a.phase=='fixed':parent.base.fixed()
    elif a.phase=='validation':parent.run(CFG['validation_end'])
    elif a.phase=='evaluate':parent.run(CFG['evaluation_end'])
    elif a.phase=='charts':charts()
    else:getattr(parent,a.phase)()


if __name__=='__main__':main()
