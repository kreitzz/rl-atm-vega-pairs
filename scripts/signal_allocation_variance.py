"""Matched risk-preference comparison on the original nine-input Q allocator.

Run under scripts.run_signal_discovery_guarded with one worker. `run` resumes
monthly checkpoints and replays all four risk preferences. `charts` only reads
saved summaries. Data and execution are inherited from the corrected allocator.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scripts import signal_capital_allocation as core

CFG=json.loads(Path('config/signal_allocation_variance.json').read_text())
ROOT=Path(CFG['output_root']);REPORT=Path(CFG['report_root'])
RISKS=np.asarray(CFG['risk_preferences']);N=core.N


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.partial')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def load():
    import joblib
    return joblib.load(Path(CFG['source_root'])/'prepared.joblib')


def fit(data,index):
    current=data['periods'][index];cutoff=current['cutoff']
    train=[i for i,m in enumerate(data['periods'][:index]) if m['end']<cutoff and m['cutoff']>=cutoff-pd.Timedelta(weeks=CFG['training_weeks'])]
    if len(train)<CFG['minimum_training_weeks']:return None
    path=ROOT/'models'/(str(cutoff.to_period('M'))+'.json')
    if path.exists():
        model=json.loads(path.read_text())
        assert model['risks']==RISKS.tolist()
        assert pd.Timestamp(model['last_training_end'])<pd.Timestamp(model['cutoff'])<=cutoff
        return model
    states=np.concatenate([np.repeat(np.column_stack([np.tile(data['periods'][i]['market'],(N,1)),data['periods'][i]['before'][:,1]]),N,axis=0) for i in train])
    following=np.concatenate([np.column_stack([np.tile(data['periods'][i+1]['market'],(N,1)),data['periods'][i]['after'][:,1]]) for i in train])
    assert all(data['periods'][i+1]['feature_date']<=current['feature_date'] for i in train)
    center=states.mean(axis=0);scale=np.maximum(states.std(axis=0),.1)
    x=core.design(states,np.tile(np.arange(N),len(train)*N),center,scale);coef=x.shape[1]
    nx=core.design(np.repeat(following,N,axis=0),np.tile(np.arange(N),len(following)),center,scale).reshape(len(following),N,coef)
    projection=x.reshape(len(train),N,N,coef).sum(axis=1).reshape(-1,coef).T/(N*N)
    ref=np.concatenate([data['periods'][i]['paths'][3,3] for i in train]);vol=max(float(ref.std(ddof=1)),1e-8)
    profit=np.concatenate([data['periods'][i]['paths'].sum(axis=-1).ravel()/vol for i in train])
    squares=np.concatenate([(data['periods'][i]['paths']**2).sum(axis=-1).ravel()/vol**2 for i in train])
    penalty=CFG['ridge_alpha']*np.eye(coef);penalty[0,0]=.01
    inverse=np.linalg.inv(x.T@x/(N*N)+penalty)
    moments=x.T@np.column_stack([profit,squares])/(N*N)
    reward=moments[:,[0]]-moments[:,[1]]*RISKS[None,:];beta=inverse@reward
    for _ in range(CFG['iterations']-1):beta=inverse@(reward+CFG['discount']*(projection@(nx@beta).max(axis=1)))
    assert np.isfinite(beta).all()
    model=dict(cutoff=str(cutoff.date()),last_training_end=str(max(data['periods'][i]['end'] for i in train).date()),training_weeks=len(train),center=center.tolist(),scale=scale.tolist(),beta=beta.tolist(),risks=RISKS.tolist(),features=core.FEATURES+['previous_straddle_weight'])
    save(path,model);return model


def run():
    data=load();periods=data['periods'];paths=[[] for _ in RISKS];actions=[[] for _ in RISKS]
    previous=np.full(len(RISKS),3,int);model=None;month=None;fitted=[]
    for index,m in enumerate(periods):
        key=str(m['cutoff'].to_period('M'))
        if key!=month or model is None:model=fit(data,index);month=key
        fitted.append(model is not None)
        for j,risk in enumerate(RISKS):
            action=3
            if model:
                raw=np.r_[m['market'],m['before'][previous[j],1]]
                scores=core.design(np.tile(raw,(N,1)),np.arange(N),np.asarray(model['center']),np.asarray(model['scale']))@np.asarray(model['beta'])[:,j]
                action=int(scores.argmax())
            paths[j].append(m['paths'][previous[j],action]);actions[j].append(action);previous[j]=action
    cash=np.concatenate([m['cash'] for m in periods]);positions=np.concatenate([m['indices'] for m in periods]);dates=data['pair'].index[positions]
    decision_dates=pd.DatetimeIndex([m['cutoff'] for m in periods]);records=[];annual=[];schedules=[]
    REPORT.mkdir(parents=True,exist_ok=True)
    for j,risk in enumerate(RISKS):
        excess=np.concatenate(paths[j]);book=pd.DataFrame(dict(excess_return=excess,cash_return=cash,nav=1e7*np.cumprod(1+excess+cash)),index=dates)
        folder=ROOT/'replays'/f'lambda{risk:g}';folder.mkdir(parents=True,exist_ok=True)
        temporary=folder/'daily.partial.parquet';book.to_parquet(temporary);temporary.replace(folder/'daily.parquet')
        for period,start,end in core.PERIODS:
            mask=(decision_dates>=start)&(decision_dates<=end);schedule=np.asarray(actions[j]);weights=core.SHARES[schedule[mask]]
            records.append(dict(variant='full7',variance_penalty=risk,period=period,state_inputs=9,critic_coefficients=30,allocation_choices=7,mean_straddle_target=float(weights.mean()),endpoint_fraction=float(((weights==0)|(weights==1)).mean()),allocation_changes=int((np.diff(schedule[mask])!=0).sum()),**core.metrics(book,start,end)))
        for year in range(2018,2026):annual.append(dict(variant='full7',variance_penalty=risk,year=year,**core.metrics(book,f'{year}-01-01',min(f'{year}-12-31',CFG['evaluation_end']))))
        for date,action,active in zip(decision_dates,actions[j],fitted):schedules.append(dict(variant='full7',variance_penalty=risk,date=date,straddle_weight=float(core.SHARES[action]),fitted=active))
    pd.DataFrame(records).to_csv(REPORT/'results.csv',index=False)
    pd.DataFrame(annual).to_csv(REPORT/'annual.csv',index=False)
    pd.DataFrame(schedules).to_csv(REPORT/'decisions.csv',index=False)
    old=pd.read_csv('reports/signal_allocation_discount/q_results.csv')
    for r in records:
        if r['variance_penalty'] not in [0.,.1,.2]:continue
        expected=old.loc[(old.gamma==.99)&(old.risk==r['variance_penalty'])&(old.period==r['period'])].iloc[0]
        assert abs(r['annualized_excess_sharpe']-expected.annualized_excess_sharpe)<1e-8
        assert abs(r['profit_above_cash']-expected.profit_above_cash)<1e-5
    save(REPORT/'verification.json',dict(original_three_preferences_reproduced=True,all_monthly_training_ends_precede_decisions=True,nine_inputs=True,seven_actions=True,policies=RISKS.tolist()))
    print(pd.DataFrame(records).to_string(index=False),flush=True)


def charts():
    selected=True
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Bbox
    q=pd.read_csv(REPORT/'results.csv')
    if selected:q=q.loc[q.variance_penalty.isin([0.,.1,.2,1.])]
    stem='original_state_selected_lambdas' if selected else 'original_state_all_lambdas'
    fixed=pd.read_csv('reports/signal_allocation_discount/fixed_results.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':11,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(16,7),sharey=True)
    fig.subplots_adjust(left=.065,right=.985,bottom=.17,top=.80,wspace=.13)
    for ax,(period,title) in zip(axes,[('validation','Validation | 2018–2021'),('evaluation','2022–August 2025')]):
        fq=q.loc[q.period.eq(period)].sort_values('variance_penalty')
        ff=fixed.loc[fixed.period.eq(period)].sort_values('profit_above_cash')
        assert len(fq)==(4 if selected else 15) and len(ff)==7
        ax.plot(ff.profit_above_cash/1e6,ff.annualized_excess_sharpe,'o--',color='#9b9fa5',lw=1.8,ms=5,label='Fixed capital splits',zorder=2)
        ax.plot(fq.profit_above_cash/1e6,fq.annualized_excess_sharpe,'o-',color='#087fa4',lw=2,ms=5.5,label='Dynamic Q · λ = 0, 0.1, 0.2, 1' if selected else 'Dynamic Q · all 15 λ values',zorder=3)
        ax.set_title(title,loc='left',fontsize=14,pad=16,weight='bold')
        ax.set_xlabel('Cumulative profit above cash ($ millions)',labelpad=10)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.15)
        ax.set_ylim(.1,2.38)
        maxprofit=max(fq.profit_above_cash.max(),ff.profit_above_cash.max())/1e6
        ax.set_xlim(-.03,maxprofit*1.13)
        ax.tick_params(axis='both',length=0,pad=7)
        ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('${x:.1f}m'))
    axes[0].set_ylabel('Annualized excess Sharpe',labelpad=12)
    fig.text(.065,.947,'The profit–Sharpe tradeoff across four risk penalties' if selected else 'Profit versus Sharpe across the full λ sweep',fontsize=21,weight='bold',color='#152b39')
    fig.text(.065,.894,'Original nine-input state · γ = 0.99 · seven allocation choices · unchanged strategy bandits and costs',fontsize=11,color='#516473')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1],labels[::-1],loc='upper right',bbox_to_anchor=(.985,.855),ncol=2,frameon=False,fontsize=10)
    fig.text(.065,.083,'Blue labels identify λ, the variance penalty. Gray labels show fixed put-pair / straddle capital weights.',fontsize=10,color='#516473')
    fig.text(.065,.042,'Selected λ values span the tested range. Lines connect successive λ settings and fixed capital splits; they do not imply a fitted efficient frontier.' if selected else 'Every tested λ is shown. Lines connect policies in order of profit for readability; they do not imply a fitted efficient frontier.',fontsize=9,color='#657681')
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for ax,period in zip(axes,['validation','evaluation']):
        items=[]
        for r in q.loc[q.period.eq(period)].itertuples():
            items.append((r.profit_above_cash/1e6,r.annualized_excess_sharpe,'λ='+f'{r.variance_penalty:g}','#087fa4',9))
        for r in fixed.loc[fixed.period.eq(period)].itertuples():
            items.append((r.profit_above_cash/1e6,r.annualized_excess_sharpe,f'{1-r.straddle_target:.0%}/{r.straddle_target:.0%}','#777e85',8))
        # Place every label explicitly, avoiding existing labels and plotted points.
        points=[ax.transData.transform((x,y)) for x,y,*_ in items]
        occupied=[]
        bounds=ax.get_window_extent()
        for x,y,label,color,size in items:
            best=None
            offsets=[(6,10),(-6,10),(6,-15),(-6,-15),(6,24),(-6,24),(6,-29),(-6,-29),(6,38),(-6,38),(20,4),(-20,4),(20,-18),(-20,-18),(6,52),(-6,52),(6,-43),(-6,-43),(40,20),(-40,20)]
            for dx,dy in offsets:
                ann=ax.annotate(label,(x,y),xytext=(dx,dy),textcoords='offset points',ha='left' if dx>=0 else 'right',va='center',fontsize=size,color=color,zorder=5)
                bb=ann.get_window_extent(renderer).expanded(1.08,1.25)
                collision=sum(bb.overlaps(old) for old in occupied)
                pointcollisions=sum(bb.contains(px,py) for px,py in points)
                outside=max(0,bounds.x0-bb.x0)+max(0,bb.x1-bounds.x1)+max(0,bounds.y0-bb.y0)+max(0,bb.y1-bounds.y1)
                score=collision*10000+pointcollisions*1000+outside*100+abs(dx)+abs(dy)
                ann.remove()
                if best is None or score<best[0]:best=(score,dx,dy,bb)
            _,dx,dy,bb=best
            ax.annotate(label,(x,y),xytext=(dx,dy),textcoords='offset points',ha='left' if dx>=0 else 'right',va='center',fontsize=size,color=color,zorder=5,arrowprops=dict(arrowstyle='-',color=color,alpha=.45,lw=.55,shrinkA=1,shrinkB=4))
            occupied.append(bb)
    for ext in ['png','pdf','svg']:
        fig.savefig(REPORT/(stem+'.'+ext),dpi=150,facecolor='white')
    plt.close(fig)
    print(str(REPORT/(stem+'.png')),flush=True)


def paper_chart():
    """Export a compact, readable single-panel figure for the paper."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    q=pd.read_csv(REPORT/'results.csv').query("period == 'evaluation'").sort_values('variance_penalty')
    fixed=pd.read_csv(REPORT/'fixed_results.csv').query("period == 'evaluation'").sort_values('straddle_target')
    fig,ax=plt.subplots(figsize=(9,5.1))
    fig.subplots_adjust(left=.10,right=.97,bottom=.20,top=.83)
    ax.plot(fixed.profit_above_cash/1e6,fixed.annualized_excess_sharpe,'o--',color='#9099a1',lw=1.6,ms=4,label='Fixed put / straddle splits')
    ax.plot(q.profit_above_cash/1e6,q.annualized_excess_sharpe,'o-',color='#087fa4',lw=2,ms=5,label='Dynamic Q · four risk preferences')
    for r in q.itertuples():
        ax.annotate(f'λ = {r.variance_penalty:g}',(r.profit_above_cash/1e6,r.annualized_excess_sharpe),xytext=(7,5),textcoords='offset points',fontsize=9,color='#087fa4')
    for r in fixed.itertuples():
        ax.annotate(f'{1-r.straddle_target:.0%}/{r.straddle_target:.0%}',(r.profit_above_cash/1e6,r.annualized_excess_sharpe),xytext=(2,9),textcoords='offset points',fontsize=8,color='#707b85')
    ax.set(xlim=(.1,2.65),ylim=(.4,2.4),xlabel='Cumulative profit above cash ($ millions)',ylabel='Annualized excess Sharpe')
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
    ax.legend(frameon=False,loc='upper left',fontsize=8.5)
    fig.text(.10,.94,'Risk preference changes the allocation tradeoff',fontsize=14,weight='bold')
    fig.text(.10,.875,'2022–August 2025 · original nine-input state · γ = 0.99',fontsize=10,color='#516473')
    fig.text(.10,.055,'Lines join discrete settings; they do not represent a fitted efficient frontier.',fontsize=8,color='#657681')
    for ext in ['png','pdf','svg']:fig.savefig(REPORT/('risk_preferences_paper.'+ext),dpi=110,facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',required=True,choices=['run','charts','paper_chart'])
    args=parser.parse_args()
    globals()[args.phase]()
