"""Compact vector figures integrating the frozen teammate economics snapshots.

Called one figure per guarded worker by options_paper_figures. No fitting or
backtest is performed. Inputs are verified by the manuscript preparation step.
"""
import csv
from datetime import datetime
from pathlib import Path

DATA=Path('reports/strategy_economics/data')


def build(name):
    import matplotlib.pyplot as plt
    import matplotlib.dates as dates
    import numpy as np
    from scripts.options_paper_figures import BLUE,TEAL,GRAY,RED,WIDTH

    def read(file):
        with (DATA/file).open() as f:return list(csv.DictReader(f))

    def value(data,key):return np.array([float(r[key]) for r in data])
    def times(data):return [datetime.fromisoformat(r['date']) for r in data]
    def axis_dates(ax):
        locator=dates.AutoDateLocator(minticks=3,maxticks=5)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(dates.ConciseDateFormatter(locator))
        ax.grid(axis='y',alpha=.15)

    if name=='straddle_bridge':
        data=read('profit_bridge.csv');amounts=[float(r['dollars'])/1e6 for r in data]
        fig,ax=plt.subplots(figsize=(WIDTH,2.15))
        fig.subplots_adjust(left=.09,right=.99,top=.90,bottom=.28)
        total=sum(amounts);running=0
        for i,amount in enumerate(amounts+[total]):
            base=0 if i==len(amounts) else running
            ax.bar(i,amount,bottom=base,color=BLUE if i==len(amounts) else TEAL if amount>=0 else RED,width=.68)
            end=base+amount
            ax.annotate(f'{amount:+.2f}',(i,end),xytext=(0,4 if amount>=0 else -12),
                        textcoords='offset points',ha='center',fontsize=8)
            if i<len(amounts):
                running+=amount
                ax.plot([i+.34,i+.66],[running,running],color=GRAY,lw=.6)
        ax.set(xticks=range(7),xticklabels=['Options','Stock\nhedge','Hedge\ncosts','Interest\nvs cash','Stock\nborrow','Hedge\ndividends','Above\ncash'],
               ylabel='Profit ($m)',ylim=(0,4.35))
        ax.tick_params(axis='x',labelsize=8.5);ax.grid(axis='y',alpha=.12)
    elif name=='straddle_paths':
        trades=read('trade_outcomes.csv')
        groups=[[r for r in trades if float(r['persistence'])<=.2],
                [r for r in trades if float(r['persistence'])>.2]]
        fig,ax=plt.subplots(figsize=(WIDTH,2.0))
        fig.subplots_adjust(left=.09,right=.99,top=.80,bottom=.23)
        for i,(field,label,color) in enumerate([('option_profit','Option',BLUE),('hedge_profit','Stock hedge',RED),('combined_profit','Combined',TEAL)]):
            vals=[sum(float(r[field]) for r in g)/1e6 for g in groups]
            ax.bar(np.arange(2)+(i-1)*.24,vals,.23,label=label,color=color)
        ax.set(xticks=[0,1],xticklabels=['More reversing (1,292)','More persistent (1,307)'],ylabel='Profit ($m)')
        ax.axhline(0,color=GRAY,lw=.65);ax.grid(axis='y',alpha=.12)
        fig.legend(loc='upper center',ncol=3,frameon=False,fontsize=9)
    elif name=='straddle_ivrv':
        trades=read('trade_outcomes.csv')
        groups=[]
        for persistent in [True,False]:
            for rich in [False,True]:
                groups.append([r for r in trades if (float(r['persistence'])>.2)==persistent
                               and (float(r['entry_iv'])>=float(r['rv']))==rich])
        vals=[sum(float(r['combined_profit']) for r in g)/1e6 for g in groups]
        fig,ax=plt.subplots(figsize=(WIDTH,2.8))
        fig.subplots_adjust(left=.11,right=.985,top=.91,bottom=.24)
        ax.bar(range(4),vals,color=[TEAL if x>=0 else RED for x in vals],width=.65)
        for i,(v,g) in enumerate(zip(vals,groups)):
            inside=v<-2
            ax.annotate(f'${v:+.2f}m\n{len(g):,} cohorts',(i,v/2 if inside else v),
                        xytext=(0,0 if inside else 5 if v>=0 else -6),textcoords='offset points',
                        ha='center',va='center' if inside else 'bottom' if v>=0 else 'top',
                        fontsize=9,color='white' if inside else 'black')
        ax.set(xticks=range(4),xticklabels=['Persistent\nIV < RV','Persistent\nIV ≥ RV','Reversing\nIV < RV','Reversing\nIV ≥ RV'],
               ylabel='Option + hedge profit ($m)',ylim=(-6,8))
        ax.axhline(0,color=GRAY,lw=.65);ax.grid(axis='y',alpha=.12)
    elif name=='straddle_examples':
        data=read('trade_examples.csv')
        fig,axes=plt.subplots(3,2,figsize=(WIDTH,4.6))
        fig.subplots_adjust(left=.105,right=.99,top=.93,bottom=.10,wspace=.30,hspace=.40)
        for j,secid in enumerate(['101803','213114']):
            b=sorted([r for r in data if r['secid']==secid],key=lambda r:r['date'])
            x=times(b);spot=value(b,'spot');carried=value(b,'shares_carried')
            assert np.allclose((carried*np.r_[0,np.diff(spot)])[1:],value(b,'hedge_profit')[1:],atol=1e-6)
            axes[0,j].plot(x,spot,color=BLUE,lw=1.4,label='Spot')
            axes[0,j].axhline(float(b[0]['strike']),color=GRAY,ls='--',lw=.8,label='Strike')
            axes[0,j].set_title(('Hedge-gain' if j==0 else 'Hedge-loss')+f' example: {secid}',fontsize=9.5)
            axes[0,j].legend(frameon=False,fontsize=7.5,ncol=2)
            axes[1,j].step(x,carried/1000,where='pre',color=RED,lw=1.2)
            for field,label,color in [('option_profit','Option',BLUE),('hedge_profit','Hedge',RED),('combined_profit','Combined',TEAL)]:
                axes[2,j].plot(x,np.cumsum(value(b,field))/1000,label=label,color=color,lw=1.2)
            for ax in axes[:,j]:
                axis_dates(ax);ax.tick_params(labelsize=8)
        for ax,label in zip(axes[:,0],['Spot / strike ($)','Shares (thousands)','Profit ($ thousands)']):ax.set_ylabel(label,fontsize=9)
        handles,labels=axes[2,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,fontsize=8,bbox_to_anchor=(.5,-.005))
    elif name=='straddle_management':
        data=read('account_daily.csv')
        books={case:sorted([r for r in data if r['case']==case and r['date']>='2021-12-31'],key=lambda r:r['date'])
               for case in ['hold_control','conditional_half','constant_lower']}
        fig,axes=plt.subplots(2,1,figsize=(WIDTH,2.7),sharex=True,gridspec_kw={'height_ratios':[1.5,1]})
        fig.subplots_adjust(left=.14,right=.985,top=.82,bottom=.16,hspace=.13)
        for case,label,color in [('hold_control','Unchanged 90/10',BLUE),('conditional_half','Conditional cuts',TEAL),('constant_lower','Constant 90/5 + cash',RED)]:
            b=books[case];nav=value(b,'nav');cash=value(b,'cash_benchmark')
            growth=(nav/nav[0])/(cash/cash[0])-1
            axes[0].plot(times(b),100*growth,label=label,color=color,lw=1.2)
        a=books['conditional_half'];b=books['hold_control']
        assert [r['date'] for r in a]==[r['date'] for r in b]
        excess=value(a,'excess_return')-value(b,'excess_return');excess[0]=0
        axes[1].plot(times(a),100*np.cumsum(excess),color=TEAL,lw=1.2)
        axes[1].axhline(0,color=GRAY,lw=.65)
        axes[0].set_ylabel('Growth vs\ncash (%)',fontsize=9)
        axes[1].set_ylabel('Accumulated\ndifference (pp)',fontsize=9)
        for ax in axes:axis_dates(ax)
        fig.legend(loc='upper center',ncol=3,frameon=False,fontsize=8.5,columnspacing=1.1)
    else:raise ValueError(name)
    return fig
