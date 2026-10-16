"""Publication figures at their printed width, using saved research only."""
from pathlib import Path
import csv

BLUE = '#244b70'
TEAL = '#087f91'
GRAY = '#7d8791'
RED = '#ad554c'
PALE = '#edf3f7'
WIDTH = 7.1


def rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def one(data, **keys):
    found = [r for r in data if all(str(r[k]) == str(v) for k, v in keys.items())]
    assert len(found) == 1, keys
    return found[0]


def build(name, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as dates
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 11, 'axes.labelsize': 10,
                         'xtick.labelsize': 9, 'ytick.labelsize': 9.5,
                         'svg.fonttype': 'none', 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.edgecolor': '#8996a1'})

    def panel(height):
        fig, ax = plt.subplots(figsize=(WIDTH, height))
        return fig, ax

    def box(ax, xy, width, height, text, accent=False, size=10):
        from matplotlib.patches import FancyBboxPatch
        x, y = xy
        ax.add_patch(FancyBboxPatch((x, y), width, height,
                     boxstyle='round,pad=0.008,rounding_size=0.015',
                     facecolor='#dff0f1' if accent else PALE,
                     edgecolor='#b9cbd7', linewidth=.8))
        ax.text(x+width/2, y+height/2, text, va='center', ha='center',
                fontsize=size, color=BLUE, linespacing=1.35)

    def arrow(ax, start, end, **kw):
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color=TEAL, lw=1.2, **kw))

    if name == 'learning_pipeline':
        fig, ax = panel(1.75)
        fig.subplots_adjust(left=.005, right=.995, top=.98, bottom=.02)
        ax.set(xlim=(0,1), ylim=(0,1)); ax.axis('off')
        for y, signal, result in [(.57, 'Relative put IV', 'Put-pair portfolio'),
                                  (.08, 'RV − IV persistence', 'Straddle portfolio')]:
            box(ax, (.01,y), .23,.34, signal)
            box(ax, (.29,y), .26,.34, 'Entry bandit\n50 / 75 / 100%')
            box(ax, (.60,y), .22,.34, result)
            arrow(ax,(.245,y+.17),(.28,y+.17));arrow(ax,(.555,y+.17),(.59,y+.17))
        box(ax, (.87,.265), .12,.47, 'Q policy\nCapital\nsplit', accent=True)
        arrow(ax,(.825,.74),(.865,.60));arrow(ax,(.825,.25),(.865,.39))

    elif name == 'signal_example':
        import pandas as pd
        d = pd.read_csv('paper/figure_data/signal_example.csv', parse_dates=['date'])
        fig, ax = panel(2.65)
        fig.subplots_adjust(left=.10,right=.985,bottom=.21,top=.78)
        x=d.date
        ax.plot(x,d.iv_spread_vol_points,color=BLUE,lw=1.5,label='AMD IV − NVDA IV')
        ax.plot(x,d.spread_20th_percentile_vol_points,color=GRAY,lw=1.1,ls='--',label='Historical 20th / 80th percentiles')
        ax.plot(x,d.spread_80th_percentile_vol_points,color=GRAY,lw=1.1,ls='--')
        ax.fill_between(x,d.iv_spread_vol_points,d.spread_80th_percentile_vol_points,
                        where=d.iv_spread_vol_points>d.spread_80th_percentile_vol_points,
                        color=TEAL,alpha=.18,interpolate=False)
        ax.fill_between(x,d.iv_spread_vol_points,d.spread_20th_percentile_vol_points,
                        where=d.iv_spread_vol_points<d.spread_20th_percentile_vol_points,
                        color=TEAL,alpha=.18,interpolate=False)
        ax.axhline(0,color=GRAY,lw=.65)
        ax.set(ylabel='IV difference\n(volatility points)',xlim=(x.min(),x.max()))
        ax.xaxis.set_major_locator(dates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(dates.DateFormatter('%b %Y'))
        ax.grid(axis='y',alpha=.15)
        fig.legend(loc='upper left',bbox_to_anchor=(.09,1),frameon=False,fontsize=9.5,ncol=1)

    elif name == 'refit_timeline':
        fig, ax = panel(2.25)
        fig.subplots_adjust(left=.01,right=.99,bottom=.02,top=.98)
        ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
        ax.text(.01,.94,'Same procedure; new completed history each month',fontsize=11,weight='bold',color=BLUE)
        for y, label, window, x, month in [(.59,'Jan 2022 fit','Jul 2020–Dec 2021',.23,'Jan'),
                                           (.34,'Feb 2022 fit','Aug 2020–Jan 2022',.28,'Feb'),
                                           (.09,'Mar 2022 fit','Sep 2020–Feb 2022',.33,'Mar')]:
            ax.text(.01,y+.085,label,va='center',fontsize=10)
            box(ax,(x,y),.49,.17,window,size=10)
            box(ax,(x+.505,y),.095,.17,month,accent=True,size=10)
            arrow(ax,(x+.493,y+.085),(x+.503,y+.085))
        ax.text(.55,.85,'18-month fit window',ha='center',color=GRAY,fontsize=9.5)
        ax.text(.88,.85,'New entries',ha='center',color=TEAL,fontsize=9.5)

    elif name == 'feature_validation':
        d=[r for r in rows('reports/p25_validation_split/validation_candidates.csv') if r['family']=='features']
        d.sort(key=lambda r:(-float(r['validation_sharpe']),r['candidate']))
        labels={'control':'Baseline: 27 inputs','iv_factor':'IV factors','term_structure':'Term structure',
                'variance_premium':'EWMA variance spread','volume_oi':'Volume / open interest',
                'smile_shape':'Smile shape','physical_tail':'Physical tails','smile_price':'Smile pricing',
                'employee':'Employee sentiment','rv_risk':'RV risk','transcripts':'Transcripts'}
        fig,ax=panel(3.5);fig.subplots_adjust(left=.32,right=.91,top=.97,bottom=.16)
        for i,r in enumerate(d):
            value=float(r['validation_sharpe']);c=TEAL if r['name']=='control' else GRAY
            ax.plot([0,value],[i,i],color=c,alpha=.25,lw=2)
            ax.plot(value,i,'o',color=c,ms=6 if r['name']=='control' else 4)
            ax.text(value+.009,i,f'{value:.3f}',va='center',fontsize=9.5,color=c)
        ax.set_yticks(range(len(d)),[labels[r['name']] for r in d]);ax.invert_yaxis()
        ax.set(xlim=(0,.66),xlabel='Annualized excess Sharpe: 2018–2021 validation')
        ax.set_xticks([0,.2,.4,.6]);ax.tick_params(axis='y',length=0)
        ax.spines['left'].set_visible(False);ax.grid(axis='x',alpha=.15)

    elif name == 'entry_policy':
        fig,ax=panel(3.35);fig.subplots_adjust(left=.01,right=.99,bottom=.02,top=.98)
        ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
        for x,w,t in [(.01,.24,'27 signal-date inputs'),(.32,.29,'Regression tree\nAt most eight states'),(.68,.30,'One size per state\n50%, 75% or 100%')]:
            box(ax,(x,.77),w,.19,t,size=10)
        arrow(ax,(.255,.865),(.31,.865));arrow(ax,(.615,.865),(.67,.865))
        ax.text(.01,.69,'Illustrative candidate maps (three states shown)',fontsize=9.5,color=GRAY)
        xs=[.045,.18,.31,.44]
        for y,row in [(.59,['Map','State 1','State 2','State 3']),(.47,['A','75%','50%','100%']),
                      (.35,['B','50%','75%','100%']),(.23,['C','100%','50%','75%'])]:
            if y==.35:ax.add_patch(plt.Rectangle((.01,y-.052),.49,.108,facecolor='#dff0f1',edgecolor='none'))
            for x,t in zip(xs,row):ax.text(x,y,t,ha='center',va='center',fontsize=10,weight='bold' if y==.59 else 'normal',color=BLUE)
        box(ax,(.58,.40),.40,.21,'Score 192 maps using\ncombined daily-profit Sharpe',size=10)
        box(ax,(.58,.13),.40,.19,'Best 19 maps update\nsize probabilities',accent=True,size=10)
        arrow(ax,(.51,.50),(.57,.50));arrow(ax,(.78,.39),(.78,.33))
        ax.plot([.78,.78,.25,.25],[.12,.055,.055,.16],color=TEAL,lw=1)
        arrow(ax,(.25,.14),(.25,.185))
        ax.text(.51,.055,'Repeat for 50 rounds',ha='center',va='center',fontsize=9.5,color=TEAL,
                bbox=dict(facecolor='white',edgecolor='none',pad=3))

    elif name == 'entry_curves':
        import pandas as pd
        fig,axes=plt.subplots(1,2,figsize=(WIDTH,2.95))
        fig.subplots_adjust(left=.085,right=.985,bottom=.20,top=.76,wspace=.32)
        books=[('Put pairs','artifacts/contracts/p25_validation_split/replays/screened_selected','artifacts/paper_code/replays/fixed100',1e3,'Profit above cash ($000)'),
               ('ATM straddles','artifacts/contracts/straddle_policy_refinement/cases/performance60/replays/tree18','artifacts/contracts/atm_rv_straddle_full/replays/straddle100',1e6,'Profit above cash ($m)')]
        for ax,(title,bandit,fixed,scale,ylabel) in zip(axes,books):
            for folder,color,label,ls in [(fixed,GRAY,'Fixed 100%','--'),(bandit,TEAL,'Bandit','-')]:
                b=pd.read_parquet(Path(folder)/'daily_nav.parquet',columns=['nav','cash_benchmark'],use_threads=False)
                delta=b.nav-b.cash_benchmark
                v=(delta.loc['2022-01-01':]-delta.loc[:'2021-12-31'].iloc[-1])/scale
                ax.plot(v.index,v,color=color,lw=1.6,label=label,ls=ls)
                ax.annotate(f'{v.iloc[-1]:,.0f}' if scale==1e3 else f'{v.iloc[-1]:.2f}',
                            (v.index[-1],v.iloc[-1]),xytext=(-3,8 if label=='Bandit' else -13),
                            textcoords='offset points',ha='right',color=color,fontsize=9.5)
            ax.set_title(title,loc='left',pad=8);ax.set_ylabel(ylabel)
            ax.axhline(0,color=GRAY,lw=.6);ax.grid(axis='y',alpha=.15)
            ax.xaxis.set_major_locator(dates.YearLocator());ax.xaxis.set_major_formatter(dates.DateFormatter('%Y'))
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,frameon=False,ncol=2,loc='upper left',bbox_to_anchor=(.07,1))

    elif name == 'entry_explanations':
        fig,axes=plt.subplots(1,2,figsize=(WIDTH,2.95))
        fig.subplots_adjust(left=.21,right=.97,bottom=.26,top=.82,wspace=1.05)
        a=rows('reports/allocation_interpretability_099/put_tree_usage.csv')
        b=rows('reports/retained_entry_explanations/straddle_feature_importance.csv')
        groups=[([(k,t) for k,t in [('profit_365d','365-day profit'),('iv_mean','Mean IV'),('commission_vol_points','Cost / vega'),('correlation','Correlation')]],
                 {r['feature']:100*float(r['path_share']) for r in a if r['period']=='evaluation'},'Put-pair tree','Visited nodes (%)',BLUE),
                ([('iv','IV'),('vix_change21','VIX change'),('normalized_memory','Normalized\nRV − IV'),('vix','VIX level')],
                 {r['feature']:100*float(r['center_ablation_size_flip_rate']) for r in b if r['period']=='evaluation'},'Straddle sizing','Sizes changed (%)',TEAL)]
        for ax,(names,values,title,xlabel,color) in zip(axes,groups):
            vals=[values[k] for k,_ in names]
            ax.barh(range(4),vals,color=color,height=.52)
            for i,v in enumerate(vals):ax.text(v+.6,i,f'{v:.1f}',va='center',fontsize=9.5,color=color)
            ax.set_yticks(range(4),[label for _,label in names]);ax.invert_yaxis()
            ax.set(xlim=(0,max(vals)*1.28),xlabel=xlabel);ax.set_title(title,loc='left',pad=10)
            ax.grid(axis='x',alpha=.12);ax.set_axisbelow(True);ax.tick_params(axis='y',length=0)

    elif name == 'allocation_risk':
        q=rows('reports/signal_allocation_variance/results.csv')
        fixed=rows('reports/signal_allocation_variance/fixed_results.csv')
        fig,axes=plt.subplots(1,2,figsize=(WIDTH,3.7),sharey=True)
        fig.subplots_adjust(left=.085,right=.975,bottom=.19,top=.81,wspace=.19)
        for ax,period,title in zip(axes,['validation','evaluation'],['Validation: 2018–2021','Evaluation: 2022–2025*']):
            f=sorted([r for r in fixed if r['period']==period],key=lambda r:float(r['straddle_target']))
            d=sorted([r for r in q if r['period']==period],key=lambda r:float(r['variance_penalty']))
            ax.plot([float(r['profit_above_cash'])/1e6 for r in f],[float(r['annualized_excess_sharpe']) for r in f],
                    'o--',color=GRAY,lw=1.2,ms=4,label='Fixed capital splits',zorder=2)
            ax.plot([float(r['profit_above_cash'])/1e6 for r in d],[float(r['annualized_excess_sharpe']) for r in d],
                    'o-',color=TEAL,lw=1.6,ms=4.5,label='Dynamic Q',zorder=3)
            label_shifts={'validation':[(9,-18),(-9,12),(0,-17),(-10,-25),(-15,-18),(0,-39),(13,-18)],
                          'evaluation':[(-7,-18),(0,12),(-7,-20),(-13,-19),(0,10),(-5,-33),(9,-16)]}
            for i,r in enumerate(f):
                x=float(r['profit_above_cash'])/1e6;y=float(r['annualized_excess_sharpe'])
                ax.annotate(f'{100*float(r["straddle_target"]):g}%',(x,y),xytext=label_shifts[period][i],
                            textcoords='offset points',ha='center',color=GRAY,fontsize=8.5,
                            arrowprops=dict(arrowstyle='-',color=GRAY,lw=.6))
            shifts={'validation':{0.:(8,12),.1:(7,12),.2:(-46,15),1.:(-47,15)},
                    'evaluation':{0.:(8,-12),.1:(8,-8),.2:(8,7),1.:(7,7)}}
            for r in d:
                risk=float(r['variance_penalty']);x=float(r['profit_above_cash'])/1e6;y=float(r['annualized_excess_sharpe'])
                ax.annotate(f'λ = {risk:g}',(x,y),xytext=shifts[period][risk],textcoords='offset points',fontsize=9.5,color=TEAL,
                            arrowprops=dict(arrowstyle='-',color=TEAL,lw=.65))
            ax.set_title(title,loc='left',fontsize=10.5,pad=10)
            ax.set(xlim=(0,2.7),ylim=(0,2.5),xlabel='Profit above cash ($m)')
            ax.set_xticks([0,1,2]);ax.grid(axis='y',alpha=.15)
        axes[0].set_ylabel('Annualized excess Sharpe')
        fig.legend(*axes[0].get_legend_handles_labels(),frameon=False,ncol=2,loc='upper left',bbox_to_anchor=(.07,1))

    elif name == 'allocation_schedule':
        import pandas as pd
        d=pd.read_csv('reports/signal_allocation_variance/decisions.csv',parse_dates=['date'])
        fig,axes=plt.subplots(2,1,figsize=(WIDTH,2.5),sharex=True)
        fig.subplots_adjust(left=.13,right=.985,bottom=.20,top=.96,hspace=.28)
        for ax,risk,color,label in zip(axes,[0.,.2],[RED,TEAL],['λ = 0','λ = 0.20']):
            sub=d.loc[(d.variance_penalty==risk)&(d.date>='2022-01-01')]
            ax.fill_between(sub.date,0,100*sub.straddle_weight,step='post',color=color,alpha=.30)
            ax.step(sub.date,100*sub.straddle_weight,where='post',color=color,lw=1.1)
            ax.set(ylabel=label+'\nStraddle %',ylim=(-5,105),yticks=[0,100])
            ax.grid(axis='x',alpha=.15);ax.tick_params(axis='x',length=0)
        axes[-1].xaxis.set_major_locator(dates.YearLocator());axes[-1].xaxis.set_major_formatter(dates.DateFormatter('%Y'))

    elif name == 'conditional_risk':
        d=rows('reports/allocation_interpretability_099/q_switch_conditional_results.csv')
        d=[one(d,period='2022-2025',source_strategy='straddle',chosen_allocation=a) for a in ['straddle_only','put_only']]
        fig,axes=plt.subplots(1,2,figsize=(WIDTH,2.8))
        fig.subplots_adjust(left=.075,right=.985,bottom=.25,top=.82,wspace=.29)
        for ax,col,title in zip(axes,['mean_daily_excess_bps','daily_excess_volatility_bps'],['Straddle mean excess return','Straddle daily volatility']):
            values=[float(r[col]) for r in d]
            ax.bar([0,1],values,color=[TEAL,GRAY],width=.52)
            for i,v in enumerate(values):ax.text(i,v+max(values)*.035,f'{v:.2f}',ha='center',fontsize=10)
            ax.set_xticks([0,1],['Q chose\nstraddles','Q chose\nput pairs'])
            ax.set(ylabel='Basis points per day',ylim=(0,max(values)*1.25))
            ax.set_title(title,loc='left',fontsize=10.5);ax.grid(axis='y',alpha=.13);ax.set_axisbelow(True)

    elif name == 'factorial':
        d=rows('reports/signal_learning_factorial/four_way.csv')
        fig,axes=plt.subplots(1,2,figsize=(WIDTH,3.0))
        fig.subplots_adjust(left=.17,right=.985,bottom=.28,top=.82,wspace=.25)
        cells=[['Fixed entry / fixed allocation','Fixed entry / Q allocation'],['Bandit entry / fixed allocation','Bandit entry / Q allocation']]
        for ax,period,title in zip(axes,['validation','evaluation'],['Validation: 2018–2021','Evaluation: 2022–2025*']):
            values=np.array([[float(one(d,cell=c,period=period,comparison_risk='0.2')['annualized_excess_sharpe']) for c in row] for row in cells])
            ax.imshow(values,cmap='Blues',vmin=0,vmax=2,aspect='auto')
            for i in range(2):
                for j in range(2):ax.text(j,i,f'{values[i,j]:.3f}',ha='center',va='center',fontsize=16,
                                           color='white' if values[i,j]>1.05 else BLUE,weight='bold')
            ax.set_xticks([0,1],['Fixed 50/50','Learned Q']);ax.set_yticks([0,1],['Fixed 100%','Bandit'])
            ax.set_title(title,loc='left',fontsize=10.5,pad=10);ax.set_xlabel('Capital allocation',labelpad=8)
            ax.tick_params(length=0,pad=8)
            for spine in ax.spines.values():spine.set_visible(False)
            ax.set_xticks([.5],minor=True);ax.set_yticks([.5],minor=True);ax.grid(which='minor',color='white',lw=3);ax.tick_params(which='minor',length=0)
        axes[0].set_ylabel('Entry sizing',labelpad=12)
        axes[1].tick_params(labelleft=False)
    else:
        raise ValueError(name)

    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf','svg'):
        fig.savefig(destination/f'{name}.{ext}',dpi=170,facecolor='white')
    plt.close(fig)
