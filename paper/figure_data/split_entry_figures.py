"""Render separate strategy panels from the original, saved plotting values.

Run with Python; no simulations or model fitting are performed.
"""
from pathlib import Path
import csv
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as dates

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / 'figures'
BLUE, TEAL, GRAY, RED = '#244b70', '#087f91', '#7d8791', '#ad554c'
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                     'axes.spines.top':False, 'axes.spines.right':False})

def rows(name):
    with (ROOT/name).open() as f:
        return list(csv.DictReader(f))

def save(fig, name):
    fig.savefig(OUT/(name+'.pdf'), facecolor='white')
    plt.close(fig)

d = rows('entry_curves.csv')
x = [datetime.fromisoformat(r['date']) for r in d]
for strategy, scale, unit in [('put', 1e3, '$000'), ('straddle', 1e6, '$m')]:
    if strategy == 'put':
        fig, ax = plt.subplots(figsize=(7.1,2.8))
        fig.subplots_adjust(left=.14,right=.97,bottom=.20,top=.82)
    else:
        fig, (ax, extra) = plt.subplots(2,1,figsize=(7.1,4.3), sharex=True,
                                        gridspec_kw={'height_ratios':[1.5,1]})
        fig.subplots_adjust(left=.14,right=.97,bottom=.13,top=.88,hspace=.15)
    for suffix, label, color, ls in [('fixed100','Fixed 100%',GRAY,'--'),
                                      ('bandit','Learned sizing',TEAL,'-')]:
        values = [float(r[strategy+'_'+suffix])/scale for r in d]
        ax.plot(x,values,label=label,color=color,ls=ls,lw=1.5)
    ax.set_ylabel('Profit above cash ('+unit+')')
    ax.axhline(0,color=GRAY,lw=.7); ax.grid(axis='y',alpha=.15)
    ax.legend(frameon=False,ncol=2,loc='lower left',bbox_to_anchor=(0,1.01))
    bottom = ax
    if strategy == 'straddle':
        gap=[float(r['straddle_bandit_minus_fixed100'])/1e3 for r in d]
        extra.plot(x,gap,color=BLUE,lw=1.3)
        extra.fill_between(x,0,gap,where=[v>=0 for v in gap],color=TEAL,alpha=.18,interpolate=True)
        extra.fill_between(x,0,gap,where=[v<0 for v in gap],color=RED,alpha=.25,interpolate=True)
        extra.axhline(0,color=GRAY,lw=.7);extra.grid(axis='y',alpha=.15)
        extra.set_ylabel('Learned minus fixed ($000)');bottom=extra
    bottom.xaxis.set_major_locator(dates.YearLocator())
    bottom.xaxis.set_major_formatter(dates.DateFormatter('%Y'))
    save(fig,strategy+'_entry_curves')

for strategy, source, measure, names, xlabel, color in [
    ('put','put_tree_usage.csv','path_share',
     [('profit_365d','365-day profit'),('iv_mean','Mean IV'),
      ('commission_vol_points','Cost / vega'),('correlation','Correlation')],
     'Share of visited tree nodes (%)',BLUE),
    ('straddle','straddle_feature_importance.csv','center_ablation_size_flip_rate',
     [('iv','IV'),('vix_change21','VIX change'),
      ('normalized_memory','Normalized RV minus IV'),('vix','VIX level')],
     'Proposal sizes changed by centering one input (%)',TEAL)]:
    values={r['feature']:100*float(r[measure]) for r in rows(source) if r['period']=='evaluation'}
    vals=[values[k] for k,_ in names]
    fig,ax=plt.subplots(figsize=(7.1,2.5))
    fig.subplots_adjust(left=.30,right=.97,bottom=.25,top=.96)
    ax.barh(range(len(vals)),vals,color=color,height=.52)
    for i,v in enumerate(vals):ax.text(v+.5,i,f'{v:.1f}',va='center',color=color)
    ax.set_yticks(range(len(vals)),[label for _,label in names]);ax.invert_yaxis()
    ax.set(xlim=(0,max(vals)*1.22),xlabel=xlabel)
    ax.grid(axis='x',alpha=.12);ax.set_axisbelow(True);ax.tick_params(axis='y',length=0)
    save(fig,strategy+'_entry_explanations')
