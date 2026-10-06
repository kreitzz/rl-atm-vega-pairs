"""One feature definition and bounded action search for all three wings."""
import numpy as np
import pandas as pd

SIZES=np.array([0.,-1.,-.75,-.5,-.25,.25,.5,.75,1.])
CORE=['iv_mean','iv_gap_relative','iv_rank']
OPTIONAL=['vrp_gap_relative','rv_mean','beta','correlation','dte_mean','dte_gap','commission_vol_points','vix']+[
    f'{prefix}_{side}_{days}d' for days in [91,365] for side in ['short','long'] for prefix in ['profit','count']]
FEATURES=CORE+OPTIONAL+[c+'_available' for c in OPTIONAL]
VARIANTS=['spread_full','spread_floor','spread_flex','memory_full','memory_floor','memory_flex',
          'rl_entry_fixed','rl_entry_sized','rl_entry_sized_full']
TRAINED=['spread_floor','spread_flex','memory_floor','memory_flex','rl_entry_fixed','rl_entry_sized']


def states(a,b,returns,screen,cal,memory,vix,fee):
    a=a.set_index('date').reindex(cal);b=b.set_index('date').reindex(cal)
    ia=a.bs_iv.where(a.bs_iv.gt(0));ib=b.bs_iv.where(b.bs_iv.gt(0));mean=(ia+ib)/2;gap=ia-ib
    rank=gap.rolling(252,min_periods=63).rank(pct=True)
    rv2=returns.reindex(cal).pow(2).rolling(20,min_periods=10).mean()*252
    x=pd.DataFrame(index=cal)
    x['iv_mean']=mean;x['iv_gap_relative']=gap/mean;x['iv_rank']=rank.fillna(.5)
    x['observed']=np.isfinite(ia)&np.isfinite(ib)
    x['rule_direction']=np.where(rank.le(.2),1,np.where(rank.gt(.8),-1,0))
    x['vrp_gap_relative']=((ia**2-rv2.iloc[:,0])-(ib**2-rv2.iloc[:,1]))/mean**2
    x['rv_mean']=np.sqrt(rv2).mean(axis=1,skipna=False)
    x['beta']=screen.beta;x['correlation']=screen.correlation;x['eligible']=screen.eligible.fillna(False)
    da=(pd.to_datetime(a.expiry)-cal.to_series()).dt.days;db=(pd.to_datetime(b.expiry)-cal.to_series()).dt.days
    x['dte_mean']=(da+db)/2;x['dte_gap']=da-db
    x['commission_vol_points']=fee/(100*a.bs_vega.where(a.bs_vega.gt(0)))+fee/(100*b.bs_vega.where(b.bs_vega.gt(0)))
    x['vix']=vix.reindex(cal)/100
    for days in [91,365]:
        for side in ['short','long']:
            count=memory[f'count_{side}_{days}d'];x[f'count_{side}_{days}d']=count
            x[f'profit_{side}_{days}d']=(memory[f'profit_{side}_{days}d']/100).where(count.gt(0))
    x=x.replace([np.inf,-np.inf],np.nan)
    for c in OPTIONAL:x[c+'_available']=x[c].notna().astype(float);x[c]=x[c].fillna(0)
    assert np.isfinite(x.loc[x.observed,FEATURES]).all().all()
    return x


def memory_ok(rows,side):
    return (rows[f'count_{side}_91d'].ge(21)&rows[f'count_{side}_365d'].ge(63)&
            rows[f'profit_{side}_91d'].gt(0)&rows[f'profit_{side}_365d'].gt(0))


def candidate_mask(rows,variant):
    if variant.startswith('rl_entry'):return np.ones(len(rows),bool)
    direction=rows.rule_direction.to_numpy(int);use=direction!=0
    if variant.startswith('memory'):
        use&=np.where(direction>0,memory_ok(rows,'long'),memory_ok(rows,'short'))
    return use


def action_menu(variant):
    if variant=='rl_entry_fixed':return np.array([0,1,8])
    if variant=='rl_entry_sized':return np.array([0,1,2,3,6,7,8])
    if variant.endswith('floor'):return np.array([2,3,4])
    if variant.endswith('flex'):return np.arange(5)
    raise ValueError(variant)


def reward_choices(labels,direction,variant):
    if variant.startswith('rl_entry'):return labels[:,action_menu(variant)]
    menu=action_menu(variant);out=np.zeros((len(labels),len(menu),labels.shape[-1]),labels.dtype)
    for j,size in enumerate(menu):
        if size:out[:,j]=labels[np.arange(len(labels)),np.where(direction>0,4+size,5-size)]
    return out


def choose(book,menu):
    ns=book.shape[1]//len(menu);na=len(menu);mean=book.mean(axis=0);cov=np.cov(book,rowvar=False,ddof=1)
    def scores(choice):
        cols=choice+np.arange(ns)[None,:]*na;m=mean[cols].sum(axis=1)
        v=cov[cols[:,:,None],cols[:,None,:]].sum(axis=(1,2))
        return np.sqrt(252)*np.divide(m,np.sqrt(np.maximum(v,0)),out=np.zeros_like(m),where=v>1e-18)
    rng=np.random.default_rng(2026);prob=np.full((ns,na),1/na);best=np.zeros(ns,int)
    anchors=np.tile(np.arange(na)[:,None],(1,ns));initial=scores(anchors);best=anchors[initial.argmax()].copy();best_score=float(initial.max())
    for _ in range(50):
        choice=(rng.random((192,ns))[:,:,None]>prob.cumsum(axis=1)[None]).sum(axis=2).clip(max=na-1)
        choice[:na]=anchors;choice[na]=best;score=scores(choice);winner=score.argmax()
        if score[winner]>best_score:best=choice[winner].copy();best_score=float(score[winner])
        elite=choice[np.argsort(score)[-19:]];frequency=np.stack([(elite==a).mean(axis=0) for a in range(na)],axis=1)
        prob=.97*(.3*prob+.7*frequency)+.03/na
    return menu[best],dict(training_shadow_daily_sharpe=best_score,best_constant_action_training_sharpe=float(initial.max()))


def apply(model,rows,variant):
    if model['tree'] is None:
        # One global fallback for all roster pairs; no pair activation test.
        codes=np.where(rows.rule_direction>0,8,np.where(rows.rule_direction<0,1,0))
        if not variant.startswith('rl_entry'):return np.full(len(rows),4,int)
        return np.asarray(codes,int)
    leaf=np.searchsorted(model['leaves'],model['tree'].apply(rows[FEATURES].to_numpy(float)))
    return model['choices'][leaf]


def decode(values,direction,variant):
    values=np.asarray(values,int)
    if variant.startswith('rl_entry'):
        if variant=='rl_entry_sized_full':values=np.where(values>=5,8,np.where(values>0,1,0))
        size=SIZES[values];return np.sign(size).astype(int),np.rint(np.abs(size)*4).astype(int)
    return np.asarray(direction,int),values
